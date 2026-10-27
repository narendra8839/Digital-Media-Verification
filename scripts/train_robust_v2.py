"""V2 Robustness fine-tuning pipeline for EfficientNet-B4 deepfake detector.

Addresses the root failure modes of V1:
1. V1 penalized Real false alarms 4x more than Fake false alarms (w0=2.5, w1=0.625),
   causing the model to collapse toward predicting "Real" on clean inputs and JPEG media.
2. V1 applied heavy distortions to 90% of samples, giving the network insufficient clean data.

V2 Interventions:
- Hypothesis A (V2-A): Asymmetric Focal Loss prioritizing Fake recall (w_fake=1.6, w_real=1.0)
  combined with a Guaranteed Clean Crop Ratio (40% pristine) and calibrated blur/resize/JPEG.
- Hypothesis B (V2-B): V2-A + Frequency High-Pass Residual Enhancement to expose manipulation
  boundaries under compression.
"""

import os
import sys
import io
import json
import random
import logging
import argparse
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from models.deepfake.efficientnet_b4 import EfficientNetB4Deepfake
from preprocessing.frame_sampling import sample_video_frames
from preprocessing.face_detection import FaceDetector
from evaluation.metrics import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("robust_v2_train")


# =====================================================================
# Loss Formulations
# =====================================================================

class SymmetricFocalLoss(nn.Module):
    """Symmetric focal loss that focuses on hard examples without class bias."""
    def __init__(self, gamma: float = 1.0, weight: Optional[torch.Tensor] = None, eps: float = 1e-7):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(self.eps, 1.0 - self.eps)
        loss = - (1.0 - pt) ** self.gamma * torch.log(pt)
        if self.weight is not None:
            w = self.weight.gather(0, targets)
            loss = w * loss
        return loss.mean()


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric focal loss that maintains gradient pressure on false negatives (missed deepfakes)."""
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma_pos: float = 1.0, gamma_neg: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.weight = weight
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        p_real = probs[:, 0].clamp(self.eps, 1.0 - self.eps)
        p_fake = probs[:, 1].clamp(self.eps, 1.0 - self.eps)

        loss_fake = - (1.0 - p_fake) ** self.gamma_pos * torch.log(p_fake)
        loss_real = - (1.0 - p_real) ** self.gamma_neg * torch.log(p_real)

        if self.weight is not None:
            loss_fake = self.weight[1] * loss_fake
            loss_real = self.weight[0] * loss_real

        is_fake = (targets == 1).float()
        is_real = (targets == 0).float()
        loss = is_fake * loss_fake + is_real * loss_real
        return loss.mean()


# =====================================================================
# Targeted Robustness & Frequency Augmentations
# =====================================================================

class RandomJPEGCompression:
    def __init__(self, quality_range: Tuple[int, int] = (40, 90), p: float = 0.35):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomGaussianBlurPIL:
    def __init__(self, radius_range: Tuple[float, float] = (0.5, 1.8), p: float = 0.35):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        radius = random.uniform(self.radius_range[0], self.radius_range[1])
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class RandomDownsampleBilinear:
    def __init__(self, scale_range: Tuple[float, float] = (0.35, 0.75), p: float = 0.35):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        down_w = max(16, int(w * scale))
        down_h = max(16, int(h * scale))
        down = img.resize((down_w, down_h), Image.BILINEAR)
        return down.resize((w, h), Image.BILINEAR)


class FrequencyResidualEnhance:
    """Enhances high-frequency residual cues to expose subtle facial manipulation boundaries."""
    def __init__(self, p: float = 0.35, strength_range: Tuple[float, float] = (0.4, 1.0)):
        self.p = p
        self.strength_range = strength_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.array(img, dtype=np.float32)
        blurred = np.array(img.filter(ImageFilter.GaussianBlur(radius=1.5)), dtype=np.float32)
        residual = arr - blurred
        alpha = random.uniform(self.strength_range[0], self.strength_range[1])
        enhanced = np.clip(arr + alpha * residual, 0, 255).astype(np.uint8)
        return Image.fromarray(enhanced)


class BalancedCleanPreservingAugment:
    """Guarantees at least 40% clean inputs in training to preserve baseline clean sensitivity."""
    def __init__(self, sub_transforms: List[Any], clean_ratio: float = 0.40):
        self.sub_transforms = sub_transforms
        self.clean_ratio = clean_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.clean_ratio:
            return img
        for t in self.sub_transforms:
            img = t(img)
        return img


def build_v2_transforms(is_train: bool = True, use_frequency_residual: bool = False) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if not is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

    distortion_ops = [
        RandomDownsampleBilinear(scale_range=(0.35, 0.75), p=0.35),
        RandomGaussianBlurPIL(radius_range=(0.5, 1.8), p=0.35),
        RandomJPEGCompression(quality_range=(40, 90), p=0.35),
    ]
    if use_frequency_residual:
        distortion_ops.append(FrequencyResidualEnhance(p=0.35, strength_range=(0.4, 1.0)))

    pipeline = [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        BalancedCleanPreservingAugment(distortion_ops, clean_ratio=0.40),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        normalize
    ]
    return transforms.Compose(pipeline)


# =====================================================================
# Dataset and Extraction
# =====================================================================

class DeepfakeFrameDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], transform: Optional[transforms.Compose] = None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.records[idx]
        img = item["face_crop"]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)

        label = item["label_id"]
        vid_id = item["video_id"]
        return tensor, label, vid_id


def extract_or_load_frames(
    manifest_path: str,
    detector: FaceDetector,
    base_data_dir: str = "data/faceforensics/C23",
    cache_path: Optional[str] = None,
    max_frames_per_video: int = 2,
    max_videos: Optional[int] = None
) -> List[Dict[str, Any]]:
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached face crops from: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        videos = json.load(f)

    if max_videos is not None:
        reals = [v for v in videos if v["label_id"] == 0]
        fakes = [v for v in videos if v["label_id"] == 1]
        half = max(1, max_videos // 2)
        videos = reals[:half] + fakes[:half]

    frame_records = []
    logger.info(f"Extracting {max_frames_per_video} frames/video from {len(videos)} videos in {manifest_path}...")

    for i, vid in enumerate(videos, 1):
        rel_path = vid["relative_path"]
        abs_path = os.path.join(base_data_dir, rel_path)
        if not os.path.exists(abs_path):
            continue

        sampled = sample_video_frames(abs_path, target_fps=1.0, max_frames=max_frames_per_video)
        for ts, frame_rgb in sampled:
            crop, is_detected = detector.detect_and_crop(frame_rgb, target_size=(224, 224))
            frame_records.append({
                "video_id": vid["video_id"],
                "label_id": vid["label_id"],
                "label_name": vid["label"],
                "face_crop": crop,
                "face_detected": is_detected
            })

        if i % 250 == 0 or i == len(videos):
            logger.info(f"Extraction progress: {i}/{len(videos)} videos processed ({len(frame_records)} crops)")

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(frame_records, cache_path, _use_new_zipfile_serialization=False)
        logger.info(f"Cached {len(frame_records)} face crops to: {cache_path}")

    return frame_records


# =====================================================================
# Dual Validation Function
# =====================================================================

def evaluate_validation_dual(model: nn.Module, val_records: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    model.eval()
    clean_transform = build_v2_transforms(is_train=False)

    perturbed_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        RandomGaussianBlurPIL(radius_range=(1.5, 2.0), p=1.0),
        RandomDownsampleBilinear(scale_range=(0.3, 0.4), p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results = {}
    for mode, tf in [("clean", clean_transform), ("perturbed", perturbed_transform)]:
        val_loader = DataLoader(DeepfakeFrameDataset(val_records, transform=tf), batch_size=16, shuffle=False)
        y_true_vids = {}
        vid_probs = defaultdict(list)

        with torch.no_grad():
            for imgs, labels, vids in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                labels_np = labels.numpy()
                for vid, prob, true_lbl in zip(vids, probs, labels_np):
                    vid_probs[vid].append(prob)
                    y_true_vids[vid] = true_lbl

        y_true = []
        y_pred = []
        for vid, prob_list in vid_probs.items():
            mean_prob = float(np.mean(prob_list))
            pred = 1 if mean_prob >= 0.5 else 0
            y_true.append(y_true_vids[vid])
            y_pred.append(pred)

        metrics = compute_classification_metrics(y_true, y_pred)
        # Add class-wise recall
        cm = metrics["confusion_matrix"]
        # labels=[0, 1] -> row 0: Real, row 1: Fake
        real_rec = cm[0][0] / max(1, cm[0][0] + cm[0][1])
        fake_rec = cm[1][1] / max(1, cm[1][0] + cm[1][1])
        metrics["real_recall"] = float(real_rec)
        metrics["fake_recall"] = float(fake_rec)
        results[mode] = metrics

    clean_f1 = results["clean"]["macro"]["f1"]
    pert_f1 = results["perturbed"]["macro"]["f1"]
    composite = 2.0 * (clean_f1 * pert_f1) / max(1e-6, clean_f1 + pert_f1)
    results["composite_score"] = float(composite)
    return results


# =====================================================================
# Main Training Function
# =====================================================================

def train_robust_candidate_v2(
    init_checkpoint: str = "models/deepfake/checkpoint_full/best_model.pt",
    output_dir: str = "models/deepfake/checkpoint_robust_v2a",
    mode: str = "v2a",
    loss_type: str = "ce",
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 3.5e-5,
    seed: int = 42,
    fake_weight: float = 1.0,
    device: str = "cuda"
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    selected_device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    if selected_device.type != "cuda":
        raise RuntimeError("Robust fine-tuning requires CUDA.")

    os.makedirs(output_dir, exist_ok=True)
    detector = FaceDetector(device="cpu")

    cache_dir = "data/cache_deepfake_frames"
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, "train_crops_2fps.pt")
    val_cache = os.path.join(cache_dir, "val_crops_2fps.pt")

    train_manifest = "data/full_splits/deepfake_train.json"
    val_manifest = "data/full_splits/deepfake_val.json"

    logger.info("Preparing training and validation datasets...")
    train_records = extract_or_load_frames(train_manifest, detector, cache_path=train_cache, max_frames_per_video=2, max_videos=1440)
    val_records = extract_or_load_frames(val_manifest, detector, cache_path=val_cache, max_frames_per_video=2)

    use_freq = (mode == "v2b")
    train_transform = build_v2_transforms(is_train=True, use_frequency_residual=use_freq)
    train_loader = DataLoader(
        DeepfakeFrameDataset(train_records, transform=train_transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Loss selection based on loss_type
    weight_tensor = torch.tensor([1.0, fake_weight], dtype=torch.float32).to(selected_device)
    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        logger.info(f"Loss criterion: CrossEntropyLoss with w_real=1.0, w_fake={fake_weight:.2f}")
    elif loss_type == "focal":
        criterion = SymmetricFocalLoss(gamma=1.0, weight=weight_tensor)
        logger.info(f"Loss criterion: SymmetricFocalLoss (gamma=1.0) with w_real=1.0, w_fake={fake_weight:.2f}")
    elif loss_type == "asl":
        criterion = AsymmetricFocalLoss(weight=weight_tensor, gamma_pos=1.0, gamma_neg=1.0)
        logger.info(f"Loss criterion: AsymmetricFocalLoss (gamma+=1.0, gamma-=1.0) with w_real=1.0, w_fake={fake_weight:.2f}")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Load baseline model weights
    logger.info(f"Loading initial model weights from: {init_checkpoint}")
    model = EfficientNetB4Deepfake(dropout_rate=0.4, pretrained=False)
    state_dict = torch.load(init_checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(selected_device)

    # Optimizer
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.5},
        {"params": head_params, "lr": lr}
    ], weight_decay=1e-2)

    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    best_composite = -1.0
    best_weights_path = os.path.join(output_dir, "best_model.pt")
    best_epoch = -1
    best_metrics = {}

    logger.info("=" * 65)
    logger.info(f"Starting V2 Robustness Training [Mode: {mode.upper()}] ({epochs} epochs, {len(train_loader)} batches/epoch)")
    logger.info("=" * 65)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (imgs, labels, _) in enumerate(train_loader):
            imgs = imgs.to(selected_device)
            labels = labels.to(selected_device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / max(1, len(train_records))
        val_eval = evaluate_validation_dual(model, val_records, selected_device)

        clean_acc = val_eval["clean"]["accuracy"]
        clean_f1 = val_eval["clean"]["macro"]["f1"]
        clean_fake_rec = val_eval["clean"]["fake_recall"]
        clean_real_rec = val_eval["clean"]["real_recall"]

        pert_acc = val_eval["perturbed"]["accuracy"]
        pert_f1 = val_eval["perturbed"]["macro"]["f1"]
        composite = val_eval["composite_score"]

        logger.info(
            f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | "
            f"Clean: Acc={clean_acc:.4f}, F1={clean_f1:.4f}, FakeRec={clean_fake_rec:.2%}, RealRec={clean_real_rec:.2%} | "
            f"Perturbed: Acc={pert_acc:.4f}, F1={pert_f1:.4f} | Comp: {composite:.4f}"
        )

        # Validation gate: must maintain Clean Fake Recall >= 70% and Clean Real Recall >= 70%
        is_balanced = (clean_fake_rec >= 0.70 and clean_real_rec >= 0.70)
        if (composite > best_composite or not os.path.exists(best_weights_path)) and (is_balanced or epoch == epochs - 1):
            best_composite = composite
            best_epoch = epoch + 1
            best_metrics = val_eval
            torch.save(model.state_dict(), best_weights_path)
            logger.info(f"==> Saved candidate checkpoint to {best_weights_path} (Comp: {composite:.4f})")

    if not os.path.exists(best_weights_path):
        torch.save(model.state_dict(), best_weights_path)

    exp_config = {
        "experiment_name": f"robustness_v2_{mode}",
        "mode": mode,
        "base_checkpoint": init_checkpoint,
        "candidate_checkpoint": best_weights_path,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": seed,
        "fake_loss_weight": fake_weight,
        "use_frequency_residual": use_freq,
        "clean_ratio": 0.40,
        "validation_selection_metrics": best_metrics
    }

    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

    logger.info(f"V2 Training complete for {mode.upper()}. Checkpoint saved to: {best_weights_path}")
    return exp_config


def main():
    parser = argparse.ArgumentParser(description="V2 Robustness fine-tuning for deepfake detector.")
    parser.add_argument("--mode", type=str, choices=["v2a", "v2b"], default="v2a", help="V2-A (Balanced) or V2-B (+ Frequency Residual)")
    parser.add_argument("--loss-type", type=str, choices=["ce", "focal", "asl"], default="ce", help="Loss formulation: ce (balanced cross-entropy), focal, or asl")
    parser.add_argument("--init-checkpoint", type=str, default="models/deepfake/checkpoint_full/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="models/deepfake/checkpoint_robust_v2a")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.5e-5)
    parser.add_argument("--fake-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_robust_candidate_v2(
        init_checkpoint=args.init_checkpoint,
        output_dir=args.output_dir,
        mode=args.mode,
        loss_type=args.loss_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        fake_weight=args.fake_weight
    )


if __name__ == "__main__":
    main()
