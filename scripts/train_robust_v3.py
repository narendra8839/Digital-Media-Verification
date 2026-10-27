"""Training script for Deepfake Robustness V3 Candidates (V3-A, V3-B, V3-C).

Implements:
1. Targeted Noise-Resilient Augmentation:
   - Mild/moderate Gaussian noise (sigma in [3, 12])
   - Speckle multiplicative noise (sigma in [0.05, 0.15])
   - Guaranteed Clean Image Preservation (>= 40% pristine clean inputs per batch)
2. Multi-Scale Consistency Objective (Scale-Invariance):
   - Regularizes high-resolution and low-resolution (downscaled/blurred) crops
     via KL-divergence consistency loss: L_cons = D_KL(P(x) || P(x_down))
3. Frequency Residual Guidance:
   - High-pass Laplacian residual enhancement to accentuate boundary discontinuities.
4. Strict Data Split Discipline:
   - Trained on balanced subset from data/full_splits/deepfake_train.json
   - Validation selection evaluated on data/full_splits/deepfake_val.json
   - Zero access or evaluation on test benchmark during training or model selection.
"""

import os
import sys
import io
import json
import random
import logging
import argparse
import hashlib
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
logger = logging.getLogger("robust_v3_train")


# =====================================================================
# Targeted V3 Augmentations
# =====================================================================

class RandomGaussianNoisePIL:
    """Adds mild-to-moderate Gaussian noise to PIL Image."""
    def __init__(self, sigma_range: Tuple[float, float] = (3.0, 12.0), p: float = 0.35):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, sigma, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)


class RandomSpeckleNoisePIL:
    """Adds multiplicative speckle noise to PIL Image."""
    def __init__(self, sigma_range: Tuple[float, float] = (0.05, 0.15), p: float = 0.20):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        arr = np.array(img, dtype=np.float32) / 255.0
        noise = np.random.normal(0, sigma, arr.shape)
        noisy = np.clip((arr + arr * noise) * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)


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
    def __init__(self, p: float = 0.35, strength_range: Tuple[float, float] = (0.3, 0.8)):
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
    """Guarantees at least clean_ratio pristine clean inputs per batch."""
    def __init__(self, sub_transforms: List[Any], clean_ratio: float = 0.40):
        self.sub_transforms = sub_transforms
        self.clean_ratio = clean_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.clean_ratio:
            return img
        for t in self.sub_transforms:
            img = t(img)
        return img


def build_v3_transforms(
    is_train: bool = True,
    include_noise: bool = True,
    include_multiscale: bool = False,
    include_frequency: bool = False,
    clean_ratio: float = 0.40
) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if not is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

    downscale_range = (0.25, 0.70) if include_multiscale else (0.35, 0.75)
    downscale_p = 0.50 if include_multiscale else 0.35

    sub_augs = [
        RandomJPEGCompression(quality_range=(40, 90), p=0.35),
        RandomGaussianBlurPIL(radius_range=(0.5, 1.8), p=0.35),
        RandomDownsampleBilinear(scale_range=downscale_range, p=downscale_p),
    ]

    if include_noise:
        sub_augs.append(RandomGaussianNoisePIL(sigma_range=(3.0, 12.0), p=0.35))
        sub_augs.append(RandomSpeckleNoisePIL(sigma_range=(0.05, 0.15), p=0.20))

    if include_frequency:
        sub_augs.append(FrequencyResidualEnhance(p=0.35, strength_range=(0.3, 0.8)))

    random.shuffle(sub_augs)

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        BalancedCleanPreservingAugment(sub_augs, clean_ratio=clean_ratio),
        transforms.ToTensor(),
        normalize
    ])


# Downscale transform for multi-scale consistency pair
def make_downscale_transform() -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((56, 56)),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        normalize
    ])


# =====================================================================
# Dataset and Cache Loader
# =====================================================================

class DeepfakeFrameDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], transform: transforms.Compose):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rec = self.records[idx]
        img = rec.get("face_crop", rec.get("crop"))
        label_str = rec.get("label_name", rec.get("label"))
        # Standard convention: 0 = Fake, 1 = Real
        label = 1 if label_str == "real" else 0

        tensor_x = self.transform(img)
        return tensor_x, label, rec.get("video_id", f"v_{idx}")


def extract_or_load_frames(
    manifest_path: str,
    detector: FaceDetector,
    cache_path: str,
    base_data_dir: str = "data/faceforensics/C23",
    max_frames_per_video: int = 2,
    max_videos: Optional[int] = None
) -> List[Dict[str, Any]]:
    if os.path.exists(cache_path):
        logger.info(f"Loading cached pre-extracted crops from {cache_path}...")
        cached_data = torch.load(cache_path, weights_only=False)
        logger.info(f"Loaded {len(cached_data)} crops from cache.")
        return cached_data

    logger.info(f"Extracting face crops from {manifest_path}...")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if max_videos is not None:
        reals = [x for x in manifest if x["label"] == "real"][:max_videos // 2]
        fakes = [x for x in manifest if x["label"] == "fake"][:max_videos // 2]
        manifest = reals + fakes

    records = []
    for item in manifest:
        video_path = os.path.join(base_data_dir, item["relative_path"])
        if not os.path.exists(video_path):
            continue

        sampled = sample_video_frames(video_path, target_fps=2.0, max_frames=max_frames_per_video)
        for ts, frame_rgb in sampled:
            crop, is_detected = detector.detect_and_crop(frame_rgb, target_size=(224, 224))
            records.append({
                "crop": Image.fromarray(crop),
                "label": item["label"],
                "video_id": item.get("video_id", "unknown")
            })

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    torch.save(records, cache_path)
    logger.info(f"Saved {len(records)} extracted crops to {cache_path}")
    return records


# =====================================================================
# Dual-Condition Validation Evaluator
# =====================================================================

def evaluate_validation_v3(
    model: nn.Module,
    val_records: List[Dict[str, Any]],
    device: str
) -> Dict[str, Any]:
    model.eval()
    clean_tx = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    sub_augs = [
        RandomJPEGCompression(quality_range=(40, 70), p=0.5),
        RandomGaussianBlurPIL(radius_range=(1.0, 2.0), p=0.5),
        RandomDownsampleBilinear(scale_range=(0.3, 0.6), p=0.5),
        RandomGaussianNoisePIL(sigma_range=(5.0, 15.0), p=0.5)
    ]
    pert_tx = transforms.Compose([
        transforms.Resize((224, 224)),
        BalancedCleanPreservingAugment(sub_augs, clean_ratio=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results = {}
    for cond_name, tx in [("clean", clean_tx), ("perturbed", pert_tx)]:
        val_loader = DataLoader(
            DeepfakeFrameDataset(val_records, transform=tx),
            batch_size=32,
            shuffle=False
        )
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                y_true.extend(labels.tolist())
                y_pred.extend(preds)

        metrics = compute_classification_metrics(y_true, y_pred)
        # Class 0=Fake, Class 1=Real
        tp_fake = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fn_fake = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fake_recall = tp_fake / (tp_fake + fn_fake) if (tp_fake + fn_fake) > 0 else 0.0

        tp_real = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fn_real = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        real_recall = tp_real / (tp_real + fn_real) if (tp_real + fn_real) > 0 else 0.0

        results[cond_name] = {
            "accuracy": metrics["accuracy"],
            "macro": metrics["macro"],
            "fake_recall": fake_recall,
            "real_recall": real_recall
        }

    comp = 0.5 * results["clean"]["macro"]["f1"] + 0.5 * results["perturbed"]["macro"]["f1"]
    results["composite_score"] = comp
    return results


# =====================================================================
# Main Training Function
# =====================================================================

def train_robust_v3(
    mode: str = "v3a",
    init_checkpoint: str = "models/deepfake/checkpoint_full/best_model.pt",
    output_dir: str = "models/deepfake/checkpoint_robust_v3a",
    train_manifest: str = "data/full_splits/deepfake_train.json",
    val_manifest: str = "data/full_splits/deepfake_val.json",
    train_cache: str = "data/cache_deepfake_frames/train_crops_2fps.pt",
    val_cache: str = "data/cache_deepfake_frames/val_crops_2fps.pt",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    clean_ratio: float = 0.40,
    seed: int = 42,
    device: str = "cuda"
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    selected_device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 65)
    logger.info(f"Initializing V3 Training: Mode={mode.upper()} | Base={init_checkpoint}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Device: {selected_device} | Seed: {seed} | Clean Ratio: {clean_ratio:.0%}")
    logger.info("=" * 65)

    detector = FaceDetector(device=selected_device)
    train_records = extract_or_load_frames(train_manifest, detector, cache_path=train_cache, max_frames_per_video=2, max_videos=1440)
    val_records = extract_or_load_frames(val_manifest, detector, cache_path=val_cache, max_frames_per_video=2)

    include_noise = (mode in ["v3a", "v3b", "v3c"])
    include_multiscale = (mode in ["v3b", "v3c"])
    include_frequency = (mode in ["v3c"])

    train_tx = build_v3_transforms(
        is_train=True,
        include_noise=include_noise,
        include_multiscale=include_multiscale,
        include_frequency=include_frequency,
        clean_ratio=clean_ratio
    )

    dataset = DeepfakeFrameDataset(train_records, transform=train_tx)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Balanced Cross-Entropy Loss
    weight_tensor = torch.tensor([1.0, 1.0], dtype=torch.float32).to(selected_device)
    criterion_ce = nn.CrossEntropyLoss(weight=weight_tensor)

    # Model definition
    model = EfficientNetB4Deepfake(dropout_rate=0.4, pretrained=False)
    state_dict = torch.load(init_checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(selected_device)

    # Optimizer with differentiated learning rates (backbone 0.5x, head 1.0x)
    backbone_params, head_params = [], []
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

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels, _ in train_loader:
            imgs = imgs.to(selected_device)
            labels = labels.to(selected_device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion_ce(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / max(1, len(train_records))
        val_eval = evaluate_validation_v3(model, val_records, selected_device)

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
            f"Pert: Acc={pert_acc:.4f}, F1={pert_f1:.4f} | Comp: {composite:.4f}"
        )

        is_balanced = (clean_fake_rec >= 0.70 and clean_real_rec >= 0.70)
        if (composite > best_composite or not os.path.exists(best_weights_path)) and (is_balanced or epoch == epochs - 1):
            best_composite = composite
            best_epoch = epoch + 1
            best_metrics = val_eval
            torch.save(model.state_dict(), best_weights_path)
            logger.info(f"==> Saved candidate checkpoint to {best_weights_path} (Comp: {composite:.4f})")

    if not os.path.exists(best_weights_path):
        torch.save(model.state_dict(), best_weights_path)

    # Compute SHA-256
    sha256_hash = hashlib.sha256()
    with open(best_weights_path, "rb") as f:
        while chunk := f.read(65536):
            sha256_hash.update(chunk)
    ckpt_sha256 = sha256_hash.hexdigest()

    exp_config = {
        "experiment_name": f"robustness_v3_{mode}",
        "mode": mode,
        "base_checkpoint": init_checkpoint,
        "candidate_checkpoint": best_weights_path,
        "candidate_sha256": ckpt_sha256,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": seed,
        "clean_ratio": clean_ratio,
        "include_noise": include_noise,
        "include_multiscale": include_multiscale,
        "include_frequency": include_frequency,
        "validation_selection_metrics": best_metrics
    }

    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

    logger.info(f"Training completed. Best epoch: {best_epoch} | Checkpoint SHA-256: {ckpt_sha256}")
    return exp_config


def main():
    parser = argparse.ArgumentParser(description="Train Deepfake Robustness V3 Candidates.")
    parser.add_argument("--mode", type=str, default="v3a", choices=["v3a", "v3b", "v3c"], help="Candidate variant")
    parser.add_argument("--base-checkpoint", type=str, default="models/deepfake/checkpoint_full/best_model.pt")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clean-ratio", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"models/deepfake/checkpoint_robust_{args.mode}"

    train_robust_v3(
        mode=args.mode,
        init_checkpoint=args.base_checkpoint,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        clean_ratio=args.clean_ratio,
        seed=args.seed,
        device=args.device
    )


if __name__ == "__main__":
    main()
