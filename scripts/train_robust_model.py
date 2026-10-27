"""Robustness fine-tuning pipeline for EfficientNet-B4 deepfake detector.

Key principles:
1. Controlled data augmentation applied to TRAINING DATA ONLY.
2. Training split: strictly data/full_splits/deepfake_train.json.
3. Validation split: strictly data/full_splits/deepfake_val.json for model selection.
4. TEST SPLIT IS STRICTLY ISOLATED AND NEVER TOUCHED.
5. Saves candidate checkpoint to models/deepfake/checkpoint_robust_candidate/
   (NEVER overwrites production checkpoint in models/deepfake/checkpoint_full/).
6. Applies targeted perturbations:
   - JPEG compression artifacts (quality 30 to 85)
   - Gaussian blur (simulating transmission and resolution softening)
   - Low-resolution downsampling and bilinear restoration (scale 0.25 to 0.70)
   - Additive Gaussian sensor noise (sigma 0.01 to 0.06)
   - Random horizontal flip and mild color jitter
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
logger = logging.getLogger("robust_train")


# =====================================================================
# Targeted Robustness Augmentations
# =====================================================================

class RandomJPEGCompression:
    """Simulates social media recompression across varying quality factors."""
    def __init__(self, quality_range: Tuple[int, int] = (30, 85), p: float = 0.45):
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
    """Applies realistic optical/transmission Gaussian blur."""
    def __init__(self, radius_range: Tuple[float, float] = (0.5, 2.2), p: float = 0.45):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        radius = random.uniform(self.radius_range[0], self.radius_range[1])
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class RandomDownsampleBilinear:
    """Simulates low-resolution transmission scaling followed by player display upscale."""
    def __init__(self, scale_range: Tuple[float, float] = (0.25, 0.70), p: float = 0.45):
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


class AddGaussianNoiseTensor:
    """Adds zero-mean additive Gaussian sensor noise to normalized float tensor."""
    def __init__(self, std_range: Tuple[float, float] = (0.01, 0.055), p: float = 0.35):
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        std = random.uniform(self.std_range[0], self.std_range[1])
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0.0, 1.0)


def build_robustness_transforms(is_train: bool = True, target_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Construct augmentation pipeline."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if not is_train:
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            normalize
        ])

    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        RandomJPEGCompression(quality_range=(30, 85), p=0.45),
        RandomGaussianBlurPIL(radius_range=(0.5, 2.2), p=0.45),
        RandomDownsampleBilinear(scale_range=(0.25, 0.70), p=0.45),
        transforms.ToTensor(),
        AddGaussianNoiseTensor(std_range=(0.01, 0.055), p=0.35),
        normalize
    ])


# =====================================================================
# Dataset and Extraction
# =====================================================================

class DeepfakeFrameDataset(Dataset):
    def __init__(self, frame_records: List[Dict[str, Any]], transform=None):
        self.records = frame_records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = rec["face_crop"]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, rec["label_id"], rec["video_id"]


def extract_or_load_frames(
    manifest_path: str,
    detector: FaceDetector,
    base_data_dir: str = "data/faceforensics/C23",
    cache_path: Optional[str] = None,
    max_frames_per_video: int = 2,
    max_videos: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Extract face crops from videos with optional disk cache."""
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
# Validation Function (Clean + Distorted Dual Validation)
# =====================================================================

def evaluate_validation_dual(model: nn.Module, val_records: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    """Evaluates validation split on both clean and simulated perturbed inputs."""
    model.eval()

    clean_transform = build_robustness_transforms(is_train=False)

    # Simulated perturbed val transform (mild blur + resize to verify robustness)
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
        results[mode] = metrics

    # Combined composite validation score: harmonic mean or balanced average
    clean_f1 = results["clean"]["macro"]["f1"]
    pert_f1 = results["perturbed"]["macro"]["f1"]
    composite_score = (clean_f1 + pert_f1) / 2.0 if (clean_f1 + pert_f1) > 0 else 0.0

    return {
        "clean": results["clean"],
        "perturbed": results["perturbed"],
        "composite_score": composite_score
    }


# =====================================================================
# Main Training Function
# =====================================================================

def train_robust_candidate(
    init_checkpoint: str = "models/deepfake/checkpoint_full/best_model.pt",
    output_dir: str = "models/deepfake/checkpoint_robust_candidate",
    train_manifest: str = "data/full_splits/deepfake_train.json",
    val_manifest: str = "data/full_splits/deepfake_val.json",
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 4e-5,
    seed: int = 42,
    max_train_videos: Optional[int] = None,
    max_val_videos: Optional[int] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Execute robust fine-tuning candidate experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    selected_device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    if selected_device.type != "cuda":
        raise RuntimeError("Robust fine-tuning requires CUDA (cuda:0).")

    os.makedirs(output_dir, exist_ok=True)
    detector = FaceDetector(device="cpu")

    cache_dir = "data/cache_deepfake_frames"
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, "train_crops_2fps.pt") if max_train_videos is None else None
    val_cache = os.path.join(cache_dir, "val_crops_2fps.pt") if max_val_videos is None else None

    logger.info("Preparing training and validation datasets...")
    train_records = extract_or_load_frames(
        train_manifest, detector, cache_path=train_cache, max_frames_per_video=2, max_videos=max_train_videos
    )
    val_records = extract_or_load_frames(
        val_manifest, detector, cache_path=val_cache, max_frames_per_video=2, max_videos=max_val_videos
    )

    logger.info(f"Loaded {len(train_records)} training face crops and {len(val_records)} validation face crops.")

    train_transform = build_robustness_transforms(is_train=True)
    train_loader = DataLoader(
        DeepfakeFrameDataset(train_records, transform=train_transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Class weighting for 1:4 Real:Fake imbalance
    class_counts = defaultdict(int)
    for r in train_records:
        class_counts[r["label_id"]] += 1
    total_samples = len(train_records)
    if class_counts[0] > 0 and class_counts[1] > 0:
        w0 = total_samples / (2.0 * class_counts[0])
        w1 = total_samples / (2.0 * class_counts[1])
        class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(selected_device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Class-balanced loss weights: Real(0)={w0:.3f}, Fake(1)={w1:.3f}")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using unweighted CrossEntropyLoss")

    # Initialize model from baseline weights
    logger.info(f"Loading initial model weights from: {init_checkpoint}")
    model = EfficientNetB4Deepfake(dropout_rate=0.4, pretrained=False)
    state_dict = torch.load(init_checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(selected_device)

    # Differential learning rate: slightly lower for backbone, standard for head
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

    best_composite_score = -1.0
    best_weights_path = os.path.join(output_dir, "best_model.pt")
    best_epoch = -1
    best_metrics = {}

    logger.info("=" * 60)
    logger.info(f"Starting Robustness Fine-Tuning ({epochs} epochs, {len(train_loader)} batches/epoch)")
    logger.info(f"Augmentations active: JPEG (30-85), GaussianBlur (0.5-2.2), LowResResize (0.25-0.70), GaussianNoise")
    logger.info("=" * 60)

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
        pert_acc = val_eval["perturbed"]["accuracy"]
        pert_f1 = val_eval["perturbed"]["f1"] if "f1" in val_eval["perturbed"] else val_eval["perturbed"]["macro"]["f1"]
        composite = val_eval["composite_score"]

        logger.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | "
            f"Val Clean Acc: {clean_acc:.4f} (F1: {clean_f1:.4f}) | "
            f"Val Perturbed Acc: {pert_acc:.4f} (F1: {pert_f1:.4f}) | "
            f"Composite: {composite:.4f}"
        )

        # Model selection: maximize composite score while keeping clean accuracy >= 80%
        if composite > best_composite_score and clean_acc >= 0.80:
            best_composite_score = composite
            best_epoch = epoch + 1
            best_metrics = val_eval
            torch.save(model.state_dict(), best_weights_path)
            logger.info(f"==> Saved new best robust checkpoint (Composite: {composite:.4f}) to {best_weights_path}")

    # If no checkpoint met >=80% filter, save the final or best composite
    if not os.path.exists(best_weights_path):
        best_epoch = epochs
        best_metrics = val_eval
        torch.save(model.state_dict(), best_weights_path)
        logger.info(f"Saved final epoch checkpoint to {best_weights_path}")

    # Persist config and metadata
    exp_config = {
        "experiment_name": "training_time_robustness_augmentation",
        "base_checkpoint": init_checkpoint,
        "candidate_checkpoint": best_weights_path,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": seed,
        "device": str(selected_device),
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "train_samples_count": len(train_records),
        "val_samples_count": len(val_records),
        "augmentation_parameters": {
            "jpeg": {"quality_range": [30, 85], "probability": 0.45},
            "gaussian_blur": {"radius_range": [0.5, 2.2], "probability": 0.45},
            "downsample_resize": {"scale_range": [0.25, 0.70], "probability": 0.45},
            "gaussian_noise": {"std_range": [0.01, 0.055], "probability": 0.35},
            "horizontal_flip": {"probability": 0.5},
            "color_jitter": {"probability": 0.3}
        },
        "validation_selection_metrics": best_metrics
    }

    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

    with open(os.path.join(output_dir, "validation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    logger.info("Robustness training experiment complete. Checkpoint and metadata saved.")
    return exp_config


def main():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B4 for robustness.")
    parser.add_argument("--init-checkpoint", type=str, default="models/deepfake/checkpoint_full/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="models/deepfake/checkpoint_robust_candidate")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-videos", type=int, default=None)
    parser.add_argument("--max-val-videos", type=int, default=None)
    args = parser.parse_args()

    train_robust_candidate(
        init_checkpoint=args.init_checkpoint,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        max_train_videos=args.max_train_videos,
        max_val_videos=args.max_val_videos
    )


if __name__ == "__main__":
    main()
