"""Reproducible training pipeline for EfficientNet-B4 binary deepfake detection.

Key features:
1. Operates on full audited dataset splits (or configurable subsets for smoke testing).
2. Robustness augmentation applied to training frames only (JPEG compression, blur, noise, jitter).
3. Strictly deterministic validation pipeline.
4. Class-balanced CrossEntropyLoss addressing 1:4 Real:Fake imbalance.
5. Checkpoint selection strictly based on validation performance (TEST SPLIT IS NEVER TOUCHED).
6. Preserves existing MVP checkpoints by writing new runs to an isolated directory.
7. Device-ready: auto-detects CUDA when available while supporting explicit CPU/CUDA configuration.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from models.deepfake.efficientnet_b4 import EfficientNetB4Deepfake
from models.deepfake.augmentation import get_deepfake_transforms
from preprocessing.frame_sampling import sample_video_frames
from preprocessing.face_detection import FaceDetector
from evaluation.metrics import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("deepfake_train")


class DeepfakeFrameDataset(Dataset):
    """Dataset serving extracted face crops and their video provenance."""

    def __init__(self, frame_records: List[Dict[str, Any]], transform=None):
        self.records = frame_records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = rec["face_crop"]
        if self.transform:
            img = self.transform(img)
        label = rec["label_id"]
        return img, label, rec["video_id"]


def extract_frames_from_manifest(
    manifest_path: str,
    detector: FaceDetector,
    base_data_dir: str = "data/faceforensics/C23",
    max_frames_per_video: int = 2,
    max_videos: Optional[int] = None,
    cache_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Extract face crops from videos listed in a split manifest."""
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached frames from {cache_path}...")
        return torch.load(cache_path, weights_only=False)

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        videos = json.load(f)

    if max_videos is not None:
        videos = videos[:max_videos]

    frame_records = []
    logger.info(f"Extracting frames from {len(videos)} videos in {manifest_path} (max_frames={max_frames_per_video})...")

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
                "timestamp": ts,
                "face_crop": crop,
                "face_detected": is_detected
            })

        if i % 250 == 0 or i == len(videos):
            logger.info(f"Extraction progress: {i}/{len(videos)} videos processed ({len(frame_records)} frames extracted)")

    logger.info(f"Successfully extracted {len(frame_records)} frames from {len(videos)} videos.")
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(frame_records, cache_path, _use_new_zipfile_serialization=False)
            logger.info(f"Saved {len(frame_records)} frames to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Could not save cache to {cache_path} (proceeding in memory): {e}")
    return frame_records


def train_deepfake_model(
    train_manifest: str = "data/full_splits/deepfake_train.json",
    val_manifest: str = "data/full_splits/deepfake_val.json",
    output_dir: str = "models/deepfake/checkpoint_full",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-4,
    seed: int = 42,
    device: Optional[str] = None,
    max_train_videos: Optional[int] = None,
    max_val_videos: Optional[int] = None,
    max_frames_per_video: int = 2,
    use_augmentation: bool = True,
    balance_classes: bool = True,
    save_checkpoint: bool = True
) -> Dict[str, Any]:
    """Train EfficientNet-B4 deepfake classifier with full reproducibility and UQ readiness."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    selected_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected_device.type != "cuda":
        raise RuntimeError("Phase 3 full training requires CUDA (cuda:0). Silent fallback to CPU is strictly prohibited.")
    logger.info(f"Initializing training on device: {selected_device}")

    os.makedirs(output_dir, exist_ok=True)
    detector = FaceDetector(device="cpu")

    # 1. Extract frames from manifests with caching in native WSL home
    cache_dir = os.path.expanduser("~/cache_deepfake_frames")
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, f"train_frames_{max_frames_per_video}fps.pt") if max_train_videos is None else None
    val_cache = os.path.join(cache_dir, f"val_frames_{max_frames_per_video}fps.pt") if max_val_videos is None else None

    train_frames = extract_frames_from_manifest(
        train_manifest, detector, max_frames_per_video=max_frames_per_video, max_videos=max_train_videos, cache_path=train_cache
    )
    val_frames = extract_frames_from_manifest(
        val_manifest, detector, max_frames_per_video=max_frames_per_video, max_videos=max_val_videos, cache_path=val_cache
    )

    if not train_frames:
        raise ValueError(f"No valid frames extracted from {train_manifest}")
    if not val_frames:
        raise ValueError(f"No valid frames extracted from {val_manifest}")

    # 2. Transforms: stochastic for train, deterministic for val
    train_transform = get_deepfake_transforms(is_train=use_augmentation, config={"augmentation_enabled": use_augmentation})
    val_transform = get_deepfake_transforms(is_train=False)

    train_loader = DataLoader(
        DeepfakeFrameDataset(train_frames, transform=train_transform),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        DeepfakeFrameDataset(val_frames, transform=val_transform),
        batch_size=batch_size,
        shuffle=False
    )

    # 3. Class balance weighting
    class_counts = defaultdict(int)
    for rec in train_frames:
        class_counts[rec["label_id"]] += 1
    total_samples = len(train_frames)
    num_classes = 2

    if balance_classes and class_counts[0] > 0 and class_counts[1] > 0:
        # Inverse class frequency weighting: total / (num_classes * count)
        w0 = total_samples / (num_classes * class_counts[0])
        w1 = total_samples / (num_classes * class_counts[1])
        class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(selected_device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Applied class-balanced loss weights: Real={w0:.3f}, Fake={w1:.3f}")
    else:
        criterion = nn.CrossEntropyLoss()

    model = EfficientNetB4Deepfake(dropout_rate=0.4, pretrained=True).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # Detailed Startup Verification Logging
    gpu_name = torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else "N/A"
    model_param_device = next(model.parameters()).device
    logger.info("=" * 60)
    logger.info("model: EfficientNet-B4")
    logger.info(f"dataset: {train_manifest} (train) | {val_manifest} (val)")
    logger.info(f"device: {selected_device}")
    logger.info(f"GPU name: {gpu_name}")
    logger.info(f"batch size: {batch_size}")
    logger.info(f"learning rate: {lr}")
    logger.info(f"epochs: {epochs}")
    logger.info(f"train frame count: {len(train_frames)}")
    logger.info(f"validation frame count: {len(val_frames)}")
    logger.info(f"seed: {seed}")
    logger.info(f"model.parameters().device: {model_param_device}")
    logger.info("=" * 60)
    assert model_param_device.type == "cuda", f"Model parameters must be on CUDA, got {model_param_device}"

    best_val_acc = -1.0
    best_epoch = -1
    best_weights_path = os.path.join(output_dir, "best_model.pt")
    first_batch_verified = False

    logger.info(f"Starting training loop ({epochs} epochs, {len(train_loader)} batches/epoch)...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(selected_device), labels.to(selected_device)
            if not first_batch_verified:
                logger.info(f"batch tensor device: {imgs.device}")
                assert imgs.device.type == "cuda", f"Batch tensors must be on CUDA, got {imgs.device}"
                first_batch_verified = True
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / max(1, len(train_frames))
        val_frame_m, val_video_m = evaluate(model, val_loader, selected_device)
        logger.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | "
            f"Val Frame Acc: {val_frame_m['accuracy']:.4f} | Val Video Acc: {val_video_m['accuracy']:.4f}"
        )

        if val_video_m["accuracy"] >= best_val_acc:
            best_val_acc = val_video_m["accuracy"]
            best_epoch = epoch + 1
            if save_checkpoint:
                torch.save(model.state_dict(), best_weights_path)
                logger.info(f"Saved new best checkpoint (Val Video Acc: {best_val_acc:.4f}) to {best_weights_path}")

    # Evaluate best model on validation split (TEST SPLIT IS NEVER TOUCHED)
    if save_checkpoint and os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, map_location=selected_device))
    val_frame_m, val_video_m = evaluate(model, val_loader, selected_device)

    label_mapping = {"0": "real", "1": "fake"}
    training_config = {
        "model": "efficientnet_b4",
        "num_classes": 2,
        "label_mapping": label_mapping,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "learning_rate": lr,
        "random_seed": seed,
        "device": str(selected_device),
        "frame_resolution": [224, 224],
        "frames_per_video_train": max_frames_per_video,
        "augmentation_enabled": use_augmentation,
        "balance_classes": balance_classes,
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "train_samples_count": len(train_frames),
        "val_samples_count": len(val_frames),
        "checkpoint_path": best_weights_path if save_checkpoint else None
    }

    metrics = {
        "validation": {
            "frame_level": val_frame_m,
            "video_level": val_video_m
        }
    }

    if save_checkpoint:
        with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(training_config, f, indent=2)
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(label_mapping, f, indent=2)

    logger.info("Deepfake training process complete. Results and configuration recorded.")
    return {
        "config": training_config,
        "metrics": metrics,
        "best_val_accuracy": best_val_acc
    }


def evaluate(model, data_loader, device):
    """Evaluate deepfake classifier at both frame and video aggregation levels."""
    model.eval()
    y_true_frames = []
    y_pred_frames = []
    video_preds = defaultdict(list)
    video_trues = {}

    with torch.no_grad():
        for imgs, labels, vids in data_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            labels_np = labels.numpy()
            y_true_frames.extend(labels_np.tolist())
            y_pred_frames.extend(preds.tolist())

            for vid, prob, true_label in zip(vids, probs, labels_np):
                video_preds[vid].append(prob)
                video_trues[vid] = true_label

    frame_metrics = compute_classification_metrics(y_true_frames, y_pred_frames)

    # Video-level aggregation: mean fake probability >= 0.5 -> fake
    y_true_videos = []
    y_pred_videos = []
    for vid, prob_list in video_preds.items():
        avg_fake_prob = float(np.mean(prob_list))
        pred_label = 1 if avg_fake_prob >= 0.5 else 0
        y_true_videos.append(video_trues[vid])
        y_pred_videos.append(pred_label)

    video_metrics = compute_classification_metrics(y_true_videos, y_pred_videos)
    return frame_metrics, video_metrics


if __name__ == "__main__":
    train_deepfake_model()
