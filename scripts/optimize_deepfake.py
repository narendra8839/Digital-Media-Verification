import os
import sys
import json
import logging
import time
from typing import Dict, Any, List
from collections import defaultdict
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from models.deepfake.efficientnet_b4 import EfficientNetB4Deepfake
from models.deepfake.augmentation import get_deepfake_transforms
from models.deepfake.train import DeepfakeFrameDataset, evaluate
from preprocessing.frame_sampling import sample_video_frames
from preprocessing.face_detection import FaceDetector
from models.deepfake.predict import predict_deepfake_crop

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("optimize_deepfake")

def evaluate_val_robustness(model, device, val_manifest="data/full_splits/deepfake_val.json", max_frames=4, real_count=50, fake_count=50):
    """Evaluate balanced robustness on validation partition (50 Real, 50 Fake)."""
    from PIL import Image, ImageFilter
    import io

    def degrade_img(arr, method="original", seed=42):
        if method == "original":
            return arr
        img = Image.fromarray(arr)
        if method == "jpeg":
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=30)
            buf.seek(0)
            img = Image.open(buf)
        elif method == "blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=2))
        elif method == "resize":
            orig_size = img.size
            img = img.resize((max(1, orig_size[0] // 4), max(1, orig_size[1] // 4)), Image.BILINEAR)
            img = img.resize(orig_size, Image.BILINEAR)
        elif method == "noise":
            arr_np = np.array(img, dtype=np.float32)
            rng = np.random.RandomState(seed)
            noise = rng.normal(0, 15, arr_np.shape)
            arr_np = np.clip(arr_np + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr_np)
        return np.array(img)

    with open(val_manifest, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    real_vids = [x for x in val_data if x["label"] == "real"][:real_count]
    fake_vids = [x for x in val_data if x["label"] == "fake"][:fake_count]
    subset = real_vids + fake_vids

    detector = FaceDetector(device=device)
    degradations = ["original", "jpeg", "blur", "resize", "noise"]
    res_table = {d: {"y_true": [], "y_pred": []} for d in degradations}

    logger.info(f"Running validation robustness on {len(subset)} videos ({len(real_vids)} Real, {len(fake_vids)} Fake)...")
    for idx, item in enumerate(subset):
        rel_path = item["relative_path"]
        vpath = os.path.join("data/faceforensics/C23", rel_path)
        gt = item["label"]
        if not os.path.exists(vpath):
            continue
        sampled = sample_video_frames(vpath, target_fps=1.0, max_frames=max_frames)
        if not sampled:
            continue

        for d in degradations:
            probs = []
            for f_idx, (ts, frame_rgb) in enumerate(sampled):
                seed = 42 + idx * 10 + f_idx
                deg_f = degrade_img(frame_rgb, method=d, seed=seed)
                crop, _ = detector.detect_and_crop(deg_f, target_size=(224, 224))
                pred_res = predict_deepfake_crop(model, crop, device)
                probs.append(pred_res["fake_probability"])

            if probs:
                mean_p = float(np.mean(probs))
                pred_label = "fake" if mean_p >= 0.5 else "real"
                res_table[d]["y_true"].append(gt)
                res_table[d]["y_pred"].append(pred_label)

    robustness_summary = {}
    orig_acc = None
    for d in degradations:
        yt = res_table[d]["y_true"]
        yp = res_table[d]["y_pred"]
        acc = float(accuracy_score(yt, yp)) if len(yt) > 0 else 0.0
        prec, rec, f1, _ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)
        c_prec, c_rec, c_f1, _ = precision_recall_fscore_support(yt, yp, labels=['real', 'fake'], zero_division=0)
        cm = confusion_matrix(yt, yp, labels=['fake', 'real']).tolist()

        if d == "original":
            orig_acc = acc
            drop = 0.0
        else:
            drop = float(orig_acc - acc) if orig_acc is not None else 0.0

        robustness_summary[d] = {
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "drop_from_original": float(drop),
            "real_f1": float(c_f1[0]),
            "fake_f1": float(c_f1[1]),
            "confusion_matrix": cm,
            "sample_count": len(yt)
        }
        logger.info(f"Val Robustness [{d:8s}]: Acc={acc:.4f} | Macro F1={f1:.4f} | Drop={drop:+.4f}")

    return robustness_summary

def run_deepfake_experiment(
    exp_name: str,
    output_dir: str,
    aug_config: Dict[str, Any],
    epochs: int = 3,
    lr: float = 1e-4,
    use_lr_decay: bool = False,
    batch_size: int = 8,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required"
    torch.cuda.reset_peak_memory_stats()

    os.makedirs(output_dir, exist_ok=True)
    best_weights_path = os.path.join(output_dir, "best_model.pt")

    cache_dir = os.path.expanduser("~/cache_deepfake_frames")
    train_cache = os.path.join(cache_dir, "train_frames_2fps.pt")
    val_cache = os.path.join(cache_dir, "val_frames_2fps.pt")

    logger.info("=" * 60)
    logger.info(f"STARTING DEEPFAKE EXPERIMENT: {exp_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Augmentation config: {json.dumps(aug_config)}")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    logger.info("=" * 60)

    logger.info(f"Loading cached frames from {train_cache} and {val_cache}...")
    train_frames = torch.load(train_cache, weights_only=False)
    val_frames = torch.load(val_cache, weights_only=False)
    logger.info(f"Loaded {len(train_frames)} train frames, {len(val_frames)} val frames.")

    train_transform = get_deepfake_transforms(is_train=True, config=aug_config)
    val_transform = get_deepfake_transforms(is_train=False)

    train_loader = DataLoader(DeepfakeFrameDataset(train_frames, transform=train_transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(DeepfakeFrameDataset(val_frames, transform=val_transform), batch_size=batch_size, shuffle=False)

    # Class weights calculation
    class_counts = defaultdict(int)
    for rec in train_frames:
        class_counts[rec["label_id"]] += 1
    total_samples = len(train_frames)
    w0 = total_samples / (2.0 * class_counts[0])
    w1 = total_samples / (2.0 * class_counts[1])
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info(f"Class-balanced weights: Real(0)={w0:.3f}, Fake(1)={w1:.3f}")

    model = EfficientNetB4Deepfake(dropout_rate=0.4, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    if use_lr_decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-6)
    else:
        scheduler = None

    best_val_video_f1 = -1.0
    best_val_metrics = None
    best_epoch = -1
    history = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels, vids in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            running_loss += loss.item() * len(imgs)

        epoch_loss = running_loss / max(1, len(train_frames))
        val_frame_m, val_video_m = evaluate(model, val_loader, device)
        video_f1 = val_video_m["macro"]["f1"]
        video_acc = val_video_m["accuracy"]
        frame_f1 = val_frame_m["macro"]["f1"]
        frame_acc = val_frame_m["accuracy"]

        logger.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | "
            f"Val Frame Acc: {frame_acc:.4f} (F1: {frame_f1:.4f}) | "
            f"Val Video Acc: {video_acc:.4f} (F1: {video_f1:.4f})"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(epoch_loss),
            "val_frame_acc": float(frame_acc),
            "val_frame_macro_f1": float(frame_f1),
            "val_video_acc": float(video_acc),
            "val_video_macro_f1": float(video_f1)
        })

        if video_f1 > best_val_video_f1:
            best_val_video_f1 = video_f1
            best_epoch = epoch + 1
            best_val_metrics = {"frame": val_frame_m, "video": val_video_m}
            torch.save(model.state_dict(), best_weights_path)
            logger.info(f"--> Saved new BEST checkpoint (Video Macro F1: {best_val_video_f1:.4f}) to {best_weights_path}")

    elapsed = time.time() - start_time
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    logger.info(f"Training completed in {elapsed:.1f}s. Peak GPU memory: {peak_mem_mb:.1f} MiB")

    # Load best model for clean validation & validation robustness evaluation
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    clean_frame_m, clean_video_m = evaluate(model, val_loader, device)

    # Confusion matrix extraction for clean video-level evaluation
    # Compute class-wise recalls: Real recall and Fake recall
    cm_video = np.array(clean_video_m["confusion_matrix"]) # row 0: Real, row 1: Fake
    real_recall = float(cm_video[0, 0] / cm_video[0, :].sum()) if cm_video[0, :].sum() > 0 else 0.0
    fake_recall = float(cm_video[1, 1] / cm_video[1, :].sum()) if cm_video[1, :].sum() > 0 else 0.0

    # Evaluate validation robustness
    val_robustness = evaluate_val_robustness(model, device)

    config_record = {
        "experiment_name": exp_name,
        "model": "efficientnet_b4",
        "epochs": epochs,
        "best_epoch": best_epoch,
        "learning_rate": lr,
        "lr_scheduler": "cosine" if use_lr_decay else "none",
        "batch_size": batch_size,
        "augmentation_config": aug_config,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_gpu_memory_mib": round(peak_mem_mb, 1),
        "training_time_seconds": round(elapsed, 2),
        "checkpoint_path": best_weights_path
    }

    full_metrics = {
        "clean_validation": {
            "frame_level": clean_frame_m,
            "video_level": clean_video_m,
            "real_recall": real_recall,
            "fake_recall": fake_recall
        },
        "robustness_validation": val_robustness
    }

    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    with open(os.path.join(output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"0": "real", "1": "fake"}, f, indent=2)
    with open(os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "exp_name": exp_name,
        "best_epoch": best_epoch,
        "clean_video_acc": float(clean_video_m["accuracy"]),
        "clean_video_macro_f1": float(clean_video_m["macro"]["f1"]),
        "clean_frame_acc": float(clean_frame_m["accuracy"]),
        "clean_frame_macro_f1": float(clean_frame_m["macro"]["f1"]),
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "robustness_original_acc": val_robustness["original"]["accuracy"],
        "robustness_blur_acc": val_robustness["blur"]["accuracy"],
        "robustness_resize_acc": val_robustness["resize"]["accuracy"],
        "robustness_jpeg_acc": val_robustness["jpeg"]["accuracy"],
        "robustness_noise_acc": val_robustness["noise"]["accuracy"],
        "peak_gpu_mem_mib": round(peak_mem_mb, 1),
        "elapsed_seconds": round(elapsed, 2)
    }

    print("\n" + "=" * 50)
    print("DEEPFAKE EXPERIMENT RESULT SUMMARY:")
    print(json.dumps(summary, indent=2))
    print("=" * 50)
    return summary

def main():
    exp_id = sys.argv[1] if len(sys.argv) > 1 else "exp1"

    if exp_id == "exp1":
        # Candidate A: Robustness-targeted augmentation (mild resize degradation p=0.35, blur kernel 5 sigma 0.2-1.8 p=0.35)
        aug_cfg = {
            "augmentation_enabled": True,
            "jpeg_p": 0.5,
            "jpeg_min_quality": 45,
            "jpeg_max_quality": 95,
            "noise_p": 0.3,
            "noise_std_max": 0.03,
            "blur_p": 0.35,
            "blur_kernel_size": 5,
            "blur_sigma_range": [0.2, 1.8],
            "resize_p": 0.35,
            "flip_p": 0.5
        }
        run_deepfake_experiment(
            exp_name="exp1_robustness_aug",
            output_dir="models/deepfake/optimization/exp1_robustness_aug",
            aug_config=aug_cfg,
            epochs=3,
            lr=1e-4,
            use_lr_decay=False
        )
    elif exp_id == "exp2":
        # Candidate B: Moderated augmentation + Cosine LR decay
        aug_cfg = {
            "augmentation_enabled": True,
            "jpeg_p": 0.5,
            "jpeg_min_quality": 50,
            "jpeg_max_quality": 95,
            "noise_p": 0.3,
            "noise_std_max": 0.03,
            "blur_p": 0.25,
            "blur_kernel_size": 3,
            "blur_sigma_range": [0.2, 1.5],
            "resize_p": 0.25,
            "flip_p": 0.5
        }
        run_deepfake_experiment(
            exp_name="exp2_moderated_decay",
            output_dir="models/deepfake/optimization/exp2_moderated_decay",
            aug_config=aug_cfg,
            epochs=3,
            lr=1e-4,
            use_lr_decay=True
        )
    else:
        print(f"Unknown experiment: {exp_id}")

if __name__ == "__main__":
    main()
