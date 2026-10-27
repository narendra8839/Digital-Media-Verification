"""Standardized and reproducible robustness evaluation pipeline for deepfake detection.

Evaluates an EfficientNet-B4 checkpoint across 5 standardized conditions:
1. Original (clean input)
2. JPEG compression (quality=30)
3. Gaussian blur (radius=2)
4. Low-resolution resize (1/4x bilinear downscale + upscale)
5. Gaussian noise (sigma=15, deterministic seed per frame)

Ensures exact frame, sample order, and random seed correspondence across all conditions.
Persists structured metrics, confusion matrices, class-wise scores, and per-video records.
"""

import os
import sys
import io
import json
import hashlib
import argparse
import warnings
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

warnings.filterwarnings("ignore")

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from torchvision import transforms
from models.deepfake.predict import load_deepfake_model
from preprocessing.frame_sampling import sample_video_frames
from preprocessing.face_detection import FaceDetector


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of a file."""
    if not os.path.exists(filepath):
        return "NOT_FOUND"
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def apply_degradation(img_arr: np.ndarray, method: str = "original", seed: int = 42) -> np.ndarray:
    """Apply deterministic degradation to an RGB numpy array (H, W, 3)."""
    if method == "original":
        return img_arr

    img = Image.fromarray(img_arr)
    if method == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=30)
        buf.seek(0)
        img = Image.open(buf)
    elif method == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
    elif method == "resize":
        w, h = img.size
        down_w = max(1, w // 4)
        down_h = max(1, h // 4)
        img = img.resize((down_w, down_h), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
    elif method == "noise":
        arr = np.array(img, dtype=np.float32)
        rng = np.random.RandomState(seed)
        noise = rng.normal(0, 15, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        raise ValueError(f"Unknown degradation method: {method}")

    return np.array(img)


def evaluate_checkpoint_robustness(
    test_manifest_path: str,
    checkpoint_path: str,
    output_json_path: str,
    base_data_dir: str = "data/faceforensics/C23",
    max_frames: int = 4,
    real_count: int = 50,
    fake_count: int = 50,
    offset_real: int = 0,
    offset_fake: int = 0,
    fake_class_idx: int = 1,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Run full standardized robustness evaluation on balanced test subset."""
    device_name = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    print("=" * 65)
    print(f"Deepfake Robustness Benchmark Evaluation")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Fake Class Index: {fake_class_idx}")
    print(f"Device: {device_name}")
    print("=" * 65)

    ckpt_hash = compute_file_sha256(checkpoint_path)
    model, loaded_device = load_deepfake_model(checkpoint_path, device_name)
    detector = FaceDetector(device=loaded_device)

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open(test_manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Deterministic selection with configurable offset
    real_pool = [x for x in data if x["label"] == "real"]
    fake_pool = [x for x in data if x["label"] == "fake"]

    selected_real = real_pool[offset_real : offset_real + real_count]
    selected_fake = fake_pool[offset_fake : offset_fake + fake_count]
    balanced_subset = selected_real + selected_fake
    total_videos = len(balanced_subset)

    print(f"Evaluation dataset: {len(selected_real)} Real (offset {offset_real}) + {len(selected_fake)} Fake (offset {offset_fake}) = {total_videos} videos")
    print(f"Sampling configuration: {max_frames} frames/video @ 1.0 FPS (reused across conditions)")

    degradations = ["original", "jpeg", "blur", "resize", "noise"]
    records_per_cond: Dict[str, List[Dict[str, Any]]] = {d: [] for d in degradations}

    # Iterate over each video once to decode frames only once
    for idx, item in enumerate(tqdm(balanced_subset, desc="Evaluating videos across conditions")):
        rel_path = item["relative_path"]
        video_path = os.path.join(base_data_dir, rel_path)
        gt_label = item["label"]
        vid_id = item.get("video_id", f"vid_{idx}")

        if not os.path.exists(video_path):
            print(f"Warning: Missing video {video_path}")
            continue

        sampled = sample_video_frames(video_path, target_fps=1.0, max_frames=max_frames)
        if not sampled:
            print(f"Warning: No frames sampled for {video_path}")
            continue

        for d in degradations:
            frame_probs = []
            for frame_idx, (ts, frame_rgb) in enumerate(sampled):
                seed = 42 + idx * 10 + frame_idx
                deg_frame = apply_degradation(frame_rgb, method=d, seed=seed)
                crop, is_detected = detector.detect_and_crop(deg_frame, target_size=(224, 224))
                if isinstance(crop, np.ndarray):
                    crop_img = Image.fromarray(crop)
                else:
                    crop_img = crop
                tensor = eval_transform(crop_img).unsqueeze(0).to(loaded_device)
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                fake_prob = float(probs[fake_class_idx])
                frame_probs.append(fake_prob)

            if frame_probs:
                mean_fake_prob = float(np.mean(frame_probs))
                pred_label = "fake" if mean_fake_prob >= 0.5 else "real"
                records_per_cond[d].append({
                    "video_id": vid_id,
                    "relative_path": rel_path,
                    "ground_truth": gt_label,
                    "predicted_label": pred_label,
                    "mean_fake_probability": mean_fake_prob,
                    "correct": (pred_label == gt_label),
                    "num_frames": len(frame_probs)
                })

    # Compute comprehensive metrics
    summary: Dict[str, Any] = {
        "metadata": {
            "evaluation_timestamp": datetime.now().isoformat(),
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": ckpt_hash,
            "test_manifest": test_manifest_path,
            "total_videos_evaluated": total_videos,
            "class_distribution": {"real": len(selected_real), "fake": len(selected_fake)},
            "offset_real": offset_real,
            "offset_fake": offset_fake,
            "fake_class_idx": fake_class_idx,
            "frames_per_video": max_frames,
            "conditions_evaluated": degradations,
            "device": str(loaded_device)
        },
        "conditions": {},
        "per_video_records": records_per_cond
    }

    orig_acc = None
    orig_macro_f1 = None

    print("\n" + "=" * 90)
    print(f"{'Condition':12s} | {'Acc':7s} | {'BalAcc':7s} | {'Macro F1':8s} | {'Drop (Acc)':10s} | {'Drop (F1)':10s} | {'CM (Fake,Real)':15s}")
    print("=" * 90)

    for d in degradations:
        cond_records = records_per_cond[d]
        y_true = [r["ground_truth"] for r in cond_records]
        y_pred = [r["predicted_label"] for r in cond_records]
        n_samples = len(y_true)

        acc = float(accuracy_score(y_true, y_pred)) if n_samples > 0 else 0.0
        bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if n_samples > 0 else 0.0
        macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        c_prec, c_rec, c_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=["real", "fake"], zero_division=0
        )
        # Confusion matrix labels=['fake', 'real']:
        # [ [TP_fake, FN_fake],
        #   [FP_fake, TN_fake] ]
        # i.e., Row 0: True Fake (classified as Fake, classified as Real)
        #       Row 1: True Real (classified as Fake, classified as Real)
        cm = confusion_matrix(y_true, y_pred, labels=["fake", "real"]).tolist()

        if d == "original":
            orig_acc = acc
            orig_macro_f1 = float(macro_f1)
            drop_acc = 0.0
            drop_f1 = 0.0
        else:
            drop_acc = float(orig_acc - acc) if orig_acc is not None else 0.0
            drop_f1 = float(orig_macro_f1 - float(macro_f1)) if orig_macro_f1 is not None else 0.0

        summary["conditions"][d] = {
            "sample_count": n_samples,
            "class_distribution": {
                "real": sum(1 for x in y_true if x == "real"),
                "fake": sum(1 for x in y_true if x == "fake")
            },
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_precision": float(macro_prec),
            "macro_recall": float(macro_rec),
            "macro_f1": float(macro_f1),
            "drop_from_original_acc": drop_acc,
            "drop_from_original_macro_f1": drop_f1,
            "per_class": {
                "real": {"precision": float(c_prec[0]), "recall": float(c_rec[0]), "f1": float(c_f1[0])},
                "fake": {"precision": float(c_prec[1]), "recall": float(c_rec[1]), "f1": float(c_f1[1])}
            },
            "confusion_matrix": cm,
            "confusion_matrix_doc": "Rows: True [Fake, Real]; Columns: Pred [Fake, Real]"
        }

        print(
            f"{d.capitalize():12s} | {acc:.4f}  | {bal_acc:.4f}  | {float(macro_f1):.4f}   | {drop_acc:+.4f}     | {drop_f1:+.4f}     | {cm}"
        )

    print("=" * 90)

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nStructured evaluation report saved to: {output_json_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake model robustness across perturbations.")
    parser.add_argument("--checkpoint", type=str, default="models/deepfake/checkpoint_full/best_model.pt", help="Path to model weights")
    parser.add_argument("--test-manifest", type=str, default="data/full_splits/deepfake_test.json", help="Path to test split manifest")
    parser.add_argument("--output", type=str, default="eval_results/baseline_robustness.json", help="Path to save output JSON")
    parser.add_argument("--real-count", type=int, default=50, help="Number of real videos")
    parser.add_argument("--fake-count", type=int, default=50, help="Number of fake videos")
    parser.add_argument("--offset-real", type=int, default=0, help="Starting index for real videos")
    parser.add_argument("--offset-fake", type=int, default=0, help="Starting index for fake videos")
    parser.add_argument("--max-frames", type=int, default=4, help="Frames per video")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--fake-class-idx", type=int, default=1, choices=[0, 1], help="Class index corresponding to Fake (1 for Baseline/V2B, 0 for V3)")
    args = parser.parse_args()

    evaluate_checkpoint_robustness(
        test_manifest_path=args.test_manifest,
        checkpoint_path=args.checkpoint,
        output_json_path=args.output,
        max_frames=args.max_frames,
        real_count=args.real_count,
        fake_count=args.fake_count,
        offset_real=args.offset_real,
        offset_fake=args.offset_fake,
        fake_class_idx=args.fake_class_idx,
        device=args.device
    )


if __name__ == "__main__":
    main()
