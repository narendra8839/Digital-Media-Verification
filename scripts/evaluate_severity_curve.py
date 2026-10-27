"""Multi-severity distortion degradation curve evaluation.

Evaluates baseline vs candidate model across multiple severity levels:
1. Gaussian Blur: radius = 1, 2, 3
2. Low-Resolution Resizing: downscale factor = 2x, 4x, 8x
3. JPEG Compression: quality = 70, 50, 30

Saves structured multi-severity metrics and summary table.
"""

import os
import sys
import io
import json
import argparse
import numpy as np
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm
import hashlib
from datetime import datetime
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from torchvision import transforms
from models.deepfake.predict import load_deepfake_model
from preprocessing.frame_sampling import sample_video_frames
from preprocessing.face_detection import FaceDetector

def compute_file_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "NOT_FOUND"
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def apply_severity_degradation(img_arr: np.ndarray, distortion_type: str, severity_val: Any, seed: int = 42) -> np.ndarray:
    img = Image.fromarray(img_arr)
    if distortion_type == "blur":
        # severity_val is blur radius (e.g. 1, 2, 3)
        img = img.filter(ImageFilter.GaussianBlur(radius=severity_val))
    elif distortion_type == "resize":
        # severity_val is downscale factor (e.g. 2, 4, 8)
        w, h = img.size
        down_w = max(1, w // severity_val)
        down_h = max(1, h // severity_val)
        img = img.resize((down_w, down_h), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
    elif distortion_type == "jpeg":
        # severity_val is quality (e.g. 70, 50, 30)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=severity_val)
        buf.seek(0)
        img = Image.open(buf)
    elif distortion_type == "noise":
        # severity_val is sigma (e.g. 5, 10, 15)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, severity_val, img_arr.shape)
        arr = np.clip(img_arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return arr
    return np.array(img)


def evaluate_severity_curves(
    checkpoint_path: str,
    test_manifest_path: str = "data/full_splits/deepfake_test.json",
    base_data_dir: str = "data/faceforensics/C23",
    real_count: int = 50,
    fake_count: int = 50,
    offset_real: int = 0,
    offset_fake: int = 0,
    max_frames: int = 4,
    fake_class_idx: int = 1,
    device: str = "cuda"
) -> Tuple[Dict[str, Any], List[str]]:
    device_name = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    model, loaded_device = load_deepfake_model(checkpoint_path, device_name)
    detector = FaceDetector(device=loaded_device)

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open(test_manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reals = [x for x in data if x["label"] == "real"][offset_real : offset_real + real_count]
    fakes = [x for x in data if x["label"] == "fake"][offset_fake : offset_fake + fake_count]
    videos = reals + fakes
    evaluated_ids = [x.get("video_id", f"vid_{i}") for i, x in enumerate(videos)]

    experiments = [
        ("blur", 1, "radius=1 (mild)"),
        ("blur", 2, "radius=2 (moderate)"),
        ("blur", 3, "radius=3 (severe)"),
        ("resize", 2, "2x downscale (mild)"),
        ("resize", 4, "4x downscale (moderate)"),
        ("resize", 8, "8x downscale (severe)"),
        ("jpeg", 70, "Q=70 (mild)"),
        ("jpeg", 50, "Q=50 (moderate)"),
        ("jpeg", 30, "Q=30 (severe)"),
        ("noise", 5, "sigma=5 (mild)"),
        ("noise", 10, "sigma=10 (moderate)"),
        ("noise", 15, "sigma=15 (severe)")
    ]

    results = {f"{dist}_{val}": {"y_true": [], "y_pred": [], "desc": desc} for dist, val, desc in experiments}

    for idx, item in enumerate(tqdm(videos, desc=f"Evaluating severities for {os.path.basename(checkpoint_path)}")):
        video_path = os.path.join(base_data_dir, item["relative_path"])
        if not os.path.exists(video_path):
            continue

        sampled = sample_video_frames(video_path, target_fps=1.0, max_frames=max_frames)
        if not sampled:
            continue

        for dist, val, desc in experiments:
            key = f"{dist}_{val}"
            probs = []
            for frame_idx, (ts, frame_rgb) in enumerate(sampled):
                seed = 42 + idx * 10 + frame_idx
                deg_img = apply_severity_degradation(frame_rgb, dist, val, seed)
                crop, _ = detector.detect_and_crop(deg_img, target_size=(224, 224))
                if isinstance(crop, np.ndarray):
                    crop_img = Image.fromarray(crop)
                else:
                    crop_img = crop
                tensor = eval_transform(crop_img).unsqueeze(0).to(loaded_device)
                with torch.no_grad():
                    logits = model(tensor)
                    soft_probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                fake_prob = float(soft_probs[fake_class_idx])
                probs.append(fake_prob)

            if probs:
                mean_p = float(np.mean(probs))
                pred_lbl = "fake" if mean_p >= 0.5 else "real"
                results[key]["y_true"].append(item["label"])
                results[key]["y_pred"].append(pred_lbl)

    summary = {}
    for key, data_dict in results.items():
        y_true = data_dict["y_true"]
        y_pred = data_dict["y_pred"]
        acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        bal_acc = balanced_accuracy_score(y_true, y_pred) if y_true else 0.0
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        c_prec, c_rec, c_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=["real", "fake"], zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=["fake", "real"]).tolist()

        summary[key] = {
            "description": data_dict["desc"],
            "sample_count": len(y_true),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(f1),
            "per_class": {
                "real": {"precision": float(c_prec[0]), "recall": float(c_rec[0]), "f1": float(c_f1[0])},
                "fake": {"precision": float(c_prec[1]), "recall": float(c_rec[1]), "f1": float(c_f1[1])}
            },
            "confusion_matrix": cm,
            "confusion_matrix_doc": "Rows: True [Fake, Real]; Columns: Pred [Fake, Real]"
        }

    return summary, evaluated_ids


def main():
    parser = argparse.ArgumentParser(description="Multi-severity distortion degradation curve evaluation.")
    parser.add_argument("--baseline-checkpoint", type=str, default="models/deepfake/checkpoint_full/best_model.pt")
    parser.add_argument("--candidate-checkpoint", type=str, default="models/deepfake/checkpoint_robust_candidate/best_model.pt")
    parser.add_argument("--test-manifest", type=str, default="data/full_splits/deepfake_test.json")
    parser.add_argument("--output", type=str, default="eval_results/severity_curve_analysis.json")
    parser.add_argument("--real-count", type=int, default=50, help="Number of real videos to evaluate")
    parser.add_argument("--fake-count", type=int, default=50, help="Number of fake videos to evaluate")
    parser.add_argument("--offset-real", type=int, default=0, help="Starting index for real videos")
    parser.add_argument("--offset-fake", type=int, default=0, help="Starting index for fake videos")
    parser.add_argument("--baseline-fake-class-idx", type=int, default=1, choices=[0, 1])
    parser.add_argument("--candidate-fake-class-idx", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()

    print(f"Evaluating Baseline Model across distortion severities ({args.real_count} Real, {args.fake_count} Fake)...")
    base_res, base_ids = evaluate_severity_curves(
        args.baseline_checkpoint,
        test_manifest_path=args.test_manifest,
        real_count=args.real_count,
        fake_count=args.fake_count,
        offset_real=args.offset_real,
        offset_fake=args.offset_fake,
        fake_class_idx=args.baseline_fake_class_idx
    )

    print(f"Evaluating Candidate Model across distortion severities ({args.real_count} Real, {args.fake_count} Fake)...")
    cand_res, cand_ids = evaluate_severity_curves(
        args.candidate_checkpoint,
        test_manifest_path=args.test_manifest,
        real_count=args.real_count,
        fake_count=args.fake_count,
        offset_real=args.offset_real,
        offset_fake=args.offset_fake,
        fake_class_idx=args.candidate_fake_class_idx
    )

    combined = {
        "metadata": {
            "evaluation_timestamp": datetime.now().isoformat(),
            "baseline_checkpoint": args.baseline_checkpoint,
            "baseline_sha256": compute_file_sha256(args.baseline_checkpoint),
            "baseline_fake_class_idx": args.baseline_fake_class_idx,
            "candidate_checkpoint": args.candidate_checkpoint,
            "candidate_sha256": compute_file_sha256(args.candidate_checkpoint),
            "candidate_fake_class_idx": args.candidate_fake_class_idx,
            "test_manifest": args.test_manifest,
            "test_manifest_sha256": compute_file_sha256(args.test_manifest),
            "total_videos_evaluated": len(base_ids),
            "class_distribution": {"real": args.real_count, "fake": args.fake_count},
            "offset_real": args.offset_real,
            "offset_fake": args.offset_fake,
            "evaluated_video_ids": base_ids,
            "sample_alignment_verified": (base_ids == cand_ids)
        },
        "baseline": base_res,
        "candidate": cand_res
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print("\n" + "=" * 90)
    print("DISTORTION SEVERITY DEGRADATION CURVE: BASELINE vs CANDIDATE")
    print("=" * 90)
    print(f"{'Condition':30s} | {'Baseline Acc':14s} | {'Candidate Acc':14s} | {'Delta Acc':10s} | {'Baseline F1':12s} | {'Candidate F1':12s}")
    print("-" * 90)
    for key in base_res:
        b = base_res[key]
        c = cand_res[key]
        d_acc = c["accuracy"] - b["accuracy"]
        print(f"{b['description']:30s} | {b['accuracy']:.2%}         | {c['accuracy']:.2%}          | {d_acc:+.2%}    | {b['macro_f1']:.4f}      | {c['macro_f1']:.4f}")
    print("=" * 90)
    print(f"Saved severity analysis to: {args.output}")


if __name__ == "__main__":
    main()
