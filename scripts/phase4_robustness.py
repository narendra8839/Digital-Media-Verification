import os
import json
import torch
import numpy as np
import io
from PIL import Image, ImageFilter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from models.deepfake.predict import load_deepfake_model, predict_deepfake_crop
from preprocessing.frame_sampling import sample_video_frames
from preprocessing.face_detection import FaceDetector

def degrade_image(img_arr, method="original", seed=42):
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
        orig_size = img.size
        # Downscale by factor of 4, then bilinear upscale back
        img = img.resize((max(1, orig_size[0] // 4), max(1, orig_size[1] // 4)), Image.BILINEAR)
        img = img.resize(orig_size, Image.BILINEAR)
    elif method == "noise":
        arr = np.array(img, dtype=np.float32)
        rng = np.random.RandomState(seed)
        noise = rng.normal(0, 15, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        
    return np.array(img)

def run_balanced_robustness(test_manifest, checkpoint_path, max_frames=4, real_count=50, fake_count=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n==========================================")
    print(f"--- Balanced Deepfake Robustness Evaluation ---")
    print(f"==========================================")
    print(f"Device: {device}")
    model, device = load_deepfake_model(checkpoint_path, device)
    detector = FaceDetector(device=device)
    
    with open(test_manifest, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Deterministic selection: first 50 Real and first 50 Fake
    real_pool = [x for x in data if x["label"] == "real"]
    fake_pool = [x for x in data if x["label"] == "fake"]
    
    selected_real = real_pool[:real_count]
    selected_fake = fake_pool[:fake_count]
    balanced_subset = selected_real + selected_fake
    total_samples = len(balanced_subset)
    
    print(f"Selected balanced evaluation subset: {len(selected_real)} Real + {len(selected_fake)} Fake = {total_samples} videos")
    print(f"Sampling depth: {max_frames} frames per video at 1.0 FPS (Identical across all conditions)")
    
    degradations = ["original", "jpeg", "blur", "resize", "noise"]
    results = {d: {"y_true": [], "y_pred": []} for d in degradations}
    
    # Process each video: decode once, then test all 5 conditions on the exact same frames
    for idx, item in enumerate(tqdm(balanced_subset, desc="Evaluating videos across conditions")):
        rel_path = item["relative_path"]
        video_path = os.path.join("data/faceforensics/C23", rel_path)
        gt = item["label"]
        
        if not os.path.exists(video_path):
            print(f"Warning: Missing video {video_path}")
            continue
            
        sampled = sample_video_frames(video_path, target_fps=1.0, max_frames=max_frames)
        if not sampled:
            print(f"Warning: No frames sampled for {video_path}")
            continue
            
        for d in degradations:
            fake_probs = []
            for frame_idx, (ts, frame_rgb) in enumerate(sampled):
                # Apply degradation with deterministic seed per frame
                seed = 42 + idx * 10 + frame_idx
                deg_frame = degrade_image(frame_rgb, method=d, seed=seed)
                crop, is_detected = detector.detect_and_crop(deg_frame, target_size=(224, 224))
                res = predict_deepfake_crop(model, crop, device)
                fake_probs.append(res["fake_probability"])
                
            if fake_probs:
                mean_fake_prob = float(np.mean(fake_probs))
                pred_label = "fake" if mean_fake_prob >= 0.5 else "real"
                results[d]["y_true"].append(gt)
                results[d]["y_pred"].append(pred_label)
                
    # Summarize metrics per condition
    summary = {
        "metadata": {
            "total_videos_evaluated": total_samples,
            "class_distribution": {"real": len(selected_real), "fake": len(selected_fake)},
            "frames_per_video": max_frames,
            "conditions_evaluated": degradations
        },
        "conditions": {}
    }
    
    print("\n--- ROBUSTNESS EVALUATION SUMMARY (BALANCED SUBSET: 50 Real, 50 Fake) ---")
    orig_acc = None
    for d in degradations:
        y_true = results[d]["y_true"]
        y_pred = results[d]["y_pred"]
        n_samples = len(y_true)
        acc = accuracy_score(y_true, y_pred) if n_samples > 0 else 0.0
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        c_prec, c_rec, c_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=['real', 'fake'], zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=['fake', 'real']).tolist()
        
        if d == "original":
            orig_acc = acc
            drop_from_orig = 0.0
        else:
            drop_from_orig = float(orig_acc - acc) if orig_acc is not None else 0.0
            
        summary["conditions"][d] = {
            "sample_count": n_samples,
            "class_distribution": {
                "real": sum(1 for x in y_true if x == "real"),
                "fake": sum(1 for x in y_true if x == "fake")
            },
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "macro_precision": float(prec),
            "macro_recall": float(rec),
            "drop_from_original": float(drop_from_orig),
            "per_class": {
                "real": {"precision": float(c_prec[0]), "recall": float(c_rec[0]), "f1": float(c_f1[0])},
                "fake": {"precision": float(c_prec[1]), "recall": float(c_rec[1]), "f1": float(c_f1[1])}
            },
            "confusion_matrix": cm
        }
        
        print(f"{d.capitalize():12s} | Acc: {acc:.4f} | Macro F1: {f1:.4f} | Drop: {drop_from_orig:+.4f} | CM (Fake/Real): {cm}")
        
    with open("phase4_robustness.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print("\nSaved balanced robustness results to phase4_robustness.json")
    return summary

def main():
    test_manifest = "data/full_splits/deepfake_test.json"
    checkpoint = "models/deepfake/checkpoint_full/best_model.pt"
    run_balanced_robustness(test_manifest, checkpoint, max_frames=4, real_count=50, fake_count=50)

if __name__ == "__main__":
    main()
