import os
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from models.deepfake.predict import load_deepfake_model, predict_deepfake_video

def evaluate_deepfake_full(test_file, checkpoint_dir, max_frames=4):
    print(f"\n==========================================")
    print(f"--- Full Deepfake Evaluation (All Videos) ---")
    print(f"==========================================")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Deepfake on {device}...")
    model, device = load_deepfake_model(checkpoint_dir, device)
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Total videos in manifest: {len(data)}")
    real_count = sum(1 for x in data if x["label"] == "real")
    fake_count = sum(1 for x in data if x["label"] == "fake")
    print(f"Manifest class distribution: Real={real_count}, Fake={fake_count}")
    
    y_true = []
    y_pred = []
    processed_details = []
    
    start_time = time.time()
    for idx, item in enumerate(tqdm(data, desc="Evaluating test videos")):
        rel_path = item["relative_path"]
        video_path = os.path.join("data/faceforensics/C23", rel_path)
        gt = item["label"]
        
        if not os.path.exists(video_path):
            print(f"Warning: File not found {video_path}")
            continue
            
        try:
            res = predict_deepfake_video(model, video_path, device, max_frames=max_frames)
            pred = res["prediction"]
            if pred != "unknown":
                y_true.append(gt)
                y_pred.append(pred)
                processed_details.append({
                    "video_id": item.get("video_id", rel_path),
                    "true_label": gt,
                    "pred_label": pred,
                    "confidence": res.get("confidence", 0.0),
                    "probabilities": res.get("probabilities", {})
                })
            else:
                print(f"Warning: Unknown prediction for {video_path}")
        except Exception as e:
            print(f"Error evaluating {video_path}: {e}")
            
    elapsed = time.time() - start_time
    total_evaluated = len(y_true)
    print(f"\nCompleted evaluation in {elapsed:.1f}s ({total_evaluated}/{len(data)} videos evaluated).")
    
    if total_evaluated == 0:
        raise RuntimeError("No videos were successfully evaluated!")
        
    acc = accuracy_score(y_true, y_pred)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Class-wise metrics with explicit label ordering: ['real', 'fake']
    c_prec, c_rec, c_f1, c_supp = precision_recall_fscore_support(
        y_true, y_pred, labels=['real', 'fake'], zero_division=0
    )
    
    # Standard scikit-learn confusion matrix with labels ['fake', 'real']
    # Row 0: True Fake, Row 1: True Real
    # Col 0: Pred Fake, Col 1: Pred Real
    cm_fake_real = confusion_matrix(y_true, y_pred, labels=['fake', 'real']).tolist()
    
    eval_real_count = sum(1 for x in y_true if x == "real")
    eval_fake_count = sum(1 for x in y_true if x == "fake")
    
    results = {
        "evaluated_count": total_evaluated,
        "class_distribution": {
            "real": eval_real_count,
            "fake": eval_fake_count
        },
        "accuracy": float(acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_prec),
        "weighted_recall": float(weighted_rec),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            "real": {
                "precision": float(c_prec[0]),
                "recall": float(c_rec[0]),
                "f1": float(c_f1[0]),
                "support": int(c_supp[0])
            },
            "fake": {
                "precision": float(c_prec[1]),
                "recall": float(c_rec[1]),
                "f1": float(c_f1[1]),
                "support": int(c_supp[1])
            }
        },
        "confusion_matrix_labels": ["fake", "real"],
        "confusion_matrix": cm_fake_real,
        "sampling_policy": {
            "target_fps": 1.0,
            "max_frames_per_video": max_frames,
            "frame_sampling_method": "deterministic_uniform_sequential"
        },
        "elapsed_seconds": round(elapsed, 2)
    }
    
    print("\n--- FINAL TEST RESULTS ---")
    print(f"Evaluated Count: {total_evaluated} (Real: {eval_real_count}, Fake: {eval_fake_count})")
    print(f"Overall Accuracy:  {acc:.4f}")
    print(f"Macro Precision:   {macro_prec:.4f}")
    print(f"Macro Recall:      {macro_rec:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Real Class: Precision={c_prec[0]:.4f}, Recall={c_rec[0]:.4f}, F1={c_f1[0]:.4f} (Support={c_supp[0]})")
    print(f"Fake Class: Precision={c_prec[1]:.4f}, Recall={c_rec[1]:.4f}, F1={c_f1[1]:.4f} (Support={c_supp[1]})")
    print(f"Confusion Matrix [[True Fake Pred Fake, Pred Real], [True Real Pred Fake, Pred Real]]:")
    print(f"  Fake: {cm_fake_real[0]}")
    print(f"  Real: {cm_fake_real[1]}")
    
    return results

def main():
    test_manifest = "data/full_splits/deepfake_test.json"
    checkpoint = "models/deepfake/checkpoint_full/best_model.pt"
    
    results = {
        "deepfake": evaluate_deepfake_full(test_manifest, checkpoint, max_frames=4)
    }
    
    with open("phase4_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print("\nSaved full evaluation results to phase4_results.json")

if __name__ == "__main__":
    main()
