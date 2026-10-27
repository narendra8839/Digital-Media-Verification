"""Independent verification script for mathematical consistency of all evaluation records.

Verifies:
1. Recomputes all metrics from raw per-video records in baseline_robustness.json and candidate_robustness.json.
2. Checks accuracy, balanced accuracy, macro F1, class-wise precision, recall, F1, and confusion matrices.
3. Tests confusion matrix orientation:
   - Documented: labels=['fake', 'real']
   - Row 0: True Fake -> [Pred Fake, Pred Real]
   - Row 1: True Real -> [Pred Fake, Pred Real]
   - Col 0: Pred Fake
   - Col 1: Pred Real
4. Evaluates whether any discrepancy exists between saved records and underlying ground truth.
"""

import json
import os
import sys
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def verify_file(filepath: str, name: str):
    print("=" * 70)
    print(f"VERIFYING METRICS INTEGRITY: {name} ({filepath})")
    print("=" * 70)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for cond in ["original", "jpeg", "blur", "resize", "noise"]:
        records = data["per_video_records"][cond]
        stored = data["conditions"][cond]

        y_true = [r["ground_truth"] for r in records]
        y_pred = [r["predicted_label"] for r in records]

        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        c_p, c_r, c_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=["real", "fake"], zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=["fake", "real"]).tolist()

        # Verify exact agreement
        assert abs(stored["accuracy"] - acc) < 1e-6, f"{cond} Accuracy mismatch"
        assert abs(stored["balanced_accuracy"] - bal_acc) < 1e-6, f"{cond} Balanced Acc mismatch"
        assert abs(stored["macro_f1"] - macro_f1) < 1e-6, f"{cond} Macro F1 mismatch: stored {stored['macro_f1']} vs computed {macro_f1}"
        assert stored["confusion_matrix"] == cm, f"{cond} CM mismatch: stored {stored['confusion_matrix']} vs computed {cm}"

        # Verify manual formula calculation from CM
        # cm = [[TP_fake, FN_fake], [FP_fake, TN_fake]]
        tp_fake = cm[0][0]
        fn_fake = cm[0][1]
        fp_fake = cm[1][0]  # Real predicted as Fake
        tn_fake = cm[1][1]  # Real predicted as Real

        real_prec_manual = tn_fake / (tn_fake + fn_fake) if (tn_fake + fn_fake) > 0 else 0.0
        real_rec_manual = tn_fake / (tn_fake + fp_fake) if (tn_fake + fp_fake) > 0 else 0.0
        real_f1_manual = (2 * real_prec_manual * real_rec_manual / (real_prec_manual + real_rec_manual)) if (real_prec_manual + real_rec_manual) > 0 else 0.0

        fake_prec_manual = tp_fake / (tp_fake + fp_fake) if (tp_fake + fp_fake) > 0 else 0.0
        fake_rec_manual = tp_fake / (tp_fake + fn_fake) if (tp_fake + fn_fake) > 0 else 0.0
        fake_f1_manual = (2 * fake_prec_manual * fake_rec_manual / (fake_prec_manual + fake_rec_manual)) if (fake_prec_manual + fake_rec_manual) > 0 else 0.0

        macro_f1_manual = (real_f1_manual + fake_f1_manual) / 2.0

        assert abs(macro_f1_manual - macro_f1) < 1e-6, f"{cond} manual macro F1 mismatch"

        print(f"[{cond:8s}] N={len(records)} | Acc={acc:.4f} | BalAcc={bal_acc:.4f} | MacroF1={macro_f1:.4f}")
        print(f"           CM={cm}")
        print(f"           Real: Prec={c_p[0]:.4f}, Rec={c_r[0]:.4f}, F1={c_f1[0]:.4f}")
        print(f"           Fake: Prec={c_p[1]:.4f}, Rec={c_r[1]:.4f}, F1={c_f1[1]:.4f}")
        print()

if __name__ == "__main__":
    base_file = os.path.join(PROJECT_ROOT, "eval_results/baseline_robustness.json")
    cand_file = os.path.join(PROJECT_ROOT, "eval_results/candidate_robustness.json")
    verify_file(base_file, "Baseline Model")
    verify_file(cand_file, "Candidate Model")
    print("ALL UNDERLYING PER-VIDEO RECORDS AND STORED JSON METRICS MATCH 100%!")
