"""Standardized evaluation metrics for classification tasks."""

from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def compute_classification_metrics(y_true: List[int], y_pred: List[int], labels: List[str] = None) -> Dict[str, Any]:
    """Compute accuracy, precision, recall, f1, and confusion matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    report = {
        "accuracy": round(acc, 4),
        "macro": {
            "precision": round(float(p_macro), 4),
            "recall": round(float(r_macro), 4),
            "f1": round(float(f1_macro), 4)
        },
        "weighted": {
            "precision": round(float(p_weight), 4),
            "recall": round(float(r_weight), 4),
            "f1": round(float(f1_weight), 4)
        },
        "confusion_matrix": cm,
        "sample_count": len(y_true)
    }
    return report
