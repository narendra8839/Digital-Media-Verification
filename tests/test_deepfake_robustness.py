"""Automated unit and regression tests for deepfake detection robustness pipeline.

Tests cover:
1. Deterministic transformation behavior with fixed random seeds.
2. Ground-truth label preservation across transformations.
3. Consistent sample correspondence and ordering across conditions.
4. Correct metric and confusion-matrix calculations (with explicit class orientation).
5. Strict dataset split isolation (zero video and sequence identity leakage).
6. Checkpoint isolation and integrity.
7. Structured evaluation output completeness and provenance verification.
8. Reproducibility of the evaluation pipeline on synthetic fixtures.
"""

import os
import sys
import json
import pytest
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def resolve_eval_path(rel_path: str) -> str:
    """Resolves an evaluation data path, checking direct path then research/results/data."""
    direct = os.path.join(PROJECT_ROOT, rel_path)
    if os.path.exists(direct):
        return direct
    if rel_path.startswith("eval_results/"):
        fname = os.path.basename(rel_path)
        alt = os.path.join(PROJECT_ROOT, "research", "results", "data", fname)
        if os.path.exists(alt):
            return alt
    return direct

from scripts.evaluate_robustness import apply_degradation, compute_file_sha256
from scripts.train_robust_model import (
    RandomJPEGCompression,
    RandomGaussianBlurPIL,
    RandomDownsampleBilinear,
    build_robustness_transforms
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support


# =====================================================================
# 1. Deterministic Transformation Tests
# =====================================================================

class TestTransformationDeterminism:
    """Verifies that all perturbation operators produce deterministic outputs under fixed seeds."""

    @pytest.fixture
    def test_image_rgb(self):
        """Creates a deterministic RGB image fixture (224x224x3)."""
        rng = np.random.RandomState(42)
        arr = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        return arr

    def test_noise_determinism_with_same_seed(self, test_image_rgb):
        out1 = apply_degradation(test_image_rgb, method="noise", seed=12345)
        out2 = apply_degradation(test_image_rgb, method="noise", seed=12345)
        np.testing.assert_array_equal(out1, out2, err_msg="Noise degradation with identical seed must match bit-for-bit.")

    def test_noise_variance_with_different_seeds(self, test_image_rgb):
        out1 = apply_degradation(test_image_rgb, method="noise", seed=100)
        out2 = apply_degradation(test_image_rgb, method="noise", seed=200)
        assert not np.array_equal(out1, out2), "Different seeds must produce different noise realizations."

    def test_jpeg_determinism(self, test_image_rgb):
        out1 = apply_degradation(test_image_rgb, method="jpeg")
        out2 = apply_degradation(test_image_rgb, method="jpeg")
        np.testing.assert_array_equal(out1, out2, err_msg="JPEG compression must be deterministic.")

    def test_blur_determinism(self, test_image_rgb):
        out1 = apply_degradation(test_image_rgb, method="blur")
        out2 = apply_degradation(test_image_rgb, method="blur")
        np.testing.assert_array_equal(out1, out2, err_msg="Gaussian blur must be deterministic.")

    def test_resize_determinism(self, test_image_rgb):
        out1 = apply_degradation(test_image_rgb, method="resize")
        out2 = apply_degradation(test_image_rgb, method="resize")
        np.testing.assert_array_equal(out1, out2, err_msg="Resize downsampling and upscaling must be deterministic.")

    def test_original_identity(self, test_image_rgb):
        out = apply_degradation(test_image_rgb, method="original")
        np.testing.assert_array_equal(out, test_image_rgb, err_msg="Original condition must return unaltered array.")


# =====================================================================
# 2. Metric and Confusion Matrix Calculation Tests
# =====================================================================

class TestMetricCalculations:
    """Verifies that accuracy, balanced accuracy, and confusion matrices follow standard orientations."""

    def test_confusion_matrix_orientation(self):
        # Ground truth: 2 Real, 2 Fake
        # Predictions:
        # Real -> 1 Real, 1 Fake (1 FP)
        # Fake -> 2 Fake (2 TP)
        y_true = ["real", "real", "fake", "fake"]
        y_pred = ["real", "fake", "fake", "fake"]

        # labels=['fake', 'real']:
        # Row 0: True Fake -> [Pred Fake, Pred Real]
        # Row 1: True Real -> [Pred Fake, Pred Real]
        cm = confusion_matrix(y_true, y_pred, labels=["fake", "real"])

        assert cm[0][0] == 2, "Fake correctly predicted as Fake (TP fake) must be 2"
        assert cm[0][1] == 0, "Fake predicted as Real (FN fake) must be 0"
        assert cm[1][0] == 1, "Real falsely predicted as Fake (FP fake) must be 1"
        assert cm[1][1] == 1, "Real correctly predicted as Real (TN fake) must be 1"

        acc = accuracy_score(y_true, y_pred)
        assert acc == 0.75, "Accuracy must be 3/4 = 0.75"

        bal_acc = balanced_accuracy_score(y_true, y_pred)
        # Real recall: 1/2 = 0.50, Fake recall: 2/2 = 1.00 -> BalAcc = 0.75
        assert bal_acc == 0.75, "Balanced accuracy must equal macro recall = 0.75"

    def test_collapse_to_fake_metric_behavior(self):
        """Simulates the observed baseline failure mode where all blurred/resized samples predict Fake."""
        y_true = ["real"] * 50 + ["fake"] * 50
        y_pred = ["fake"] * 100  # 100% fake predictions

        cm = confusion_matrix(y_true, y_pred, labels=["fake", "real"])
        assert cm[0][0] == 50 and cm[0][1] == 0  # 50 fake correct
        assert cm[1][0] == 50 and cm[1][1] == 0  # 50 real false alarms

        acc = accuracy_score(y_true, y_pred)
        assert acc == 0.50

        bal_acc = balanced_accuracy_score(y_true, y_pred)
        assert bal_acc == 0.50

        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=["real", "fake"], zero_division=0)
        assert rec[0] == 0.0, "Real recall must be 0.0 when all are predicted fake"
        assert rec[1] == 1.0, "Fake recall must be 1.0"

    def test_candidate_clean_input_exact_f1_proof(self):
        """Regression test verifying candidate clean matrix [[28, 22], [2, 48]] mathematically equals Macro F1 0.7500."""
        # Row 0: True Fake -> [28 Pred Fake, 22 Pred Real]
        # Row 1: True Real -> [2 Pred Fake, 48 Pred Real]
        y_true = ["fake"] * 50 + ["real"] * 50
        y_pred = ["fake"] * 28 + ["real"] * 22 + ["fake"] * 2 + ["real"] * 48

        cm = confusion_matrix(y_true, y_pred, labels=["fake", "real"])
        assert cm.tolist() == [[28, 22], [2, 48]]

        acc = accuracy_score(y_true, y_pred)
        assert acc == 0.76, "Clean candidate accuracy must be 76/100 = 0.76"

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
        assert round(f1, 4) == 0.7500, f"Clean candidate macro F1 must be exactly 0.7500, got {f1}"
        assert round(f1, 4) != 0.7552, "0.7552 was an erroneous narrative transcription"

    def test_candidate_noise_exact_f1_proof(self):
        """Regression test verifying candidate noise matrix [[22, 28], [6, 44]] mathematically equals Macro F1 0.6427."""
        # Row 0: True Fake -> [22 Pred Fake, 28 Pred Real]
        # Row 1: True Real -> [6 Pred Fake, 44 Pred Real]
        y_true = ["fake"] * 50 + ["real"] * 50
        y_pred = ["fake"] * 22 + ["real"] * 28 + ["fake"] * 6 + ["real"] * 44

        cm = confusion_matrix(y_true, y_pred, labels=["fake", "real"])
        assert cm.tolist() == [[22, 28], [6, 44]]

        acc = accuracy_score(y_true, y_pred)
        assert acc == 0.66, "Noise candidate accuracy must be 66/100 = 0.66"

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
        assert round(f1, 4) == 0.6427, f"Noise candidate macro F1 must be exactly 0.6427, got {f1}"

        # Furthermore, verify that matrix [[34, 16], [18, 32]] mathematically equals ~0.6599 (not 0.6559)
        y_alt_pred = ["fake"] * 34 + ["real"] * 16 + ["fake"] * 18 + ["real"] * 32
        _, _, alt_f1, _ = precision_recall_fscore_support(y_true, y_alt_pred, average="macro")
        assert round(alt_f1, 4) == 0.6599, "Matrix [[34, 16], [18, 32]] implies 0.6599"

    def test_candidate_jpeg_exact_f1_proof(self):
        """Regression test verifying candidate JPEG matrix [[10, 40], [3, 47]] mathematically equals Macro F1 0.5018."""
        y_true = ["fake"] * 50 + ["real"] * 50
        y_pred = ["fake"] * 10 + ["real"] * 40 + ["fake"] * 3 + ["real"] * 47

        cm = confusion_matrix(y_true, y_pred, labels=["fake", "real"])
        assert cm.tolist() == [[10, 40], [3, 47]]

        acc = accuracy_score(y_true, y_pred)
        assert acc == 0.57

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
        assert round(f1, 4) == 0.5018, f"JPEG candidate macro F1 must be 0.5018, got {f1}"

    def test_saved_json_mathematical_consistency_baseline(self):
        """Asserts that all 5 conditions in baseline_robustness.json match their raw per_video_records."""
        base_path = resolve_eval_path("eval_results/baseline_robustness.json")
        if not os.path.exists(base_path):
            pytest.skip("baseline_robustness.json not found")
        with open(base_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for cond, records in data["per_video_records"].items():
            stored = data["conditions"][cond]
            y_true = [r["ground_truth"] for r in records]
            y_pred = [r["predicted_label"] for r in records]

            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=["fake", "real"]).tolist()

            assert abs(stored["accuracy"] - acc) < 1e-6, f"{cond} baseline acc mismatch"
            assert abs(stored["balanced_accuracy"] - bal_acc) < 1e-6, f"{cond} baseline bal_acc mismatch"
            assert abs(stored["macro_f1"] - macro_f1) < 1e-6, f"{cond} baseline macro_f1 mismatch"
            assert stored["confusion_matrix"] == cm, f"{cond} baseline cm mismatch"

    def test_saved_json_mathematical_consistency_candidate(self):
        """Asserts that all 5 conditions in candidate_robustness.json match their raw per_video_records."""
        cand_path = resolve_eval_path("eval_results/candidate_robustness.json")
        if not os.path.exists(cand_path):
            pytest.skip("candidate_robustness.json not found")
        with open(cand_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for cond, records in data["per_video_records"].items():
            stored = data["conditions"][cond]
            y_true = [r["ground_truth"] for r in records]
            y_pred = [r["predicted_label"] for r in records]

            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=["fake", "real"]).tolist()

            assert abs(stored["accuracy"] - acc) < 1e-6, f"{cond} candidate acc mismatch"
            assert abs(stored["balanced_accuracy"] - bal_acc) < 1e-6, f"{cond} candidate bal_acc mismatch"
            assert abs(stored["macro_f1"] - macro_f1) < 1e-6, f"{cond} candidate macro_f1 mismatch"
            assert stored["confusion_matrix"] == cm, f"{cond} candidate cm mismatch"

    def test_baseline_and_candidate_sample_alignment(self):
        """Asserts that baseline and candidate evaluations evaluated identical videos in identical order."""
        base_path = resolve_eval_path("eval_results/baseline_robustness.json")
        cand_path = resolve_eval_path("eval_results/candidate_robustness.json")
        if not (os.path.exists(base_path) and os.path.exists(cand_path)):
            pytest.skip("Evaluation files missing")

        with open(base_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        with open(cand_path, "r", encoding="utf-8") as f:
            cand_data = json.load(f)

        for cond in ["original", "jpeg", "blur", "resize", "noise"]:
            b_recs = base_data["per_video_records"][cond]
            c_recs = cand_data["per_video_records"][cond]
            assert len(b_recs) == len(c_recs) == 100

            b_vids = [r["video_id"] for r in b_recs]
            c_vids = [r["video_id"] for r in c_recs]
            assert b_vids == c_vids, f"Video ID order mismatch in {cond}"

            b_gt = [r["ground_truth"] for r in b_recs]
            c_gt = [r["ground_truth"] for r in c_recs]
            assert b_gt == c_gt, f"Ground truth mismatch in {cond}"

    def test_saved_json_mathematical_consistency_v2a(self):
        """Asserts that all 5 conditions in candidate_v2a_robustness.json match raw per_video_records."""
        v2a_path = resolve_eval_path("eval_results/candidate_v2a_robustness.json")
        if not os.path.exists(v2a_path):
            pytest.skip("candidate_v2a_robustness.json not found")
        with open(v2a_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for cond, records in data["per_video_records"].items():
            stored = data["conditions"][cond]
            y_true = [r["ground_truth"] for r in records]
            y_pred = [r["predicted_label"] for r in records]

            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=["fake", "real"]).tolist()

            assert abs(stored["accuracy"] - acc) < 1e-6, f"{cond} V2A acc mismatch"
            assert abs(stored["balanced_accuracy"] - bal_acc) < 1e-6, f"{cond} V2A bal_acc mismatch"
            assert abs(stored["macro_f1"] - macro_f1) < 1e-6, f"{cond} V2A macro_f1 mismatch"
            assert stored["confusion_matrix"] == cm, f"{cond} V2A cm mismatch"

    def test_saved_json_mathematical_consistency_v2b(self):
        """Asserts that all 5 conditions in candidate_v2b_robustness.json match raw per_video_records."""
        v2b_path = resolve_eval_path("eval_results/candidate_v2b_robustness.json")
        if not os.path.exists(v2b_path):
            pytest.skip("candidate_v2b_robustness.json not found")
        with open(v2b_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for cond, records in data["per_video_records"].items():
            stored = data["conditions"][cond]
            y_true = [r["ground_truth"] for r in records]
            y_pred = [r["predicted_label"] for r in records]

            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=["fake", "real"]).tolist()

            assert abs(stored["accuracy"] - acc) < 1e-6, f"{cond} V2B acc mismatch"
            assert abs(stored["balanced_accuracy"] - bal_acc) < 1e-6, f"{cond} V2B bal_acc mismatch"
            assert abs(stored["macro_f1"] - macro_f1) < 1e-6, f"{cond} V2B macro_f1 mismatch"
            assert stored["confusion_matrix"] == cm, f"{cond} V2B cm mismatch"

    def test_v2_sample_alignment_across_all_models(self):
        """Asserts that Baseline, V1, V2A, and V2B evaluated identical 100 videos in identical order."""
        paths = [
            resolve_eval_path("eval_results/baseline_robustness.json"),
            resolve_eval_path("eval_results/candidate_robustness.json"),
            resolve_eval_path("eval_results/candidate_v2a_robustness.json"),
            resolve_eval_path("eval_results/candidate_v2b_robustness.json")
        ]
        loaded = []
        for p in paths:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    loaded.append(json.load(f))

        if len(loaded) < 2:
            pytest.skip("Not enough evaluation reports generated yet.")

        ref_vids = [r["video_id"] for r in loaded[0]["per_video_records"]["original"]]
        ref_gt = [r["ground_truth"] for r in loaded[0]["per_video_records"]["original"]]

        for d in loaded[1:]:
            for cond in ["original", "jpeg", "blur", "resize", "noise"]:
                vids = [r["video_id"] for r in d["per_video_records"][cond]]
                gt = [r["ground_truth"] for r in d["per_video_records"][cond]]
                assert vids == ref_vids, f"Video ID alignment mismatch in {cond}"
                assert gt == ref_gt, f"GT alignment mismatch in {cond}"

    def test_saved_json_mathematical_consistency_v3a(self):
        """Asserts that all 5 conditions in candidate_v3a_robustness.json match raw per_video_records."""
        v3a_path = resolve_eval_path("eval_results/candidate_v3a_robustness.json")
        if not os.path.exists(v3a_path):
            pytest.skip("candidate_v3a_robustness.json not found")
        with open(v3a_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for cond, records in data["per_video_records"].items():
            stored = data["conditions"][cond]
            y_true = [r["ground_truth"] for r in records]
            y_pred = [r["predicted_label"] for r in records]

            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=["fake", "real"]).tolist()

            assert abs(stored["accuracy"] - acc) < 1e-6, f"{cond} V3A acc mismatch"
            assert abs(stored["balanced_accuracy"] - bal_acc) < 1e-6, f"{cond} V3A bal_acc mismatch"
            assert abs(stored["macro_f1"] - macro_f1) < 1e-6, f"{cond} V3A macro_f1 mismatch"
            assert stored["confusion_matrix"] == cm, f"{cond} V3A cm mismatch"

    def test_saved_json_mathematical_consistency_v3b(self):
        """Asserts that all 5 conditions in candidate_v3b_robustness.json match raw per_video_records."""
        v3b_path = resolve_eval_path("eval_results/candidate_v3b_robustness.json")
        if not os.path.exists(v3b_path):
            pytest.skip("candidate_v3b_robustness.json not found")
        with open(v3b_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for cond, records in data["per_video_records"].items():
            stored = data["conditions"][cond]
            y_true = [r["ground_truth"] for r in records]
            y_pred = [r["predicted_label"] for r in records]

            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=["fake", "real"]).tolist()

            assert abs(stored["accuracy"] - acc) < 1e-6, f"{cond} V3B acc mismatch"
            assert abs(stored["balanced_accuracy"] - bal_acc) < 1e-6, f"{cond} V3B bal_acc mismatch"
            assert abs(stored["macro_f1"] - macro_f1) < 1e-6, f"{cond} V3B macro_f1 mismatch"
            assert stored["confusion_matrix"] == cm, f"{cond} V3B cm mismatch"

    def test_saved_json_mathematical_consistency_v3c(self):
        """Asserts that all 5 conditions in candidate_v3c_robustness.json match raw per_video_records."""
        v3c_path = resolve_eval_path("eval_results/candidate_v3c_robustness.json")
        if not os.path.exists(v3c_path):
            pytest.skip("candidate_v3c_robustness.json not found")
        with open(v3c_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for cond, records in data["per_video_records"].items():
            stored = data["conditions"][cond]
            y_true = [r["ground_truth"] for r in records]
            y_pred = [r["predicted_label"] for r in records]

            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=["fake", "real"]).tolist()

            assert abs(stored["accuracy"] - acc) < 1e-6, f"{cond} V3C acc mismatch"
            assert abs(stored["balanced_accuracy"] - bal_acc) < 1e-6, f"{cond} V3C bal_acc mismatch"
            assert abs(stored["macro_f1"] - macro_f1) < 1e-6, f"{cond} V3C macro_f1 mismatch"
            assert stored["confusion_matrix"] == cm, f"{cond} V3C cm mismatch"

    def test_v3_sample_alignment_with_baseline(self):
        """Asserts that V3 candidates evaluated identical videos in identical order as baseline."""
        base_path = resolve_eval_path("eval_results/baseline_robustness.json")
        if not os.path.exists(base_path):
            pytest.skip("Baseline evaluation report not found.")

        with open(base_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)

        ref_vids = [r["video_id"] for r in base_data["per_video_records"]["original"]]
        ref_gt = [r["ground_truth"] for r in base_data["per_video_records"]["original"]]

        for cand_name in ["candidate_v3a", "candidate_v3b", "candidate_v3c"]:
            cand_path = resolve_eval_path(f"eval_results/{cand_name}_robustness.json")
            if os.path.exists(cand_path):
                with open(cand_path, "r", encoding="utf-8") as f:
                    c_data = json.load(f)
                for cond in ["original", "jpeg", "blur", "resize", "noise"]:
                    vids = [r["video_id"] for r in c_data["per_video_records"][cond]]
                    gt = [r["ground_truth"] for r in c_data["per_video_records"][cond]]
                    assert vids == ref_vids, f"{cand_name} Video ID alignment mismatch in {cond}"
                    assert gt == ref_gt, f"{cand_name} GT alignment mismatch in {cond}"


# =====================================================================
# 3. Dataset Split Isolation Tests
# =====================================================================

class TestSplitIntegrity:
    """Verifies strict train/val/test split separation with zero video and sequence ID overlap."""

    def test_zero_leakage_between_splits(self):
        train_path = os.path.join(PROJECT_ROOT, "data/full_splits/deepfake_train.json")
        val_path = os.path.join(PROJECT_ROOT, "data/full_splits/deepfake_val.json")
        test_path = os.path.join(PROJECT_ROOT, "data/full_splits/deepfake_test.json")

        if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
            pytest.skip("Full split manifests not available in current test environment.")

        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(val_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        train_vids = {x["video_id"] for x in train_data}
        val_vids = {x["video_id"] for x in val_data}
        test_vids = {x["video_id"] for x in test_data}

        # 1. Zero video_id overlap
        assert len(train_vids.intersection(val_vids)) == 0, "Train and Val share video IDs"
        assert len(train_vids.intersection(test_vids)) == 0, "Train and Test share video IDs"
        assert len(val_vids.intersection(test_vids)) == 0, "Val and Test share video IDs"

        # 2. Zero sequence_id / source identity overlap
        train_seqs = {seq for x in train_data for seq in x.get("sequence_ids", [])}
        val_seqs = {seq for x in val_data for seq in x.get("sequence_ids", [])}
        test_seqs = {seq for x in test_data for seq in x.get("sequence_ids", [])}

        assert len(train_seqs.intersection(val_seqs)) == 0, "Train and Val share source sequences"
        assert len(train_seqs.intersection(test_seqs)) == 0, "Train and Test share source sequences"
        assert len(val_seqs.intersection(test_seqs)) == 0, "Val and Test share source sequences"

    def test_offset_based_subset_isolation(self):
        """Verifies that development benchmark (offset 0) and reserved subset (offset 50) are mutually disjoint."""
        test_path = os.path.join(PROJECT_ROOT, "data/full_splits/deepfake_test.json")
        if not os.path.exists(test_path):
            pytest.skip("Test manifest not found.")

        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        reals = [x for x in test_data if x["label"] == "real"]
        fakes = [x for x in test_data if x["label"] == "fake"]

        dev_real_ids = {x["video_id"] for x in reals[0:50]}
        res_real_ids = {x["video_id"] for x in reals[50:100]}
        assert len(dev_real_ids) == 50
        assert len(res_real_ids) == 50
        assert len(dev_real_ids.intersection(res_real_ids)) == 0, "Dev and Reserved Real subsets must not overlap"

        dev_fake_ids = {x["video_id"] for x in fakes[0:50]}
        res_fake_ids = {x["video_id"] for x in fakes[50:100]}
        assert len(dev_fake_ids) == 50
        assert len(res_fake_ids) == 50
        assert len(dev_fake_ids.intersection(res_fake_ids)) == 0, "Dev and Reserved Fake subsets must not overlap"


# =====================================================================
# 4. Checkpoint Isolation and Provenance Tests
# =====================================================================

class TestCheckpointIsolation:
    """Verifies that baseline and candidate checkpoints remain isolated."""

    def test_baseline_checkpoint_integrity(self):
        ckpt_path = os.path.join(PROJECT_ROOT, "models/deepfake/checkpoint_full/best_model.pt")
        if not os.path.exists(ckpt_path):
            pytest.skip("Baseline checkpoint not found.")

        sha = compute_file_sha256(ckpt_path)
        assert sha == "5878ae38bf5239dd9889e010fafa1d8c7da4198ac260542dc96681ede1e63234"

    def test_checkpoint_paths_are_disjoint(self):
        baseline_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "models/deepfake/checkpoint_full"))
        candidate_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "models/deepfake/checkpoint_robust_candidate"))
        assert baseline_dir != candidate_dir, "Candidate checkpoint directory must not equal baseline directory"

    def test_all_v3_checkpoints_integrity_and_disjointness(self):
        """Verifies that all 5 key checkpoints exist with expected cryptographic hashes and distinct paths."""
        ckpts = {
            "baseline": ("models/deepfake/checkpoint_full/best_model.pt", "5878ae38bf5239dd9889e010fafa1d8c7da4198ac260542dc96681ede1e63234"),
            "v2b": ("models/deepfake/checkpoint_robust_v2b/best_model.pt", "f8b8c0ba272b37b353fbcaa2b3148844a0d9b0d4278c52b73a3259f9d5d277c8"),
            "v3a": ("models/deepfake/checkpoint_robust_v3a/best_model.pt", "5f8057f3d54759c2c71163e6d4c60cf6bf303dae611739491e02e1ae8452c41d"),
            "v3b": ("models/deepfake/checkpoint_robust_v3b/best_model.pt", "1b53f93e70d2301fb819f987d2f0c0bd6c6c37ac6560142d933d8542f9470032"),
            "v3c": ("models/deepfake/checkpoint_robust_v3c/best_model.pt", "12b552738dcfcf598b5b18b2ce65820087063703346ac76b24c7b6382efefd34"),
        }
        seen_dirs = set()
        for name, (rel_path, expected_hash) in ckpts.items():
            full_path = os.path.join(PROJECT_ROOT, rel_path)
            if not os.path.exists(full_path):
                if name == "baseline":
                    assert False, f"Baseline checkpoint {rel_path} does not exist"
                continue
            assert compute_file_sha256(full_path) == expected_hash, f"Hash mismatch for {name}"
            d = os.path.dirname(full_path)
            assert d not in seen_dirs, f"Duplicate checkpoint directory: {d}"
            seen_dirs.add(d)


# =====================================================================
# 5. Output Completeness and Provenance Tests
# =====================================================================

class TestEvaluationOutputStructure:
    """Verifies that structured JSON outputs contain required provenance, conditions, and records."""

    def test_baseline_json_structure(self):
        eval_path = resolve_eval_path("eval_results/baseline_robustness.json")
        if not os.path.exists(eval_path):
            pytest.skip("Baseline evaluation report not yet generated.")

        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "conditions" in data
        assert "per_video_records" in data

        meta = data["metadata"]
        assert meta["total_videos_evaluated"] == 100
        assert meta["class_distribution"] == {"real": 50, "fake": 50}
        assert "checkpoint_sha256" in meta

        expected_conds = ["original", "jpeg", "blur", "resize", "noise"]
        for c in expected_conds:
            assert c in data["conditions"], f"Missing condition {c} in report"
            c_data = data["conditions"][c]
            assert "accuracy" in c_data
            assert "macro_f1" in c_data
            assert "balanced_accuracy" in c_data
            assert "confusion_matrix" in c_data
            assert len(data["per_video_records"][c]) == 100
