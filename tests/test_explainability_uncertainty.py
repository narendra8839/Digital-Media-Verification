"""Comprehensive test suite for Explainability (Grad-CAM, Token Attribution, Rationale Alignment)
and Uncertainty Quantification (MC Dropout, Human Review Flag, Result Schema).
"""

import os
import sys
import pytest
import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from explainability.gradcam import GradCAM
from explainability.text_attention import TokenAttribution, evaluate_rationale_alignment
from explainability.explanation_generator import ExplanationGenerator, build_verification_result
from uncertainty.variance import (
    compute_predictive_mean,
    compute_predictive_variance,
    compute_predictive_std,
    compute_predictive_entropy,
    summarize_uncertainty
)
from uncertainty.mc_dropout import enable_mc_dropout, MCDropoutEstimator
from uncertainty.uncertainty_engine import UncertaintyEngine
from models.deepfake.predict import load_deepfake_model
from models.propaganda.predict import load_propaganda_model
from models.hate_speech.predict import load_hate_speech_model


class TestGradCAM:
    @classmethod
    def setup_class(cls):
        cls.model, cls.device = load_deepfake_model()
        cls.gradcam = GradCAM(model=cls.model, device=cls.device)

    def test_gradcam_generation(self):
        dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        res = self.gradcam.explain(dummy_img)

        assert "prediction" in res
        assert "confidence" in res
        assert res["prediction"] in ("real", "fake")
        assert len(res["heatmap_shape"]) == 2
        assert res["heatmap_shape"] == [7, 7]
        assert "disclaimer" in res
        assert "post-hoc" in res["disclaimer"]

    def test_gradcam_overlay_saving(self, tmp_path):
        dummy_img = Image.new("RGB", (224, 224), color=(200, 100, 50))
        save_file = str(tmp_path / "test_overlay.png")
        res = self.gradcam.explain(dummy_img, output_path=save_file)

        assert os.path.exists(save_file)
        saved_img = Image.open(save_file)
        assert saved_img.size == (224, 224)


class TestTokenAttribution:
    @classmethod
    def setup_class(cls):
        cls.pr_model, cls.pr_tok, cls.pr_labels, cls.device = load_propaganda_model()
        cls.hs_model, cls.hs_tok, cls.hs_labels, _ = load_hate_speech_model()
        cls.pr_explainer = TokenAttribution(cls.pr_model, cls.pr_tok, device=cls.device)
        cls.hs_explainer = TokenAttribution(cls.hs_model, cls.hs_tok, device=cls.device)

    def test_propaganda_token_attribution(self):
        text = "This is a dangerous and crooked conspiracy."
        res = self.pr_explainer.attribute(text, label_mapping=self.pr_labels)

        assert "prediction" in res
        assert "tokens" in res
        assert len(res["tokens"]) > 0
        assert "token_scores" in res
        assert len(res["top_tokens"]) > 0
        for item in res["top_tokens"]:
            assert 0.0 <= item["score"] <= 1.0

    def test_hate_speech_token_attribution(self):
        text = "go back to your country you filthy immigrant"
        res = self.hs_explainer.attribute(text, label_mapping=self.hs_labels)

        assert "prediction" in res
        assert res["prediction"] in ("normal", "offensive", "hatespeech")
        assert len(res["top_tokens"]) > 0
        assert res["top_tokens"][0]["score"] >= res["top_tokens"][-1]["score"]

    def test_rationale_alignment(self):
        attributed_tokens = [
            {"token": "go", "score": 0.1},
            {"token": "back", "score": 0.2},
            {"token": "filthy", "score": 0.95},
            {"token": "immigrant", "score": 0.85}
        ]
        ground_truth_rationales = [
            [0, 0, 1, 1],
            [0, 0, 1, 0]
        ]
        post_tokens = ["go", "back", "filthy", "immigrant"]

        alignment = evaluate_rationale_alignment(
            attributed_tokens=attributed_tokens,
            ground_truth_rationales=ground_truth_rationales,
            post_tokens=post_tokens,
            top_k=2
        )

        assert alignment["rationale_available"] is True
        assert alignment["overlap_count"] == 2
        assert set(alignment["overlap_tokens"]) == {"filthy", "immigrant"}
        assert alignment["precision"] == 1.0
        assert alignment["recall"] == 1.0
        assert alignment["f1"] == 1.0


class TestMCDropoutAndUncertainty:
    @classmethod
    def setup_class(cls):
        cls.df_model, cls.device = load_deepfake_model()
        cls.hs_model, cls.hs_tok, cls.hs_labels, _ = load_hate_speech_model()
        cls.pr_model, cls.pr_tok, cls.pr_labels, _ = load_propaganda_model()
        cls.estimator = MCDropoutEstimator(mc_samples=20, device=cls.device)

    def test_enable_mc_dropout(self):
        count_df = enable_mc_dropout(self.df_model)
        assert count_df >= 1, "At least 1 dropout layer in EfficientNet-B4"
        count_hs = enable_mc_dropout(self.hs_model)
        assert count_hs >= 1, "At least 1 dropout layer in BERT"
        count_pr = enable_mc_dropout(self.pr_model)
        assert count_pr >= 1, "At least 1 dropout layer in RoBERTa"

    def test_stochastic_variance_deepfake(self):
        dummy_crop = Image.new("RGB", (224, 224), color=(100, 150, 200))
        res = self.estimator.estimate_deepfake_crop(self.df_model, dummy_crop)

        assert res["mc_samples"] == 20
        assert res["max_variance"] > 0, "MC Dropout must produce non-zero variance!"
        assert res["is_stochastic"] is True
        assert len(res["variance"]) == 2

    def test_stochastic_variance_nlp(self):
        text = "This message may or may not contain controversial rhetoric."
        res_hs = self.estimator.estimate_text(self.hs_model, self.hs_tok, text, label_mapping=self.hs_labels)
        assert res_hs["mc_samples"] == 20
        assert res_hs["max_variance"] > 0, "BERT MC Dropout must produce non-zero variance"

        res_pr = self.estimator.estimate_text(self.pr_model, self.pr_tok, text, label_mapping=self.pr_labels)
        assert res_pr["mc_samples"] == 20
        assert res_pr["max_variance"] > 0, "RoBERTa MC Dropout must produce non-zero variance"

    def test_uncertainty_math(self):
        # 3 samples, 2 classes
        probs = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3]
        ])
        summary = summarize_uncertainty(probs)
        assert abs(summary["mean_probability"][0] - 0.8) < 1e-4
        assert abs(summary["mean_probability"][1] - 0.2) < 1e-4
        assert summary["variance"][0] > 0
        assert summary["entropy"] > 0


class TestHumanReviewAndSchema:
    def test_human_review_flag_logic(self):
        engine = UncertaintyEngine(variance_threshold=0.01, entropy_threshold=0.8)

        # High uncertainty case
        high_uq = {
            "max_variance": 0.05,
            "entropy": 0.95
        }
        res_high = engine.evaluate_policy(high_uq)
        assert res_high["review_required"] is True
        assert res_high["decision_status"] == "REVIEW_RECOMMENDED"
        assert "exceeds threshold" in res_high["review_justification"]

        # Low uncertainty case
        low_uq = {
            "max_variance": 0.001,
            "entropy": 0.3
        }
        res_low = engine.evaluate_policy(low_uq)
        assert res_low["review_required"] is False
        assert res_low["decision_status"] == "CONFIDENT_PREDICTION"

    def test_build_verification_result_schema(self):
        res = build_verification_result(
            modality="deepfake",
            prediction="fake",
            confidence=0.92,
            uncertainty_info={
                "mean_probability": [0.08, 0.92],
                "variance": [0.002, 0.002],
                "std": [0.045, 0.045],
                "entropy": 0.402,
                "max_variance": 0.002,
                "mc_samples": 20
            },
            review_info={
                "review_required": False,
                "decision_status": "CONFIDENT_PREDICTION",
                "review_justification": "All uncertainty parameters within bounds."
            },
            explanation_info={
                "heatmap_shape": [7, 7],
                "overlay_path": "reports/explanations/deepfake/test.png"
            }
        )

        assert res["modality"] == "deepfake"
        assert res["prediction"] == "fake"
        assert res["confidence"] == 0.92
        assert "uncertainty" in res
        assert res["uncertainty"]["mc_samples"] == 20
        assert res["review_required"] is False
        assert res["review_status"] == "CONFIDENT_PREDICTION"
        assert "explanation" in res
