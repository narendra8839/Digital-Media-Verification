"""Comprehensive verification script executing real samples across all XAI and UQ components."""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import torch
import numpy as np
from PIL import Image

from explainability.gradcam import GradCAM
from explainability.text_attention import TokenAttribution, evaluate_rationale_alignment
from explainability.explanation_generator import ExplanationGenerator, build_verification_result
from uncertainty.mc_dropout import enable_mc_dropout, MCDropoutEstimator
from uncertainty.uncertainty_engine import UncertaintyEngine
from models.deepfake.predict import load_deepfake_model
from models.propaganda.predict import load_propaganda_model
from models.hate_speech.predict import load_hate_speech_model
from preprocessing.frame_sampling import sample_video_frames
from preprocessing.face_detection import FaceDetector


def run_real_deepfake_gradcam():
    print("=" * 70)
    print("1. DEEPFAKE GRAD-CAM EXECUTION (REAL & FAKE VALIDATION EXAMPLES)")
    print("=" * 70)

    model, device = load_deepfake_model()
    gradcam = GradCAM(model=model, device=device)
    detector = FaceDetector(device="cpu")

    # Verify target layer
    target_layer = gradcam.target_layer
    print(f"Verified Target Layer: {target_layer.__class__.__name__}")
    assert "Conv2dNormActivation" in str(target_layer), "Target layer is not Conv2dNormActivation!"

    with open("data/mvp/deepfake/val_subset.json", "r", encoding="utf-8") as f:
        val_subset = json.load(f)

    # Pick 1 real video (index 0) and 1 fake video (index 10)
    real_sample = val_subset[0]
    fake_sample = val_subset[10]

    os.makedirs("reports/explanations/deepfake", exist_ok=True)

    for sample in [real_sample, fake_sample]:
        vid_id = sample["video_id"]
        gt_label = sample["label"]
        vpath = os.path.join("data/faceforensics/C23", sample["relative_path"])

        frames = sample_video_frames(vpath, target_fps=1.0, max_frames=1)
        ts, rgb = frames[0]
        crop, face_found = detector.detect_and_crop(rgb)
        assert face_found, f"Face not detected in {vid_id}"

        out_path = f"reports/explanations/deepfake/{vid_id}_{gt_label}_gradcam.png"
        res = gradcam.explain(crop, output_path=out_path)

        print(f"\nVideo ID: {vid_id} | Ground Truth: {gt_label}")
        print(f"  Predicted Label: {res['prediction']} (Confidence: {res['confidence']:.4f})")
        print(f"  Class Probabilities: {res['probabilities']}")
        print(f"  Heatmap Shape: {res['heatmap_shape']} | Min: {np.min(res['heatmap']):.4f}, Max: {np.max(res['heatmap']):.4f}")
        print(f"  Overlay Saved: {out_path} (Exists: {os.path.exists(out_path)})")
        print(f"  Captured Activations Shape: {gradcam.activations.shape}")
        print(f"  Captured Gradients Shape: {gradcam.gradients.shape}")

        assert gradcam.activations.numel() > 0, "Activations are empty!"
        assert gradcam.gradients.numel() > 0, "Gradients are empty!"
        assert os.path.exists(out_path), "Overlay image file was not created!"

    print("\n--> DEEPFAKE GRAD-CAM EXECUTION: PASSED AND VERIFIED!\n")


def run_real_propaganda_token_attribution():
    print("=" * 70)
    print("2. PROPAGANDA TOKEN ATTRIBUTION EXECUTION (SEMEVAL 14-CLASS)")
    print("=" * 70)

    model, tokenizer, label_mapping, device = load_propaganda_model()
    explainer = TokenAttribution(model, tokenizer, device=device)

    with open("data/mvp/semeval2020_task11/val_subset.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)

    sample = val_data[0]
    text = sample["text_span"]
    gt = sample["technique"]

    res = explainer.attribute(text, label_mapping=label_mapping)

    print(f"Sample Text: '{text}'")
    print(f"Ground Truth Technique: {gt}")
    print(f"Predicted Technique: {res['prediction']} (Confidence: {res['confidence']:.4f})")
    print("Tokens with Model Gradient Attribution Scores:")
    for item in res["token_scores"]:
        print(f"  {item['clean_token']:15s} : score = {item['score']:.4f}")

    print("\nTop 3 Influential Tokens:")
    for item in res["top_tokens"][:3]:
        print(f"  * {item['clean_token']:12s} (saliency: {item['score']:.4f})")

    assert len(res["top_tokens"]) > 0, "No top tokens extracted!"
    print("\n--> PROPAGANDA TOKEN ATTRIBUTION: PASSED AND VERIFIED!\n")


def run_real_hate_speech_and_rationale_alignment():
    print("=" * 70)
    print("3. HATE SPEECH ATTRIBUTION & RATIONALE ALIGNMENT (HATEXPLAIN 3-CLASS)")
    print("=" * 70)

    model, tokenizer, label_mapping, device = load_hate_speech_model()
    explainer = TokenAttribution(model, tokenizer, device=device)

    with open("data/mvp/hatexplain/test_subset.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Select a real hatespeech sample with ground-truth rationales
    sample = [x for x in test_data if x["label"] == "hatespeech"][0]
    text = sample["text"]
    gt = sample["label"]
    post_tokens = sample["post_tokens"]
    rationales = sample.get("rationales", [])

    print(f"Sample Post: '{text}'")
    print(f"Ground Truth Label: {gt}")
    print(f"Post Tokens ({len(post_tokens)} tokens): {post_tokens}")
    print(f"Ground Truth Annotator Rationale Masks: {rationales}")

    res = explainer.attribute(text, label_mapping=label_mapping)
    print(f"\nModel Prediction: {res['prediction']} (Confidence: {res['confidence']:.4f})")
    print(f"Class Probabilities: {res['probabilities']}")

    print("\nInfluential Tokens (Model Attribution):")
    for item in res["top_tokens"]:
        print(f"  * {item['clean_token']:12s} (saliency: {item['score']:.4f})")

    # Evaluate rationale alignment against human annotations
    alignment = evaluate_rationale_alignment(
        attributed_tokens=res["token_scores"],
        ground_truth_rationales=rationales,
        post_tokens=post_tokens,
        top_k=3
    )

    print("\nEmpirical Human Rationale Alignment Evaluation:")
    print(f"  Ground Truth Marked Tokens Count: {alignment['ground_truth_count']}")
    print(f"  Model Selected Tokens Count:      {alignment['model_selected_count']}")
    print(f"  Overlap Count:                   {alignment['overlap_count']}")
    print(f"  Overlapping Rationale Tokens:    {alignment['overlap_tokens']}")
    print(f"  Rationale Precision:             {alignment['precision']:.4f}")
    print(f"  Rationale Recall:                {alignment['recall']:.4f}")
    print(f"  Rationale F1:                    {alignment['f1']:.4f}")

    assert alignment["overlap_count"] > 0, "No rationale overlap found!"
    print("\n--> HATE SPEECH ATTRIBUTION & RATIONALE ALIGNMENT: PASSED AND VERIFIED!\n")


def run_real_mc_dropout_20_passes():
    print("=" * 70)
    print("4. MC DROPOUT UNCERTAINTY EXECUTION (T=20 STOCHASTIC FORWARD PASSES)")
    print("=" * 70)

    estimator = MCDropoutEstimator(mc_samples=20, device="cpu")

    # Test 1: Deepfake MC Dropout
    df_model, device = load_deepfake_model()
    dummy_crop = Image.new("RGB", (224, 224), color=(120, 140, 160))

    enable_mc_dropout(df_model)
    df_res = estimator.estimate_deepfake_crop(df_model, dummy_crop)

    print(f"A. Deepfake EfficientNet-B4 (20 stochastic passes):")
    print(f"   Prediction: {df_res['prediction']} (Mean Conf: {df_res['confidence']:.4f})")
    print(f"   Mean Probabilities: {df_res['probabilities']}")
    print(f"   Class Variances:    {df_res['class_variances']}")
    print(f"   Predictive Std:     {df_res['std']}")
    print(f"   Shannon Entropy:    {df_res['entropy']:.4f} bits")
    print(f"   Is Stochastic:      {df_res['is_stochastic']} (Max Var: {df_res['max_variance']:.6f})")
    assert df_res["is_stochastic"], "Deepfake outputs are deterministic!"

    # Test 2: Hate Speech BERT MC Dropout with 20 printed probability vectors
    hs_model, hs_tok, hs_labels, _ = load_hate_speech_model()
    text = "they should not be allowed here get them out"

    enable_mc_dropout(hs_model)
    hs_res = estimator.estimate_text(hs_model, hs_tok, text, label_mapping=hs_labels)

    print(f"\nB. Hate Speech BERT (20 stochastic passes):")
    print(f"   Input: '{text}'")
    print(f"   Prediction: {hs_res['prediction']} (Mean Conf: {hs_res['confidence']:.4f})")
    print(f"   Mean Probabilities: {hs_res['probabilities']}")
    print(f"   Class Variances:    {hs_res['class_variances']}")
    print(f"   Shannon Entropy:    {hs_res['entropy']:.4f} bits")
    print(f"   Is Stochastic:      {hs_res['is_stochastic']} (Max Var: {hs_res['max_variance']:.6f})")
    assert hs_res["is_stochastic"], "Hate Speech outputs are deterministic!"

    # Test 3: Propaganda RoBERTa MC Dropout
    pr_model, pr_tok, pr_labels, _ = load_propaganda_model()
    pr_text = "The dishonest media will never tell you the truth."
    pr_res = estimator.estimate_text(pr_model, pr_tok, pr_text, label_mapping=pr_labels)

    print(f"\nC. Propaganda RoBERTa (20 stochastic passes):")
    print(f"   Input: '{pr_text}'")
    print(f"   Prediction: {pr_res['prediction']} (Mean Conf: {pr_res['confidence']:.4f})")
    print(f"   Shannon Entropy:    {pr_res['entropy']:.4f} bits")
    print(f"   Is Stochastic:      {pr_res['is_stochastic']} (Max Var: {pr_res['max_variance']:.6f})")
    assert pr_res["is_stochastic"], "Propaganda outputs are deterministic!"

    print("\n--> MC DROPOUT 20-PASS EXECUTION: PASSED AND VERIFIED ON ALL 3 MODELS!\n")


def run_real_human_review_and_schema():
    print("=" * 70)
    print("5. HUMAN REVIEW POLICY & STANDARDIZED RESULT SCHEMA EXECUTION")
    print("=" * 70)

    engine = UncertaintyEngine(variance_threshold=0.01, entropy_threshold=0.85)

    # Case A: Low Uncertainty
    low_uq = {"max_variance": 0.002, "entropy": 0.35}
    policy_low = engine.evaluate_policy(low_uq, component_name="Deepfake Detector")
    print("Case A: Low Uncertainty Sample")
    print(f"  Review Required: {policy_low['review_required']}")
    print(f"  Decision Status: {policy_low['decision_status']}")
    print(f"  Justification:   {policy_low['review_justification']}")

    # Case B: High Uncertainty (triggers human review)
    high_uq = {"max_variance": 0.045, "entropy": 1.12}
    policy_high = engine.evaluate_policy(high_uq, component_name="Deepfake Detector")
    print("\nCase B: High Uncertainty Sample (Triggers Human Review)")
    print(f"  Review Required: {policy_high['review_required']}")
    print(f"  Decision Status: {policy_high['decision_status']}")
    print(f"  Trigger Component: {policy_high['review_trigger_component']}")
    print(f"  Justification:   {policy_high['review_justification']}")

    assert policy_low["review_required"] is False
    assert policy_high["review_required"] is True
    assert policy_high["review_trigger_component"] == "Deepfake Detector"

    # Assemble complete unified output schema
    result = build_verification_result(
        modality="hate_speech",
        prediction="hatespeech",
        confidence=0.8207,
        uncertainty_info={
            "mean_probability": [0.0675, 0.1118, 0.8207],
            "variance": [0.00093, 0.00071, 0.00008],
            "std": [0.0305, 0.0266, 0.0089],
            "entropy": 0.7642,
            "max_variance": 0.00093,
            "mc_samples": 20
        },
        review_info=policy_low,
        explanation_info={
            "top_tokens": [
                {"token": "##gger", "clean_token": "gger", "score": 1.0},
                {"token": "mandela", "clean_token": "mandela", "score": 0.5319}
            ],
            "rationale_alignment": {
                "overlap_count": 1,
                "overlap_tokens": ["nigger"],
                "precision": 0.5,
                "recall": 1.0,
                "f1": 0.6667
            },
            "disclaimer": "Token attribution is a post-hoc explanatory signal and should not be interpreted as causal proof."
        }
    )

    print("\nUnified Output Schema JSON:")
    print(json.dumps(result, indent=2))
    assert result["modality"] == "hate_speech"
    assert result["uncertainty"]["mc_samples"] == 20
    assert "explanation" in result
    assert "review_required" in result

    print("\n--> HUMAN REVIEW POLICY & UNIFIED SCHEMA: PASSED AND VERIFIED!\n")


if __name__ == "__main__":
    run_real_deepfake_gradcam()
    run_real_propaganda_token_attribution()
    run_real_hate_speech_and_rationale_alignment()
    run_real_mc_dropout_20_passes()
    run_real_human_review_and_schema()
    print("=" * 70)
    print("ALL REAL COMPONENT EXECUTIONS COMPLETED AND VERIFIED SUCCESSFULLY!")
    print("=" * 70)
