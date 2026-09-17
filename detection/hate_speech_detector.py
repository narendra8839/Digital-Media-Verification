"""Hate speech detector module integrating BERT 3-class model, Token Attribution, and MC Dropout."""

from typing import Dict, Any, Optional, List
from models.hate_speech.predict import load_hate_speech_model
from explainability.text_attention import TokenAttribution, evaluate_rationale_alignment
from uncertainty.uncertainty_engine import UncertaintyEngine


class HateSpeechDetector:
    """Orchestrates 3-class hate speech classification, token attribution, and uncertainty estimation."""

    def __init__(self, checkpoint_dir: str = "models/hate_speech/checkpoint",
                 tokenizer_dir: str = "models/hate_speech/tokenizer",
                 device: Optional[str] = None,
                 variance_threshold: float = 0.02,
                 entropy_threshold: float = 0.85):
        self.model, self.tokenizer, self.label_mapping, self.device = load_hate_speech_model(
            checkpoint_dir=checkpoint_dir,
            tokenizer_dir=tokenizer_dir,
            device=device
        )
        self.explainer = TokenAttribution(model=self.model, tokenizer=self.tokenizer, device=self.device)
        self.uncertainty_engine = UncertaintyEngine(
            mc_samples=20,
            variance_threshold=variance_threshold,
            entropy_threshold=entropy_threshold,
            device=self.device
        )

    def detect_text(self, text: Optional[str],
                    ground_truth_rationales: Optional[List[List[int]]] = None,
                    post_tokens: Optional[List[str]] = None) -> Dict[str, Any]:
        """Classify hate speech, extract token attributions, and compute rationale alignment if available.

        Args:
            text: Input text string.
            ground_truth_rationales: Optional binary rationale masks from HateXplain.
            post_tokens: Optional post tokens from HateXplain.

        Returns:
            Dictionary containing prediction, confidence, explanation, uncertainty, and review status.
        """
        if not text or not text.strip():
            return {
                "status": "NO_TEXT",
                "prediction": None,
                "confidence": 0.0,
                "note": "No text content available for hate speech verification.",
                "explanation": None,
                "uncertainty": None,
                "review_required": False,
                "decision_status": "NOT_APPLICABLE"
            }

        clean_text = text.strip()

        # 1. Uncertainty & stochastic inference via MC Dropout (T=20)
        uq_result = self.uncertainty_engine.process_and_evaluate(
            model=self.model,
            input_data=clean_text,
            modality="hate_speech",
            tokenizer=self.tokenizer,
            label_mapping=self.label_mapping
        )

        # 2. Gradient-based token attribution
        attrib_result = self.explainer.attribute(
            text=clean_text,
            label_mapping=self.label_mapping
        )

        # 3. Rationale alignment (if ground-truth rationales provided)
        alignment_result = None
        if ground_truth_rationales is not None and post_tokens is not None:
            alignment_result = evaluate_rationale_alignment(
                attributed_tokens=attrib_result["token_scores"],
                ground_truth_rationales=ground_truth_rationales,
                post_tokens=post_tokens,
                top_k=3
            )

        explanation_dict = {
            "top_tokens": attrib_result["top_tokens"],
            "token_scores": attrib_result["token_scores"],
            "disclaimer": attrib_result["disclaimer"]
        }
        if alignment_result is not None:
            explanation_dict["rationale_alignment"] = alignment_result

        return {
            "status": "SUCCESS",
            "prediction": uq_result["prediction"],
            "confidence": uq_result["confidence"],
            "probabilities": uq_result["probabilities"],
            "explanation": explanation_dict,
            "uncertainty": uq_result["uncertainty"],
            "review_required": uq_result["review_required"],
            "decision_status": uq_result["decision_status"],
            "review_trigger_component": uq_result.get("review_trigger_component"),
            "review_justification": uq_result["review_justification"]
        }
