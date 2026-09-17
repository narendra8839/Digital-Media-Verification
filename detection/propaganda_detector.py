"""Propaganda technique detector module integrating RoBERTa 14-class model, Token Attribution, and MC Dropout."""

from typing import Dict, Any, Optional
from models.propaganda.predict import load_propaganda_model
from explainability.text_attention import TokenAttribution
from uncertainty.uncertainty_engine import UncertaintyEngine


class PropagandaDetector:
    """Orchestrates 14-class propaganda classification, token attribution, and uncertainty estimation."""

    def __init__(self, checkpoint_dir: str = "models/propaganda/checkpoint",
                 tokenizer_dir: str = "models/propaganda/tokenizer",
                 device: Optional[str] = None,
                 variance_threshold: float = 0.02,
                 entropy_threshold: float = 0.85):
        self.model, self.tokenizer, self.label_mapping, self.device = load_propaganda_model(
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

    def detect_text(self, text: Optional[str]) -> Dict[str, Any]:
        """Classify propaganda technique and extract token attributions.

        Args:
            text: Input text string.

        Returns:
            Dictionary containing prediction, confidence, explanation, uncertainty, and review status.
        """
        if not text or not text.strip():
            return {
                "status": "NO_TEXT",
                "prediction": None,
                "confidence": 0.0,
                "note": "No text content available for propaganda verification.",
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
            modality="propaganda",
            tokenizer=self.tokenizer,
            label_mapping=self.label_mapping
        )

        # 2. Gradient-based token attribution
        attrib_result = self.explainer.attribute(
            text=clean_text,
            label_mapping=self.label_mapping
        )

        return {
            "status": "SUCCESS",
            "prediction": uq_result["prediction"],
            "confidence": uq_result["confidence"],
            "probabilities": uq_result["probabilities"],
            "explanation": {
                "top_tokens": attrib_result["top_tokens"],
                "token_scores": attrib_result["token_scores"],
                "disclaimer": attrib_result["disclaimer"]
            },
            "uncertainty": uq_result["uncertainty"],
            "review_required": uq_result["review_required"],
            "decision_status": uq_result["decision_status"],
            "review_trigger_component": uq_result.get("review_trigger_component"),
            "review_justification": uq_result["review_justification"]
        }
