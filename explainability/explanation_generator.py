"""Unified explanation generator for multimodal media verification."""

from typing import Dict, Any, Optional, List, Union
from PIL import Image
import numpy as np
import torch

from explainability.gradcam import GradCAM
from explainability.text_attention import TokenAttribution, evaluate_rationale_alignment


class ExplanationGenerator:
    """Unified generator for visual (Grad-CAM) and textual (Token Attribution) explanations."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._gradcam_cache = {}
        self._attrib_cache = {}

    def explain_deepfake(self, model: torch.nn.Module,
                         image_crop: Union[Image.Image, np.ndarray],
                         target_class: Optional[int] = None,
                         output_path: Optional[str] = None,
                         label_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate Grad-CAM explanation for deepfake face crop."""
        explainer = GradCAM(model=model, device=self.device)
        return explainer.explain(
            image_crop=image_crop,
            target_class=target_class,
            output_path=output_path,
            label_mapping=label_mapping
        )

    def explain_text(self, model: torch.nn.Module, tokenizer: Any,
                     text: str, target_class: Optional[int] = None,
                     label_mapping: Optional[Dict[str, str]] = None,
                     ground_truth_rationales: Optional[List[List[int]]] = None,
                     post_tokens: Optional[List[str]] = None,
                     top_k: int = 5) -> Dict[str, Any]:
        """Generate token-level attribution explanation for text models (BERT / RoBERTa)."""
        explainer = TokenAttribution(model=model, tokenizer=tokenizer, device=self.device)
        res = explainer.attribute(text=text, target_class=target_class, label_mapping=label_mapping)

        # Attach rationale alignment if ground truth is supplied (e.g. HateXplain)
        if ground_truth_rationales is not None and post_tokens is not None:
            alignment = evaluate_rationale_alignment(
                attributed_tokens=res["token_scores"],
                ground_truth_rationales=ground_truth_rationales,
                post_tokens=post_tokens,
                top_k=top_k
            )
            res["rationale_alignment"] = alignment

        return res


def build_verification_result(modality: str,
                              prediction: str,
                              confidence: float,
                              uncertainty_info: Dict[str, Any],
                              review_info: Dict[str, Any],
                              explanation_info: Dict[str, Any],
                              metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble standardized verification result conforming to project schema.

    Schema:
    {
        "modality": "...",
        "prediction": "...",
        "confidence": 0.XX,
        "uncertainty": {
            "mean_probability": [...],
            "variance": [...],
            "std": [...],
            "entropy": ...,
            "mc_samples": 20
        },
        "review_required": true/false,
        "review_status": "...",
        "review_justification": "...",
        "explanation": {...},
        "metadata": {...}
    }
    """
    return {
        "modality": modality,
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "uncertainty": {
            "mean_probability": uncertainty_info.get("mean_probability", []),
            "variance": uncertainty_info.get("variance", []),
            "std": uncertainty_info.get("std", []),
            "entropy": uncertainty_info.get("entropy", 0.0),
            "max_variance": uncertainty_info.get("max_variance", 0.0),
            "mc_samples": uncertainty_info.get("mc_samples", 20)
        },
        "review_required": bool(review_info.get("review_required", False)),
        "review_status": review_info.get("decision_status", "CONFIDENT_PREDICTION"),
        "review_justification": review_info.get("review_justification", ""),
        "explanation": explanation_info,
        "metadata": metadata or {}
    }


def build_multimodal_verification_result(input_meta: Dict[str, Any],
                                        visual_result: Optional[Dict[str, Any]] = None,
                                        text_summary: Optional[Dict[str, Any]] = None,
                                        propaganda_result: Optional[Dict[str, Any]] = None,
                                        hate_speech_result: Optional[Dict[str, Any]] = None,
                                        uncertainty_map: Optional[Dict[str, Any]] = None,
                                        governance_info: Optional[Dict[str, Any]] = None,
                                        audit_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble complete multimodal pipeline verification result matching the full project schema.

    Schema:
    {
        "input": {...},
        "visual": {...},
        "text": {...},
        "propaganda": {...},
        "hate_speech": {...},
        "uncertainty": {
            "deepfake": {...},
            "propaganda": {...},
            "hate_speech": {...}
        },
        "governance": {
            "review_required": bool,
            "decision_status": str,
            "review_trigger_component": str
        },
        "audit": {...}
    }
    """
    return {
        "input": input_meta,
        "visual": visual_result or {"status": "NOT_APPLICABLE"},
        "text": text_summary or {"sources": [], "segments": [], "combined_text": "", "text_status": "NO_TEXT_AVAILABLE"},
        "propaganda": propaganda_result or {"status": "NOT_APPLICABLE"},
        "hate_speech": hate_speech_result or {"status": "NOT_APPLICABLE"},
        "uncertainty": uncertainty_map or {},
        "governance": governance_info or {
            "review_required": False,
            "decision_status": "CONFIDENT_PREDICTION",
            "review_trigger_component": None
        },
        "audit": audit_info or {}
    }


