"""Uncertainty policy evaluation and Human-in-the-loop review decision layer."""

from typing import Dict, Any, Optional
from uncertainty.mc_dropout import MCDropoutEstimator


class UncertaintyEngine:
    """Evaluates predictive uncertainty and applies Responsible AI human-review thresholds."""

    def __init__(self, mc_samples: int = 20,
                 variance_threshold: float = 0.02,
                 entropy_threshold: float = 0.85,
                 device: str = "cpu"):
        self.mc_samples = mc_samples
        self.variance_threshold = variance_threshold
        self.entropy_threshold = entropy_threshold
        self.device = device
        self.estimator = MCDropoutEstimator(mc_samples=mc_samples, device=device)

    def evaluate_policy(self, uncertainty_result: Dict[str, Any],
                        variance_threshold: Optional[float] = None,
                        entropy_threshold: Optional[float] = None,
                        component_name: Optional[str] = None) -> Dict[str, Any]:
        """Apply human-review thresholding to an MC Dropout uncertainty result.

        Args:
            uncertainty_result: Dictionary output from MCDropoutEstimator
            variance_threshold: Override for max_variance threshold
            entropy_threshold: Override for entropy threshold
            component_name: Optional name of the model/component evaluated

        Returns:
            Dictionary with review_required flag, decision status, reason, component, and thresholds applied.
        """
        var_thresh = variance_threshold if variance_threshold is not None else self.variance_threshold
        ent_thresh = entropy_threshold if entropy_threshold is not None else self.entropy_threshold

        max_var = uncertainty_result.get("max_variance", 0.0)
        entropy = uncertainty_result.get("entropy", 0.0)

        high_variance = max_var >= var_thresh
        high_entropy = entropy >= ent_thresh

        review_required = high_variance or high_entropy

        reasons = []
        if high_variance:
            reasons.append(f"predictive variance ({max_var:.4f}) exceeds threshold ({var_thresh:.4f})")
        if high_entropy:
            reasons.append(f"predictive entropy ({entropy:.4f} bits) exceeds threshold ({ent_thresh:.4f} bits)")

        comp_prefix = f"[{component_name.upper()}] " if component_name else ""
        if review_required:
            status = "REVIEW_RECOMMENDED"
            explanation_note = (
                f"{comp_prefix}Flagged for human review: Model exhibits elevated uncertainty based on {', and '.join(reasons)}. "
                "Per Responsible AI guidelines, elevated uncertainty mandates human oversight rather than automated action."
            )
        else:
            status = "CONFIDENT_PREDICTION"
            explanation_note = (
                f"{comp_prefix}Predictive variance ({max_var:.4f}) and entropy ({entropy:.4f} bits) are within acceptable bounds. "
                "Automated classification meets threshold criteria."
            )

        return {
            "review_required": review_required,
            "decision_status": status,
            "review_reasons": reasons,
            "review_justification": explanation_note,
            "review_trigger_component": component_name if review_required else None,
            "thresholds_applied": {
                "variance_threshold": var_thresh,
                "entropy_threshold": ent_thresh
            },
            "threshold_disclaimer": "Development/MVP threshold — not clinically, legally, or production calibrated."
        }

    def process_and_evaluate(self, model, input_data, modality: str,
                             tokenizer=None, label_mapping=None,
                             variance_threshold=None, entropy_threshold=None) -> Dict[str, Any]:
        """Run MC Dropout and evaluate review policy in a single unified step.

        Args:
            model: PyTorch model
            input_data: Image crop or text string
            modality: 'deepfake', 'propaganda', or 'hate_speech'
            tokenizer: Tokenizer for NLP modalities
            label_mapping: Label index to name dict
            variance_threshold: Custom variance threshold
            entropy_threshold: Custom entropy threshold

        Returns:
            Complete combined prediction, uncertainty, and review dictionary.
        """
        if modality == "deepfake":
            uq = self.estimator.estimate_deepfake_crop(model, input_data, label_mapping=label_mapping)
        elif modality in ("propaganda", "hate_speech"):
            if tokenizer is None:
                raise ValueError(f"Tokenizer must be provided for text modality: {modality}")
            uq = self.estimator.estimate_text(model, tokenizer, input_data, label_mapping=label_mapping)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        policy = self.evaluate_policy(uq, variance_threshold, entropy_threshold, component_name=modality)

        return {
            "modality": modality,
            "prediction": uq["prediction"],
            "class_index": uq["class_index"],
            "confidence": uq["confidence"],
            "probabilities": uq["probabilities"],
            "uncertainty": {
                "mean_probability": uq["mean_probability"],
                "variance": uq["variance"],
                "std": uq["std"],
                "entropy": uq["entropy"],
                "max_variance": uq["max_variance"],
                "mean_variance": uq["mean_variance"],
                "mc_samples": uq["mc_samples"],
                "is_stochastic": uq["is_stochastic"]
            },
            "review_required": policy["review_required"],
            "review_status": policy["decision_status"],
            "decision_status": policy["decision_status"],
            "review_trigger_component": policy.get("review_trigger_component"),
            "review_justification": policy["review_justification"],
            "thresholds_applied": policy["thresholds_applied"],
            "disclaimers": {
                "uncertainty": uq["disclaimer"],
                "policy": policy["threshold_disclaimer"]
            }
        }
