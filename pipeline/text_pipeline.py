"""Text verification pipeline: Direct text routing to RoBERTa and BERT with attribution and MC Dropout."""

import os
from typing import Dict, Any, Optional

from detection.propaganda_detector import PropagandaDetector
from detection.hate_speech_detector import HateSpeechDetector
from extraction.text_aggregator import TextAggregator
from explainability.explanation_generator import build_multimodal_verification_result


class TextPipeline:
    """Processing pipeline for direct text input."""

    def __init__(self, propaganda_detector: Optional[PropagandaDetector] = None,
                 hate_speech_detector: Optional[HateSpeechDetector] = None,
                 device: Optional[str] = None):
        self.device = device or "cpu"
        self.propaganda_detector = propaganda_detector or PropagandaDetector(device=self.device)
        self.hate_speech_detector = hate_speech_detector or HateSpeechDetector(device=self.device)

    def process_text(self, text: str,
                     source_name: str = "direct_input") -> Dict[str, Any]:
        """Execute text verification pipeline.

        Args:
            text: Raw input text string or path to text file.
            source_name: Optional identifier for text origin.

        Returns:
            Unified multimodal verification result dictionary.
        """
        raw_text = text
        filename = source_name
        size_bytes = len(raw_text.encode("utf-8"))

        if isinstance(text, str) and os.path.exists(text) and os.path.isfile(text):
            try:
                filename = os.path.basename(text)
                size_bytes = os.path.getsize(text)
                with open(text, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
            except Exception:
                raw_text = text

        input_meta = {
            "input_type": "text",
            "filename": filename,
            "char_count": len(raw_text),
            "size_bytes": size_bytes,
            "status": "PROCESSED"
        }

        # 1. Visual analysis is skipped for direct text
        visual_result = {
            "status": "NOT_APPLICABLE",
            "note": "Visual analysis skipped for text-only input.",
            "prediction": None,
            "confidence": 0.0,
            "probabilities": None,
            "explanation": None,
            "uncertainty": None,
            "review_required": False,
            "decision_status": "NOT_APPLICABLE"
        }

        # 2. Text Aggregation
        text_summary = TextAggregator.aggregate(user_text=raw_text)

        # 3. Propaganda & Hate Speech Detection
        propaganda_result = None
        hate_speech_result = None
        uncertainty_map = {}
        review_triggers = []

        if text_summary.get("text_status") == "AVAILABLE":
            combined_txt = text_summary["combined_text"]
            propaganda_result = self.propaganda_detector.detect_text(combined_txt)
            hate_speech_result = self.hate_speech_detector.detect_text(combined_txt)

            if propaganda_result.get("uncertainty"):
                uncertainty_map["propaganda"] = propaganda_result["uncertainty"]
                if propaganda_result.get("review_required"):
                    review_triggers.append("propaganda")

            if hate_speech_result.get("uncertainty"):
                uncertainty_map["hate_speech"] = hate_speech_result["uncertainty"]
                if hate_speech_result.get("review_required"):
                    review_triggers.append("hate_speech")
        else:
            propaganda_result = {"status": "NO_TEXT_AVAILABLE"}
            hate_speech_result = {"status": "NO_TEXT_AVAILABLE"}

        # 4. Governance & Human Review Policy
        review_required = len(review_triggers) > 0
        trigger_comp = ", ".join(review_triggers) if review_triggers else None

        if review_required:
            decision_status = "REVIEW_RECOMMENDED"
            disclaimer = "Human review is a responsible AI safeguard triggered by predictive uncertainty."
        elif text_summary.get("text_status") != "AVAILABLE":
            decision_status = "ANALYSIS_INCOMPLETE"
            disclaimer = (
                "No usable text was available for natural language verification; "
                "the system refrains from an ungrounded prediction."
            )
        else:
            decision_status = "CONFIDENT_PREDICTION"
            disclaimer = "Model confidence and statistical consistency meet baseline criteria."

        governance_info = {
            "review_required": review_required,
            "decision_status": decision_status,
            "review_trigger_component": trigger_comp,
            "disclaimer": disclaimer
        }

        # 5. Audit Metadata
        audit_info = {
            "models_executed": ["RoBERTa", "BERT"] if text_summary.get("text_status") == "AVAILABLE" else [],
            "face_detected": False,
            "ocr_executed": False,
            "asr_executed": False,
            "mc_samples": 20
        }

        return build_multimodal_verification_result(
            input_meta=input_meta,
            visual_result=visual_result,
            text_summary=text_summary,
            propaganda_result=propaganda_result,
            hate_speech_result=hate_speech_result,
            uncertainty_map=uncertainty_map,
            governance_info=governance_info,
            audit_info=audit_info
        )
