"""Image verification pipeline: Face detection, EfficientNet-B4, Grad-CAM, MC Dropout, and OCR."""

import os
from typing import Dict, Any, Optional, Union
from PIL import Image
import numpy as np

from detection.deepfake_detector import DeepfakeDetector
from detection.propaganda_detector import PropagandaDetector
from detection.hate_speech_detector import HateSpeechDetector
from extraction.ocr import EasyOCRReader
from extraction.text_aggregator import TextAggregator
from explainability.explanation_generator import build_multimodal_verification_result


class ImagePipeline:
    """End-to-end processing pipeline for still images."""

    def __init__(self, deepfake_detector: Optional[DeepfakeDetector] = None,
                 propaganda_detector: Optional[PropagandaDetector] = None,
                 hate_speech_detector: Optional[HateSpeechDetector] = None,
                 ocr_reader: Optional[EasyOCRReader] = None,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if False else "cpu")
        self.deepfake_detector = deepfake_detector or DeepfakeDetector(device=self.device)
        self.propaganda_detector = propaganda_detector or PropagandaDetector(device=self.device)
        self.hate_speech_detector = hate_speech_detector or HateSpeechDetector(device=self.device)
        self.ocr_reader = ocr_reader or EasyOCRReader(gpu=False)

    def process_image(self, image: Union[Image.Image, np.ndarray, str],
                      caption: Optional[str] = None,
                      output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Execute image verification pipeline.

        Args:
            image: PIL Image, RGB numpy array, or image file path.
            caption: Optional text caption provided with the image.
            output_dir: Optional directory to save explanation artifacts.

        Returns:
            Unified multimodal verification result dictionary.
        """
        # Metadata
        filename = "<in-memory image>"
        size_bytes = 0
        if isinstance(image, str) and os.path.exists(image):
            filename = os.path.basename(image)
            size_bytes = os.path.getsize(image)
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            size_bytes = pil_img.width * pil_img.height * 3
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
            size_bytes = int(image.nbytes)
        else:
            raise ValueError(f"Unsupported image input type: {type(image)}")

        input_meta = {
            "input_type": "image",
            "filename": filename,
            "dimensions": [pil_img.width, pil_img.height],
            "size_bytes": size_bytes,
            "status": "PROCESSED"
        }

        # 1. Visual Deepfake Analysis (Face Detection + EfficientNet-B4 + Grad-CAM + MC Dropout)
        visual_result = self.deepfake_detector.detect_image(pil_img, output_dir=output_dir, generate_gradcam=True)

        # 2. Text Extraction via OCR (runs independently even if no face is found)
        ocr_result = self.ocr_reader.extract_text(pil_img)
        ocr_list = [ocr_result] if ocr_result.get("text") else []

        # 3. Text Aggregation (Caption + OCR)
        text_summary = TextAggregator.aggregate(user_text=caption, ocr_results=ocr_list)

        # 4. Text Analysis (Propaganda & Hate Speech) if text exists
        propaganda_result = None
        hate_speech_result = None
        uncertainty_map = {}
        review_triggers = []

        if visual_result.get("uncertainty"):
            uncertainty_map["deepfake"] = visual_result["uncertainty"]
            if visual_result.get("review_required"):
                review_triggers.append(visual_result.get("review_trigger_component") or "deepfake")

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

        # 5. Governance & Human Review Policy
        review_required = len(review_triggers) > 0
        trigger_comp = ", ".join(review_triggers) if review_triggers else None

        if review_required:
            decision_status = "REVIEW_RECOMMENDED"
            disclaimer = "Human review is a responsible AI safeguard triggered by predictive uncertainty."
        elif visual_result.get("status") == "NO_FACE_DETECTED":
            decision_status = "ANALYSIS_INCOMPLETE"
            disclaimer = (
                "Automated facial deepfake detection skipped because no human face was detected. "
                "The system refrains from an ungrounded prediction; governance status indicates analysis is incomplete."
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

        # 6. Audit Metadata
        audit_info = {
            "models_executed": ["EfficientNet-B4"] + (["RoBERTa", "BERT"] if text_summary.get("text_status") == "AVAILABLE" else []),
            "face_detected": visual_result.get("face_detected", False),
            "ocr_executed": True,
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
