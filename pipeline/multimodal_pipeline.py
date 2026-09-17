"""Master Multimodal Verification Pipeline Orchestrator.

Routes Image, Video, and Text inputs through the appropriate sub-pipelines,
sharing model instances to avoid redundant memory allocations and initialization costs.
"""

import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union
from PIL import Image
import numpy as np

from detection.deepfake_detector import DeepfakeDetector
from detection.propaganda_detector import PropagandaDetector
from detection.hate_speech_detector import HateSpeechDetector
from extraction.ocr import EasyOCRReader
from extraction.asr import WhisperASR
from pipeline.router import MediaRouter
from pipeline.image_pipeline import ImagePipeline
from pipeline.video_pipeline import VideoPipeline
from pipeline.text_pipeline import TextPipeline
from explainability.explanation_generator import build_multimodal_verification_result


class MultimodalPipeline:
    """End-to-end multimodal media verification pipeline.

    Intelligently routes inputs:
    - Image: Face Detection -> EfficientNet-B4 -> Grad-CAM -> OCR -> MC Dropout
    - Video: Frame Sampling -> Per-frame Face Detection -> Video Aggregation -> Selective Grad-CAM -> OCR -> Whisper ASR -> NLP Models
    - Text: Direct NLP -> RoBERTa Propaganda -> BERT Hate-Speech -> Token Attribution -> MC Dropout
    """

    def __init__(self,
                 deepfake_detector: Optional[DeepfakeDetector] = None,
                 propaganda_detector: Optional[PropagandaDetector] = None,
                 hate_speech_detector: Optional[HateSpeechDetector] = None,
                 ocr_reader: Optional[EasyOCRReader] = None,
                 whisper_asr: Optional[WhisperASR] = None,
                 device: Optional[str] = None):
        """Initialize pipeline with shared detector instances."""
        self.device = device or "cpu"

        # Shared component singletons
        self.deepfake_detector = deepfake_detector or DeepfakeDetector(device=self.device)
        self.propaganda_detector = propaganda_detector or PropagandaDetector(device=self.device)
        self.hate_speech_detector = hate_speech_detector or HateSpeechDetector(device=self.device)
        self.ocr_reader = ocr_reader or EasyOCRReader(gpu=False)
        self.whisper_asr = whisper_asr or WhisperASR(device=self.device)

        # Specialized sub-pipelines sharing detector instances
        self.image_pipeline = ImagePipeline(
            deepfake_detector=self.deepfake_detector,
            propaganda_detector=self.propaganda_detector,
            hate_speech_detector=self.hate_speech_detector,
            ocr_reader=self.ocr_reader,
            device=self.device
        )

        self.video_pipeline = VideoPipeline(
            deepfake_detector=self.deepfake_detector,
            propaganda_detector=self.propaganda_detector,
            hate_speech_detector=self.hate_speech_detector,
            ocr_reader=self.ocr_reader,
            whisper_asr=self.whisper_asr,
            device=self.device
        )

        self.text_pipeline = TextPipeline(
            propaganda_detector=self.propaganda_detector,
            hate_speech_detector=self.hate_speech_detector,
            device=self.device
        )

    def verify(self,
               media_input: Union[str, Image.Image, np.ndarray],
               caption: Optional[str] = None,
               output_dir: Optional[str] = None,
               **kwargs) -> Dict[str, Any]:
        """Verify media input through the multimodal verification pipeline.

        Args:
            media_input: File path (image/video), raw text string, PIL.Image, or numpy array.
            caption: Optional user-supplied caption associated with image or video.
            output_dir: Optional directory to store visual explanation heatmaps.
            **kwargs: Additional parameters forwarded to sub-pipelines (e.g. sample_fps, max_frames).

        Returns:
            Standardized multimodal verification result matching the unified verification schema.
        """
        start_time = time.time()
        timestamp = datetime.now(timezone.utc).isoformat()
        errors = []

        # 1. Media Type Inspection
        inspection = MediaRouter.inspect_input(media_input)
        input_type = inspection.get("input_type", "unknown")

        if input_type == "unknown":
            error_msg = f"Unsupported media type or format: {inspection.get('status')}"
            return build_multimodal_verification_result(
                input_meta=inspection,
                governance_info={
                    "review_required": True,
                    "decision_status": "INPUT_ERROR",
                    "review_trigger_component": "router",
                    "error": error_msg
                },
                audit_info={
                    "timestamp": timestamp,
                    "execution_time_seconds": round(time.time() - start_time, 3),
                    "errors": [error_msg]
                }
            )

        # 2. Dispatch to the appropriate sub-pipeline with graceful error fallback
        result = None
        try:
            if input_type == "image":
                result = self.image_pipeline.process_image(
                    image=media_input,
                    caption=caption,
                    output_dir=output_dir
                )
            elif input_type == "video":
                sample_fps = kwargs.get("sample_fps", 1.0)
                max_frames = kwargs.get("max_frames", 16)
                result = self.video_pipeline.process_video(
                    video_path=media_input,
                    caption=caption,
                    sample_fps=sample_fps,
                    max_frames=max_frames,
                    output_dir=output_dir
                )
            elif input_type == "text":
                result = self.text_pipeline.process_text(
                    text=media_input
                )
        except Exception as exc:
            errors.append(f"Pipeline error during {input_type} processing: {str(exc)}")
            # Fallback error container
            result = build_multimodal_verification_result(
                input_meta=inspection,
                governance_info={
                    "review_required": True,
                    "decision_status": "COMPONENT_FAILURE",
                    "review_trigger_component": input_type,
                    "error": str(exc)
                }
            )

        # 3. Enrich Audit Metadata
        audit = result.get("audit", {})
        audit.update({
            "timestamp": timestamp,
            "execution_time_seconds": round(time.time() - start_time, 3),
            "input_type": input_type,
            "filename": inspection.get("filename", "unknown"),
            "checkpoint_versions": {
                "deepfake": "EfficientNet-B4 (models/deepfake/checkpoint/best_model.pt)",
                "propaganda": "RoBERTa-base (models/propaganda/checkpoint)",
                "hate_speech": "BERT-base-uncased (models/hate_speech/checkpoint)",
                "asr": f"Whisper-base ({self.whisper_asr.model_id})"
            },
            "errors": errors
        })
        result["audit"] = audit

        return result

    def verify_image(self, image: Union[str, Image.Image, np.ndarray],
                     caption: Optional[str] = None,
                     output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Convenience method for image verification."""
        return self.verify(media_input=image, caption=caption, output_dir=output_dir)

    def verify_video(self, video_path: str,
                     caption: Optional[str] = None,
                     sample_fps: float = 1.0,
                     max_frames: int = 16,
                     output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Convenience method for video verification."""
        return self.verify(
            media_input=video_path,
            caption=caption,
            sample_fps=sample_fps,
            max_frames=max_frames,
            output_dir=output_dir
        )

    def verify_text(self, text: str) -> Dict[str, Any]:
        """Convenience method for text verification."""
        return self.verify(media_input=text)
