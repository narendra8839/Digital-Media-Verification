"""Video verification pipeline: Frame sampling, EfficientNet-B4, Selective Grad-CAM, OCR, Audio/Whisper ASR."""

import os
from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image

from preprocessing.frame_sampling import sample_video_frames
from preprocessing.face_detection import FaceDetector
from detection.deepfake_detector import DeepfakeDetector
from detection.propaganda_detector import PropagandaDetector
from detection.hate_speech_detector import HateSpeechDetector
from extraction.audio_extraction import has_audio_stream, extract_audio_to_wav
from extraction.asr import WhisperASR
from extraction.ocr import EasyOCRReader
from extraction.text_aggregator import TextAggregator
from explainability.explanation_generator import build_multimodal_verification_result


class VideoPipeline:
    """End-to-end processing pipeline for video media."""

    def __init__(self, deepfake_detector: Optional[DeepfakeDetector] = None,
                 propaganda_detector: Optional[PropagandaDetector] = None,
                 hate_speech_detector: Optional[HateSpeechDetector] = None,
                 ocr_reader: Optional[EasyOCRReader] = None,
                 whisper_asr: Optional[WhisperASR] = None,
                 device: Optional[str] = None):
        self.device = device or "cpu"
        self.deepfake_detector = deepfake_detector or DeepfakeDetector(device=self.device)
        self.propaganda_detector = propaganda_detector or PropagandaDetector(device=self.device)
        self.hate_speech_detector = hate_speech_detector or HateSpeechDetector(device=self.device)
        self.ocr_reader = ocr_reader or EasyOCRReader(gpu=False)
        self.whisper_asr = whisper_asr or WhisperASR(device=self.device)
        self.face_detector = FaceDetector(device=self.device)

    def process_video(self, video_path: str,
                      caption: Optional[str] = None,
                      sample_fps: float = 1.0,
                      max_frames: int = 16,
                      output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Execute video verification pipeline.

        Args:
            video_path: Absolute or relative path to video file.
            caption: Optional user-supplied caption string.
            sample_fps: Frame sampling rate in frames per second (default: 1.0).
            max_frames: Maximum number of frames to sample (prevents OOM).
            output_dir: Optional directory to save representative Grad-CAM explanation.

        Returns:
            Unified multimodal verification result dictionary.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        filename = os.path.basename(video_path)
        size_bytes = os.path.getsize(video_path)

        input_meta = {
            "input_type": "video",
            "filename": filename,
            "file_path": os.path.abspath(video_path),
            "size_bytes": size_bytes,
            "sample_fps": sample_fps,
            "status": "PROCESSED"
        }

        # 1. Frame Sampling at target FPS (without loading full video into memory)
        sampled_frames = sample_video_frames(video_path, target_fps=sample_fps, max_frames=max_frames)
        total_sampled = len(sampled_frames)

        # 2. Per-Frame Face Detection & Deepfake Inference
        frame_predictions = []
        fake_probs = []
        valid_crops = [] # (frame_index, timestamp, crop_pil, fake_prob)

        for idx, (ts, frame_rgb) in enumerate(sampled_frames):
            crop, is_face = self.face_detector.detect_and_crop(frame_rgb, target_size=(224, 224))
            if is_face and crop is not None:
                # Fast single crop inference
                crop_res = self.deepfake_detector.detect_image(crop, generate_gradcam=False)
                fake_p = crop_res["probabilities"]["fake"]
                fake_probs.append(fake_p)
                frame_predictions.append({
                    "frame_index": idx,
                    "timestamp": round(float(ts), 2),
                    "face_detected": True,
                    "prediction": crop_res["prediction"],
                    "confidence": crop_res["confidence"],
                    "fake_probability": fake_p
                })
                valid_crops.append((idx, ts, crop, fake_p))
            else:
                frame_predictions.append({
                    "frame_index": idx,
                    "timestamp": round(float(ts), 2),
                    "face_detected": False,
                    "prediction": None,
                    "confidence": 0.0,
                    "fake_probability": None
                })

        # 3. Video-level Aggregation (mean fake probability >= 0.5 -> fake)
        visual_result = {}
        if fake_probs:
            mean_fake_prob = float(np.mean(fake_probs))
            mean_real_prob = 1.0 - mean_fake_prob
            video_pred = "fake" if mean_fake_prob >= 0.5 else "real"
            video_conf = mean_fake_prob if video_pred == "fake" else mean_real_prob

            # Identify representative frame (most confident according to video prediction)
            if video_pred == "fake":
                rep_crop_tuple = max(valid_crops, key=lambda x: x[3])
            else:
                rep_crop_tuple = min(valid_crops, key=lambda x: x[3])

            rep_idx, rep_ts, rep_crop, rep_p = rep_crop_tuple

            # Run Grad-CAM ONLY on the single representative frame
            rep_save_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                rep_save_path = os.path.join(output_dir, f"{filename}_frame{rep_idx}_gradcam.png")

            rep_gradcam = self.deepfake_detector.gradcam.explain(
                image_crop=rep_crop,
                output_path=rep_save_path,
                label_mapping=self.deepfake_detector.label_mapping
            )

            # MC Dropout on representative frame
            rep_uq = self.deepfake_detector.uncertainty_engine.process_and_evaluate(
                model=self.deepfake_detector.model,
                input_data=rep_crop,
                modality="deepfake",
                label_mapping=self.deepfake_detector.label_mapping
            )

            visual_result = {
                "status": "SUCCESS",
                "prediction": video_pred,
                "confidence": round(video_conf, 4),
                "probabilities": {
                    "real": round(mean_real_prob, 4),
                    "fake": round(mean_fake_prob, 4)
                },
                "frames_analyzed": len(valid_crops),
                "total_frames_sampled": total_sampled,
                "representative_frame": {
                    "frame_index": rep_idx,
                    "timestamp": round(float(rep_ts), 2),
                    "fake_probability": round(rep_p, 4),
                    "explanation": rep_gradcam
                },
                "frame_level_predictions": frame_predictions,
                "uncertainty": rep_uq["uncertainty"],
                "review_required": rep_uq["review_required"],
                "decision_status": rep_uq["decision_status"],
                "review_trigger_component": rep_uq.get("review_trigger_component"),
                "review_justification": rep_uq["review_justification"]
            }
        else:
            visual_result = {
                "status": "NO_FACE_DETECTED",
                "prediction": None,
                "confidence": 0.0,
                "note": "No faces were detected in any of the sampled video frames.",
                "frames_analyzed": 0,
                "total_frames_sampled": total_sampled,
                "frame_level_predictions": frame_predictions,
                "uncertainty": None,
                "review_required": False,
                "decision_status": "NOT_APPLICABLE"
            }

        # 4. OCR on Key Video Frames (up to 4 evenly spaced frames)
        ocr_results = self.ocr_reader.extract_from_frames(sampled_frames, max_frames=4)

        # 5. Audio Stream Extraction and Whisper ASR
        asr_result = {"text": "", "source": "ASR", "status": "NO_AUDIO"}
        temp_wav = None
        try:
            if has_audio_stream(video_path):
                temp_wav = extract_audio_to_wav(video_path)
                if temp_wav:
                    asr_result = self.whisper_asr.transcribe(temp_wav)
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass

        # 6. Text Aggregation
        text_summary = TextAggregator.aggregate(
            user_text=caption,
            ocr_results=ocr_results,
            asr_result=asr_result
        )

        # 7. Text Analysis (Propaganda & Hate Speech) if text exists
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

        # 8. Governance & Human Review Policy
        review_required = len(review_triggers) > 0
        trigger_comp = ", ".join(review_triggers) if review_triggers else None

        if review_required:
            decision_status = "REVIEW_RECOMMENDED"
            disclaimer = "Human review is a responsible AI safeguard triggered by predictive uncertainty."
        elif visual_result.get("status") == "NO_FACE_DETECTED":
            decision_status = "ANALYSIS_INCOMPLETE"
            disclaimer = (
                "Automated facial deepfake verification was bypassed because no faces were detected in sampled frames. "
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

        # 9. Audit Metadata
        audit_info = {
            "models_executed": ["EfficientNet-B4"] + (["RoBERTa", "BERT"] if text_summary.get("text_status") == "AVAILABLE" else []),
            "frames_sampled": total_sampled,
            "faces_analyzed": len(valid_crops),
            "ocr_executed": True,
            "ocr_frames_inspected": min(4, total_sampled),
            "asr_executed": asr_result.get("status") != "NO_AUDIO",
            "has_audio": has_audio_stream(video_path),
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
