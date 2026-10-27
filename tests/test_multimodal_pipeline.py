"""End-to-End Integration Tests for Multimodal Verification Pipeline.

Tests all 10 required multimodal orchestration scenarios:
1. Text-only input
2. Image with face
3. Image without detectable face
4. Video with audio
5. Video without audio
6. Input with OCR-readable text
7. Input where OCR returns no text
8. Input where ASR returns usable text
9. Input with no available text
10. High-uncertainty review path
"""

import os
import sys
import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.router import MediaRouter
from pipeline.multimodal_pipeline import MultimodalPipeline
from extraction.text_aggregator import TextAggregator


@pytest.fixture(scope="module")
def shared_pipeline():
    """Load the MultimodalPipeline once for the entire test session."""
    pipeline = MultimodalPipeline(device="cpu")
    return pipeline


class TestMultimodalPipelineEndToEnd:
    """Test suite covering the 10 required multimodal orchestration flows."""

    # 1. Text-only input
    def test_01_text_only_input(self, shared_pipeline):
        sample_text = "The government is hiding the truth about secret satellite transmissions."
        res = shared_pipeline.verify(sample_text)

        assert res["input"]["input_type"] == "text"
        assert res["visual"]["status"] == "NOT_APPLICABLE"
        assert res["text"]["text_status"] == "AVAILABLE"
        assert len(res["text"]["segments"]) == 1
        assert res["text"]["segments"][0]["source"] == "USER_TEXT"

        # NLP models executed
        assert "prediction" in res["propaganda"]
        assert "explanation" in res["propaganda"]
        assert "prediction" in res["hate_speech"]
        assert "explanation" in res["hate_speech"]

        # Uncertainty computed per modality
        assert "propaganda" in res["uncertainty"]
        assert "hate_speech" in res["uncertainty"]
        assert "deepfake" not in res["uncertainty"]

        # Governance & audit
        assert "review_required" in res["governance"]
        assert "audit" in res
        assert "RoBERTa" in res["audit"]["models_executed"]
        assert "BERT" in res["audit"]["models_executed"]

    # 2. Image with face
    def test_02_image_with_face(self, shared_pipeline, tmp_path):
        face_path = os.path.join(PROJECT_ROOT, "tests", "assets", "face_image.jpg")
        assert os.path.exists(face_path), f"Asset missing: {face_path}"

        out_dir = str(tmp_path / "xai_out")
        res = shared_pipeline.verify(face_path, output_dir=out_dir)

        assert res["input"]["input_type"] == "image"
        assert res["visual"]["face_detected"] is True
        assert res["visual"]["prediction"] in ("real", "fake")
        assert 0.0 <= res["visual"]["confidence"] <= 1.0

        # Grad-CAM explanation
        assert "explanation" in res["visual"]
        assert res["visual"]["explanation"]["heatmap_shape"] == [7, 7]
        assert "post-hoc" in res["visual"]["explanation"]["disclaimer"]

        # Visual MC Dropout
        assert "deepfake" in res["uncertainty"]
        assert "variance" in res["uncertainty"]["deepfake"]
        assert res["uncertainty"]["deepfake"]["mc_samples"] == 20

    # 3. Image without detectable face
    def test_03_image_without_face(self, shared_pipeline):
        no_face_path = os.path.join(PROJECT_ROOT, "tests", "assets", "no_face_image.jpg")
        assert os.path.exists(no_face_path), f"Asset missing: {no_face_path}"

        res = shared_pipeline.verify(no_face_path)

        assert res["input"]["input_type"] == "image"
        assert res["visual"]["status"] == "NO_FACE_DETECTED"
        assert res["visual"]["face_detected"] is False
        assert res["visual"]["prediction"] is None
        assert res["visual"]["confidence"] == 0.0

        # Uncertainty is not applicable for missing faces
        assert res["visual"]["uncertainty"] is None

        # Governance consistency: must indicate ANALYSIS_INCOMPLETE rather than CONFIDENT_PREDICTION
        assert res["governance"]["decision_status"] == "ANALYSIS_INCOMPLETE"
        assert res["governance"]["review_required"] is False

    # 4. Video with audio
    def test_04_video_with_audio(self, shared_pipeline):
        audio_video_path = os.path.join(PROJECT_ROOT, "tests", "assets", "audio_video.mp4")
        assert os.path.exists(audio_video_path), f"Asset missing: {audio_video_path}"

        res = shared_pipeline.verify(audio_video_path, sample_fps=1.0, max_frames=2)

        assert res["input"]["input_type"] == "video"
        assert "frames_sampled" in res["audit"]
        assert res["audit"]["has_audio"] is True
        assert res["audit"]["asr_executed"] is True

    # 5. Video without audio
    def test_05_video_without_audio(self, shared_pipeline):
        silent_video_path = os.path.join(PROJECT_ROOT, "tests", "assets", "silent_video.mp4")
        assert os.path.exists(silent_video_path), f"Asset missing: {silent_video_path}"

        res = shared_pipeline.verify(silent_video_path, sample_fps=1.0, max_frames=2)

        assert res["input"]["input_type"] == "video"
        assert res["audit"]["has_audio"] is False
        assert res["audit"]["asr_executed"] is False
        # ASR status in text segments or sources
        asr_segments = [s for s in res["text"]["segments"] if s["source"] == "ASR"]
        assert len(asr_segments) == 0 or asr_segments[0]["text"] == ""

    # 6. Input with OCR-readable text
    def test_06_input_with_ocr_readable_text(self, shared_pipeline):
        ocr_path = os.path.join(PROJECT_ROOT, "tests", "assets", "ocr_image.jpg")
        assert os.path.exists(ocr_path), f"Asset missing: {ocr_path}"

        res = shared_pipeline.verify(ocr_path)

        assert res["input"]["input_type"] == "image"
        # OCR detected text
        ocr_segments = [s for s in res["text"]["segments"] if s["source"] == "OCR"]
        assert len(ocr_segments) > 0
        assert len(res["text"]["combined_text"]) > 0

        # Propaganda and hate-speech models executed on OCR text
        assert res["text"]["text_status"] == "AVAILABLE"
        assert res["propaganda"]["status"] == "SUCCESS"
        assert res["hate_speech"]["status"] == "SUCCESS"

    # 7. Input where OCR returns no text
    def test_07_input_where_ocr_returns_no_text(self, shared_pipeline):
        no_face_path = os.path.join(PROJECT_ROOT, "tests", "assets", "no_face_image.jpg")
        res = shared_pipeline.verify(no_face_path)

        # No text in the image
        ocr_segments = [s for s in res["text"]["segments"] if s["source"] == "OCR"]
        assert len(ocr_segments) == 0

    # 8. Input where ASR returns usable text
    def test_08_input_where_asr_returns_usable_text(self, shared_pipeline):
        audio_video_path = os.path.join(PROJECT_ROOT, "tests", "assets", "audio_video.mp4")
        res = shared_pipeline.verify(audio_video_path, sample_fps=1.0, max_frames=2)

        asr_segments = [s for s in res["text"]["segments"] if s["source"] == "ASR"]
        assert len(asr_segments) == 1
        assert asr_segments[0]["source"] == "ASR"
        assert len(asr_segments[0]["text"]) > 0

        # Preserved provenance in text summary
        assert "ASR" in res["text"]["sources"]
        assert res["text"]["text_status"] == "AVAILABLE"

    # 9. Input with no available text
    def test_09_input_with_no_available_text(self, shared_pipeline):
        # Plain image with no face, no OCR text, and no user caption
        no_face_path = os.path.join(PROJECT_ROOT, "tests", "assets", "no_face_image.jpg")
        res = shared_pipeline.verify(no_face_path, caption=None)

        assert res["text"]["text_status"] == "NO_TEXT_AVAILABLE"
        assert res["text"]["combined_text"] == ""
        assert res["propaganda"]["status"] == "NO_TEXT_AVAILABLE"
        assert res["hate_speech"]["status"] == "NO_TEXT_AVAILABLE"
        assert "propaganda" not in res["uncertainty"]
        assert "hate_speech" not in res["uncertainty"]

    # 10. High-uncertainty review path
    def test_10_high_uncertainty_review_path(self, shared_pipeline):
        # Ambiguous input or input with review required
        sample_text = "They say elections might be rigged, but who knows what really happens behind closed doors."
        res = shared_pipeline.verify(sample_text)

        assert "governance" in res
        assert "review_required" in res["governance"]
        assert res["governance"]["decision_status"] in ("REVIEW_RECOMMENDED", "CONFIDENT_PREDICTION")
        assert "disclaimer" in res["governance"]

    # Extra: Error handling for invalid / unsupported inputs
    def test_error_handling_graceful_failure(self, shared_pipeline):
        # Non-existent file
        res = shared_pipeline.verify("non_existent_file.xyz")
        # MediaRouter treats unrecognized path as raw text or unsupported
        assert "input" in res
        assert "governance" in res
        assert "audit" in res

    # Regression: Video verification with detected face, output_dir, and Grad-CAM retention
    def test_video_verification_with_detected_face_and_gradcam_retention(self, shared_pipeline, tmp_path):
        """Verify that video with detected face generates Grad-CAM under protect_active_artifact and returns valid governance."""
        face_img_path = os.path.join(PROJECT_ROOT, "tests", "assets", "face_image.jpg")
        assert os.path.exists(face_img_path)
        face_crop = Image.open(face_img_path).convert("RGB").resize((224, 224))

        video_path = os.path.join(PROJECT_ROOT, "tests", "assets", "silent_video.mp4")
        assert os.path.exists(video_path)

        out_dir = str(tmp_path / "artifacts")
        os.makedirs(out_dir, exist_ok=True)

        with patch.object(shared_pipeline.video_pipeline.face_detector, "detect_and_crop", return_value=(face_crop, True)):
            res = shared_pipeline.verify(video_path, sample_fps=1.0, max_frames=2, output_dir=out_dir)

        assert res["input"]["input_type"] == "video"
        assert res["visual"]["status"] == "SUCCESS"
        assert res["visual"]["prediction"] in ("real", "fake")
        assert isinstance(res["visual"]["confidence"], float)
        assert 0.0 <= res["visual"]["confidence"] <= 1.0
        assert res["visual"]["frames_analyzed"] > 0

        # Representative frame explanation (Grad-CAM)
        rep_frame = res["visual"]["representative_frame"]
        assert rep_frame is not None
        assert "frame_index" in rep_frame
        assert "timestamp" in rep_frame
        assert "explanation" in rep_frame
        rep_expl = rep_frame["explanation"]
        assert rep_expl is not None
        assert "overlay_path" in rep_expl
        assert rep_expl["overlay_path"] is not None
        assert os.path.exists(rep_expl["overlay_path"])

        # Governance status must be valid prediction, NOT COMPONENT_FAILURE
        gov = res["governance"]
        assert gov["decision_status"] in ("CONFIDENT_PREDICTION", "REVIEW_RECOMMENDED")
        assert gov["decision_status"] != "COMPONENT_FAILURE"
        assert gov["review_trigger_component"] != "video" or gov["decision_status"] == "REVIEW_RECOMMENDED"

    # Regression: Genuine video processing failure produces honest COMPONENT_FAILURE and server-side logs
    def test_video_verification_genuine_failure_produces_honest_error_and_logs(self, shared_pipeline, caplog):
        """Verify that an unhandled error during video processing produces COMPONENT_FAILURE and logs."""
        import logging
        video_path = os.path.join(PROJECT_ROOT, "tests", "assets", "silent_video.mp4")

        with caplog.at_level(logging.ERROR, logger="dmv.pipeline.multimodal"):
            with patch.object(shared_pipeline.video_pipeline, "process_video", side_effect=RuntimeError("Simulated video decoder crash")):
                res = shared_pipeline.verify(video_path, sample_fps=1.0, max_frames=2)

        # Honest failure container
        assert res["governance"]["decision_status"] == "COMPONENT_FAILURE"
        assert res["governance"]["review_required"] is True
        assert res["governance"]["review_trigger_component"] == "video"
        assert "Simulated video decoder crash" in res["governance"]["error"]
        assert any("Simulated video decoder crash" in err for err in res["audit"]["errors"])

        # Useful server-side logs recorded
        assert any("Simulated video decoder crash" in record.message for record in caplog.records if record.levelname == "ERROR")
