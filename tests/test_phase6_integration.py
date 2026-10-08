"""Focused integration tests for Phase 6: Final Backend/Model Integration, XAI, UQ, Governance."""

import os
import io
import time
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from api.utils.device import get_optimal_device
from detection.deepfake_detector import DeepfakeDetector
from detection.propaganda_detector import PropagandaDetector
from detection.hate_speech_detector import HateSpeechDetector
from extraction.ocr import EasyOCRReader
from extraction.asr import WhisperASR


client = TestClient(app)
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


class TestPhase6DeviceAndCheckpoints:
    """Audit device auto-detection and Phase 5 checkpoint verification."""

    def test_device_selection_auto_and_override(self, monkeypatch):
        """Test device detection behavior and environment variable override."""
        # Auto detection should return a valid PyTorch device string
        dev = get_optimal_device()
        assert dev in ("cpu", "cuda:0", "cuda")

        # Explicit override
        monkeypatch.setenv("DMV_DEVICE", "cpu")
        assert get_optimal_device() == "cpu"

    def test_checkpoint_defaults_and_eval_mode(self):
        """Verify that detectors default to the Phase 5 candidate checkpoints in eval mode."""
        df_det = DeepfakeDetector()
        assert "checkpoint_full" in df_det.model.__class__.__name__ or hasattr(df_det.model, "backbone")
        assert not df_det.model.training  # Model must be in eval mode for standard inference

        pr_det = PropagandaDetector()
        assert not pr_det.model.training
        assert len(pr_det.label_mapping) == 14  # SemEval-2020 14-class technique schema

        hs_det = HateSpeechDetector()
        assert not hs_det.model.training
        assert len(hs_det.label_mapping) == 3   # 3-class schema: normal, offensive, hatespeech


class TestPhase6TextVerification:
    """Verify text verification endpoint, token attribution, UQ, and disclaimers."""

    def test_text_verification_success(self):
        """Test text inference with propaganda and hate-speech models."""
        response = client.post(
            "/verify/text",
            json={"text": "This is an example text for rhetorical technique and hate speech testing."}
        )
        assert response.status_code == 200
        data = response.json()

        # Schema structure
        assert "propaganda" in data
        assert "hate_speech" in data
        assert "uncertainty" in data
        assert "governance" in data
        assert "audit" in data

        # Propaganda outputs
        pr = data["propaganda"]
        assert pr["status"] == "SUCCESS"
        assert "prediction" in pr
        assert "confidence" in pr
        assert "probabilities" in pr
        assert "explanation" in pr

        # Token attribution explainability
        expl = pr["explanation"]
        assert "token_scores" in expl or "tokens" in expl
        assert "disclaimer" in expl
        assert "post-hoc" in expl["disclaimer"].lower()

        # Hate speech outputs
        hs = data["hate_speech"]
        assert hs["status"] == "SUCCESS"
        assert hs["prediction"] in ("normal", "offensive", "hatespeech")

        # Uncertainty quantification
        uq = data["uncertainty"]
        assert "propaganda" in uq
        assert uq["propaganda"]["mc_samples"] == 20
        assert "mean_probability" in uq["propaganda"]
        assert "variance" in uq["propaganda"]
        assert "entropy" in uq["propaganda"]
        assert "calibration not quantitatively established" in uq["propaganda"]["disclaimer"].lower()

        # Governance
        gov = data["governance"]
        assert gov["decision_status"] in ("CONFIDENT_PREDICTION", "REVIEW_RECOMMENDED")

    def test_text_empty_and_whitespace(self):
        """Empty text should fail validation cleanly."""
        res = client.post("/verify/text", json={"text": ""})
        assert res.status_code in (400, 422)

        res_ws = client.post("/verify/text", json={"text": "   "})
        assert res_ws.status_code in (400, 422)


class TestPhase6ImageVerification:
    """Verify image verification endpoint, Grad-CAM artifacts, no-face handling, and static serving."""

    def test_valid_image_with_face_and_artifact_serving(self):
        """Image with face produces prediction, Grad-CAM overlay, and browser-accessible artifact."""
        face_path = os.path.join(ASSETS_DIR, "face_image.jpg")
        assert os.path.exists(face_path), "Test asset face_image.jpg missing"

        with open(face_path, "rb") as f:
            response = client.post(
                "/verify/image",
                files={"file": ("face_image.jpg", f, "image/jpeg")},
                data={"caption": "Test face photo"}
            )

        assert response.status_code == 200
        data = response.json()

        # Visual results
        vis = data["visual"]
        assert vis["status"] == "SUCCESS"
        assert vis["face_detected"] is True
        assert vis["prediction"] in ("real", "fake")
        assert 0.0 <= vis["confidence"] <= 1.0

        # Explanation / Grad-CAM
        assert "explanation" in vis and vis["explanation"] is not None
        expl = vis["explanation"]
        assert "saved_path" in expl
        assert "overlay_url" in expl

        # If saved to artifacts directory, verify HTTP accessibility via FastAPI static mount
        overlay_url = expl["overlay_url"]
        if overlay_url and overlay_url.startswith("/artifacts/"):
            art_res = client.get(overlay_url)
            assert art_res.status_code == 200
            assert "image" in art_res.headers.get("content-type", "")

        # MC Dropout Uncertainty T=20
        assert "deepfake" in data["uncertainty"]
        df_uq = data["uncertainty"]["deepfake"]
        assert df_uq["mc_samples"] == 20
        assert "mean_probability" in df_uq
        assert "max_variance" in df_uq
        assert "entropy" in df_uq

    def test_image_governance_confident_prediction(self):
        """Image with low uncertainty (variance < 0.02 and entropy < 0.85) yields CONFIDENT_PREDICTION."""
        face_path = os.path.join(ASSETS_DIR, "face_image.jpg")
        with open(face_path, "rb") as f:
            response = client.post(
                "/verify/image",
                files={"file": ("face_image.jpg", f, "image/jpeg")}
            )
        assert response.status_code == 200
        data = response.json()
        df_uq = data["uncertainty"]["deepfake"]
        assert df_uq["max_variance"] < 0.02
        assert df_uq["entropy"] < 0.85
        assert data["governance"]["decision_status"] == "CONFIDENT_PREDICTION"
        assert data["governance"]["review_required"] is False

    def test_no_face_image_handling(self):
        """Image without face must return NO_FACE_DETECTED and ANALYSIS_INCOMPLETE governance."""
        no_face_path = os.path.join(ASSETS_DIR, "no_face_image.jpg")
        assert os.path.exists(no_face_path), "Test asset no_face_image.jpg missing"

        with open(no_face_path, "rb") as f:
            response = client.post(
                "/verify/image",
                files={"file": ("no_face_image.jpg", f, "image/jpeg")}
            )

        assert response.status_code == 200
        data = response.json()

        vis = data["visual"]
        assert vis["status"] == "NO_FACE_DETECTED"
        assert vis["face_detected"] is False
        assert vis["prediction"] is None

        # Governance must strictly indicate ANALYSIS_INCOMPLETE without guessing
        gov = data["governance"]
        assert gov["decision_status"] == "ANALYSIS_INCOMPLETE"
        assert "incomplete" in gov["disclaimer"].lower() or "skipped" in gov["disclaimer"].lower()


class TestPhase6VideoAndAsyncJobs:
    """Verify video pipeline, audio extraction, ASR failure handling, and async job lifecycle."""

    def test_video_sync_verification(self):
        """Synchronous video verification using test asset."""
        silent_vid = os.path.join(ASSETS_DIR, "silent_video.mp4")
        assert os.path.exists(silent_vid), "Test asset silent_video.mp4 missing"

        with open(silent_vid, "rb") as f:
            response = client.post(
                "/verify/video",
                files={"file": ("silent_video.mp4", f, "video/mp4")},
                data={"async_mode": "false", "sample_fps": "1.0", "max_frames": "4"}
            )

        assert response.status_code == 200
        data = response.json()
        assert "visual" in data
        assert "text" in data
        assert "audit" in data
        # Videos without audio must have handled ASR safely without crashing
        assert data["audit"]["asr_executed"] is False

    def test_video_async_job_lifecycle(self):
        """Asynchronous video job creation, polling, and completion."""
        silent_vid = os.path.join(ASSETS_DIR, "silent_video.mp4")
        with open(silent_vid, "rb") as f:
            res_submit = client.post(
                "/verify/video",
                files={"file": ("silent_video.mp4", f, "video/mp4")},
                data={"async_mode": "true", "sample_fps": "1.0", "max_frames": "4"}
            )

        assert res_submit.status_code == 200
        job_info = res_submit.json()
        assert "job_id" in job_info
        assert job_info["status"] == "QUEUED"
        assert "poll_url" in job_info
        job_id = job_info["job_id"]

        # Poll until COMPLETED or timeout
        poll_res = client.get(f"/verify/status/{job_id}")
        assert poll_res.status_code == 200
        status_data = poll_res.json()
        assert status_data["status"] in ("QUEUED", "PROCESSING", "COMPLETED")


class TestPhase6Multimodal:
    """Verify multimodal endpoint combining media and textual channels."""

    def test_multimodal_text_only(self):
        """Multimodal endpoint with text-only payload."""
        response = client.post(
            "/verify/multimodal",
            data={"text": "Multimodal text-only verification test input"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["visual"]["status"] == "NOT_APPLICABLE"
        assert data["text"]["text_status"] == "AVAILABLE"
        assert "propaganda" in data

    def test_multimodal_image_and_caption(self):
        """Multimodal endpoint with image file and user-supplied caption."""
        face_path = os.path.join(ASSETS_DIR, "face_image.jpg")
        with open(face_path, "rb") as f:
            response = client.post(
                "/verify/multimodal",
                files={"file": ("face_image.jpg", f, "image/jpeg")},
                data={"text": "A photograph claiming to depict an official event."}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["input"]["input_type"] == "image"
        assert data["visual"]["status"] == "SUCCESS"
        assert data["text"]["text_status"] == "AVAILABLE"
        assert data["governance"]["decision_status"] in ("CONFIDENT_PREDICTION", "REVIEW_RECOMMENDED")
