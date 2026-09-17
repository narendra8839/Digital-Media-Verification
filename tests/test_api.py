"""Comprehensive test suite for FastAPI service layer.

Covers:
1. GET /health
2. GET /ready
3. POST /verify/image (valid image with face)
4. POST /verify/image (image without face)
5. POST /verify/video (valid video with audio, sync mode)
6. POST /verify/video (valid video in async mode & GET /verify/status/{job_id})
7. POST /verify/video (silent video without audio)
8. POST /verify/text (valid text)
9. POST /verify/text (empty / whitespace text -> 422)
10. Invalid image format (rejected with 400)
11. Invalid video format (rejected with 400)
12. Empty file upload (rejected with 400)
13. Oversized file upload (rejected with 413)
14. POST /verify/multimodal (media + caption)
15. POST /verify/multimodal (no input provided -> 400)
16. Structured error responses
17. JSON serialization of results
"""

import os
import io
import sys
import json
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app
from api.dependencies.pipeline_dep import get_pipeline


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifespan pre-warming."""
    # Pre-warm pipeline explicitly for tests
    get_pipeline()
    with TestClient(app) as c:
        yield c


class TestHealthAndReadiness:
    """Test suite for health and readiness endpoints."""

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "digital-media-verification"

    def test_ready_endpoint(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ready", "initializing")
        assert "deepfake" in data["models"]
        assert "propaganda" in data["models"]
        assert "hate_speech" in data["models"]
        assert "asr" in data["models"]


class TestTextVerificationEndpoint:
    """Test suite for POST /verify/text."""

    def test_valid_text_verification(self, client):
        payload = {"text": "The government is systematically hiding the true election ballots."}
        response = client.post("/verify/text", json=payload)
        assert response.status_code == 200
        data = response.json()

        # Check schema structure
        assert "input" in data
        assert data["input"]["input_type"] == "text"
        assert "propaganda" in data
        assert data["propaganda"]["status"] == "SUCCESS"
        assert "hate_speech" in data
        assert data["hate_speech"]["status"] == "SUCCESS"
        assert "uncertainty" in data
        assert "governance" in data
        assert "audit" in data

        # JSON serializable
        json.dumps(data)

    def test_empty_text_rejected(self, client):
        response = client.post("/verify/text", json={"text": "   "})
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"] == "Validation Error"

    def test_missing_text_field(self, client):
        response = client.post("/verify/text", json={})
        assert response.status_code == 422
        data = response.json()
        assert data["status_code"] == 422


class TestImageVerificationEndpoint:
    """Test suite for POST /verify/image."""

    def test_valid_image_with_face(self, client):
        face_path = os.path.join(PROJECT_ROOT, "tests", "assets", "face_image.jpg")
        assert os.path.exists(face_path), f"Asset missing: {face_path}"

        with open(face_path, "rb") as f:
            files = {"file": ("face.jpg", f, "image/jpeg")}
            data = {"caption": "Breaking news photograph"}
            response = client.post("/verify/image", files=files, data=data)

        assert response.status_code == 200
        res = response.json()
        assert res["input"]["input_type"] == "image"
        assert res["visual"]["face_detected"] is True
        assert res["visual"]["prediction"] in ("real", "fake")
        assert "explanation" in res["visual"]
        assert "deepfake" in res["uncertainty"]
        assert "governance" in res
        json.dumps(res)

    def test_image_without_face(self, client):
        no_face_path = os.path.join(PROJECT_ROOT, "tests", "assets", "no_face_image.jpg")
        assert os.path.exists(no_face_path), f"Asset missing: {no_face_path}"

        with open(no_face_path, "rb") as f:
            files = {"file": ("scenery.jpg", f, "image/jpeg")}
            response = client.post("/verify/image", files=files)

        assert response.status_code == 200
        res = response.json()
        assert res["input"]["input_type"] == "image"
        assert res["visual"]["status"] == "NO_FACE_DETECTED"
        assert res["visual"]["face_detected"] is False
        assert res["visual"]["prediction"] is None
        # Governance consistency: faceless media must NOT be reported as CONFIDENT_PREDICTION
        assert res["governance"]["decision_status"] == "ANALYSIS_INCOMPLETE"
        assert res["governance"]["review_required"] is False

    def test_invalid_image_extension(self, client):
        fake_content = io.BytesIO(b"Not an image")
        files = {"file": ("test.pdf", fake_content, "application/pdf")}
        response = client.post("/verify/image", files=files)

        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "Request Error"
        assert "Unsupported file format" in data["detail"]

    def test_empty_image_file(self, client):
        empty_content = io.BytesIO(b"")
        files = {"file": ("empty.jpg", empty_content, "image/jpeg")}
        response = client.post("/verify/image", files=files)

        assert response.status_code == 400
        data = response.json()
        assert "empty" in data["detail"].lower()


class TestVideoVerificationEndpoint:
    """Test suite for POST /verify/video and GET /verify/status/{job_id}."""

    def test_valid_video_with_audio_sync(self, client):
        video_path = os.path.join(PROJECT_ROOT, "tests", "assets", "audio_video.mp4")
        assert os.path.exists(video_path), f"Asset missing: {video_path}"

        with open(video_path, "rb") as f:
            files = {"file": ("test_vid.mp4", f, "video/mp4")}
            data = {"async_mode": "false", "sample_fps": "1.0", "max_frames": "2"}
            response = client.post("/verify/video", files=files, data=data)

        assert response.status_code == 200
        res = response.json()
        assert res["input"]["input_type"] == "video"
        assert res["audit"]["has_audio"] is True
        assert res["audit"]["asr_executed"] is True
        json.dumps(res)

    def test_silent_video_sync(self, client):
        video_path = os.path.join(PROJECT_ROOT, "tests", "assets", "silent_video.mp4")
        assert os.path.exists(video_path), f"Asset missing: {video_path}"

        with open(video_path, "rb") as f:
            files = {"file": ("silent.mp4", f, "video/mp4")}
            data = {"async_mode": "false", "sample_fps": "1.0", "max_frames": "2"}
            response = client.post("/verify/video", files=files, data=data)

        assert response.status_code == 200
        res = response.json()
        assert res["input"]["input_type"] == "video"
        assert res["audit"]["has_audio"] is False
        assert res["audit"]["asr_executed"] is False

    def test_valid_video_async_mode(self, client):
        video_path = os.path.join(PROJECT_ROOT, "tests", "assets", "audio_video.mp4")
        assert os.path.exists(video_path), f"Asset missing: {video_path}"

        with open(video_path, "rb") as f:
            files = {"file": ("async_vid.mp4", f, "video/mp4")}
            data = {"async_mode": "true", "sample_fps": "1.0", "max_frames": "2"}
            response = client.post("/verify/video", files=files, data=data)

        assert response.status_code == 200
        init_res = response.json()
        assert "job_id" in init_res
        assert init_res["status"] in ("QUEUED", "PROCESSING", "COMPLETED")
        job_id = init_res["job_id"]

        # Poll status
        status_res = client.get(f"/verify/status/{job_id}")
        assert status_res.status_code == 200
        status_data = status_res.json()
        assert status_data["job_id"] == job_id
        assert status_data["status"] in ("QUEUED", "PROCESSING", "COMPLETED")

    def test_invalid_job_id_returns_404(self, client):
        response = client.get("/verify/status/non-existent-uuid-12345")
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_invalid_video_extension(self, client):
        fake_content = io.BytesIO(b"Not a video")
        files = {"file": ("test.exe", fake_content, "application/octet-stream")}
        response = client.post("/verify/video", files=files)

        assert response.status_code == 400
        data = response.json()
        assert "Unsupported file format" in data["detail"]


class TestMultimodalEndpoint:
    """Test suite for POST /verify/multimodal."""

    def test_multimodal_image_and_text(self, client):
        face_path = os.path.join(PROJECT_ROOT, "tests", "assets", "face_image.jpg")
        with open(face_path, "rb") as f:
            files = {"file": ("face.jpg", f, "image/jpeg")}
            data = {"text": "Vaccine chips conspiracy statement"}
            response = client.post("/verify/multimodal", files=files, data=data)

        assert response.status_code == 200
        res = response.json()
        assert res["input"]["input_type"] == "image"
        # Provenance: USER_TEXT was preserved
        sources = res["text"]["sources"]
        assert "USER_TEXT" in sources

    def test_multimodal_text_only(self, client):
        data = {"text": "The election was corrupted by external forces."}
        response = client.post("/verify/multimodal", data=data)

        assert response.status_code == 200
        res = response.json()
        assert res["input"]["input_type"] == "text"
        assert res["propaganda"]["status"] == "SUCCESS"

    def test_multimodal_no_input_rejected(self, client):
        response = client.post("/verify/multimodal")
        assert response.status_code == 400
        data = response.json()
        assert "At least one media input" in data["detail"]


class TestErrorResponsesAndLimits:
    """Test structured error responses and upload limits."""

    def test_structured_error_response_format(self, client):
        response = client.get("/verify/status/missing-id")
        assert response.status_code == 404
        data = response.json()
        assert set(data.keys()) == {"error", "detail", "status_code"}
        assert data["status_code"] == 404

    def test_oversized_image_upload_rejected(self, client, monkeypatch):
        # Temporarily reduce MAX_IMAGE_SIZE_BYTES to test 413
        from api.utils import file_utils
        monkeypatch.setattr(file_utils, "MAX_IMAGE_SIZE_BYTES", 100)

        oversized_data = io.BytesIO(b"A" * 200)
        files = {"file": ("large.jpg", oversized_data, "image/jpeg")}
        response = client.post("/verify/image", files=files)

        assert response.status_code == 413
        data = response.json()
        assert data["status_code"] == 413
        assert "exceeded maximum" in data["detail"].lower()
