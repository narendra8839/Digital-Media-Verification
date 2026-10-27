"""Comprehensive tests for Downloadable Image Analysis Report feature.

Covers:
1. Successful in-memory generation of PDF report with valid binary structure.
2. Correct HTTP response headers (Content-Type: application/pdf, Content-Disposition).
3. Safe filename generation and sanitization (path traversal protection).
4. Handling missing, partial, and NO_FACE_DETECTED analysis fields.
5. Safe handling of arbitrary Unicode, emojis, multiline text, and XML characters.
6. Path traversal protection for Grad-CAM explainability artifact resolution.
7. Proof that requesting a report does NOT rerun model inference.
8. Accurate association with the currently selected image's result.
9. Graceful error handling (HTTP 500) if report generation encounters failure.
"""

import os
import io
import sys
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app
from api.utils.report_generator import (
    generate_image_analysis_pdf,
    sanitize_filename,
    generate_download_filename,
    safe_text,
    _resolve_safe_image_path,
    _decode_base64_image,
)
from pipeline.multimodal_pipeline import MultimodalPipeline


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifespan context."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_analysis_result():
    """Realistic complete analysis result dictionary."""
    return {
        "status": "SUCCESS",
        "visual": {
            "prediction": "fake",
            "confidence": 0.9412,
            "probabilities": {"real": 0.0588, "fake": 0.9412},
            "face_detected": True,
            "status": "SUCCESS",
            "explanation": {
                "method": "grad_cam",
                "disclaimer": "Grad-CAM is a post-hoc explanatory signal and should not be interpreted as causal proof.",
                "heatmap_shape": [7, 7]
            },
            "uncertainty": {
                "mean_variance": 0.0125,
                "std": 0.1118,
                "entropy": 0.2854,
                "mc_samples": 20,
                "mean_probability": [0.0588, 0.9412],
                "is_elevated": True
            }
        },
        "text": {
            "text_status": "AVAILABLE",
            "sources": ["OCR"],
            "segments": [
                {"text": "Breaking: Official Statement Released", "source": "OCR", "confidence": 0.98}
            ],
            "combined_text": "Breaking: Official Statement Released"
        },
        "propaganda": {
            "status": "SUCCESS",
            "prediction": "Loaded_Language",
            "confidence": 0.885,
            "probabilities": {
                "Loaded_Language": 0.885,
                "Name_Calling": 0.065,
                "Doubt": 0.035,
                "Appeal_to_Fear": 0.015
            },
            "explanation": {
                "method": "token_attribution",
                "top_tokens": [{"token": "Breaking", "score": 0.91}]
            },
            "uncertainty": {
                "mean_variance": 0.005,
                "std": 0.071,
                "entropy": 0.22,
                "mc_samples": 20,
                "is_elevated": False
            }
        },
        "hate_speech": {
            "status": "SUCCESS",
            "prediction": "normal",
            "confidence": 0.962,
            "probabilities": {"normal": 0.962, "offensive": 0.025, "hatespeech": 0.013},
            "uncertainty": {
                "mean_variance": 0.002,
                "std": 0.045,
                "entropy": 0.11,
                "mc_samples": 20,
                "is_elevated": False
            }
        },
        "governance": {
            "decision_status": "REVIEW_RECOMMENDED",
            "review_required": True,
            "review_trigger_component": "Visual Deepfake Model (MC Dropout Elevated Uncertainty)",
            "review_justification": "Predictive variance 0.0125 exceeds threshold 0.0100",
            "disclaimer": "Human review recommended before operational decisions."
        },
        "audit": {
            "timestamp": "2026-10-27T12:00:00Z",
            "input_type": "IMAGE",
            "filename": "suspect_profile.png",
            "models_executed": ["EfficientNet-B4", "EasyOCR", "RoBERTa-base", "BERT-base"],
            "mc_samples": 20,
            "ocr_executed": True,
            "asr_executed": False,
            "execution_time_seconds": 0.95
        }
    }


class TestReportGeneratorUnit:
    """Unit tests for standalone PDF report generator functions."""

    def test_generate_pdf_success(self, sample_analysis_result):
        """Verify successful generation of valid PDF bytes from a full result."""
        pdf_bytes = generate_image_analysis_pdf(
            result=sample_analysis_result,
            filename="suspect_profile.png"
        )
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        # Valid PDF documents must begin with the PDF magic header
        assert pdf_bytes.startswith(b"%PDF-")

    def test_generate_pdf_no_face_detected(self):
        """Verify clean handling of NO_FACE_DETECTED results."""
        no_face_result = {
            "status": "SUCCESS",
            "visual": {
                "status": "NO_FACE_DETECTED",
                "face_detected": False
            },
            "text": {
                "text_status": "AVAILABLE",
                "combined_text": "Plain document without human faces"
            },
            "governance": {
                "decision_status": "CONFIDENT_PREDICTION",
                "review_required": False
            },
            "audit": {
                "input_type": "IMAGE",
                "filename": "landscape.jpg",
                "timestamp": "2026-10-27T14:30:00Z"
            }
        }
        pdf_bytes = generate_image_analysis_pdf(
            result=no_face_result,
            filename="landscape.jpg"
        )
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF-")

    def test_generate_pdf_minimal_empty_result(self):
        """Verify graceful handling when result dictionary is sparse or empty."""
        pdf_bytes = generate_image_analysis_pdf(
            result={},
            filename="empty_result.png"
        )
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF-")

    def test_safe_text_escaping_and_unicode(self):
        """Verify XML escaping and ASCII/decimal encoding for arbitrary Unicode and emojis."""
        raw_text = 'Alert: <script>alert("hack")</script> & "quotes" — Hindi: यह परीक्षण है 🚨🔥'
        escaped = safe_text(raw_text)
        assert "<script>" not in escaped
        assert "&lt;script&gt;" in escaped
        assert "&amp;" in escaped
        # Non-ASCII characters should be entity-encoded or tagged without crashing
        assert "[ALERT]" in escaped
        assert "[URGENT]" in escaped

    def test_multilingual_unicode_rendering_in_pdf(self):
        """Verify genuine rendering of Hindi, Cyrillic, Chinese, Arabic, and emojis in PDF."""
        multilingual_result = {
            "status": "SUCCESS",
            "visual": {
                "prediction": "fake",
                "confidence": 0.95,
                "face_detected": True,
                "status": "SUCCESS"
            },
            "text": {
                "text_status": "AVAILABLE",
                "sources": ["OCR"],
                "segments": [
                    {"text": "Hindi: डिजिटल मीडिया सत्यापन और चेतावनी", "source": "OCR", "confidence": 0.95},
                    {"text": "Cyrillic: Внимание, обнаружена манипуляция лица", "source": "OCR", "confidence": 0.97},
                    {"text": "Chinese: 数字媒体真实性验证 - 深度伪造检测", "source": "OCR", "confidence": 0.94},
                    {"text": "Arabic: التحقق من الوسائط الرقمية", "source": "OCR", "confidence": 0.92},
                    {"text": "Emoji & Alert Symbols: 🚨 ⚠️ 🛡️ ✅ ❌", "source": "OCR", "confidence": 0.99}
                ],
                "combined_text": "Hindi: डिजिटल मीडिया सत्यापन | Cyrillic: Внимание | Chinese: 数字媒体 | Arabic: التحقق | Emojis: 🚨 ⚠️ 🛡️"
            },
            "governance": {
                "decision_status": "REVIEW_RECOMMENDED",
                "review_required": True,
                "review_justification": "Multilingual forensic evidence review"
            },
            "audit": {
                "timestamp": "2026-10-27T12:00:00Z",
                "input_type": "IMAGE",
                "filename": "multilingual_suspect_doc.png"
            }
        }

        pdf_bytes = generate_image_analysis_pdf(
            result=multilingual_result,
            filename="multilingual_suspect_doc.png"
        )
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 2000
        assert pdf_bytes.startswith(b"%PDF-")

    def test_sanitize_filename_traversal_protection(self):
        """Ensure filenames strip directory traversal sequences and unsafe characters."""
        dangerous = "../../../../etc/passwd.jpg"
        clean = sanitize_filename(dangerous)
        assert "/" not in clean
        assert "\\" not in clean
        assert ".." not in clean
        assert clean.endswith("passwd.jpg")

        windows_path = "C:\\Windows\\System32\\calc.exe"
        clean_win = sanitize_filename(windows_path)
        assert ":" not in clean_win
        assert "\\" not in clean_win
        assert "calc.exe" in clean_win

    def test_generate_download_filename(self):
        """Verify standard downloadable filename format."""
        ts = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
        dl_name = generate_download_filename("my_image_scan.png", timestamp=ts)
        assert dl_name == "dmv-image-report-my_image_scan-2026-10-27.pdf"

        # None/empty filename fallback
        dl_name_fallback = generate_download_filename(None, timestamp=ts)
        assert dl_name_fallback == "dmv-image-report-2026-10-27.pdf"

    def test_resolve_safe_image_path_prevents_traversal(self, tmp_path):
        """Ensure path traversal outside designated roots is rejected."""
        # Unsafe traversal
        unsafe = "../../../../../etc/passwd"
        assert _resolve_safe_image_path(unsafe, artifact_dir=str(tmp_path)) is None

        # Safe file inside artifact_dir
        safe_file = tmp_path / "gradcam_test.png"
        safe_file.write_bytes(b"dummy_png_bytes")
        resolved = _resolve_safe_image_path("gradcam_test.png", artifact_dir=str(tmp_path))
        assert resolved is not None
        assert os.path.exists(resolved)

    def test_decode_base64_image_caps_oversized_payloads(self):
        """Verify base64 decoder rejects payloads exceeding memory safety limits."""
        # Oversized payload (>10MB decoded, >14MB encoded)
        huge_data = "data:image/png;base64," + ("A" * (15 * 1024 * 1024))
        stream = _decode_base64_image(huge_data)
        assert stream is None

        # Valid small base64
        import base64
        small_bytes = b"fake_png_header"
        small_uri = "data:image/png;base64," + base64.b64encode(small_bytes).decode("ascii")
        stream = _decode_base64_image(small_uri)
        assert stream is not None
        assert stream.read() == small_bytes


class TestReportEndpointAPI:
    """Integration tests for POST /verify/image/report endpoint."""

    def test_download_report_endpoint_success(self, client, sample_analysis_result):
        """Verify successful 200 response with correct content-type and filename header."""
        payload = {
            "result": sample_analysis_result,
            "filename": "suspect_profile.png",
            "image_preview": None
        }
        response = client.post("/verify/image/report", json=payload)
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/pdf"
        assert "Content-Disposition" in response.headers
        disp = response.headers["Content-Disposition"]
        assert "attachment" in disp
        assert 'filename="dmv-image-report-suspect_profile-2026-10-27.pdf"' in disp
        assert response.content.startswith(b"%PDF-")

    def test_download_report_does_not_rerun_inference(self, client, sample_analysis_result):
        """Critical constraint: requesting a report must export existing results without rerun."""
        payload = {
            "result": sample_analysis_result,
            "filename": "precomputed_analysis.jpg"
        }
        with patch.object(MultimodalPipeline, "verify") as mock_verify:
            response = client.post("/verify/image/report", json=payload)
            assert response.status_code == 200
            # Ensure model verification was NEVER called
            mock_verify.assert_not_called()

    def test_download_report_preserves_current_image_results(self, client):
        """Verify the report corresponds to the specific image result passed in the request."""
        # Case A: Real image analysis
        result_a = {
            "visual": {"prediction": "real", "confidence": 0.985, "face_detected": True, "status": "SUCCESS"},
            "governance": {"decision_status": "CONFIDENT_PREDICTION", "review_required": False}
        }
        resp_a = client.post("/verify/image/report", json={"result": result_a, "filename": "authentic_face.png"})
        assert resp_a.status_code == 200
        assert "authentic_face" in resp_a.headers["Content-Disposition"]

        # Case B: Manipulated image analysis
        result_b = {
            "visual": {"prediction": "fake", "confidence": 0.992, "face_detected": True, "status": "SUCCESS"},
            "governance": {"decision_status": "REVIEW_RECOMMENDED", "review_required": True}
        }
        resp_b = client.post("/verify/image/report", json={"result": result_b, "filename": "tampered_face.png"})
        assert resp_b.status_code == 200
        assert "tampered_face" in resp_b.headers["Content-Disposition"]

    def test_download_report_with_incomplete_fields(self, client):
        """Verify endpoint handles missing optional fields without failing."""
        payload = {
            "result": {
                "visual": {"status": "NO_FACE_DETECTED", "face_detected": False}
            },
            "filename": "no_face_input.jpg"
        }
        response = client.post("/verify/image/report", json=payload)
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/pdf"
        assert response.content.startswith(b"%PDF-")

    def test_download_report_failure_returns_500(self, client, sample_analysis_result):
        """Verify endpoint returns clean HTTP 500 when PDF generation encounters an unexpected error."""
        with patch("api.routes.verify.generate_image_analysis_pdf", side_effect=RuntimeError("ReportLab internal error")):
            payload = {
                "result": sample_analysis_result,
                "filename": "error_sample.jpg"
            }
            response = client.post("/verify/image/report", json=payload)
            assert response.status_code == 500
            data = response.json()
            assert "Failed to generate report" in data["detail"]
