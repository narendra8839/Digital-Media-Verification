# Digital Media Verification — System Design & Service Contracts

## 1. Gateway & API Design

The backend service is built using **FastAPI** (`api/main.py`), utilizing asynchronous ASGI routing, Pydantic data validation (`api/schemas/requests.py`), and a modular route structure (`api/routes/verify.py`).

### 1.1 Endpoint Specifications

| Endpoint | HTTP Method | Input Content-Type | Response Type | Description |
| :--- | :---: | :--- | :--- | :--- |
| `/api/health` | `GET` | N/A | `application/json` | Liveness health probe returning system status and device mode |
| `/api/ready` | `GET` | N/A | `application/json` | Readiness probe verifying loaded model checkpoint availability |
| `/api/verify/text` | `POST` | `application/json` | `VerificationResult` | Analyzes text for propaganda techniques and hate speech |
| `/api/verify/image` | `POST` | `multipart/form-data` | `VerificationResult` | Analyzes image for visual deepfakes, OCR text, and NLP flags |
| `/api/verify/video` | `POST` | `multipart/form-data` | `VerificationResult` / Job ID | Analyzes video frames (1.0 FPS) and demuxed audio transcript |
| `/api/verify/video/async` | `POST` | `multipart/form-data` | `{"job_id": str}` | Enqueues long-running video verification for background worker |
| `/api/verify/jobs/{id}` | `GET` | N/A | `application/json` | Polls background job status, progress, and result payload |
| `/api/artifacts/{id}` | `GET` | N/A | `image/png` | Serves transient Grad-CAM visualization heatmaps |
| `/api/verify/report/pdf` | `POST` | `application/json` | `application/pdf` | Generates on-the-fly downloadable PDF report for analyzed image |

---

## 2. Data Contracts and Schemas

All verification requests produce an immutable, strongly-typed **`VerificationResult`** schema:

```json
{
  "media_type": "image",
  "overall_verdict": "Likely Manipulated",
  "overall_risk_score": 0.88,
  "confidence": 0.92,
  "human_review_required": false,
  "uncertainty_score": 0.042,
  "visual_result": {
    "is_deepfake": true,
    "confidence": 0.92,
    "predictive_variance": 0.042,
    "faces_detected": 1,
    "artifact_url": "/api/artifacts/artifact_abc123.png",
    "frames_analyzed": 1
  },
  "text_result": {
    "text": "Detected caption or OCR snippet",
    "propaganda": {
      "detected": true,
      "techniques": [
        {"name": "Loaded Language", "confidence": 0.85, "span": [12, 28]}
      ]
    },
    "hate_speech": {
      "detected": false,
      "category": "Normal",
      "confidence": 0.98
    }
  },
  "processing_time_sec": 0.428,
  "timestamp": "2026-10-27T17:30:00Z"
}
```

---

## 3. Artifact Retention and Lifecycle Management

Visual explanations (Grad-CAM overlays) are generated as temporary PNG artifacts. To prevent unbounded disk growth while guaranteeing artifact availability during active user sessions, `api/utils/artifact_manager.py` manages artifact lifecycles:

1. **Configurable Directory:** Governed by `DMV_ARTIFACT_DIR` environment variable (defaults to `artifacts/`).
2. **Session Protection:** `protect_active_artifact(artifact_id)` shields in-use artifacts from eviction.
3. **Background Pruning:** A daemon thread runs periodically, removing unshielded files older than the TTL (default: 3,600 seconds / 1 hour).
4. **Security Defenses:**
   - Strict filename regex validation (`artifact_[a-zA-Z0-9_\-]+\.png`) rejecting directory traversal sequences (`../`, `..\\`).
   - Resolution checking ensuring resolved paths reside strictly within the designated artifact directory.
   - Symlink traversal protection.

---

## 4. Zero-Inference Downloadable Image Analysis Report

To allow forensic analysts, journalists, and end users to export self-contained audit records, the system provides a PDF report generator (`api/utils/report_generator.py`):

1. **Zero-Inference Principle:** Generates the audit report directly from the client's precomputed `VerificationResult` and cached Grad-CAM image. No redundant GPU model passes are executed.
2. **Defensive Image Decoding:** Base64-encoded visual evidence is validated, capped at 10 MB, and decoded with PIL error guards. If Grad-CAM artifacts have expired or no faces were found, the report renders a clean informational fallback panel without failing.
3. **Multilingual Unicode PDF Engine:**
   - Uses ReportLab Platypus layout engine (`SimpleDocTemplate`).
   - Dynamically discovers system TrueType fonts (`Nirmala.ttc` for Devanagari/Hindi, `STSong-Light` CID font for Chinese/Japanese, `DejaVuSans` / `Arial` for Cyrillic and Arabic).
   - Sanitizes text and maps color emoji glyphs to text tags (`[ALERT]`, `[SHIELD]`, `[WARNING]`).
4. **Path Disclosure Immunity:** If an unexpected error occurs during rendering, internal server paths and stack traces are suppressed, returning a generic safe error response.
