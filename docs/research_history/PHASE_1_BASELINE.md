# Phase 1 Baseline

## Git
- **Current Branch:** `swaraj` (Verified via `git branch --show-current`)
- **Working Tree Status:** Clean development state on `swaraj`; uncommitted user modifications in `frontend/src/` (`App.css`, `Header.jsx`, `ResultDashboard.jsx`, `TokenAttributionView.jsx`, `index.css`) and untracked documentation files safely preserved without resets or overwrites.
- **Latest Commit:** `79a0767 Implement MVP digital media verification system with FastAPI, React frontend, XAI, UQ, and governance`

## Repository Structure
- **Active Directories:**
  - `api/`: FastAPI REST service layer (`main.py`, `routes/`, `schemas/`, `dependencies/`, `utils/`).
  - `frontend/`: React 19 + Vite 8 analyst workstation (`src/components/`, `src/hooks/`, `src/services/`, `src/__tests__/`).
  - `pipeline/`: Multimodal orchestration (`multimodal_pipeline.py`, `router.py`, `image_pipeline.py`, `video_pipeline.py`, `text_pipeline.py`).
  - `detection/`: Core detector classes (`deepfake_detector.py`, `propaganda_detector.py`, `hate_speech_detector.py`).
  - `models/`: Neural architectures, checkpoints, tokenizers, training and inference routines (`models/deepfake/`, `models/propaganda/`, `models/hate_speech/`).
  - `extraction/`: Signal extraction utilities (`ocr.py` EasyOCR, `asr.py` Whisper-base via Transformers, `text_aggregator.py`).
  - `preprocessing/`: Frame extraction (`frame_sampling.py`), face detection (`face_detection.py` MTCNN).
  - `explainability/`: Grad-CAM (`gradcam.py`) and token attribution (`token_attribution.py`, `explanation_generator.py`).
  - `uncertainty/`: Monte Carlo Dropout uncertainty engine (`uncertainty_engine.py`) and decision governance policy.
  - `evaluation/`: Evaluation metrics (`metrics.py`).
  - `data/`: Local dataset storage (`datasets/` FaceForensics++, HateXplain, SemEval-2020), splits, and sample test fixtures (`tests/assets/`).
  - `config/`: System configuration (`config/config.yaml`).
  - `scripts/`: Diagnostic, verification, and smoke-testing utilities.
  - `tests/`: End-to-end backend tests (`test_api.py`, `test_multimodal_pipeline.py`, `test_explainability_uncertainty.py`).
  - `reports/`: Audit logs and implementation milestone reports.
  - `docs/`: System documentation, viva preparation, and context files.
- **Legacy Directories / Files:**
  - `app.py`: 7-line Streamlit stub (retained intact for safety; inactive).
  - `dashboard/`: Streamlit scaffold stubs (`components.py`, `layout.py`, `state.py`, `visualizations.py`) (retained intact; inactive).
  - `governance/`: Initial scaffold stubs (`audit_logger.py`, `decision_policy.py`, `human_review.py`, `safety_checks.py`) (retained intact; active governance logic operates in `uncertainty/uncertainty_engine.py` and `pipeline/`).
  - `utils/`: Initial scaffold stubs (`common.py`, `file_utils.py`, `logger.py`) (retained intact; active utilities reside in `api/utils/`).
  - `evaluation/` stubs: `calibration.py`, `evaluate_deepfake.py`, `evaluate_hate_speech.py`, `evaluate_propaganda.py` (retained; active metrics operate in `evaluation/metrics.py`).
  - `preprocessing/` stubs: `audio_preprocessing.py`, `normalisation.py`, `video_utils.py` (retained; active modules are `face_detection.py` and `frame_sampling.py`).
  - `config/thresholds.yaml`: Legacy configuration file not referenced by active code.
- **Organization Changes Made:**
  - Preserved the existing directory layout without disruptive file moves to avoid breaking active dependencies (e.g. `from detection...` and `from preprocessing...`).
  - Kept `config/` unchanged per explicit instruction.
  - Retained all legacy scaffold files safely without deletion.

## Environment
- **Python Version:** 3.14.2 (`C:\Program Files\Python314\python.exe`)
- **PyTorch Version:** 2.14.0+cpu (`C:\Users\Swaraj Shedge\AppData\Roaming\Python\Python314\site-packages`)
- **CUDA Availability in PyTorch:** `False` (`torch.cuda.is_available() == False`)
- **CUDA Version in PyTorch:** `None`
- **GPU Count in PyTorch:** `0`
- **GPU Name in PyTorch:** `None`
- **Physical Hardware (nvidia-smi):**
  - Driver Version: `592.82`
  - Supported CUDA API: `13.1`
  - GPU: `NVIDIA GeForce RTX 4050 Laptop GPU` (6,141 MiB VRAM)
  - Current Reality: The active Python environment uses a CPU-only PyTorch wheel. PyTorch cannot access the RTX 4050 GPU in this environment.
- **Important Dependency Status:**
  - `fastapi`: 0.135.3 (Verified importable & operational)
  - `uvicorn`: 0.41.0 (Verified importable & operational)
  - `torch`: 2.14.0+cpu (Verified importable & operational)
  - `torchvision`: 0.29.0 (Verified importable & operational)
  - `transformers`: 5.10.2 (Verified importable & operational)
  - `cv2` (OpenCV): 4.13.0 (Verified importable & operational)
  - `facenet_pytorch` (MTCNN): 2.6.0 (Verified importable & operational)
  - `easyocr`: 1.7.2 (Verified importable & operational)
  - `imageio_ffmpeg`: 0.6.0 (Verified importable & operational)
  - `whisper`: The codebase utilizes Hugging Face `transformers.pipeline("automatic-speech-recognition", model="openai/whisper-base")` via `extraction.asr`, which imports and executes successfully.
  - `pytest`: System PATH `pytest` points to Anaconda; executing via `python -m pytest` correctly uses the Python 3.14 environment with installed dependencies.

## Model Artifacts
- **Checkpoint Status:**
  - Deepfake: `models/deepfake/checkpoint/best_model.pt` (70.9 MB) — EfficientNet-B4 binary classifier (Present & verified).
  - Propaganda: `models/propaganda/checkpoint/model.safetensors` (498.6 MB) — RoBERTa-base sequence classification (Present & verified).
  - Hate Speech: `models/hate_speech/checkpoint/model.safetensors` (438.0 MB) — BERT-base-uncased sequence classification (Present & verified).
  - ASR (Whisper): Hugging Face pre-cached `openai/whisper-base` model weights (~290 MB) loaded through `extraction.asr` (Present & verified).
- **Tokenizer / Config Status:**
  - Propaganda: `models/propaganda/checkpoint/config.json`, `models/propaganda/tokenizer/tokenizer.json`, `models/propaganda/tokenizer/tokenizer_config.json`, `models/propaganda/label_mapping.json` (14 classes: 0 to 13) (Present & verified).
  - Hate Speech: `models/hate_speech/checkpoint/config.json`, `models/hate_speech/tokenizer/tokenizer.json`, `models/hate_speech/tokenizer/tokenizer_config.json`, `models/hate_speech/label_mapping.json` (3 classes: normal, hatespeech, offensive) (Present & verified).
- **Inference Compatibility:** Checkpoint architectures and tokenizer vocabularies align with the inference classes in `models/` and `detection/`.

## Backend
- **Entry Point:** `api/main.py` (`uvicorn api.main:app`)
- **Endpoints Verified:**
  - `GET /health`: 200 OK — Liveness probe returning `{"status": "healthy", "service": "digital-media-verification"}`.
  - `GET /ready`: 200 OK — Readiness probe verifying pre-warmed models and pipelines.
  - `POST /verify/text`: 200 OK — Analyzes text for propaganda techniques and hate speech with token attribution and MC Dropout.
  - `POST /verify/image`: 200 OK — Analyzes image for facial deepfakes (EfficientNet-B4 + Grad-CAM) and embedded text (EasyOCR + NLP).
  - `POST /verify/video`: 200 OK — Processes video synchronously or queues asynchronous background jobs.
  - `POST /verify/multimodal`: 200 OK — Unified endpoint accepting media and/or text with channel provenance.
  - `GET /verify/status/{job_id}`: 200 OK — Retrieves progress or results for background video jobs.
- **Startup Device Behavior:**
  - Explicitly hardcoded to CPU in `api/main.py:28`: `init_pipeline(device="cpu")`.
- **Video Processing Mechanism:**
  - Synchronous mode: Direct execution in request thread via `pipeline.verify(temp_path, ...)`.
  - Asynchronous mode: Dispatched to FastAPI `BackgroundTasks` via worker `_run_video_background()`.
- **Job Storage Mechanism:**
  - In-memory dictionary `job_store: Dict[str, Dict[str, Any]]` in `api/routes/verify.py`. Jobs are transient and reset on process restart.
- **Error Handling:**
  - Structured JSON exception handlers for `HTTPException`, `RequestValidationError` (422), and fallback `Exception` (500) returning `{ "error": ..., "detail": ..., "status_code": ... }`.

## Frontend
- **Framework & Tooling:** React 19.2.8, React DOM 19.2.8, Vite 8.2.2, Vitest 4.1.11, Lucide React 1.44.0.
- **Entry Point:** `frontend/src/main.jsx` and `frontend/src/App.jsx`.
- **Main UI Structure:**
  - Top Header: Project branding, active backend health/liveness indicator.
  - Mode Switcher: 4 verification modes (Image, Video, Text, Multimodal).
  - Input Views: `ImageInput.jsx`, `VideoInput.jsx`, `TextInput.jsx`, `MultimodalInput.jsx`.
  - Result Dashboard:
    - Visual results (`VisualResultView.jsx`): Face bounding boxes, deepfake probability, Grad-CAM heatmaps.
    - NLP results (`NLPResultView.jsx`): Multi-class propaganda distributions and hate speech classification.
    - Explainability (`TokenAttributionView.jsx`): Interactive word-level attribution heatmaps.
    - Uncertainty (`UncertaintyCard.jsx`): Shannon entropy, predictive variance, MC Dropout stats.
    - Governance (`GovernancePanel.jsx`): Human review escalation status (`AUTOMATED_ACTION_PERMITTED`, `REVIEW_RECOMMENDED`, `ANALYSIS_INCOMPLETE`).
    - Extracted Text (`TextExtractionView.jsx`): EasyOCR recognized text and Whisper ASR transcripts.
    - Audit Trail (`AuditDetails.jsx`): SHA-256 media hashes, model versions, timestamped audit metadata.
- **Build & Test Status:**
  - Production build: `npm run build` succeeds cleanly in 882ms (`dist/index.html`, `dist/assets/`).
  - Unit/Component tests: `npm test` passes 10/10 tests in 3.19s.

## Tests
- **Backend Result:**
  - Command: `python -m pytest -q tests`
  - Total Tests: 41
  - Passed: 41
  - Failed: 0
  - Skipped: 0
  - Execution Time: 82.72s
  - Warnings: 11 (Torch quantization deprecations in EasyOCR, Starlette HTTP 422 deprecation notice, pin_memory notice without GPU accelerator).
- **Frontend Result:**
  - Command: `cd frontend && npm test`
  - Total Tests: 10
  - Passed: 10
  - Failed: 0
  - Skipped: 0
  - Execution Time: 3.19s

## Baseline Issues Confirmed
1. **CPU Inference Hardcoded (VERIFIED)**: `api/main.py:28` hardcodes `init_pipeline(device="cpu")`.
2. **PyTorch CPU-Only Build (VERIFIED)**: `torch-2.14.0+cpu` is installed. Even though the system has an NVIDIA RTX 4050 Laptop GPU, PyTorch cannot leverage CUDA in the current environment.
3. **Transient In-Memory Video Job Store (VERIFIED)**: `job_store` in `api/routes/verify.py` is an in-memory dictionary. Jobs do not persist across server restarts.
4. **Environment PATH Precedence for Pytest (VERIFIED)**: Invoking bare `pytest` executes Anaconda's pytest binary from PATH, which fails due to missing modules in Anaconda. Running via `python -m pytest` resolves to the correct environment.
5. **Legacy Scaffold Files Present (VERIFIED)**: `app.py`, `dashboard/`, `governance/`, `utils/`, and `config/thresholds.yaml` exist as legacy scaffolding alongside the active FastAPI/React codebase.
6. **No Real Face Tracking Across Video Frames (VERIFIED)**: Frame extraction detects faces per frame independently; no multi-face identity association is performed across time.
7. **Development-Default Thresholds for Uncertainty (VERIFIED)**: Uncertainty thresholds (`variance > 0.02`, `entropy > 0.85`) are development defaults rather than statistically calibrated.

## Phase 1 Changes
- `docs/PHASE_1_BASELINE.md`: Created comprehensive Phase 1 baseline audit document detailing Git, Repository Structure, Environment, Model Artifacts, Backend, Frontend, Tests, and Verified Issues.
- Preserved existing uncommitted user files in `frontend/src/` and untracked context documents.
- Preserved repository layout, active modules (`detection/`, `preprocessing/`, `pipeline/`), and legacy stubs (`app.py`, `dashboard/`, `governance/`, `utils/`) without breaking changes.

## Phase 1 Status
COMPLETE
