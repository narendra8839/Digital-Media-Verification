# Current MVP Audit: Digital Media Verification System

> **Project Title:** Digital Media Verification: Explainable and Uncertainty-Aware Deepfake, Propaganda and Hate-Speech Detection  
> **Repository Branch:** `swaraj`  
> **Audit Date:** October 2026  
> **Audit Mode:** Pure Observation & Reverse-Engineering (Zero Code Modifications)

---

## 1. Executive Summary

This document presents a comprehensive technical audit of the **Digital Media Verification** repository on branch `swaraj`. The audit was conducted by reverse-engineering the codebase directly from disk, tracing execution pathways across Python modules and React components, inspecting physical weights and dataset manifests, and validating test suites.

### Core Audit Findings

1. **System Implementation Reality (Full-Stack Operational)**:
   - The project is **not** an abstract prototype or planning scaffold. The core MVP is **implemented, fine-tuned, and fully operational**.
   - The architecture comprises a **FastAPI (Python 3.14 / Uvicorn)** REST backend service connected to an interactive **React 19 + Vite 8** forensic analyst dashboard.
   - All three core fine-tuned deep learning models (`EfficientNet-B4`, `RoBERTa-base`, `BERT-base-uncased`) have physical weights stored on disk and are actively loaded by detector classes.
   - Ancillary extraction modules (`MTCNN` for facial cropping, `EasyOCR` for text recognition, `OpenAI Whisper-base` for 16 kHz speech transcription, and `imageio-ffmpeg` for audio probing) are integrated into a master pipeline.
   - Explainable AI (spatial Grad-CAM and token-level Gradient $\times$ Input attribution) and Uncertainty Quantification (Monte Carlo Dropout, $T=20$) execute in real time on inference requests.
   - Automated test suites verify stability: **41 backend tests (Pytest)** and **10 frontend tests (Vitest)** pass with 100% success.

2. **Scaffold Stubs vs. Implemented Code**:
   - A critical discovery during this audit is the existence of **scaffold stubs** from initial planning:
     - `app.py` is a 7-line placeholder from an early Streamlit draft (`st.info("Project scaffold created...")`).
     - `dashboard/*.py` (`results.py`, `review_panel.py`, `upload.py`, `visualizations.py`) are single-line docstring stubs.
     - `governance/*.py` (`audit_logger.py`, `decision_policy.py`, `human_review.py`, `safety_checks.py`) are single-line docstring stubs.
     - `preprocessing/*.py` (`image_preprocessing.py`, `text_cleaning.py`, `video_preprocessing.py`) are single-line docstring stubs.
     - `evaluation/*.py` (`calibration.py`, `evaluate_deepfake.py`, `evaluate_hate_speech.py`, `evaluate_propaganda.py`) are single-line docstring stubs.
     - `utils/*.py` (`common.py`, `file_utils.py`, `logger.py`) are single-line docstring stubs.
   - **Crucial Clarification**: The actual working implementation does **not** rely on these stubs. The real logic was implemented directly within:
     - Backend API: `api/` (`main.py`, `routes/`, `dependencies/`, `utils/`)
     - Pipeline Orchestration: `pipeline/` (`router.py`, `image_pipeline.py`, `video_pipeline.py`, `text_pipeline.py`, `multimodal_pipeline.py`)
     - Model Detectors: `detection/` (`deepfake_detector.py`, `propaganda_detector.py`, `hate_speech_detector.py`)
     - Model Definitions & Training: `models/` (`deepfake/`, `propaganda/`, `hate_speech/`)
     - Feature Extraction: `extraction/` (`asr.py`, `audio_extraction.py`, `ocr.py`, `text_aggregator.py`)
     - Vision Preprocessing: `preprocessing/` (`face_detection.py`, `frame_sampling.py`)
     - Explainability: `explainability/` (`gradcam.py`, `text_attention.py`, `explanation_generator.py`)
     - Uncertainty & Governance: `uncertainty/` (`mc_dropout.py`, `variance.py`, `uncertainty_engine.py`)
     - User Interface: `frontend/src/` (`App.jsx`, `components/`, `hooks/`, `services/`)

3. **Responsible AI & Governance Integrity**:
   - The system functions strictly as **decision-support telemetry** for human reviewers. It nowhere attempts autonomous content censorship, account banning, or definitive truth claims.
   - The governance consistency policy is enforced: when media contains no human faces, the visual pipeline returns `visual.status = "NO_FACE_DETECTED"` and the governance layer assigns `decision_status = "ANALYSIS_INCOMPLETE"` (refusing to falsely report faceless images as confident predictions).

4. **Empirical Results Transparency**:
   - The audited metrics on disk confirm genuine MVP training behavior:
     - **Deepfake**: 97.50% train video accuracy, but drops to **50.00% test video accuracy** (Macro F1: 0.4857). This weak generalization across subject-disjoint splits is an authentic finding due to small MVP subset size (160 frames from 40 identities) and validates the necessity of uncertainty-aware human triage.
     - **Propaganda**: 47.68% train accuracy, 29.29% validation accuracy across 14 fine-grained classes (random baseline is 7.14%).
     - **Hate Speech**: 81.33% train accuracy, 60.00% test accuracy across 3 classes (random baseline is 33.33%).

---

## 2. Exact Folder Structure

The repository contains 9,095 files totaling ~1.16 GB (dominated by 3,000 FaceForensics++ videos, HuggingFace safetensors, and node_modules). Excluding `.git`, `node_modules`, `__pycache__`, and temporary caches, the active project structure is detailed below.

### Status Taxonomy
- **IMPLEMENTED**: Active, fully functional, imported, and covered by test suites or execution scripts.
- **PARTIALLY IMPLEMENTED**: Functional for primary flows but has minor omissions or unintegrated options.
- **PLACEHOLDER**: A stub or scaffold file containing only docstrings or empty passes; not actively imported.
- **UNVERIFIED**: Code exists but lacks standalone automated tests or verified runtime execution.
- **BROKEN**: Code exists but fails on execution or has fatal syntax/import errors.

```
Digital-Media-Verification/
├── api/                                      # FastAPI REST Service Layer
│   ├── dependencies/
│   │   ├── __init__.py                       # Package marker
│   │   └── pipeline_dep.py                   # [IMPLEMENTED] Singleton pipeline dependency provider (High Confidence)
│   ├── routes/
│   │   ├── __init__.py                       # Package marker
│   │   ├── health.py                         # [IMPLEMENTED] GET /health and GET /ready endpoints (High Confidence)
│   │   └── verify.py                         # [IMPLEMENTED] POST /verify/text, /image, /video, /multimodal (High Confidence)
│   ├── schemas/
│   │   ├── __init__.py                       # Package marker
│   │   ├── requests.py                       # [IMPLEMENTED] Pydantic request models (TextVerificationRequest) (High Confidence)
│   │   └── responses.py                      # [IMPLEMENTED] Pydantic response models (Health, Ready, AsyncJob) (High Confidence)
│   ├── utils/
│   │   ├── __init__.py                       # Package marker
│   │   ├── file_utils.py                     # [IMPLEMENTED] Streaming upload, extension & size validation (High Confidence)
│   │   └── serialization.py                  # [IMPLEMENTED] Recursive NumPy/PyTorch JSON sanitization (High Confidence)
│   ├── __init__.py                           # Package marker
│   └── main.py                               # [IMPLEMENTED] FastAPI app instance, CORS, lifespan pre-warming, exceptions (High Confidence)
│
├── config/                                   # Configuration Files
│   ├── config.yaml                           # [IMPLEMENTED] Main system config (models, thresholds, seed=42) (High Confidence)
│   └── thresholds.yaml                       # [PLACEHOLDER] Legacy scaffold threshold config (Unused in code) (High Confidence)
│
├── dashboard/                                # [PLACEHOLDER] Legacy Streamlit Dashboard Stubs
│   ├── results.py                            # [PLACEHOLDER] Docstring stub only (High Confidence)
│   ├── review_panel.py                       # [PLACEHOLDER] Docstring stub only (High Confidence)
│   ├── upload.py                             # [PLACEHOLDER] Docstring stub only (High Confidence)
│   └── visualizations.py                     # [PLACEHOLDER] Docstring stub only (High Confidence)
│
├── data/                                     # Datasets & MVP Subsets
│   ├── faceforensics/C23/                    # [IMPLEMENTED] FF++ C23 development copy (3,000 videos, official splits) (High Confidence)
│   ├── hatexplain/                           # [IMPLEMENTED] HateXplain benchmark dataset (20,148 annotated posts) (High Confidence)
│   ├── semeval2020_task11/                   # [IMPLEMENTED] SemEval propaganda corpus (articles & span labels) (High Confidence)
│   └── mvp/                                  # [IMPLEMENTED] Balanced MVP subsets for model fine-tuning (High Confidence)
│       ├── deepfake/                         # Subsets: train (40 vids), val (18 vids), test (18 vids) (High Confidence)
│       ├── hatexplain/                       # Subsets: train (600 posts), val (150), test (150) (High Confidence)
│       ├── semeval2020_task11/               # Subsets: train (560 spans), val (140 spans) (High Confidence)
│       ├── dataset_report.json               # [IMPLEMENTED] Dataset provenance and validation report (High Confidence)
│       └── README.md                         # [IMPLEMENTED] Subset documentation (High Confidence)
│
├── detection/                                # High-level Model Detectors
│   ├── deepfake_detector.py                  # [IMPLEMENTED] EfficientNet-B4 + MTCNN + Grad-CAM + UQ (High Confidence)
│   ├── hate_speech_detector.py               # [IMPLEMENTED] BERT-base 3-class + Token Attribution + UQ (High Confidence)
│   └── propaganda_detector.py                # [IMPLEMENTED] RoBERTa-base 14-class + Token Attribution + UQ (High Confidence)
│
├── docs/                                     # Documentation
│   ├── current_project_context_for_gemini.md # [IMPLEMENTED] Comprehensive current project context for slides (High Confidence)
│   ├── demo_walkthrough.md                   # [IMPLEMENTED] Step-by-step live demo instructions (High Confidence)
│   └── viva_questions.md                     # [IMPLEMENTED] 23 viva defense questions & answers (High Confidence)
│
├── evaluation/                               # Evaluation Utilities
│   ├── calibration.py                        # [PLACEHOLDER] Docstring stub only (High Confidence)
│   ├── evaluate_deepfake.py                  # [PLACEHOLDER] Docstring stub only (High Confidence)
│   ├── evaluate_hate_speech.py               # [PLACEHOLDER] Docstring stub only (High Confidence)
│   ├── evaluate_propaganda.py                # [PLACEHOLDER] Docstring stub only (High Confidence)
│   └── metrics.py                            # [IMPLEMENTED] compute_classification_metrics() (High Confidence)
│
├── explainability/                           # Explainable AI (XAI)
│   ├── explanation_generator.py              # [IMPLEMENTED] Unified XAI orchestrator & schema builder (High Confidence)
│   ├── gradcam.py                            # [IMPLEMENTED] EfficientNet-B4 final conv layer Grad-CAM & overlay (High Confidence)
│   └── text_attention.py                     # [IMPLEMENTED] Grad x Input token attribution & rationale alignment (High Confidence)
│
├── extraction/                               # Feature & Content Extraction
│   ├── asr.py                                # [IMPLEMENTED] Whisper-base ASR wrapper with PCM audio decoding (High Confidence)
│   ├── audio_extraction.py                   # [IMPLEMENTED] imageio-ffmpeg stream probe & WAV extraction (High Confidence)
│   ├── ocr.py                                # [IMPLEMENTED] EasyOCR CRAFT text detector wrapper (High Confidence)
│   └── text_aggregator.py                    # [IMPLEMENTED] Multi-source text aggregator with provenance tracking (High Confidence)
│
├── frontend/                                 # React 19 + Vite 8 Analyst Workstation
│   ├── public/                               # Static assets (favicons, SVG icons)
│   ├── src/
│   │   ├── assets/                           # Branding assets
│   │   ├── components/
│   │   │   ├── AuditDetails.jsx              # [IMPLEMENTED] Collapsible reproducibility & checkpoint telemetry (High Confidence)
│   │   │   ├── GovernancePanel.jsx           # [IMPLEMENTED] Responsible AI decision status & justifications (High Confidence)
│   │   │   ├── Header.jsx                    # [IMPLEMENTED] Top navigation & academic branding bar (High Confidence)
│   │   │   ├── HealthIndicator.jsx           # [IMPLEMENTED] Pulsing backend liveness indicator (High Confidence)
│   │   │   ├── ImageInput.jsx                # [IMPLEMENTED] Drag-and-drop image dropzone & preview (High Confidence)
│   │   │   ├── InputSelector.jsx             # [IMPLEMENTED] 4-modality navigation tabs (High Confidence)
│   │   │   ├── MultimodalInput.jsx           # [IMPLEMENTED] Combined media + caption input panel (High Confidence)
│   │   │   ├── NLPResultView.jsx             # [IMPLEMENTED] Propaganda & Hate Speech results display (High Confidence)
│   │   │   ├── ProcessingIndicator.jsx       # [IMPLEMENTED] Loading spinner & stage telemetry (High Confidence)
│   │   │   ├── ResultDashboard.jsx           # [IMPLEMENTED] Master result coordination view (High Confidence)
│   │   │   ├── TextExtractionView.jsx        # [IMPLEMENTED] Multi-source provenance text view (High Confidence)
│   │   │   ├── TextInput.jsx                 # [IMPLEMENTED] Textarea with sample benchmark chips (High Confidence)
│   │   │   ├── TokenAttributionView.jsx      # [IMPLEMENTED] Saliency highlighted token pill chips (High Confidence)
│   │   │   ├── UncertaintyCard.jsx           # [IMPLEMENTED] MC Dropout variance, std, entropy card (High Confidence)
│   │   │   ├── VideoInput.jsx                # [IMPLEMENTED] Video upload, FPS selector & async toggle (High Confidence)
│   │   │   └── VisualResultView.jsx          # [IMPLEMENTED] Deepfake result & Grad-CAM viewer (High Confidence)
│   │   ├── hooks/
│   │   │   ├── useBackendHealth.js           # [IMPLEMENTED] Non-blocking health polling hook (High Confidence)
│   │   │   └── useVerification.js            # [IMPLEMENTED] Verification state machine & async video polling (High Confidence)
│   │   ├── services/
│   │   │   └── api.js                        # [IMPLEMENTED] Fetch wrapper for FastAPI endpoints (High Confidence)
│   │   ├── types/
│   │   │   └── schemas.js                    # [IMPLEMENTED] Domain constants, status enums & formatters (High Confidence)
│   │   ├── __tests__/
│   │   │   └── App.test.jsx                  # [IMPLEMENTED] 10 Vitest integration tests (High Confidence)
│   │   ├── App.css                           # Component styling overrides
│   │   ├── App.jsx                           # [IMPLEMENTED] Main application shell (High Confidence)
│   │   ├── index.css                         # [IMPLEMENTED] Clean Vanilla CSS design system (High Confidence)
│   │   ├── main.jsx                          # React 19 DOM entry point
│   │   └── setupTests.js                     # Testing Library Jest-DOM setup
│   ├── .env / .env.example                   # Environment configuration (VITE_API_BASE_URL)
│   ├── package.json                          # Dependencies: React 19, Lucide, Vitest, Vite 8
│   └── vite.config.js                        # Vite build & test configuration
│
├── governance/                               # [PLACEHOLDER] Legacy Governance Module Stubs
│   ├── audit_logger.py                       # [PLACEHOLDER] Docstring stub only (High Confidence)
│   ├── decision_policy.py                    # [PLACEHOLDER] Docstring stub only (High Confidence)
│   ├── human_review.py                       # [PLACEHOLDER] Docstring stub only (High Confidence)
│   └── safety_checks.py                      # [PLACEHOLDER] Docstring stub only (High Confidence)
│
├── models/                                   # Model Architecture Definitions & Trained Checkpoints
│   ├── deepfake/
│   │   ├── checkpoint/best_model.pt          # [IMPLEMENTED] Fine-tuned PyTorch state dict (70.9 MB) (High Confidence)
│   │   ├── efficientnet_b4.py                # [IMPLEMENTED] EfficientNet-B4 binary classification head (High Confidence)
│   │   ├── label_mapping.json                # [IMPLEMENTED] {"0": "real", "1": "fake"} (High Confidence)
│   │   ├── metrics.json                      # [IMPLEMENTED] Verified validation & test metrics (High Confidence)
│   │   ├── predict.py                        # [IMPLEMENTED] Deepfake inference & checkpoint loader (High Confidence)
│   │   ├── train.py                          # [IMPLEMENTED] MVP training script (High Confidence)
│   │   └── training_config.json              # [IMPLEMENTED] Saved hyperparameter config (High Confidence)
│   ├── hate_speech/
│   │   ├── checkpoint/model.safetensors      # [IMPLEMENTED] Fine-tuned BERT weights (438.0 MB) (High Confidence)
│   │   ├── checkpoint/config.json            # [IMPLEMENTED] HuggingFace model config (High Confidence)
│   │   ├── tokenizer/tokenizer.json          # [IMPLEMENTED] WordPiece tokenizer assets (High Confidence)
│   │   ├── bert.py                           # [IMPLEMENTED] BERT-base-uncased 3-class architecture (High Confidence)
│   │   ├── label_mapping.json                # [IMPLEMENTED] {"0": "normal", "1": "offensive", "2": "hatespeech"} (High Confidence)
│   │   ├── metrics.json                      # [IMPLEMENTED] Verified validation & test metrics (High Confidence)
│   │   ├── predict.py                        # [IMPLEMENTED] Hate speech inference & checkpoint loader (High Confidence)
│   │   ├── train.py                          # [IMPLEMENTED] MVP training script (High Confidence)
│   │   └── training_config.json              # [IMPLEMENTED] Saved hyperparameter config (High Confidence)
│   └── propaganda/
│       ├── checkpoint/model.safetensors      # [IMPLEMENTED] Fine-tuned RoBERTa weights (498.6 MB) (High Confidence)
│       ├── checkpoint/config.json            # [IMPLEMENTED] HuggingFace model config (High Confidence)
│       ├── tokenizer/tokenizer.json          # [IMPLEMENTED] Byte-level BPE tokenizer assets (High Confidence)
│       ├── roberta.py                        # [IMPLEMENTED] RoBERTa-base 14-class architecture (High Confidence)
│       ├── label_mapping.json                # [IMPLEMENTED] 14 official SemEval propaganda technique names (High Confidence)
│       ├── metrics.json                      # [IMPLEMENTED] Verified validation metrics (High Confidence)
│       ├── predict.py                        # [IMPLEMENTED] Propaganda inference & checkpoint loader (High Confidence)
│       ├── train.py                          # [IMPLEMENTED] MVP training script (High Confidence)
│       └── training_config.json              # [IMPLEMENTED] Saved hyperparameter config (High Confidence)
│
├── pipeline/                                 # Master Orchestration Pipelines
│   ├── __init__.py                           # Package exports
│   ├── router.py                             # [IMPLEMENTED] Ingestion & media format inspection (High Confidence)
│   ├── image_pipeline.py                     # [IMPLEMENTED] MTCNN -> EfficientNet -> Grad-CAM -> OCR -> NLP (High Confidence)
│   ├── video_pipeline.py                     # [IMPLEMENTED] 1 FPS Sampling -> MTCNN -> Rep Frame Grad-CAM -> OCR -> Whisper (High Confidence)
│   ├── text_pipeline.py                      # [IMPLEMENTED] RoBERTa + BERT + Token Attribution + UQ (High Confidence)
│   └── multimodal_pipeline.py                # [IMPLEMENTED] Master singleton pipeline orchestrator (High Confidence)
│
├── preprocessing/                            # Preprocessing Utilities
│   ├── face_detection.py                     # [IMPLEMENTED] MTCNN face detection & 224x224 cropping (High Confidence)
│   ├── frame_sampling.py                     # [IMPLEMENTED] OpenCV video frame sampling at target FPS (High Confidence)
│   ├── image_preprocessing.py                # [PLACEHOLDER] Docstring stub only (High Confidence)
│   ├── text_cleaning.py                      # [PLACEHOLDER] Docstring stub only (High Confidence)
│   └── video_preprocessing.py                # [PLACEHOLDER] Docstring stub only (High Confidence)
│
├── reports/                                  # Verified Validation & Technical Reports
│   ├── explanations/deepfake/                # Generated Grad-CAM overlay PNG images (High Confidence)
│   ├── explainability_uncertainty.md         # [IMPLEMENTED] XAI & UQ technical documentation (High Confidence)
│   ├── fastapi_backend.md                    # [IMPLEMENTED] API service layer documentation (High Confidence)
│   ├── frontend.md                           # [IMPLEMENTED] React dashboard technical report (High Confidence)
│   ├── multimodal_pipeline.md                # [IMPLEMENTED] Pipeline orchestration report (High Confidence)
│   ├── mvp_model_validation.md               # [IMPLEMENTED] Training sanity check report (High Confidence)
│   └── system_validation.md                  # [IMPLEMENTED] Full-stack regression report (High Confidence)
│
├── scripts/                                  # Standalone Verification & Execution Scripts
│   ├── check_leakage.py                      # [IMPLEMENTED] Split overlap verification (High Confidence)
│   ├── prepare_mvp_subsets.py                # [IMPLEMENTED] Deterministic subset sampler (seed=42) (High Confidence)
│   ├── test_gradcam_demo.py                  # [IMPLEMENTED] Standalone Grad-CAM demo script (High Confidence)
│   ├── test_model_reloads.py                 # [IMPLEMENTED] Standalone checkpoint reload test (High Confidence)
│   ├── test_token_attribution_demo.py        # [IMPLEMENTED] Standalone token attribution demo (High Confidence)
│   ├── verify_api_live.py                    # [IMPLEMENTED] Live HTTP API integration test script (High Confidence)
│   ├── verify_real_execution.py              # [IMPLEMENTED] Real execution across all models & XAI (High Confidence)
│   └── verify_whisper_base.py                # [IMPLEMENTED] Audio probing & Whisper-base verification (High Confidence)
│
├── tests/                                    # Automated Test Suites & Test Media Assets
│   ├── assets/                               # Real media assets used in automated tests & demos
│   │   ├── audio_video.mp4                   # [IMPLEMENTED] 1-second video with spoken audio track (High Confidence)
│   │   ├── silent_video.mp4                  # [IMPLEMENTED] 1-second video with no audio track (High Confidence)
│   │   ├── face_image.jpg                    # [IMPLEMENTED] Image with clear human face (High Confidence)
│   │   ├── no_face_image.jpg                 # [IMPLEMENTED] Faceless landscape image (High Confidence)
│   │   └── ocr_image.jpg                     # [IMPLEMENTED] Image with legible text ("BREAKING NEWS") (High Confidence)
│   ├── test_api.py                           # [IMPLEMENTED] 19 FastAPI endpoint tests (High Confidence)
│   ├── test_explainability_uncertainty.py    # [IMPLEMENTED] 11 XAI & MC Dropout tests (High Confidence)
│   └── test_multimodal_pipeline.py           # [IMPLEMENTED] 11 end-to-end multimodal pipeline tests (High Confidence)
│
├── uncertainty/                              # Uncertainty Quantification (UQ)
│   ├── mc_dropout.py                         # [IMPLEMENTED] MCDropoutEstimator (T=20 stochastic passes) (High Confidence)
│   ├── uncertainty_engine.py                 # [IMPLEMENTED] Human-review thresholding & policy evaluator (High Confidence)
│   └── variance.py                           # [IMPLEMENTED] Predictive mean, variance, std dev, Shannon entropy (High Confidence)
│
├── utils/                                    # [PLACEHOLDER] Legacy Utility Stubs
│   ├── common.py                             # [PLACEHOLDER] Docstring stub only (High Confidence)
│   ├── file_utils.py                         # [PLACEHOLDER] Docstring stub only (Active utils are in api/utils) (High Confidence)
│   └── logger.py                             # [PLACEHOLDER] Docstring stub only (High Confidence)
│
├── app.py                                    # [PLACEHOLDER] Legacy 7-line Streamlit scaffold entry point (High Confidence)
├── project_structure_mapping.md              # Early proposal architecture document (High Confidence)
├── README.md                                 # Main project documentation (High Confidence)
└── requirements.txt                          # Python dependencies (High Confidence)
```

---

## 3. Actual System Architecture

The following Mermaid diagram maps the **actual runtime execution flow** reverse-engineered directly from active imports and module calls.

```mermaid
flowchart TD
    subgraph UI["React 19 + Vite 8 Analyst Frontend (Port 5173)"]
        UserAction[User Upload: Image / Video / Text / Multimodal]
        Selector[InputSelector: 4 Modality Tabs]
        Dashboard[ResultDashboard: Decision Support Workstation]
        GovPanel[GovernancePanel: REVIEW_RECOMMENDED / ANALYSIS_INCOMPLETE]
        XAI_View[VisualResultView: Grad-CAM / NLPResultView: Token Pills]
        UQ_Card[UncertaintyCard: Variance, Std, Entropy, Probabilities]
    end

    subgraph API_LAYER["FastAPI REST Backend (Port 8000)"]
        AppFastAPI[FastAPI Application: api/main.py]
        Lifespan[Lifespan Startup: Pre-warms Pipeline Singleton]
        RouteHealth["GET /health & GET /ready"]
        RouteVerify["POST /verify/{text, image, video, multimodal}"]
        RouteStatus["GET /verify/status/{job_id}"]
        StreamUpload[Chunked Upload Streaming: 64KB chunks to Temp Disk]
        Sanitizer[api.utils.serialization.sanitize_for_json]
        JobStore[(In-Memory Job Store: BackgroundTasks)]
    end

    subgraph PIPELINE["Master Orchestrator: pipeline/multimodal_pipeline.py"]
        Router[pipeline.router.MediaRouter: Inspects MIME/Headers]
        ImgPipe[pipeline.image_pipeline.ImagePipeline]
        VidPipe[pipeline.video_pipeline.VideoPipeline]
        TxtPipe[pipeline.text_pipeline.TextPipeline]
    end

    subgraph EXTRACTION["Pre-processing & Extraction Engines"]
        MTCNN[preprocessing.face_detection.FaceDetector: MTCNN]
        Sampler[preprocessing.frame_sampling.sample_video_frames: OpenCV 1 FPS]
        EasyOCR[extraction.ocr.EasyOCRReader: CRAFT + ResNet-BiLSTM]
        AudioProbe{extraction.audio_extraction.has_audio_stream}
        FFmpegExtract[extraction.audio_extraction.extract_audio_to_wav: 16kHz PCM]
        WhisperASR[extraction.asr.WhisperASR: openai/whisper-base]
        TextAgg[extraction.text_aggregator.TextAggregator: Provenance Tracking]
    end

    subgraph MODELS["Fine-Tuned Deep Learning Detectors"]
        DeepfakeModel[models.deepfake: EfficientNet-B4 Binary Classifier]
        PropagandaModel[models.propaganda: RoBERTa-base 14-Class Classifier]
        HateSpeechModel[models.hate_speech: BERT-base 3-Class Classifier]
    end

    subgraph XAI_UQ["Explainability & Epistemic Uncertainty"]
        GradCAM[explainability.gradcam.GradCAM: Conv features[-1] 7x7 -> 224x224]
        TokenAttr[explainability.text_attention.TokenAttribution: Grad x Input]
        MCDropout[uncertainty.mc_dropout.MCDropoutEstimator: T=20 Passes]
        UQStats[uncertainty.variance: Mean, Variance, Std, Shannon Entropy]
        UQEngine[uncertainty.uncertainty_engine.UncertaintyEngine: Policy Evaluator]
    end

    %% Flow Connections
    UserAction --> Selector --> AppFastAPI
    AppFastAPI --> Lifespan
    Lifespan --> Router
    AppFastAPI --> RouteHealth
    AppFastAPI --> RouteVerify
    RouteVerify --> StreamUpload --> Router

    RouteVerify -.->|async_mode=true| JobStore
    RouteStatus -.->|Poll status| JobStore

    Router -->|Image| ImgPipe
    Router -->|Video| VidPipe
    Router -->|Text| TxtPipe

    %% Image flow
    ImgPipe --> MTCNN
    MTCNN -->|Face detected: crop 224x224| DeepfakeModel
    MTCNN -->|No face: NO_FACE_DETECTED| ImgPipe
    ImgPipe --> EasyOCR
    EasyOCR --> TextAgg

    %% Video flow
    VidPipe --> Sampler
    Sampler --> MTCNN
    MTCNN --> DeepfakeModel
    VidPipe --> EasyOCR
    VidPipe --> AudioProbe
    AudioProbe -->|Has audio| FFmpegExtract --> WhisperASR --> TextAgg
    AudioProbe -->|No audio: NO_AUDIO| TextAgg

    %% Text & NLP flow
    TextAgg --> TxtPipe
    TxtPipe --> PropagandaModel
    TxtPipe --> HateSpeechModel

    %% XAI & UQ
    DeepfakeModel --> GradCAM
    DeepfakeModel --> MCDropout
    PropagandaModel --> TokenAttr
    PropagandaModel --> MCDropout
    HateSpeechModel --> TokenAttr
    HateSpeechModel --> MCDropout

    MCDropout --> UQStats --> UQEngine

    %% Aggregation & Return
    DeepfakeModel & PropagandaModel & HateSpeechModel & GradCAM & TokenAttr & UQEngine --> Sanitizer
    Sanitizer --> RouteVerify --> Dashboard

    %% UI Rendering
    Dashboard --> GovPanel
    Dashboard --> XAI_View
    Dashboard --> UQ_Card
```

### Component Status Classification
- **Implemented (100% Operational)**:
  - `api/main.py`, `api/routes/verify.py`, `api/routes/health.py`, `api/utils/`
  - `pipeline/multimodal_pipeline.py`, `image_pipeline.py`, `video_pipeline.py`, `text_pipeline.py`, `router.py`
  - `detection/deepfake_detector.py`, `propaganda_detector.py`, `hate_speech_detector.py`
  - `models/deepfake/`, `models/propaganda/`, `models/hate_speech/`
  - `extraction/ocr.py`, `extraction/asr.py`, `extraction/audio_extraction.py`, `extraction/text_aggregator.py`
  - `preprocessing/face_detection.py`, `preprocessing/frame_sampling.py`
  - `explainability/gradcam.py`, `text_attention.py`, `explanation_generator.py`
  - `uncertainty/mc_dropout.py`, `variance.py`, `uncertainty_engine.py`
  - `frontend/src/` (Entire React 19 application)
- **Partially Implemented**:
  - HateXplain Rationale Alignment (`explainability/text_attention.py`): Fully implemented and tested on benchmark samples, but operates in live user inference only when ground-truth rationales are provided (which in-the-wild uploads do not have).
  - Background Job Queue (`api/routes/verify.py`): Background video tasks run asynchronously via FastAPI `BackgroundTasks`, but state is held in an in-memory dictionary (`job_store`) rather than persistent Redis/Celery.
- **Placeholder / Mock**:
  - `app.py` & `dashboard/*.py` (Legacy Streamlit files).
  - `governance/*.py` (Legacy stubs; real governance is inside `uncertainty/uncertainty_engine.py`).
  - `preprocessing/{image_preprocessing, text_cleaning, video_preprocessing}.py` (Legacy stubs).
  - `evaluation/{calibration, evaluate_deepfake, evaluate_hate_speech, evaluate_propaganda}.py` (Legacy stubs; evaluation runs inside `train.py`).
  - `utils/{common, file_utils, logger}.py` (Legacy stubs).
- **Unverified / Broken**:
  - None of the active pipeline files are broken. All 51 tests pass.

---

## 4. Image Pipeline

The image verification workflow executes deterministically as follows:

```
Upload (UI) 
  ──► POST /verify/image (multipart/form-data: file, optional caption)
  ──► save_upload_to_temp() (Validates ext: jpg, png, webp, bmp, tiff; enforces <= 15 MB)
  ──► MediaRouter.inspect_input() (Confirms input_type: "image")
  ──► ImagePipeline.process_image()
        ├── 1. Face Detection: MTCNN (facenet-pytorch)
        │       ├── Face Found (prob > 0.8): Aligns and crops to 224×224 RGB
        │       │     ├── EfficientNet-B4 forward pass
        │       │     ├── GradCAM.explain() (features[-1] Conv2d -> 7×7 map -> 224×224 JET overlay)
        │       │     └── MCDropoutEstimator.estimate_deepfake_crop() (T=20 stochastic passes)
        │       └── No Face: Returns status="NO_FACE_DETECTED", confidence=0.0, deepfake bypassed
        ├── 2. OCR Extraction: EasyOCRReader.extract_text() (Runs independently on full image)
        ├── 3. Text Aggregation: TextAggregator.aggregate(user_text=caption, ocr_results=[ocr])
        ├── 4. Text Verification: If text available:
        │       ├── PropagandaDetector.detect_text() (RoBERTa 14-class + Token Attribution + UQ T=20)
        │       └── HateSpeechDetector.detect_text() (BERT 3-class + Token Attribution + UQ T=20)
        │     If no text available:
        │       └── status="NO_TEXT_AVAILABLE" (NLP gracefully bypassed)
        ├── 5. Governance Evaluation:
        │       ├── If any variance >= 0.02 or entropy >= 0.85: decision_status="REVIEW_RECOMMENDED"
        │       ├── If visual status="NO_FACE_DETECTED" and no NLP triggers: decision_status="ANALYSIS_INCOMPLETE"
        │       └── If all models below thresholds: decision_status="CONFIDENT_PREDICTION"
        └── 6. Result Assembly: build_multimodal_verification_result()
  ──► cleanup_temp_file() (Guaranteed temp unlinking in finally block)
  ──► sanitize_for_json() (Converts tensors/arrays to standard JSON primitives)
  ──► Frontend ResultDashboard Render (VisualResultView, TextExtractionView, NLPResultView, GovernancePanel)
```

### Exact Code & Files Involved
- Route: [api/routes/verify.py:L88-114](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/api/routes/verify.py#L88-L114)
- Upload Validation: [api/utils/file_utils.py:L38-90](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/api/utils/file_utils.py#L38-L90)
- Ingestion Router: [pipeline/router.py:L17-94](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/pipeline/router.py#L17-L94)
- Sub-Pipeline: [pipeline/image_pipeline.py:L30-149](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/pipeline/image_pipeline.py#L30-L149)
- Face Detector: [preprocessing/face_detection.py:L9-44](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/preprocessing/face_detection.py#L9-L44)
- Deepfake Detector: [detection/deepfake_detector.py:L33-108](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/detection/deepfake_detector.py#L33-L108)
- OCR Engine: [extraction/ocr.py:L29-96](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/extraction/ocr.py#L29-L96)
- Text Aggregator: [extraction/text_aggregator.py:L10-77](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/extraction/text_aggregator.py#L10-L77)
- Frontend View: [frontend/src/components/VisualResultView.jsx](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/frontend/src/components/VisualResultView.jsx)

---

## 5. Video Pipeline

The video verification pipeline is engineered for temporal memory efficiency, preventing Out-Of-Memory (OOM) failures:

```
Upload (UI) 
  ──► POST /verify/video (file, caption, async_mode, sample_fps, max_frames)
  ──► save_upload_to_temp() (Validates ext: mp4, avi, mov, mkv, webm; enforces <= 100 MB)
  ──► If async_mode == True:
  │     ├── Generates UUID job_id; records status="QUEUED" in job_store
  │     ├── Dispatches _run_video_background() via FastAPI BackgroundTasks
  │     └── Returns 200 OK immediately with AsyncJobResponse (job_id, poll_url)
  │         └── UI initiates interval polling on GET /verify/status/{job_id} (every 2000ms)
  └── If async_mode == False: Executes synchronously on current thread
        ──► VideoPipeline.process_video()
              ├── 1. Frame Sampling: sample_video_frames() (OpenCV: native_fps / sample_fps; max 16 frames)
              ├── 2. Per-Frame Face Detection & Scoring:
              │       ├── MTCNN detects and crops primary face per frame
              │       └── Single forward pass computes P(fake) for each face
              ├── 3. Video-Level Probability Aggregation:
              │       ├── Mean Fake Probability: P_bar = (1/N) * sum(P_i(fake))
              │       ├── Video Prediction: "fake" if P_bar >= 0.5 else "real"
              │       └── Representative Frame Selection:
              │             ├── If "fake": selects frame with maximum P(fake)
              │             └── If "real": selects frame with minimum P(fake)
              │       └── Grad-CAM and MC Dropout (T=20) run ONLY on representative frame
              ├── 4. Keyframe OCR: EasyOCR runs on up to 4 evenly spaced sampled frames
              ├── 5. Audio Probing & Whisper ASR:
              │       ├── has_audio_stream() probes container using imageio-ffmpeg
              │       ├── If audio stream found:
              │       │     ├── extract_audio_to_wav() dumps 16 kHz mono 16-bit PCM WAV
              │       │     ├── WhisperASR.transcribe() processes audio with openai/whisper-base
              │       │     └── Temp WAV file unlinked in finally block
              │       └── If no audio stream found:
              │             └── Whisper is BYPASSED cleanly; status="NO_AUDIO"
              ├── 6. Text Aggregation: Aggregates OCR keyframes + Whisper ASR + caption
              ├── 7. Text NLP Models: RoBERTa (14-class) + BERT (3-class) on aggregated text
              ├── 8. Governance Evaluation: Evaluates max variance & entropy thresholds across detectors
              └── 9. Final Payload Assembly & Response
```

### Exact Code & Files Involved
- Route & Async Worker: [api/routes/verify.py:L35-67, L121-177](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/api/routes/verify.py#L35-L67)
- Sub-Pipeline: [pipeline/video_pipeline.py:L37-277](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/pipeline/video_pipeline.py#L37-L277)
- Frame Sampler: [preprocessing/frame_sampling.py:L8-34](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/preprocessing/frame_sampling.py#L8-L34)
- Audio Probe & Extractor: [extraction/audio_extraction.py:L15-82](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/extraction/audio_extraction.py#L15-L82)
- Speech Recognizer: [extraction/asr.py:L23-95](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/extraction/asr.py#L23-L95)
- Video Polling Hook: [frontend/src/hooks/useVerification.js:L59-94](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/frontend/src/hooks/useVerification.js#L59-L94)

---

## 6. Text Pipeline

The direct text verification pipeline evaluates raw natural language claims:

```
Direct Text / Claim Input 
  ──► POST /verify/text (JSON: {"text": "..."})
  ──► Pydantic Validation: TextVerificationRequest (min_length=1, max_length=10000, whitespace stripped)
  ──► TextPipeline.process_text()
        ├── 1. Visual Status: Set to "NOT_APPLICABLE" (bypassed)
        ├── 2. Text Aggregator: TextAggregator.aggregate(user_text=text) -> text_status="AVAILABLE"
        ├── 3. Propaganda Classification & XAI:
        │       ├── RoBERTa-base forward pass over 14 SemEval classes
        │       ├── TokenAttribution.attribute() (Grad x Input on word embeddings)
        │       └── MCDropoutEstimator (T=20 stochastic forward passes)
        ├── 4. Hate Speech Classification & XAI:
        │       ├── BERT-base-uncased forward pass over 3 toxicity classes
        │       ├── TokenAttribution.attribute() (Grad x Input on word embeddings)
        │       └── MCDropoutEstimator (T=20 stochastic forward passes)
        ├── 5. Governance Evaluation:
        │       ├── Compares max_variance against 0.02 and entropy against 0.85
        │       └── Assigns "REVIEW_RECOMMENDED" or "CONFIDENT_PREDICTION"
        └── 6. Schema Assembly: build_multimodal_verification_result()
  ──► sanitize_for_json() -> 200 OK Response
  ──► Frontend Render (NLPResultView, TokenAttributionView with interactive saliency chips, GovernancePanel)
```

### Exact Code & Files Involved
- Route: [api/routes/verify.py:L69-86](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/api/routes/verify.py#L69-L86)
- Request Model: [api/schemas/requests.py:L6-18](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/api/schemas/requests.py#L6-L18)
- Sub-Pipeline: [pipeline/text_pipeline.py:L22-137](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/pipeline/text_pipeline.py#L22-L137)
- Token Attribution: [explainability/text_attention.py:L8-115](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/explainability/text_attention.py#L8-L115)
- Frontend View: [frontend/src/components/NLPResultView.jsx](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/frontend/src/components/NLPResultView.jsx) and [TokenAttributionView.jsx](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/frontend/src/components/TokenAttributionView.jsx)

---

## 7. Model Audit

Every fine-tuned model checkpoint referenced in the code physically exists on disk and was independently reloaded and tested.

| Property | Deepfake Detector | Propaganda Detector | Hate Speech Detector | Speech Recognizer (ASR) |
| :--- | :--- | :--- | :--- | :--- |
| **Model Component** | `detection/deepfake_detector.py` | `detection/propaganda_detector.py` | `detection/hate_speech_detector.py` | `extraction/asr.py` |
| **Base Architecture** | `EfficientNet-B4` (`torchvision`) | `RoBERTa-base` (`transformers`) | `BERT-base-uncased` (`transformers`) | `openai/whisper-base` (74M params) |
| **Physical Checkpoint Path** | `models/deepfake/checkpoint/best_model.pt` | `models/propaganda/checkpoint/model.safetensors` | `models/hate_speech/checkpoint/model.safetensors` | Hugging Face cache (`openai/whisper-base`) |
| **Checkpoint Status on Disk** | **EXISTS (70,969,363 bytes)** | **EXISTS (498,649,736 bytes)** | **EXISTS (437,961,700 bytes)** | **EXISTS / PRELOADED** |
| **Tokenizer Path** | N/A (Image transforms) | `models/propaganda/tokenizer/` (3.56 MB) | `models/hate_speech/tokenizer/` (711 KB) | Built-in Whisper tokenizer |
| **Classification Head** | `nn.Dropout(p=0.4) -> Linear(1792, 2)` | `Linear(768, 14)` | `Linear(768, 3)` | Encoder-Decoder Sequence Generation |
| **Target Task / Taxonomy** | Binary classification (`real`, `fake`) | 14 fine-grained propaganda techniques | 3 toxicity classes (`normal`, `offensive`, `hatespeech`) | Zero-shot speech-to-text transcription |
| **Input Specifications** | $224 \times 224 \times 3$ RGB tensor, ImageNet normalized | String tokenized with max length 128 | String tokenized with max length 128 | 16 kHz mono 16-bit PCM waveform |
| **Decision Threshold** | $P(\text{fake}) \ge 0.5 \implies \text{fake}$ | $\text{argmax}_{c} P(c)$ | $\text{argmax}_{c} P(c)$ | Non-empty decoded transcript |
| **Inference Mode** | `eval()` with selective dropout training | `eval()` with selective dropout training | `eval()` with selective dropout training | HuggingFace pipeline evaluation |
| **Hardware Device** | CPU (auto-switchable to CUDA) | CPU (auto-switchable to CUDA) | CPU (auto-switchable to CUDA) | CPU (auto-switchable to CUDA) |

---

## 8. Extraction Components

### 1. MTCNN Face Detection (`preprocessing/face_detection.py`)
- **Library**: `facenet-pytorch.MTCNN`
- **Configuration**: `keep_all=False`, `select_largest=True`, `device="cpu"`
- **Margin**: 0.2 expansion factor around bounding box
- **Confidence Threshold**: $P(\text{face}) > 0.8$
- **Output**: Aligned, cropped PIL Image resized to $224 \times 224$ via bilinear interpolation.
- **Fallback**: Center crop if no face exceeds 0.8 confidence, returning `face_detected = False`.

### 2. Video Frame Sampling (`preprocessing/frame_sampling.py`)
- **Library**: OpenCV (`cv2.VideoCapture`)
- **Sampling Logic**: Computes `frame_interval = round(native_fps / target_fps)`
- **Defaults**: `target_fps = 1.0`, capped at `max_frames = 16`
- **Output**: List of `(timestamp_sec, frame_rgb_numpy)` tuples.
- **Memory Safety**: Video frames are iterated on the fly; the complete video is never decoded into RAM simultaneously.

### 3. Audio Extraction & Probing (`extraction/audio_extraction.py`)
- **Tool**: Bundled `imageio-ffmpeg` binary executable (bypasses reliance on system PATH).
- **Stream Probing**: Runs `ffmpeg -i video_path` and inspects `stderr` for `"Audio:"`. Silent videos bypass ASR completely.
- **WAV Extraction**: Runs `ffmpeg -y -i input -vn -acodec pcm_s16le -ar 16000 -ac 1 temp.wav`.
- **Deterministic Cleanup**: Temporary WAV files are unlinked in `finally` blocks.

### 4. EasyOCR Text Detection (`extraction/ocr.py`)
- **Library**: `easyocr.Reader(["en"], gpu=False)` (CRAFT detector + ResNet-BiLSTM recognizer).
- **Execution**: Runs independently on image uploads and on up to 4 keyframes in video uploads.
- **Deduplication**: Case-insensitive and whitespace-normalized deduplication across video frames.
- **Output**: Combined text string, mean confidence, and bounding box coordinates.

### 5. Multi-Source Text Aggregator (`extraction/text_aggregator.py`)
- **Function**: Merges text from `USER_TEXT`, `OCR`, and `ASR`.
- **Provenance Preservation**: Every segment explicitly preserves its origin tag.
- **Missing Text Flag**: Returns `text_status = "NO_TEXT_AVAILABLE"` when all sources are empty, prompting downstream NLP detectors to bypass cleanly without errors.

---

## 9. Explainability Audit

### 1. Spatial Grad-CAM (`explainability/gradcam.py`)
- **Target Model**: `EfficientNet-B4` binary deepfake detector.
- **Target Layer**: Final stage convolutional block `model.backbone.features[-1]` (`Conv2dNormActivation`), outputting activation maps of shape $(1, 1792, 7, 7)$ for $224 \times 224$ face crops.
- **Methodology**:
  - Global average pooling of backward gradients: $\alpha_k^c = \frac{1}{Z}\sum_{i}\sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$.
  - Rectified linear combination: $\text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$.
  - Bilinear upsampling to $224 \times 224$, min-max normalized to $[0, 1]$.
  - OpenCV `COLORMAP_JET` colorization blended with the face crop at $\alpha = 0.5$.
- **Artifact Generation**: Saves overlay PNGs when `output_dir` is passed (e.g. `reports/explanations/deepfake/`).
- **Frontend Display**: In `VisualResultView.jsx`, supports interactive view toggling (`Overlay` vs. `Side-by-Side` comparison with original input). If file path is not exposed via HTTP, the UI gracefully renders a heatmap fallback box.
- **Authenticity**: **Genuine model-derived output** computed from live PyTorch backward gradient tensors.

### 2. Textual Token Attribution (`explainability/text_attention.py`)
- **Target Models**: `RoBERTa-base` (Propaganda) and `BERT-base-uncased` (Hate Speech).
- **Methodology (Gradient $\times$ Input)**:
  - Hooks input embedding vectors $e_i \in \mathbb{R}^D$.
  - Computes gradient of predicted class logit with respect to embeddings: $G_i = \nabla_{e_i} y^c$.
  - Computes saliency: $S_i = \| G_i \odot e_i \|_2$, normalized across tokens to $[0, 1]$.
  - Subword markers (`##` for WordPiece, `Ġ` for BPE) are cleaned for display.
- **Frontend Display**: In `TokenAttributionView.jsx`, tokens are rendered as interactive inline pills with red background tints proportional to attribution score.
- **Authenticity**: **Genuine model-derived output** computed via autograd backward passes.

### 3. HateXplain Human Rationale Alignment (`evaluate_rationale_alignment`)
- **Methodology**: Evaluates model-selected top tokens against human-annotated binary rationale masks from HateXplain:
  - Computes token overlap count, Rationale Precision, Rationale Recall, and Rationale F1.
  - Verified on HateXplain test samples ($F1_{\text{rationale}} = 0.6667$).
- **Runtime Reality**: In live user verification, rationale alignment is `None` because user-uploaded media lacks human annotations. It activates only during benchmark evaluation scripts.

### 4. Mandatory Disclaimer
Every visual and textual explanation carries the non-dismissable disclaimer:
> *"Grad-CAM and token attribution are post-hoc explanatory signals and should not be interpreted as causal proof."*

---

## 10. Uncertainty Audit

### Monte Carlo (MC) Dropout Implementation (`uncertainty/mc_dropout.py`)
- **Stochastic Passes**: Exactly $T = 20$ stochastic forward passes per sample.
- **Layer Handling**:
  - `enable_mc_dropout(model)` iterates through all submodules.
  - Sets `isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d))` explicitly to `m.train()`.
  - Normalization layers (`BatchNorm2d`, `LayerNorm`) remain in `eval()` mode with frozen running statistics.
- **Mathematical Formulations (`uncertainty/variance.py`)**:
  - Predictive Mean: $\bar{p}_c = \frac{1}{T} \sum_{t=1}^T p_c^{(t)}$
  - Predictive Variance: $\sigma_c^2 = \frac{1}{T} \sum_{t=1}^T (p_c^{(t)} - \bar{p}_c)^2$
  - Predictive Standard Deviation: $\sigma_c = \sqrt{\sigma_c^2}$
  - Shannon Predictive Entropy: $H(\bar{p}) = -\sum_{c=1}^C \bar{p}_c \log_2(\bar{p}_c + \epsilon)$
- **Active Models**:
  - Deepfake `EfficientNet-B4`: Verified stochastic variance ($\sigma^2 > 0$).
  - Propaganda `RoBERTa-base`: Verified stochastic variance ($\sigma^2 > 0$).
  - Hate Speech `BERT-base`: Verified stochastic variance ($\sigma^2 > 0$).
- **Thresholding & Decision Policy (`uncertainty/uncertainty_engine.py`)**:
  - `variance_threshold = 0.02`
  - `entropy_threshold = 0.85 bits`
  - If $\max(\sigma^2) \ge 0.02$ OR $H \ge 0.85$: flags `review_required = True` and `decision_status = "REVIEW_RECOMMENDED"`.
  - Component Isolation: Uncertainty is evaluated independently per model and is **never cross-averaged across modalities**.
- **Calibration Status**: Explicitly documented in `config/config.yaml` and code:
  > *"Development/MVP threshold — not clinically, legally, or production calibrated."*

---

## 11. Responsible AI / Governance Audit

| Dimension | Implemented Mechanism | Code Location | Status |
| :--- | :--- | :--- | :--- |
| **Accountability** | Full audit trail: timestamps, latency, executed models, checkpoint versions, error logs. | `pipeline/multimodal_pipeline.py:L154-169` | **Implemented** |
| **Responsibility** | Strict decision-support mandate: no automated content deletion, no account banning, no censorship actions. | `uncertainty/uncertainty_engine.py:L55-58` | **Implemented** |
| **Transparency** | Mathematical justifications accompany every flagged prediction (e.g. variance and entropy breach values). | `uncertainty/uncertainty_engine.py:L47-51` | **Implemented** |
| **Human-in-the-Loop** | Elevated epistemic uncertainty triggers `REVIEW_RECOMMENDED`, mandating human inspection. | `pipeline/image_pipeline.py:L111`, `video_pipeline.py:L236` | **Implemented** |
| **Truth Claim Refusal** | Predictions expressed strictly as statistical likelihoods; refuses to assert "definitely real" or "definitely fake". | `docs/viva_questions.md:L84-89` | **Implemented** |
| **Faceless Media Policy** | Images/videos with no face are marked `ANALYSIS_INCOMPLETE` instead of defaulting to confident prediction. | `reports/system_validation.md:L52-70`, `pipeline/*.py` | **Implemented** |
| **Provenance Isolation** | Extracted text preserves strict origin tags (`USER_TEXT`, `OCR`, `ASR`), preventing attribution conflation. | `extraction/text_aggregator.py:L23-65` | **Implemented** |
| **Disclaimer Integration** | Non-dismissable post-hoc XAI and uncalibrated UQ disclaimers displayed across API and UI. | `frontend/src/components/GovernancePanel.jsx` | **Implemented** |

> [!NOTE]
> **Code vs. Documentation Distinction**: The `governance/` directory on disk contains 4 placeholder stubs (`audit_logger.py`, `decision_policy.py`, `human_review.py`, `safety_checks.py`). However, the actual governance logic is **genuinely implemented** inside `uncertainty/uncertainty_engine.py`, `pipeline/`, and `frontend/src/components/GovernancePanel.jsx`.

---

## 12. API Audit

Implemented in `api/main.py` using **FastAPI 0.136.1** running on **Python 3.14** / Uvicorn.

### Endpoint Catalog

| Method | Endpoint | Request Format | Content-Type | Response Model | Sync / Async |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | None | `application/json` | `HealthResponse` (`status="healthy"`) | Synchronous (2ms) |
| `GET` | `/ready` | None | `application/json` | `ReadyResponse` (status of all 5 models) | Synchronous (4ms) |
| `POST` | `/verify/text` | JSON `{"text": "..."}` | `application/json` | Unified Verification JSON | Synchronous (2.3s) |
| `POST` | `/verify/image` | FormData (`file`, `caption`) | `multipart/form-data` | Unified Verification JSON | Synchronous (6.1s) |
| `POST` | `/verify/video` | FormData (`file`, `caption`, `async_mode`, `sample_fps`, `max_frames`) | `multipart/form-data` | Direct JSON or `AsyncJobResponse` | Sync or Async |
| `GET` | `/verify/status/{job_id}` | Path parameter `job_id` | `application/json` | `JobStatusResponse` | Synchronous |
| `POST` | `/verify/multimodal` | FormData (`file`, `text`) | `multipart/form-data` | Unified Verification JSON | Synchronous |

### Lifecycle, Streaming & Error Handling
- **Pre-warming Lifecycle**: `FastAPI.lifespan` initializes `MultimodalPipeline(device="cpu")` once on startup, caching model weights in RAM.
- **Upload Streaming**: Files are read in 64 KB chunks directly to disk (`tempfile.NamedTemporaryFile`) using `save_upload_to_temp()`.
- **Upload Size Limits**: Hard limit of **15 MB for images** and **100 MB for videos**. Violations trigger `HTTP 413 Payload Too Large`.
- **Deterministic Cleanup**: All endpoints execute `file_utils.cleanup_temp_file(temp_path)` in `finally` blocks, guaranteeing scratch disk deletion.
- **JSON Sanitization**: `sanitize_for_json()` recursively converts `np.float32`, `np.bool_`, `NaN`, `Inf`, and tensors to JSON-safe primitives.
- **Structured Error Handlers**: Intercepts `HTTPException`, `RequestValidationError` (422), and unhandled server errors (500) without leaking stack traces.
- **Interactive OpenAPI Documentation**: Available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.

---

## 13. Frontend Audit

Implemented in `frontend/` using **React 19.2.8** and bundled with **Vite 8.2.2**.

### Architecture & Routing
- **Single-Page Application (SPA)**: Operates without a heavy router library; state-driven modality switching via `InputSelector.jsx` (`IMAGE`, `VIDEO`, `TEXT`, `MULTIMODAL`).
- **Design System**: Vanilla CSS (`frontend/src/index.css`) utilizing dark slate tones, responsive CSS grid/flexbox, `Inter` typography, and `JetBrains Mono` code telemetry.
- **State Hooks**:
  - `useBackendHealth.js`: Non-blocking periodic polling checking `GET /health` with visual indicator badge.
  - `useVerification.js`: Orchestrates API calls, manages loading state, error alerts, and background job polling for async video.

### View Rendering Capabilities
- **Visual Saliency Viewer (`VisualResultView.jsx`)**: Renders EfficientNet-B4 prediction, probability distribution, and Grad-CAM overlays. Features interactive view toggling (`Overlay` vs. `Side-by-Side`).
- **Text Provenance Viewer (`TextExtractionView.jsx`)**: Separates text segments with color-coded origin tags (`OCR`, `ASR`, `USER_TEXT`).
- **NLP Analysis Viewer (`NLPResultView.jsx`)**: Renders 14-class propaganda and 3-class toxicity predictions with confidence progress bars.
- **Token Attribution Viewer (`TokenAttributionView.jsx`)**: Renders subword chips with proportional heat-map highlighting and HateXplain alignment metrics.
- **Uncertainty Card (`UncertaintyCard.jsx`)**: Displays predictive variance, standard deviation, Shannon entropy, and MC sample badges.
- **Human Oversight Panel (`GovernancePanel.jsx`)**: Renders status badges (`CONFIDENT_PREDICTION`, `REVIEW_RECOMMENDED`, `ANALYSIS_INCOMPLETE`), identifying triggering components and ethics justifications.
- **Audit Telemetry (`AuditDetails.jsx`)**: Collapsible accordion displaying execution latency, checkpoint paths, and hardware metadata.

---

## 14. Configuration Audit

| Configuration Target | Configuration Source | Configured Value | Status / Conflict Note |
| :--- | :--- | :--- | :--- |
| **Random Seed** | `config/config.yaml:L15` | `42` | Strictly enforced across subset scripts and training. |
| **Deepfake Model** | `config/config.yaml:L5` | `efficientnet_b4` | Active in `models/deepfake/predict.py`. |
| **Propaganda Model** | `config/config.yaml:L6` | `roberta_base` | Active in `models/propaganda/predict.py`. |
| **Hate Speech Model** | `config/config.yaml:L7` | `bert_base_uncased` | Active in `models/hate_speech/predict.py`. |
| **Whisper Model** | `config/config.yaml:L8` | `"openai/whisper-base"` | Active in `extraction/asr.py`. |
| **MC Dropout Passes** | `config/config.yaml:L11, L32` | `20` | Active in `uncertainty/mc_dropout.py`. |
| **Variance Threshold** | `config/config.yaml:L33` | `0.02` | **Active threshold** in `UncertaintyEngine` (0.02). |
| **Entropy Threshold** | `config/config.yaml:L34` | `0.85` | **Active threshold** in `UncertaintyEngine` (0.85 bits). |
| **Legacy Thresholds** | `config/thresholds.yaml` | `0.05` | **INACTIVE SCENARIO**: `thresholds.yaml` is an unused scaffold file; code does not load it. |
| **Video Sampling FPS** | `config/config.yaml:L13` | `1.0 FPS` | Active default in `VideoPipeline`. |
| **Max Video Frames** | `api/routes/verify.py:L127` | `16 frames` | Default parameter in API and pipeline. |
| **Image Upload Limit** | `api/utils/file_utils.py:L13` | `15 MB` (`15 * 1024 * 1024`) | Enforced during chunked upload. |
| **Video Upload Limit** | `api/utils/file_utils.py:L14` | `100 MB` (`100 * 1024 * 1024`)| Enforced during chunked upload. |
| **Backend Port** | `api/main.py` / CLI | `8000` | Standard Uvicorn port. |
| **Frontend Port** | `frontend/vite.config.js` | `5173` | Standard Vite dev server port. |
| **API Base URL** | `frontend/.env` | `http://localhost:8000` | Injected via `import.meta.env.VITE_API_BASE_URL`. |

---

## 15. Dataset & Checkpoint Audit

### 1. Physical Dataset Verification

#### A. FaceForensics++ C23
- **Local Path**: `data/faceforensics/C23/`
- **Integrity**: Exactly **3,000 video files** verified on disk across 5 folders:
  - `original_sequences/youtube/c23/videos/`: 1,000 MP4 files
  - `manipulated_sequences/Deepfakes/c23/videos/`: 1,000 MP4 files
  - `manipulated_sequences/Face2Face/c23/videos/`: 1,000 MP4 files
  - `manipulated_sequences/FaceSwap/c23/videos/`: 1,000 MP4 files
  - `manipulated_sequences/NeuralTextures/c23/videos/`: 1,000 MP4 files
- **Splits Defined**: `splits/train.json`, `val.json`, `test.json` present. Zero video ID overlap.
- **MVP Subsets Prepared**:
  - Train: 40 videos (20 real, 20 fake) $\rightarrow$ `data/mvp/deepfake/train_subset.json`
  - Validation: 18 videos (10 real, 8 fake) $\rightarrow$ `data/mvp/deepfake/val_subset.json`
  - Test: 18 videos (10 real, 8 fake) $\rightarrow$ `data/mvp/deepfake/test_subset.json`
- **Academic Provenance Note**: Sourced from Kaggle development mirror (`xdxd003/ff-c23`) for engineering development; official TUM access pending.

#### B. SemEval-2020 Task 11 (Propaganda Techniques)
- **Local Path**: `data/semeval2020_task11/`
- **Integrity**: 371 training articles, 75 development articles, 371 task-1 span label files, 371 task-2 technique label files.
- **MVP Subsets Prepared**:
  - Train: 560 spans (exactly 40 per technique across 14 classes, 285 articles)
  - Validation: 140 spans (exactly 10 per technique across 14 classes, 72 articles)
  - Article Isolation: Overlap between train and validation articles is exactly **0**.

#### C. HateXplain Benchmark
- **Local Path**: `data/hatexplain/`
- **Integrity**: 5 files including `dataset.json` (20,148 posts) and official `post_id_divisions.json`.
- **MVP Subsets Prepared**:
  - Train: 600 samples (200 normal, 200 offensive, 200 hatespeech)
  - Validation: 150 samples (50 normal, 50 offensive, 50 hatespeech)
  - Test: 150 samples (50 normal, 50 offensive, 50 hatespeech)
  - Ground-Truth Rationales: 100% retained for offensive and hatespeech samples.

### 2. Physical Checkpoint Verification

| Checkpoint Identifier | Physical File Path | File Size | Checkpoint Status | Associated Detector |
| :--- | :--- | :--- | :--- | :--- |
| **Deepfake Weights** | `models/deepfake/checkpoint/best_model.pt` | 70,969,363 bytes | **EXISTS** | `DeepfakeDetector` |
| **Propaganda Weights** | `models/propaganda/checkpoint/model.safetensors` | 498,649,736 bytes | **EXISTS** | `PropagandaDetector` |
| **Propaganda Config** | `models/propaganda/checkpoint/config.json` | 1,736 bytes | **EXISTS** | `PropagandaDetector` |
| **Propaganda Tokenizer** | `models/propaganda/tokenizer/tokenizer.json` | 3,558,895 bytes | **EXISTS** | `PropagandaDetector` |
| **Hate Speech Weights** | `models/hate_speech/checkpoint/model.safetensors` | 437,961,700 bytes | **EXISTS** | `HateSpeechDetector` |
| **Hate Speech Config** | `models/hate_speech/checkpoint/config.json` | 1,018 bytes | **EXISTS** | `HateSpeechDetector` |
| **Hate Speech Tokenizer**| `models/hate_speech/tokenizer/tokenizer.json` | 711,649 bytes | **EXISTS** | `HateSpeechDetector` |
| **Speech ASR Model** | HuggingFace cache (`openai/whisper-base`) | ~290,000,000 bytes | **EXISTS** | `WhisperASR` |

---

## 16. Evaluation Audit

### Implemented Evaluation Routines
- `evaluation/metrics.py`: Contains `compute_classification_metrics()`, computing:
  - Accuracy
  - Macro Precision, Recall, F1
  - Weighted Precision, Recall, F1
  - Multi-class Confusion Matrix
- Training scripts (`models/*/train.py`): Execute evaluation loops at each epoch on validation and test subsets, outputting verified metrics to `models/*/metrics.json`.
- Rationale Alignment (`explainability/text_attention.py`): Computes token overlap, rationale precision, recall, and F1 against HateXplain human annotations.

### Verified Empirical Model Performance

```
┌─────────────────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
│ Metric / Split                  │ Deepfake Detector    │ Propaganda Detector  │ Hate Speech Detector │
├─────────────────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Training Accuracy               │ 97.50% (Video)       │ 47.68% (14 classes)  │ 81.33% (3 classes)   │
│ Training Macro F1               │ 0.9750               │ 0.4619               │ 0.8067               │
│ Validation Accuracy             │ 55.56% (Video)       │ 29.29%               │ 60.67%               │
│ Validation Macro F1             │ 0.5500               │ 0.2481               │ 0.5942               │
│ Test Accuracy                   │ 50.00% (Video)       │ N/A (Dev only)       │ 60.00%               │
│ Test Macro F1                   │ 0.4857               │ N/A                  │ 0.5773               │
│ Baseline Guessing Accuracy      │ 50.00%               │ 7.14% (1/14)         │ 33.33% (1/3)         │
└─────────────────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘
```

### Documented But Not Verified / Stubs
- `evaluation/calibration.py`: Docstring stub only. Formal uncertainty calibration (e.g. temperature scaling, Platt scaling, or Brier score evaluation) is **not implemented**.
- `evaluation/evaluate_*.py`: Standalone evaluation CLI scripts are docstring stubs. Evaluation was executed within `train.py` and `scripts/verify_real_execution.py`.

---

## 17. Testing Audit

The test suite contains **4 automated test files** across backend (pytest) and frontend (vitest), totaling **51 passing tests**.

### 1. Backend Pytest Suite (41 Tests)
Command: `python -m pytest tests -v` (Execution duration: ~62s on CPU)

- **`tests/test_api.py` (19 tests)**:
  - Validates `GET /health` and `GET /ready`.
  - Validates `POST /verify/text` (valid text, empty text 422 rejection, missing fields).
  - Validates `POST /verify/image` (face image, faceless image `ANALYSIS_INCOMPLETE`, invalid extension 400, empty file 400).
  - Validates `POST /verify/video` (synchronous video with audio, silent video, asynchronous mode `QUEUED` and status polling, 404 on bad job ID, invalid extension).
  - Validates `POST /verify/multimodal` (image+text, text-only, empty input 400 rejection).
  - Validates structured error responses and upload limit rejection (413).
  - *Mocking*: Uses real FastAPI `TestClient` and real models; mocks only `MAX_IMAGE_SIZE_BYTES` temporarily to test 413 rejections.

- **`tests/test_multimodal_pipeline.py` (11 tests)**:
  - Tests 10 multimodal orchestration scenarios (text-only, image with face, image without face, video with audio, video without audio, OCR readable, OCR empty, ASR readable, no text available, high-uncertainty review path).
  - Tests error handling and graceful pipeline failure recovery.
  - *Mocking*: **Zero mocks**. Runs real PyTorch checkpoints, EasyOCR, and Whisper on CPU using real test assets in `tests/assets/`.

- **`tests/test_explainability_uncertainty.py` (11 tests)**:
  - Validates Grad-CAM heatmap generation, activation shapes, and PNG file saving.
  - Validates token attribution on RoBERTa and BERT embeddings.
  - Validates HateXplain human rationale precision, recall, and F1 calculations.
  - Validates `enable_mc_dropout()` layer switching.
  - Validates non-zero stochastic variance across $T=20$ forward passes for visual and text models.
  - Validates Shannon entropy math and human review threshold escalation logic.
  - *Mocking*: **Zero mocks**. Uses real model weights and test assets.

### 2. Frontend Vitest Suite (10 Tests)
Command: `cd frontend && npm test` (Execution duration: ~2.3s in jsdom)

- **`frontend/src/__tests__/App.test.jsx` (10 tests)**:
  - Tests page loads with academic header, branding, and navigation tabs.
  - Tests image upload UI, drag-and-drop, and preview rendering.
  - Tests text input UI, character counter, and benchmark sample chips.
  - Tests API success response rendering across all dashboard panels.
  - Tests API error alert banners and dismiss controls.
  - Tests governance panel rendering `REVIEW_RECOMMENDED`, `ANALYSIS_INCOMPLETE`, and ethics principles.
  - Tests Grad-CAM overlay rendering, disclaimers, and view toggle buttons.
  - Tests token attribution pills and HateXplain alignment metrics.
  - Tests uncertainty cards rendering variance, std, entropy, and MC sample badges.
  - Tests asynchronous video polling cycle (`QUEUED` $\rightarrow$ `PROCESSING` $\rightarrow$ `COMPLETED`).
  - *Mocking*: Mocks `fetch` network calls via `src/services/api.js` to isolate frontend component logic.

---

## 18. Current Demo Flow

The system can be demonstrated live directly from VS Code using real test assets stored in `tests/assets/`.

```
STEP 1: Start Backend (Terminal 1)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
--> Pre-warms MultimodalPipeline singleton; endpoints live at http://localhost:8000/docs

STEP 2: Start Frontend (Terminal 2)
cd frontend && npm run dev
--> Vite development server live at http://localhost:5173

STEP 3: Live Verification Demos
├── IMAGE DEMO (tests/assets/face_image.jpg)
│     1. User uploads face image.
│     2. Frontend POSTs to /verify/image.
│     3. Backend crops face via MTCNN, runs EfficientNet-B4, computes Grad-CAM and MC Dropout (T=20).
│     4. ResultDashboard renders classification (Authentic / Manipulated), Grad-CAM heatmap, and variance.
│
├── FACELESS MEDIA DEMO (tests/assets/no_face_image.jpg)
│     1. User uploads landscape image.
│     2. MTCNN finds no face (visual.status = "NO_FACE_DETECTED").
│     3. Governance panel renders info badge: "ANALYSIS_INCOMPLETE (Safe Bypass)".
│     4. Refuses to make an ungrounded guess, demonstrating Responsible AI governance.
│
├── TEXT DEMO (Benchmark Sample Chip)
│     1. User selects "Propaganda Rhetoric" demo sample chip.
│     2. Frontend POSTs to /verify/text.
│     3. RoBERTa predicts fine-grained technique; BERT evaluates toxicity; TokenAttribution highlights tokens.
│     4. ResultDashboard renders highlighted token stream with red saliency tints and MC Dropout entropy.
│
└── VIDEO DEMO (tests/assets/audio_video.mp4)
      1. User uploads video, sets 1.0 FPS, toggles Async Mode = True.
      2. Backend enqueues BackgroundTask, returning job_id.
      3. UI displays ProcessingIndicator; polls /verify/status/{job_id} every 2000ms.
      4. Backend extracts 16 kHz audio via FFmpeg, transcribes speech with Whisper-base, aggregates text.
      5. Polling completes; full multimodal analysis with provenance breakdown is rendered.
```

---

## 19. Current Output Schema

Every pipeline execution produces a standardized JSON dictionary conforming to the unified schema:

```json
{
  "input": {
    "input_type": "image",
    "filename": "face_image.jpg",
    "size_bytes": 35811,
    "dimensions": [224, 224],
    "status": "PROCESSED"
  },
  "visual": {
    "status": "SUCCESS",
    "face_detected": true,
    "prediction": "real",
    "confidence": 0.8842,
    "probabilities": {
      "real": 0.8842,
      "fake": 0.1158
    },
    "explanation": {
      "method": "grad_cam",
      "heatmap_shape": [7, 7],
      "overlay_path": null,
      "disclaimer": "Grad-CAM is a post-hoc feature attribution method and should not be interpreted as causal proof."
    },
    "uncertainty": {
      "mean_probability": [0.8842, 0.1158],
      "variance": [0.000185, 0.000185],
      "std": [0.0136, 0.0136],
      "entropy": 0.5185,
      "max_variance": 0.000185,
      "mean_variance": 0.000185,
      "mc_samples": 20,
      "is_stochastic": true,
      "disclaimer": "MC Dropout uncertainty is a model uncertainty signal and has not been formally calibrated for deployment."
    }
  },
  "text": {
    "sources": ["OCR", "USER_TEXT"],
    "segments": [
      {
        "source": "OCR",
        "text": "BREAKING NEWS HEADLINE",
        "confidence": 0.9421
      }
    ],
    "combined_text": "BREAKING NEWS HEADLINE",
    "text_status": "AVAILABLE",
    "segment_count": 1
  },
  "propaganda": {
    "status": "SUCCESS",
    "prediction": "Loaded_Language",
    "confidence": 0.7612,
    "probabilities": {
      "Loaded_Language": 0.7612,
      "Name_Calling,Labeling": 0.0841,
      "Doubt": 0.0412
    },
    "explanation": {
      "top_tokens": [
        {"token": "BREAKING", "clean_token": "BREAKING", "score": 0.9421}
      ],
      "disclaimer": "Token attribution is a post-hoc explanatory signal and should not be interpreted as causal proof."
    },
    "uncertainty": {
      "mean_probability": [0.7612],
      "max_variance": 0.00084,
      "entropy": 0.4125,
      "mc_samples": 20
    }
  },
  "hate_speech": {
    "status": "SUCCESS",
    "prediction": "normal",
    "confidence": 0.9125,
    "probabilities": {
      "normal": 0.9125,
      "offensive": 0.0612,
      "hatespeech": 0.0263
    },
    "explanation": {
      "top_tokens": [],
      "disclaimer": "Token attribution is a post-hoc explanatory signal..."
    },
    "uncertainty": {
      "max_variance": 0.00032,
      "entropy": 0.3124,
      "mc_samples": 20
    }
  },
  "uncertainty": {
    "deepfake": { "max_variance": 0.000185, "entropy": 0.5185 },
    "propaganda": { "max_variance": 0.00084, "entropy": 0.4125 },
    "hate_speech": { "max_variance": 0.00032, "entropy": 0.3124 }
  },
  "governance": {
    "review_required": false,
    "decision_status": "CONFIDENT_PREDICTION",
    "review_trigger_component": null,
    "disclaimer": "Model confidence and statistical consistency meet baseline criteria."
  },
  "audit": {
    "timestamp": "2026-10-05T02:00:00Z",
    "execution_time_seconds": 1.452,
    "input_type": "image",
    "filename": "face_image.jpg",
    "models_executed": ["EfficientNet-B4", "RoBERTa", "BERT"],
    "checkpoint_versions": {
      "deepfake": "EfficientNet-B4 (models/deepfake/checkpoint/best_model.pt)",
      "propaganda": "RoBERTa-base (models/propaganda/checkpoint)",
      "hate_speech": "BERT-base-uncased (models/hate_speech/checkpoint)",
      "asr": "Whisper-base (openai/whisper-base)"
    },
    "errors": []
  }
}
```

---

## 20. PPT Alignment Check

Comparison between the intended project architecture/technology stack and the actual implemented codebase:

| Target Component | Intended Architecture / Tech Stack | Actual Implemented State | Audit Verdict |
| :--- | :--- | :--- | :---: |
| **Frontend Workstation** | React-based human review dashboard | React 19 + Vite 8 workstation with Vanilla CSS, dark mode, responsive cards | ✅ **Implemented** |
| **Backend Service** | FastAPI REST API service | FastAPI 0.136 on Python 3.14 with Uvicorn, pre-warming, chunked uploads | ✅ **Implemented** |
| **Deepfake Model** | EfficientNet-B4 binary classifier | Pretrained EfficientNet-B4 fine-tuned on FF++ C23 (`best_model.pt`) | ✅ **Implemented** |
| **Face Detection** | MTCNN face detection & cropping | `facenet-pytorch.MTCNN` with 0.2 margin cropping to $224 \times 224$ | ✅ **Implemented** |
| **Frame Sampling** | Temporal video frame extraction | OpenCV sampling at configurable FPS (default 1.0 FPS, max 16) | ✅ **Implemented** |
| **OCR Text Extraction** | EasyOCR optical character recognition | `EasyOCRReader` running CRAFT on images and video keyframes | ✅ **Implemented** |
| **Audio Extraction** | Bundled FFmpeg audio probing & extraction | `imageio-ffmpeg` probing streams and extracting 16 kHz mono PCM WAV | ✅ **Implemented** |
| **Speech ASR** | OpenAI Whisper-base speech-to-text | `openai/whisper-base` (74M parameters) transcription with PCM buffer decoding | ✅ **Implemented** |
| **Text Aggregation** | Multi-source text merger with provenance | `TextAggregator` merging `USER_TEXT`, `OCR`, and `ASR` | ✅ **Implemented** |
| **Propaganda Detection** | RoBERTa fine-grained classifier | Fine-tuned `RoBERTa-base` across 14 SemEval-2020 Task 11 classes | ✅ **Implemented** |
| **Hate Speech Detection**| BERT 3-class toxicity classifier | Fine-tuned `BERT-base-uncased` across HateXplain (`normal`, `offensive`, `hatespeech`) | ✅ **Implemented** |
| **Visual Explainability**| Spatial Grad-CAM saliency heatmaps | Grad-CAM on final Conv block `features[-1]`, bilinear upsampling & colormap blend | ✅ **Implemented** |
| **Text Explainability** | Token-level feature attribution | Gradient $\times$ Input ($S_i = \|\nabla_{e_i} y \odot e_i\|$) with subword reconstruction | ✅ **Implemented** |
| **Rationale Alignment** | HateXplain human rationale scoring | Precision/Recall/F1 evaluation against HateXplain masks (active in benchmarks) | 🟡 **Partially Implemented** |
| **Uncertainty Method** | Monte Carlo Dropout ($T=20$) | $T=20$ stochastic passes, selective layer activation, variance, entropy | ✅ **Implemented** |
| **Review Policy** | Epistemic thresholding review flag | Threshold evaluation ($\sigma^2 \ge 0.02$, $H \ge 0.85 \implies \text{REVIEW\_RECOMMENDED}$) | ✅ **Implemented** |
| **Governance Safeguards**| Decision support, faceless media handling | `ANALYSIS_INCOMPLETE` on faceless media; non-punitive, decision support only | ✅ **Implemented** |
| **Async Video Queue** | Background video job processing | FastAPI `BackgroundTasks` + polling endpoint (uses in-memory `job_store`) | 🟡 **Partially Implemented** |
| **Streamlit GUI** | Streamlit web application (`app.py`) | Replaced by React 19 + Vite 8 frontend; `app.py` is a dormant scaffold stub | ❌ **Replaced / Obsolete** |
| **Uncertainty Calibration**| Formally calibrated thresholds | Thresholds are development defaults; formal calibration curves not implemented | ❌ **Missing / Future** |
| **Cross-Modal Fusion** | Deep cross-modal attention fusion | Parallel independent modality inspection with textual aggregation | ❌ **Missing / Future** |

---

## 21. Technical Gaps

A prioritized breakdown of system limitations and architectural gaps:

### P0 — Critical Functional / Breaking Bugs
- **None**: All core pipelines, API endpoints, model inferences, and frontend components execute without crashing. All 51 tests pass.

### P1 — Required for Final Live Demonstration Polish
1. **Grad-CAM Overlay Base64 Streaming**: In `VisualResultView.jsx`, if `output_dir` is not explicitly passed, the overlay image path is `null` and the UI renders the heatmap fallback box. Exposing the generated overlay as a base64 Data URL directly in `visual.explanation.overlay_url` would ensure the blended heatmap image always displays in the browser without requiring static file serving.
2. **Persistent Video Task Store**: The video background task queue (`job_store`) resides in volatile Python RAM. If the server reloads (e.g. via `--reload`), queued or completed jobs are wiped.

### P2 — Required for Academic Completeness & Project Hygiene
1. **Pruning Scaffold Stubs**: The repository contains obsolete placeholder files (`app.py`, `dashboard/*.py`, `governance/*.py`, `preprocessing/image_preprocessing.py`, `evaluation/*.py`, `utils/*.py`). They should either be documented as legacy scaffolds or cleaned up to avoid confusing future auditors.
2. **Formal Uncertainty Calibration**: Variance (0.02) and entropy (0.85) thresholds are development defaults. Formal calibration (e.g. temperature scaling or Platt scaling against validation error curves) remains an academic open loop.
3. **Deepfake Generalization Bottleneck**: Expanding training from 160 frames (40 identities) to a larger slice of FaceForensics++ to improve generalization beyond 50.00% test video accuracy.

### P3 — Future Improvements / Production Readiness
1. **Distributed Task Queue**: Replacing FastAPI `BackgroundTasks` with Redis and Celery for enterprise-scale video processing.
2. **Cross-Modal Transformer Fusion**: Moving from parallel modality extraction to joint cross-attention between audio spectrograms, video frames, and text tokens.
3. **Temporal Video Forensics**: Adding 3D ConvNets or Video Transformers to detect inter-frame flickering and lip-sync inconsistency.

---

## 22. Final Current-State Summary

1. **What the MVP Can Actually Do Today**:
   The system ingests images, videos, and raw text; extracts faces via MTCNN; samples video frames via OpenCV; transcribes speech via Whisper-base; extracts on-screen text via EasyOCR; classifies deepfakes (EfficientNet-B4), propaganda (RoBERTa 14-class), and hate speech (BERT 3-class); computes spatial Grad-CAM heatmaps and token-level Gradient $\times$ Input attributions; estimates epistemic uncertainty via 20 MC Dropout passes; flags ambiguous media for human review (`REVIEW_RECOMMENDED`); safely handles faceless media (`ANALYSIS_INCOMPLETE`); serves all functionality through a validated FastAPI backend; and visualizes results in a responsive React 19 dashboard.

2. **What It Cannot Do Today**:
   It cannot perform autonomous content moderation, deletions, or account bans; it cannot establish ground-truth physical reality; it cannot process non-English audio or text; it cannot verify non-facial deepfakes (e.g., scene inpainting); it does not fuse multimodal features into a unified cross-attention transformer; and it does not maintain persistent background jobs across server restarts.

3. **What Is Only Documented (or Scaffolded)**:
   The `governance/`, `dashboard/`, `utils/`, and `evaluation/` directories contain docstring stubs. `config/thresholds.yaml` is unused. Streamlit (`app.py`) is completely dormant. Uncertainty calibration is only documented as an MVP limitation, not empirically calibrated.

4. **What Is Genuinely Verified**:
   All three fine-tuned checkpoints (`best_model.pt`, `model.safetensors`), tokenizers, 3,000 FF++ videos, SemEval corpus, and HateXplain benchmark exist physically on disk. 41 backend pytest tests and 10 frontend vitest tests execute and pass 100%.

5. **Most Important Technical Gaps**:
   - Deepfake model test video generalization is at baseline (50.00%) due to small MVP subset training.
   - Background video jobs are held in volatile RAM.
   - Grad-CAM overlay image in web UI falls back to heatmap dimensions if static overlay paths are not served.

6. **Current Architecture in One Paragraph**:
   The system is a decoupled client-server architecture where a React 19 single-page application communicates over HTTP/REST with a FastAPI service layer running a pre-warmed singleton `MultimodalPipeline`. Ingested media is routed to specialized Image, Video, or Text sub-pipelines that invoke MTCNN, EasyOCR, and Whisper-base in parallel, feeding extracted representations into fine-tuned EfficientNet-B4, RoBERTa-base, and BERT-base classifiers. Every prediction is augmented with 20 Monte Carlo Dropout stochastic passes and gradient-based explainability before an uncertainty governance layer triages the outcome into `CONFIDENT_PREDICTION`, `REVIEW_RECOMMENDED`, or `ANALYSIS_INCOMPLETE` for human forensic oversight.

7. **Current Final Demo Flow in One Paragraph**:
   A live demonstration begins by running Uvicorn on port 8000 and Vite on port 5173, after which the analyst accesses the dashboard in the browser, observes the live pulsing backend health indicator, and evaluates test assets from `tests/assets/`: verifying `face_image.jpg` to display deepfake probabilities and Grad-CAM saliency; uploading `no_face_image.jpg` to demonstrate responsible governance refusal via `ANALYSIS_INCOMPLETE`; inputting benchmark propaganda and hate speech text to showcase token-level attribution pills and entropy telemetry; and submitting `audio_video.mp4` in asynchronous mode to demonstrate non-blocking background video processing with speech transcription and multi-source provenance tracking.

8. **Recommended Next Phase Roadmap**:
   - **Phase 1 (Presentation & Hygiene)**: Use [docs/current_project_context_for_gemini.md](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/docs/current_project_context_for_gemini.md) to upgrade academic presentation slides, clearly highlighting the completed React+FastAPI architecture and candidly presenting the 50.00% deepfake generalization as scientific justification for human-in-the-loop uncertainty triage.
   - **Phase 2 (UI Polish)**: Ensure Grad-CAM overlays return directly as base64 data URLs to guarantee immediate image rendering in web browsers.
   - **Phase 3 (Model Scaling)**: Train EfficientNet-B4 on larger slices of FaceForensics++ to improve generalization, and formalize uncertainty calibration curves.
