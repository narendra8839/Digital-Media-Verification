# Phase 7: Final Frontend Refinement, End-to-End Validation, Demo Readiness, and System Documentation

**Project:** Digital Media Verification: Explainable and Uncertainty-Aware Deepfake, Propaganda and Hate-Speech Detection  
**Branch:** `swaraj`  
**Date:** October 2026  
**Status:** Completed and Verified  

---

## 1. Executive Summary & Project Positioning

Phase 7 marks the final implementation, frontend refinement, full end-to-end validation, and demo preparation phase of the **Digital Media Verification** platform.

### Core Project Identity:
The platform is designed and evaluated strictly as a **decision-support system for media analysts, fact-checkers, and investigators**.
- **It is NOT:**
  - A "truth detector" or oracle of absolute reality.
  - An automated content moderation or censorship engine.
  - An automated deletion, takedown, or account banning mechanism.
- **It IS:**
  - An explainable, uncertainty-aware decision-support framework following the paradigm:
    $$\text{Analyze} \longrightarrow \text{Explain} \longrightarrow \text{Quantify Uncertainty} \longrightarrow \text{Recommend Human Review}$$
  - A responsible AI system adhering to ART (Accountability, Responsibility, and Transparency) principles.

All pre-existing user frontend modifications have been meticulously audited and preserved. The React frontend and FastAPI backend have been fully reconciled and validated across every supported modality under real model execution.

---

## 2. Final System Architecture & Verification Workflow

```
                          [User Upload / Input]
                                    │
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
       [Image Upload]         [Video Upload]          [Raw Text]
             │                      │                      │
             │                      ├─ Audio Extraction    │
             │                      │  (FFmpeg Probe)      │
             │                      │  └─ Whisper ASR      │
             │                      │     (Mono 16kHz)     │
             │                      │                      │
             │                      ├─ Keyframe OCR        │
             │                      │  (EasyOCR CRAFT)     │
             │                      │                      │
             │                      └─ 1 FPS Frame Sample  │
             │                             │               │
             ▼                             ▼               │
      [MTCNN Face Crop]             [MTCNN Face Crop]      │
             │                             │               │
    ┌────────┴────────┐           ┌────────┴────────┐      │
    │ Face Detected?  │           │ Face Detected?  │      │
    └──┬───────────┬──┘           └──┬───────────┬──┘      │
       │ No        │ Yes             │ No        │ Yes     │
       ▼           ▼                 ▼           ▼         │
  [ANALYSIS_   [EfficientNet-B4] [ANALYSIS_  [Per-Frame    │
  INCOMPLETE]      │             INCOMPLETE] Probabilities]│
                   │                             │         │
                   │                             ▼         │
                   │                     [Mean Probability │
                   │                      Aggregation]     │
                   ▼                             ▼         │
             [Grad-CAM Saliency Map]                       │
             [MC Dropout: T=20 Passes]                     │
                   │                                       │
                   └───────────────────┬───────────────────┘
                                       │
                        ┌──────────────┴──────────────┐
                        ▼                             ▼
                 [RoBERTa-base]                [BERT-base]
             (14-Class Propaganda)         (3-Class Hate Speech)
                        │                             │
                        ├─ Token Attribution          ├─ Token Attribution
                        │  (Grad x Input)             │  (Grad x Input)
                        └─ MC Dropout: T=20           └─ MC Dropout: T=20
                                       │
                                       ▼
                       [Responsible AI Governance Engine]
                                       │
             ┌─────────────────────────┼─────────────────────────┐
             ▼                         ▼                         ▼
   [CONFIDENT_PREDICTION]    [REVIEW_RECOMMENDED]      [ANALYSIS_INCOMPLETE]
    σ² < 0.02, H < 0.85       σ² >= 0.02 or H >= 0.85    Face / Audio Absent
    Baseline Confidence       Logs Specific Trigger     Safe Non-Speculative
    Criteria Satisfied        (e.g., "propaganda")      Bypass
```

---

## 3. Pre-Existing User Frontend Work Preservation Audit

Before any Phase 7 adjustments were made, the existing workspace state was audited to protect user frontend code:

| File | Status Before Phase 7 | Changes in Phase 7 | Verification Status |
| :--- | :--- | :--- | :--- |
| `frontend/src/App.css` | User design styling | **Preserved intact** (0 lines altered) | Verified identical |
| `frontend/src/index.css` | User global typography & variables | **Preserved intact** (0 lines altered) | Verified identical |
| `frontend/src/components/Header.jsx` | User header component & logo | **Preserved intact** (0 lines altered) | Verified identical |
| `frontend/src/components/ResultDashboard.jsx` | User dashboard structure | **Preserved intact** (0 lines altered) | Verified identical |
| `frontend/src/components/TokenAttributionView.jsx` | User saliency highlights | **Preserved intact** (0 lines altered) | Verified identical |
| `frontend/src/components/UncertaintyCard.jsx` | Metric card presentation | Added explicit heuristic UQ disclaimer | Verified compliant |
| `frontend/src/components/VisualResultView.jsx` | Image & Grad-CAM viewer | URL proxy normalization for `/artifacts/` | Verified compliant |
| `frontend/vite.config.js` | Dev server configuration | Added `/artifacts` proxy to `http://localhost:8000` | Verified functional |
| `frontend/index.html` | Page shell | Updated descriptive title & meta tags | Verified compliant |

No reset, stash, checkout, or destructive Git operations were performed.

---

## 4. End-to-End API / UI Contract Verification

The frontend consumes the FastAPI backend responses across all 4 operational modes:

### Mode 1: Image Verification
- **Input:** Single image upload (PNG, JPEG, WebP) with optional contextual caption.
- **Face Detected:**
  - Renders primary prediction: `Manipulated / Deepfake` or `Authentic / Real`.
  - Confidence metric and class probabilities (`real`, `fake`).
  - Grad-CAM heatmap visualization with side-by-side or overlay toggle modes.
  - Uncertainty card: Predictive variance ($\sigma^2$), standard deviation ($\sigma$), Shannon entropy ($H$), and $T=20$ MC Dropout passes.
  - Governance status: `CONFIDENT_PREDICTION` or `REVIEW_RECOMMENDED`.
- **No Face Detected:**
  - Safely bypasses classification.
  - Visual status rendered: `NO_FACE_DETECTED`.
  - Prediction fabricated: **No** (`null` / suppressed).
  - Governance status: `ANALYSIS_INCOMPLETE`.

### Mode 2: Video Verification
- **Input:** MP4, WebM, AVI, MOV file.
- **Workflow:** Supports both synchronous execution and background asynchronous job processing (`/verify/video` with `async_mode=true` and `/verify/status/{job_id}`).
- **Sampling & Aggregation:** Uniform deterministic 1 FPS sampling (up to 16 frames); video-level classification computed as arithmetic mean of per-frame fake probabilities.
- **Audio Stream Handling:** Inspects audio tracks cleanly. Silent videos automatically set `has_audio=False` with zero errors.
- **Multimodal Extraction:** Keyframe OCR and Whisper ASR are extracted and displayed with clear provenance tags.

### Mode 3: Text Verification
- **Input:** Raw text input with live character counter and demo sample chips.
- **Propaganda Analysis:** RoBERTa 14-class technique prediction, probability distribution, and token attribution highlights.
- **Hate Speech Analysis:** BERT 3-class classification (`normal`, `offensive`, `hatespeech`), confidence, and HateXplain rationale alignment metrics.
- **Disclaimers:** Displays: *"Token attribution is a post-hoc explanatory signal and should not be interpreted as causal proof."*

### Mode 4: Multimodal Combined Verification
- **Input:** Media file (Image or Video) plus accompanying text or caption.
- **Component Separation:** Distinguishes **Visual Evidence**, **Textual Evidence**, **Extracted Content (OCR / ASR)**, **Uncertainty Quantification**, and **Governance**.
- **Review Trigger Traceability:** When uncertainty is elevated in one specific modality, governance clearly highlights which component triggered the review recommendation (e.g. `Review Triggering Component: propaganda`).

---

## 5. Official Benchmark Evaluation Metrics (Accepted Baseline)

The following metrics are the accepted benchmark numbers established in Phase 4 and Phase 5 and confirmed across all evaluations:

### A. Deepfake Detection (FaceForensics++ C23)
- **Evaluation Partition:** Full untouched test split ($N = 700$ videos: 140 Real, 560 Fake)
- **Model Checkpoint:** `models/deepfake/checkpoint_full/best_model.pt` (EfficientNet-B4)
- **Sampling Policy:** Deterministic 4-frame uniform sampling
- **Video Accuracy:** **0.8800** (88.00%)
- **Macro F1 Score:** **0.8321**
- **Weighted F1 Score:** **0.8859**
- **Real Class Metrics:** Precision = **0.6505**, Recall = **0.8643**, F1 = **0.7424**
- **Fake Class Metrics:** Precision = **0.9634**, Recall = **0.8839**, F1 = **0.9220**

### B. Propaganda Technique Detection (SemEval-2020 Task 11)
- **Evaluation Partition:** Held-out article-disjoint validation split ($N = 1,448$ spans across 71 articles; 100% disjoint from the 286 training articles)
- **Model Checkpoint:** `models/propaganda/optimization/exp1_sqrt_weighted_scheduled` (RoBERTa-base)
- **Span Accuracy:** **0.6519** (65.19%)
- **Macro F1 Score:** **0.5132**
- **Weighted F1 Score:** **0.6477**
- *Note:* Official SemEval-2020 Task 11 test labels remain unreleased by task organizers; reported evaluation strictly represents held-out article-disjoint validation.

### C. Hate Speech & Toxicity Detection (HateXplain)
- **Evaluation Partition:** Full untouched test split ($N = 1,924$ posts: 782 Normal, 548 Offensive, 594 Hate Speech)
- **Model Checkpoint:** `models/hate_speech/checkpoint_full` (BERT-base-uncased)
- **Test Accuracy:** **0.6741** (67.41%)
- **Macro F1 Score:** **0.6617**
- **Weighted F1 Score:** **0.6738**
- **Normal Class F1:** **0.7554**
- **Offensive Class F1:** **0.5756**
- **Hate Speech Class F1:** **0.6541**

---

## 6. Verification Test Matrix

### A. Frontend Test Suite (`frontend/src/__tests__/App.test.jsx`)
Ran via Vitest in JSDOM environment:
- **Baseline Tests (10/10):** All passing without regression.
- **Phase 7 Focused Additions (2):**
  - Test 11: Multimodal verification with component-specific review trigger rendering.
  - Test 12: Faceless image verification rendering `NO_FACE_DETECTED` and `ANALYSIS_INCOMPLETE` without fabricating predictions.
- **Total Frontend Test Result:** **12 passed (100% PASS RATE)**.

### B. Backend Test Suite (`tests/`)
Ran via PyTorch on Python 3.14 + CUDA in WSL2 environment:
- `tests/test_api.py` (41 tests): PASS
- `tests/test_phase6_integration.py` (11 integration tests): PASS
- **Total Backend Test Result:** **52 passed (100% PASS RATE)**.

### C. Live End-to-End Execution Results (Live Server: `http://localhost:8000`)
Executed with real model checkpoints on NVIDIA GeForce RTX 4050 Laptop GPU:

| Scenario | Input Tested | Latency | Primary Output | UQ / XAI Output | Governance Status | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Image with Face** | `tests/assets/face_image.jpg` | 5,482 ms | `real` (87.83% conf) | Grad-CAM URL HTTP 200 OK; $\sigma^2=0.000185$ | `CONFIDENT_PREDICTION` | **PASS** |
| **B. Image without Face** | `tests/assets/no_face_image.jpg` | 344 ms | `null` (suppressed) | Bypassed; no fabrication | `ANALYSIS_INCOMPLETE` | **PASS** |
| **C. Text Input** | Political rhetoric text | 1,073 ms | Propaganda: `Doubt`<br>Toxicity: `offensive` | Token highlights generated; $T=20$ | `REVIEW_RECOMMENDED`<br>(Trigger: `hate_speech`) | **PASS** |
| **D. Silent Video** | `tests/assets/silent_video.mp4` | 382 ms | Clean audio bypass | Keyframe OCR executed; ASR skipped | `ANALYSIS_INCOMPLETE` | **PASS** |
| **E. Multimodal** | Face image + claim text | 1,880 ms | Visual: `real`<br>Propaganda: `Doubt` | Component-specific UQ | `REVIEW_RECOMMENDED`<br>(Trigger: `propaganda`) | **PASS** |

---

## 7. Recommended 5–7 Minute Live Demo Script

### Flow Overview:
1. **Introduction & System Philosophy (1 min):**
   - Present the core concept: Decision Support System following ART principles.
   - Clarify: Not a truth detector, not automated censorship.
2. **Image Verification & Grad-CAM (1.5 min):**
   - Upload sample face image.
   - Show face detection, binary classification, and confidence.
   - Demonstrate the Grad-CAM saliency overlay and side-by-side comparison.
   - Highlight the post-hoc explanation disclaimer.
3. **Uncertainty Quantification & Confident Governance (1 min):**
   - Show the Uncertainty Quantification card ($T=20$ MC Dropout passes).
   - Point out predictive variance $\sigma^2$ and Shannon entropy $H$.
   - Note the `CONFIDENT_PREDICTION` badge and explain the heuristic threshold rule ($\sigma^2 < 0.02, H < 0.85$ bits).
4. **Safety & Ungrounded Media Handling (1 min):**
   - Upload a landscape or object image without a human face.
   - Show `NO_FACE_DETECTED` and `ANALYSIS_INCOMPLETE`.
   - Emphasize academic integrity: the model refuses to guess authenticity without grounded facial features.
5. **Text Verification & Token Attribution (1.5 min):**
   - Navigate to Text Verification tab; select "Propaganda Rhetoric" demo sample chip.
   - Observe RoBERTa 14-class technique prediction (`Doubt` or `Loaded_Language`) and BERT toxicity class (`normal` / `offensive`).
   - Show Token Attribution saliency highlights and the HateXplain rationale alignment metrics.
   - Point out the `REVIEW_RECOMMENDED` governance status with the specific triggering component badge.
6. **Video & Multimodal Integration (1 min):**
   - Briefly demonstrate multimodal combined verification.
   - Explain OCR/ASR provenance tracking and asynchronous background processing capabilities.
   - Conclude with human-in-the-loop decision support responsibility.

---

## 8. Responsible AI Guardrails & Unsupported Claims Audit

A full repository-wide text audit confirmed that the system contains **zero unsupported marketing claims**:
- `"100% accurate"`: **0 occurrences**
- `"detects truth"`: **0 occurrences**
- `"guaranteed deepfake detection"`: **0 occurrences**
- `"fully reliable"`: **0 occurrences**
- `"calibrated uncertainty"`: **0 occurrences**
- `"causal explanation"`: **0 occurrences**
- `"automatically moderates"`: **0 occurrences**
- `"eliminates human review"`: **0 occurrences**

All explanations carry the explicit disclaimer:
> *"Token attribution / Grad-CAM is a post-hoc explanatory signal and should not be interpreted as causal proof."*

All uncertainty panels carry the explicit disclaimer:
> *"Heuristic development threshold — calibration not quantitatively established. MC Dropout predictive uncertainty represents model epistemic uncertainty (T = 20 passes), not error probability."*

---

## 9. Final Limitations & Academic Disclaimers

1. **Uncertainty Calibration:** Epistemic uncertainty is estimated via $T=20$ Monte Carlo Dropout iterations. While variance and entropy distinguish high-confidence predictions from ambiguous edge cases, quantitative temperature scaling or conformal calibration has not been formally evaluated.
2. **Facial Boundary Scope:** Visual deepfake detection operates exclusively within cropped facial bounding boxes detected by MTCNN. Non-facial visual manipulations (background scene synthesis, generative image backgrounds, object insertions) are outside the model's receptive field.
3. **Monolingual Constraints:** Speech-to-text (Whisper), scene text extraction (EasyOCR), and NLP models (RoBERTa, BERT) are trained and validated exclusively on English discourse.
4. **Post-Hoc Explanations:** Grad-CAM heatmaps and gradient-based token attributions reflect features that influenced the model's activations, not causal evidence of deliberate human editing.
5. **Asynchronous Video Persistence:** Video background tasks utilize an in-memory job state store designed for local academic evaluation. Production deployments require persistent message broker backends (e.g., Redis / Celery).
