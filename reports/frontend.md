# Frontend Interface Technical Report: React + Vite Dashboard

## 1. Overview & Architectural Philosophy

The **Digital Media Verification System Frontend** provides an academic, human-in-the-loop verification workstation for digital forensics, ethical AI, and misinformation research. Developed as part of a B.Tech capstone project, the user interface adheres to strict **Responsible AI** guidelines:
- It functions purely as a **decision-support telemetry system** rather than an automated censorship tool.
- It displays calibrated epistemic uncertainty and post-hoc feature attributions without making definitive truth claims.
- It provides complete multi-source provenance isolation between Optical Character Recognition (OCR), Whisper-base Speech Recognition (ASR), and user-provided claims.

The interface is built using **React 19** and bundled with **Vite 8**, adopting a curated dark slate aesthetic with responsive CSS grid and flexbox layouts, accessible typography (`Inter` and `JetBrains Mono`), and glassmorphism styling.

---

## 2. Directory & Component Structure

```
frontend/
├── index.html                   # HTML5 shell with metadata and Google Fonts
├── vite.config.js               # Vite config with dev server (port 5173) and Vitest jsdom
├── package.json                 # React 19, Lucide icons, Vitest, Testing Library
├── .env / .env.example          # Environment variables (VITE_API_BASE_URL)
└── src/
    ├── main.jsx                 # React root injection
    ├── App.jsx                  # Primary application shell & state orchestration
    ├── index.css                # Polished Vanilla CSS design system
    ├── setupTests.js            # Testing library Jest-DOM setup
    │
    ├── services/
    │   └── api.js               # Centralized HTTP & FormData API communication layer
    │
    ├── types/
    │   └── schemas.js           # Domain constants, enum definitions & formatting helpers
    │
    ├── hooks/
    │   ├── useBackendHealth.js  # Non-blocking background health check polling
    │   └── useVerification.js   # Verification lifecycle, state machine & async video polling
    │
    ├── components/
    │   ├── Header.jsx           # Top application bar with academic badge & health status
    │   ├── HealthIndicator.jsx  # Pulsing live indicator for backend liveness
    │   ├── InputSelector.jsx    # 4-mode selector (Image, Video, Text, Multimodal)
    │   ├── ImageInput.jsx       # Image drag-and-drop, preview & optional caption
    │   ├── VideoInput.jsx       # Video file upload, FPS selector & async toggle
    │   ├── TextInput.jsx        # Textarea with character counter & demo sample chips
    │   ├── MultimodalInput.jsx  # Unified media file + accompanying claim input
    │   ├── ProcessingIndicator.jsx # Spinner with active stage telemetry & job metadata
    │   ├── ResultDashboard.jsx  # Master coordination view with action bar
    │   ├── GovernancePanel.jsx  # Responsible AI decision support & review status
    │   ├── VisualResultView.jsx # EfficientNet-B4 classification & Grad-CAM visualizer
    │   ├── TextExtractionView.jsx # Provenance isolation (OCR vs ASR vs USER_TEXT)
    │   ├── NLPResultView.jsx    # RoBERTa propaganda & BERT toxicity analysis
    │   ├── TokenAttributionView.jsx # Subword saliency highlight chips & HateXplain metrics
    │   ├── UncertaintyCard.jsx  # MC Dropout (T=20) metrics: variance, std, entropy
    │   └── AuditDetails.jsx     # Collapsible reproducibility telemetry & checkpoint paths
    │
    └── __tests__/
        └── App.test.jsx         # 10 Vitest integration tests covering full specifications
```

---

## 3. API Communication Layer (`services/api.js`)

All interactions with the FastAPI backend (`http://localhost:8000`) are centralized in `src/services/api.js`. The endpoint base URL is dynamically injected from `import.meta.env.VITE_API_BASE_URL` with a sensible default fallback, preventing hardcoded references across components:

| Method | Endpoint | Description | Request Format |
|---|---|---|---|
| `GET` | `/health` | Liveness check | Query |
| `GET` | `/ready` | Model readiness check | Query |
| `POST` | `/verify/text` | Natural language propaganda & hate speech | JSON (`{ text }`) |
| `POST` | `/verify/image` | Facial deepfake detection + EasyOCR | `multipart/form-data` (`file`, `caption`) |
| `POST` | `/verify/video` | Deepfake sampling, OCR & Whisper ASR | `multipart/form-data` (`file`, `caption`, `sample_fps`, `max_frames`, `async_mode`) |
| `GET` | `/verify/status/{id}`| Async video job polling | Path parameter |
| `POST` | `/verify/multimodal` | Unified multimodal router | `multipart/form-data` (`file`, `text`) |

### Error Parsing & Resiliency
Non-200 responses are automatically unpacked into user-friendly diagnostic messages (`response.json().detail`), avoiding raw stack traces or cryptic HTTP status codes in user-facing dialogs.

---

## 4. Input Modality Workspaces

### A. Image Verification
- **Dropzone**: Drag-and-drop or file picker accepting JPG, PNG, WEBP, BMP, TIFF (up to 15 MB).
- **Preview**: Client-side object URL generation with dimension and file size telemetry.
- **Accompanying Caption**: Optional context field enabling multimodal cross-referencing.
- **Faceless Media Handling**: If MTCNN detects no human faces, the UI gracefully renders a `NO_FACE_DETECTED` status and safe bypass banner rather than erroring or guessing.

### B. Video Verification
- **File Ingestion**: Accepts MP4, AVI, MOV, MKV, WebM up to 100 MB.
- **Frame Sampling Control**: Dropdown allows toggling between 0.5 FPS (Sparse), 1.0 FPS (Standard), and 2.0 FPS (Dense).
- **Async Processing Toggle**: Allows offloading long video runs to the FastAPI `BackgroundTasks` engine with non-blocking progress polling.

### C. Direct Text Verification
- **Textarea**: 10,000-character input box with live character counting.
- **Demo Sample Chips**: Instant one-click loading of benchmark examples (Propaganda Rhetoric, Election Fraud Claim, Neutral News Statement).

### D. Unified Multimodal Verification
- Accepts both an uploaded media asset (image or video) and accompanying social media post / claim text for cross-modal routing.

---

## 5. Result Dashboard & Responsible AI Layout

The Result Dashboard implements a hierarchical, academic card layout:

```
┌─────────────────────────────────────────────────────────────┐
│ Action Bar: Modality • Filename • Latency • Reset Button   │
├─────────────────────────────────────────────────────────────┤
│ 1. Responsible AI / Human Oversight Panel                  │
│    - Status: CONFIDENT_PREDICTION vs REVIEW_RECOMMENDED     │
│    - Triggering Component (e.g. Visual MC Dropout Variance) │
│    - Justification & Decision Support Principle             │
├─────────────────────────────────────────────────────────────┤
│ 2. Visual Deepfake & Grad-CAM Analysis (if applicable)     │
│    - Prediction: Manipulated / Deepfake vs Authentic / Real │
│    - Confidence: Stochastic Softmax Score                   │
│    - Grad-CAM: Overlay / Side-by-Side comparison views      │
│    - Disclaimer: Post-hoc saliency attribution              │
│    - Uncertainty: Predictive variance, std dev, entropy     │
├─────────────────────────────────────────────────────────────┤
│ 3. Extracted Text Provenance (if applicable)                │
│    - Multi-Source isolation (OCR vs ASR vs USER_TEXT)       │
│    - Aggregated textual payload for NLP inference           │
├─────────────────────────────────────────────────────────────┤
│ 4. NLP Verification (if applicable)                         │
│    - RoBERTa: 14 Propaganda Techniques + Token Saliency     │
│    - BERT: Hate Speech Classification + Token Saliency      │
│    - HateXplain Benchmark: Human Rationale Alignment Metric │
├─────────────────────────────────────────────────────────────┤
│ 5. Audit & Reproducibility Accordion                       │
│    - Checkpoint paths, execution time, MC samples, errors   │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Key Explainability & Uncertainty Telemetry

### Grad-CAM Convolutional Saliency
- Displays actual overlays produced by `GradCAM` from the EfficientNet-B4 final convolutional block (`features[-1]`).
- Supports interactive view mode switching (`Overlay` vs `Side-by-Side` comparison with original input).
- Explicit post-hoc disclaimer prevents confusing saliency heatmap activations with causal proof of manual tampering.

### Token-Level Gradient Attribution
- Saliency values calculated via $Grad \times Input$ across transformer word embeddings.
- Renders subword chips with proportional heat-map highlighting.
- HateXplain Ground-Truth Rationale Alignment displays precision, recall, and rationale F1 scores against human annotators.

### Monte Carlo Dropout ($T = 20$)
- Simple numeric cards avoiding misleading pseudo-calibrated dials:
  - Mean Variance ($\sigma^2$)
  - Standard Deviation ($\sigma$)
  - Shannon Entropy ($H$ in bits)
  - Stochastic MC Passes ($T = 20$)
- Predictive mean probability distribution across classes rendered via discrete percentage bars.
- Evaluates review threshold criteria to flag elevated epistemic uncertainty.

---

## 7. Asynchronous Video Processing Workflow

When asynchronous mode is requested:
1. `POST /verify/video` returns `{"job_id": "...", "status": "QUEUED"}`.
2. `useVerification` initializes a polling interval checking `/verify/status/{job_id}` every 2000 ms.
3. The UI presents an animated `ProcessingIndicator` detailing the active stage (temporal frame extraction, EasyOCR, Whisper-base ASR).
4. Upon receiving `COMPLETED`, polling terminates cleanly, and the complete verification payload is rendered in the dashboard.
5. If `FAILED`, polling ceases and the error is surfaced in a dismissable error banner.

---

## 8. Verification & Test Suite

The frontend includes 10 comprehensive unit/integration tests running via **Vitest** in a `jsdom` environment:

1. **Page Loads**: Confirms header, academic sub-heading, backend online status badge, and modality navigation.
2. **Image Upload UI**: Tests file drop/selection, object URL preview, caption inputs, and button enabling.
3. **Text Input UI**: Tests textarea input, character counter, and benchmark demo sample chips.
4. **API Success Response**: Validates full rendering of predictions, confidence scores, and reset controls.
5. **API Failure Handling**: Verifies graceful display and dismissal of error banners without raw stack traces.
6. **Governance Panel**: Asserts rendering of `REVIEW_RECOMMENDED`, trigger components, and ethics disclaimers.
7. **Grad-CAM Display**: Verifies overlay image rendering, disclaimer notes, and side-by-side mode toggle.
8. **Token Attribution UI**: Verifies highlighted token pills, saliency percentages, and HateXplain alignment.
9. **Uncertainty UI**: Verifies predictive variance, standard deviation, Shannon entropy, and MC sample badges.
10. **Async Video Polling**: Simulates background job lifecycle (`QUEUED` $\rightarrow$ `PROCESSING` $\rightarrow$ `COMPLETED`).

**Result: 10 / 10 Tests Passing.**

---

## 9. Running Instructions

### 1. Launch FastAPI Backend
```bash
# In repository root:
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Verify endpoints at `http://localhost:8000/docs` and `http://localhost:8000/health`.

### 2. Launch React Frontend
```bash
# In frontend/ directory:
npm install
npm run dev
```
Open browser at `http://localhost:5173`.

### 3. Run Frontend Tests
```bash
npm test
```
