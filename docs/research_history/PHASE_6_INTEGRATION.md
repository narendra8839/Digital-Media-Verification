# Phase 6: Final Backend / Model Integration, Inference Pipeline, XAI, UQ, and Governance Report

**Project:** Digital Media Verification  
**Branch:** `swaraj`  
**Date:** October 2026  
**Status:** Completed and Verified  

---

## 1. Executive Summary

Phase 6 executed the final end-to-end integration of the selected Phase 5 model checkpoints into the live FastAPI application service layer (`api/`), pipeline orchestrator (`pipeline/`), and detection subsystems (`detection/`). 

All hardcoded MVP defaults and CPU-only initializations were audited and updated to support:
1. **Dynamic Hardware Acceleration:** Safe auto-detection of NVIDIA CUDA acceleration (`cuda:0`) with verified fallback to `cpu`. Both CUDA inference (WSL2 with RTX 4050 Laptop GPU) and CPU fallback (Windows) were verified via actual API routes.
2. **Selected Phase 5 Checkpoints:** Native loading of the finalized candidate models across all modalities without modification of weights or architecture.
3. **Browser-Accessible Explainability Artifacts:** Grad-CAM saliency maps are written to a mounted static media directory (`/artifacts/`), returning web-consumable URLs and Base64 data URIs for immediate browser rendering.
4. **Epistemic Uncertainty Quantification:** Full Monte Carlo Dropout ($T=20$ stochastic forward passes) across deepfake, propaganda, and hate speech models, with explicit heuristic disclaimers (*"Calibration not quantitatively established"*).
5. **Responsible AI Governance Layer:** Explicit human-in-the-loop review policy adhering to ART (Accountability, Responsibility, Transparency) principles, enforcing `REVIEW_RECOMMENDED` on high uncertainty and `ANALYSIS_INCOMPLETE` on ungrounded media (e.g., faceless inputs).
6. **Graceful Multimodal Degradation:** Robust fallback pipelines ensuring that OCR or ASR failures do not cascade or invalidate visual/textual verification.

---

## 2. Checkpoint Audit & Physical Loading Verification

The application now loads the selected Phase 5 checkpoints:

| Modality | Target Task | Training Architecture | Final Checkpoint Path Loaded | Split Evaluated In Phase 4/5 | Key Metric |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Deepfake** | Facial Manipulation Detection | EfficientNet-B4 (dropout 0.4, 224x224) | `models/deepfake/checkpoint_full/best_model.pt` | Full Test Split ($N=700$ videos: 140 Real, 560 Fake) | Acc: **88.00%**, Macro F1: **0.8321**, Weighted F1: **0.8859**, Real Rec: **86.43%** |
| **Propaganda** | 14-Class Rhetorical Technique Classification | RoBERTa-base (seq_len 128, sqrt-weighted scheduled) | `models/propaganda/optimization/exp1_sqrt_weighted_scheduled/` | Held-out Validation Split ($N=1,448$ spans across 71 articles) | Acc: **65.19%**, Macro F1: **0.5132**, Weighted F1: **0.6477** (Val) |
| **Hate Speech** | 3-Class Toxicity Classification | BERT-base-uncased (seq_len 128) | `models/hate_speech/checkpoint_full/` | Full Test Split ($N=1,924$ posts: 782 Normal, 548 Offensive, 594 Hate Speech) | Acc: **67.41%**, Macro F1: **0.6617**, Weighted F1: **0.6738** |
| **ASR** | Speech-to-Text Transcription | OpenAI Whisper-base | Hugging Face cache (`openai/whisper-base`) | N/A | Character/Word transcript |
| **OCR** | Scene & Overlay Text Extraction | EasyOCR CRAFT + PyTorch CRNN | Local EasyOCR cache (English) | N/A | Bounding boxes + Text string |

*Evaluation Reconciliation Audit:*
1. **Propaganda ($N = 1,448$ spans):** Evaluated strictly against the official article-disjoint held-out validation split (`data/full_splits/propaganda_val.json`). SemEval-2020 Task 11 official test annotations remain unreleased. The previously reported $N = 612$ was a reporting typo referencing an earlier unmerged development slice; the physical manifest on disk contains 1,448 spans across 71 articles (100% article-disjoint from the 286 training articles). The selected Phase 5 Exp 1 checkpoint achieved **Acc = 0.6519**, **Macro F1 = 0.5132**, and **Weighted F1 = 0.6477**.
2. **Hate Speech ($N = 1,924$ posts):** Evaluated strictly on the untouched HateXplain test split (`data/full_splits/hatexplain_test.json`), achieving **Acc = 0.6741**, **Macro F1 = 0.6617**, and **Weighted F1 = 0.6738**. The previously reported $N = 2,878$ was a reporting count error citing the validation partition count, and Macro F1 was typed as 0.6558 instead of the verified test split metric of **0.6617**.
3. **Deepfake ($N = 700$ videos):** Evaluated on the full untouched FaceForensics++ C23 test split (`data/full_splits/deepfake_test.json`: 140 Real, 560 Fake) using uniform deterministic 4-frame sampling, achieving **Acc = 0.8800**, **Macro F1 = 0.8321**, **Weighted F1 = 0.8859**, **Real Precision = 0.6505**, and **Real Recall = 0.8643**.

### Preprocessing & Architecture Verification:
- **Deepfake Classifier:** Matches training dimensions ($224 \times 224 \times 3$), standard ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), label mapping `{"0": "real", "1": "fake"}`. Standard prediction explicitly invokes `model.eval()`.
- **Propaganda Classifier:** Tokenizer loads directly from `models/propaganda/optimization/exp1_sqrt_weighted_scheduled/tokenizer`, matching training vocabulary with max sequence length $128$. Label mapping dynamically loads all 14 rhetorical techniques.
- **Hate Speech Classifier:** Tokenizer loads from `models/hate_speech/checkpoint_full/tokenizer`, matching BERT-base vocabulary with max sequence length $128$. Label mapping: `{"0": "normal", "1": "offensive", "2": "hatespeech"}`.

---

## 3. Device Integration & Hardware Acceleration Audit

Previously, `api/main.py` and `api/dependencies/pipeline_dep.py` hardcoded `device="cpu"`. A safe device resolution utility was introduced in `api/utils/device.py`:

```python
def get_optimal_device() -> str:
    # 1. Check explicit override
    env_device = os.environ.get("DMV_DEVICE", "").strip()
    if env_device:
        return env_device

    # 2. Check CUDA capability and test memory allocation
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda:0")
            del t
            return "cuda:0"
        except Exception:
            return "cpu"
    return "cpu"
```

### Verified Runtime Behavior:
1. **Windows 11 Native Host (CPU Fallback):**
   - Detected Device: `cpu`
   - Model Device: `cpu`
   - Test Status: `Clean CPU fallback confirmed through API path!`
2. **WSL2 Ubuntu Runtime (NVIDIA RTX 4050 Laptop GPU):**
   - Detected Device: `cuda:0`
   - GPU Model: `NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM)`
   - Model Device: `cuda:0` for EfficientNet-B4, RoBERTa, and BERT.
   - VRAM Allocated: `1,642.58 MB`
   - Text Inference Latency: Dropped from `2.267s` (CPU) to `1.439s` (GPU).
   - Test Status: `Genuine CUDA GPU execution confirmed through API path!`

---

## 4. Explainable AI (XAI) & Artifact Serving

### Visual Saliency (Grad-CAM):
- **Target Layer:** Final convolutional stage of EfficientNet-B4 (`model.backbone.features[-1]`).
- **Artifact Serving:** FastAPI now mounts the static directory `/artifacts` via `app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")`.
- **Browser Accessibility:** Grad-CAM generation produces:
  - An image artifact saved on disk: `artifacts/gradcam_{uuid}.png`
  - A browser-accessible relative path: `overlay_url = "/artifacts/gradcam_{uuid}.png"` (verified via HTTP GET 200 OK)
  - A Base64 data URI: `saved_path = "data:image/png;base64,..."` for immediate frontend display without network proxy dependencies.
- **Fault Tolerance:** Grad-CAM invocation is wrapped in defensive try/except handling. If saliency map computation encounters an error, visual prediction and uncertainty are preserved, and an explanation fallback note is returned.

### Textual Attribution (Grad $\times$ Input):
- Token-level gradient attribution maps gradient magnitudes through model word embeddings (`roberta.embeddings.word_embeddings` / `bert.embeddings.word_embeddings`).
- Special tokens (`<s>`, `</s>`, `[CLS]`, `[SEP]`, `<pad>`) are filtered from top ranked influencers.
- Explanations carry the explicit disclaimer: *"Token attribution is a post-hoc explanatory signal and should not be interpreted as causal proof."*

---

## 5. Monte Carlo Dropout Uncertainty Quantification (UQ)

- **Sampling Passes:** Exactly $T = 20$ stochastic forward passes are executed via `enable_mc_dropout(model)`, which sets `nn.Dropout` modules to `.train()` while preserving deterministic `BatchNorm` and `LayerNorm` in `.eval()`.
- **Stochasticity Check:** `is_stochastic` flag is computed by testing whether maximum predictive variance exceeds $10^{-8}$.
- **Statistical Outputs:** Mean probability vector, predictive variance per class, standard deviation per class, entropy ($H(p) = -\sum p_c \log_2 p_c$ bits), and maximum variance.
- **Threshold Policy:**
  - Heuristic Variance Threshold: $\sigma^2 > 0.02$
  - Heuristic Entropy Threshold: $H > 0.85$ bits
- **Mandatory Disclaimer:** The uncertainty engine and verification payloads explicitly return:
  > `"threshold_disclaimer": "Heuristic development threshold — calibration not quantitatively established."`
  > `"disclaimer": "MC Dropout uncertainty is a model uncertainty signal; calibration not quantitatively established."`

---

## 6. Responsible AI Governance & Multi-Component Policy

Governance enforces Human-in-the-Loop decision support aligned with ART principles:

```
[Input Inspection & Modality Routing]
       │
       ├── Faceless Media (Image or Video with 0 detected faces)
       │       └── Decision: ANALYSIS_INCOMPLETE / NO_FACE_DETECTED
       │           (Refrains from ungrounded authenticity predictions)
       │
       ├── Face / Text Present
       │       │
       │       ├── Any Active Modality Variance >= 0.02 OR Entropy >= 0.85 bits
       │       │       └── Decision: REVIEW_RECOMMENDED
       │       │           (Elevated uncertainty mandates human verification;
       │       │            trigger logged in `review_trigger_component`)
       │       │
       │       └── All Active Modalities: Variance < 0.02 AND Entropy < 0.85 bits
       │               └── Decision: CONFIDENT_PREDICTION
       │                   (Model confidence meets baseline criteria)
```

### Governance Hierarchy & Resolution Rules:
1. **Single-Modality Verification (Pure Image / Pure Text):**
   - For image verification without text, governance reflects the visual detector: if a face is detected and $\sigma^2 < 0.02$ with $H < 0.85$ bits, governance returns `CONFIDENT_PREDICTION` with `review_required = False`.
2. **Multimodal Verification (Media + Caption / ASR):**
   - Governance aggregates uncertainty across all evaluated modalities (`deepfake`, `propaganda`, `hate_speech`). If any evaluated component exceeds the uncertainty thresholds, overall governance flags `REVIEW_RECOMMENDED`, logging the specific trigger(s) (e.g., `review_trigger_component = "propaganda"`).
3. **OCR Discourse Sufficiency Rule:**
   - Isolated OCR watermarks, channel logos, or timecodes (e.g. `"NDTV"`, $< 2$ words and $< 10$ characters) without a user caption do not constitute textual discourse for rhetorical propaganda or hate-speech analysis. Filtering isolated watermarks preserves clean visual verification and prevents spurious NLP uncertainty from overriding grounded facial predictions. Full OCR text is preserved with provenance in `text.segments`.

The system strictly acts as an interpretable decision-support platform and never claims to determine objective truth or execute automated punitive censorship.

---

## 7. Video Pipeline & Modality Fallback Audit

### Video Aggregation Architecture:
1. **Sampling:** Uniform deterministic frame sampling at `sample_fps=1.0` (up to `max_frames=16`).
2. **Per-frame Face Analysis:** MTCNN detects face crops; crops are evaluated for deepfake probability $P(\text{fake})$.
3. **Aggregation Methodology:** Video-level prediction is computed as the arithmetic mean of per-frame fake probabilities:
   $$\bar{P}(\text{fake}) = \frac{1}{N} \sum_{i=1}^N P_i(\text{fake})$$
   $$\text{Prediction} = \begin{cases} \text{fake} & \text{if } \bar{P}(\text{fake}) \ge 0.50 \\ \text{real} & \text{otherwise} \end{cases}$$
4. **Representative Frame Saliency:** The single frame exhibiting the highest fake probability (if classified fake) or lowest fake probability (if classified real) is selected for Grad-CAM overlay and $T=20$ MC Dropout estimation, minimizing inference latency.
5. **Keyframe OCR:** Up to 4 sampled frames are processed by EasyOCR; extracted text is deduplicated and merged.
6. **Audio Whisper ASR:** Audio stream is inspected via FFmpeg probe. If audio exists, a 16kHz mono WAV stream is transcribed via Whisper-base.
7. **Modality Failure Handling:**
   - **No Audio:** Handled safely with `has_audio=False` and `status="NO_AUDIO"`. Visual analysis and OCR proceed unaffected.
   - **No Faces:** Returns `NO_FACE_DETECTED` with `ANALYSIS_INCOMPLETE` governance. Any extracted OCR/audio text continues into NLP models.
   - **OCR Failure / Blank Text:** Gracefully bypasses NLP models without error.

---

## 8. Test Suite & Verification Results

### Regression Testing:
All 41 baseline tests continue to pass with 0 regressions.

### New Integration Suite (`tests/test_phase6_integration.py`):
11 new comprehensive integration tests were implemented and passed:
1. `test_device_selection_auto_and_override`: Validates auto-detection and `DMV_DEVICE` environment overrides.
2. `test_checkpoint_defaults_and_eval_mode`: Validates default loading of Phase 5 checkpoints in eval mode.
3. `test_text_verification_success`: Validates NLP text route, RoBERTa 14-class technique, BERT hate speech, token attribution, UQ ($T=20$), and disclaimers.
4. `test_text_empty_and_whitespace`: Validates 422/400 validation error handling.
5. `test_valid_image_with_face_and_artifact_serving`: Validates facial deepfake inference, Grad-CAM generation, and HTTP static artifact accessibility.
6. `test_image_governance_confident_prediction`: Validates that image with low predictive uncertainty ($\sigma^2 < 0.02, H < 0.85$) strictly yields `CONFIDENT_PREDICTION`.
7. `test_no_face_image_handling`: Validates faceless image routing to `ANALYSIS_INCOMPLETE`.
8. `test_video_sync_verification`: Validates synchronous video analysis with silent video handling.
9. `test_video_async_job_lifecycle`: Validates async job submission, polling endpoint, and state transitions.
10. `test_multimodal_text_only`: Validates text-only multimodal routing.
11. `test_multimodal_image_and_caption`: Validates combined image and caption multimodal verification.

**Combined PyTorch Test Suite Result:** `52 passed, 16 warnings in 66.85s` (100% PASS RATE).

---

## 9. End-to-End Live API Smoke Test Results

Executed via `scripts/smoke_test_api.py` with real model checkpoints:

```json
{
  "timestamp": "2026-10-07T12:57:18Z",
  "runtime_device": "cpu",
  "endpoints": {
    "readiness": {
      "status": "ready",
      "models": {
        "deepfake": "EfficientNet-B4 (models/deepfake/checkpoint_full/best_model.pt)",
        "propaganda": "RoBERTa-base (models/propaganda/optimization/exp1_sqrt_weighted_scheduled)",
        "hate_speech": "BERT-base-uncased (models/hate_speech/checkpoint_full)",
        "asr": "Whisper-base (openai/whisper-base)",
        "ocr": "EasyOCR (English CRAFT detector)"
      },
      "pipeline_device": "cpu"
    },
    "text": {
      "status_code": 200,
      "latency_seconds": 2.948,
      "propaganda": { "prediction": "Doubt", "confidence": 0.8614 },
      "hate_speech": { "prediction": "normal", "confidence": 0.6025 },
      "uncertainty": { "mc_samples": 20, "max_variance": 0.004854, "entropy": 0.9969 },
      "governance": { "decision_status": "REVIEW_RECOMMENDED" }
    },
    "image_with_face": {
      "status_code": 200,
      "latency_seconds": 3.731,
      "visual": { "face_detected": true, "prediction": "real", "confidence": 0.8766 },
      "gradcam": {
        "generated": true,
        "overlay_url": "/artifacts/gradcam_7d459c73.png",
        "artifact_http_accessible": true,
        "artifact_status_code": 200
      },
      "uncertainty": { "mc_samples": 20, "max_variance": 0.000204, "entropy": 0.5392 },
      "governance": {
        "decision_status": "CONFIDENT_PREDICTION",
        "review_required": false,
        "review_trigger_component": null
      }
    },
    "image_no_face": {
      "status_code": 200,
      "latency_seconds": 0.365,
      "visual": { "status": "NO_FACE_DETECTED", "face_detected": false, "prediction": null },
      "governance": { "decision_status": "ANALYSIS_INCOMPLETE" }
    },
    "video": {
      "status_code": 200,
      "latency_seconds": 0.384,
      "visual": { "status": "NO_FACE_DETECTED", "frames_analyzed": 0, "total_frames_sampled": 1 },
      "ocr_executed": true,
      "asr_executed": false,
      "has_audio": false,
      "governance": { "decision_status": "ANALYSIS_INCOMPLETE" }
    },
    "multimodal": {
      "status_code": 200,
      "latency_seconds": 4.237,
      "visual_status": "SUCCESS",
      "text_status": "AVAILABLE",
      "propaganda_technique": "Loaded_Language",
      "hate_speech_class": "normal",
      "governance_decision": "REVIEW_RECOMMENDED",
      "governance_trigger_component": "propaganda"
    }
  }
}
```

---

## 10. Frontend Compatibility Verification

The frontend API contract was audited against `frontend/src/services/api.js` and all consuming components:
- `ResultDashboard.jsx`: Correctly displays primary prediction, confidence percentage, execution latency, and governance status badges (`CONFIDENT_PREDICTION`, `REVIEW_RECOMMENDED`, `ANALYSIS_INCOMPLETE`).
- `VisualResultView.jsx`: Consumes `explanation?.saved_path || explanation?.overlay_url`. With Base64 and `/artifacts/...` serving, Grad-CAM heatmaps render immediately. Faceless inputs trigger the `NO_FACE_DETECTED` notice.
- `TokenAttributionView.jsx`: Renders `explanation.token_scores` as soft red highlights proportional to attribution magnitude, with disclaimers displayed.
- `UncertaintyCard.jsx`: Visualizes variance, entropy, and standard deviation bars with heuristic disclaimer notices.
- **Frontend Safety:** No user frontend files (`App.css`, `index.css`, `Header.jsx`, `ResultDashboard.jsx`, `TokenAttributionView.jsx`) were modified or reverted. All existing work is preserved.

---

## 11. Known Limitations & Academic Disclaimers

1. **Uncertainty Calibration:** Heuristic variance ($0.02$) and entropy ($0.85$ bits) thresholds serve as developmental reference markers. Calibration has not been quantitatively established.
2. **In-Memory Job Store:** Asynchronous video jobs use an in-memory Python dictionary tracking store suitable for academic demonstration; jobs do not persist across server restarts.
3. **Facial Boundary:** The deepfake classifier operates exclusively on cropped facial bounding boxes detected by MTCNN. Non-facial manipulations (background synthesis, object insertion) are out of scope.
4. **Language Scope:** OCR and NLP models are constrained to English. Multilingual or cross-lingual verification remains future scope.
