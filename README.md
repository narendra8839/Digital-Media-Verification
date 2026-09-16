# Digital Media Verification

An explainable, uncertainty-aware decision-support system for detecting likely deepfakes, propaganda techniques, and hate speech across images, videos, and text.

## Intended workflow

1. Route an uploaded image, video, or caption to the appropriate pipeline.
2. Extract faces, frames, on-screen text, and speech as applicable.
3. Run deepfake, propaganda, and hate-speech classifiers.
4. Produce Grad-CAM and text-attribution explanations.
5. Estimate epistemic uncertainty with Monte Carlo dropout.
6. Escalate inconclusive results for human review and audit logging.

## Architecture & Stack

- **Backend**: FastAPI (Python 3.14 / Uvicorn)
  - EfficientNet-B4 (Deepfake detection)
  - RoBERTa-base (SemEval-2020 Propaganda technique detection)
  - BERT-base-uncased (HateXplain hate speech classification)
  - OpenAI Whisper-base (16 kHz Speech Recognition ASR)
  - EasyOCR (Optical Character Recognition)
  - Grad-CAM & Gradient × Input Token Attribution
  - Monte Carlo Dropout ($T = 20$) Epistemic Uncertainty Quantification
  - Human Oversight & Responsible AI Governance Layer
- **Frontend**: React 19 + Vite 8
  - Desktop-optimized verification workstation
  - Modality workspaces: Image, Video, Text, Multimodal
  - Interactive Grad-CAM and Token Attribution viewers
  - Real-time backend health monitor and async video polling

## Quick start

### 1. Backend Service
```powershell
pip install -r requirements.txt
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation available at: `http://localhost:8000/docs`

### 2. Frontend Interface
```powershell
cd frontend
npm install
npm run dev
```
Verification Dashboard available at: `http://localhost:5173`

### 3. Testing
```powershell
# Backend pytest suite (41 tests):
python -m pytest tests -v

# Frontend Vitest suite (10 tests):
cd frontend && npm test
```

## Datasets, Provenance & Academic Scope

- **FaceForensics++ (FF++ c23)**:
  - *Engineering MVP Dataset*: 3,000 video subset verified against official metadata.
  - *Provenance Note*: Sourced from a public development mirror (`xdxd003/ff-c23`) strictly for local engineering and architecture verification. Official authorized institutional access is being pursued. Results are labeled as MVP/development baselines.
  - *Leakage Prevention*: Video-level, subject-disjoint train/val/test splits strictly enforced.
- **SemEval-2020 Task 11**:
  - Fine-grained 14-class propaganda and rhetorical technique detection using `RoBERTa-base`.
- **HateXplain Benchmark**:
  - 3-class toxicity classification (`normal`, `offensive`, `hatespeech`) with word-level human rationale alignment using `BERT-base-uncased`.
- **Speech Recognition (ASR)**:
  - `openai/whisper-base` processing 16 kHz audio streams.
- **Epistemic Uncertainty Quantification**:
  - Monte Carlo Dropout ($T = 20$) stochastic sampling for predictive variance, standard deviation, and Shannon entropy.
- **Responsible AI Mandate**:
  - Decision support telemetry with automated human review escalation (`REVIEW_RECOMMENDED` and `ANALYSIS_INCOMPLETE`); no automated punitive actions or physical truth claims.

Model checkpoints and datasets are deliberately excluded from version control. Place them in the matching `checkpoint` and `data/datasets` folders.
