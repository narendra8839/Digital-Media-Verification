# Digital Media Verification

An explainable, uncertainty-aware decision-support framework for detecting visual deepfakes, textual propaganda techniques, and online hate speech across image, video, and text modalities.

---

## Overview

Digital Media Verification (DMV) is an academic decision-support platform designed to assist content analysts, fact-checkers, and platform evaluators in assessing manipulated digital content. Rather than operating as an automated censorship or truth-detection engine, DMV provides multi-faceted forensic evidence:
- Visual facial manipulation detection using deep convolutional neural networks.
- Multi-label rhetorical propaganda technique classification via fine-tuned language representation models.
- Targeted hate speech and offensive content detection.
- Multimodal extraction integrating automatic speech recognition (Whisper) and optical character recognition (EasyOCR).
- Explainable AI (XAI) featuring Grad-CAM visual heatmaps and gradient-based token attribution.
- Epistemic uncertainty quantification via Monte Carlo Dropout paired with human-in-the-loop governance triage.

---

## Features

- **Multimodal Ingestion**: Supports single-frame images (`.jpg`, `.png`), video containers (`.mp4`, `.avi`, `.mov`), and raw text inputs.
- **Visual Tampering Analysis**: Evaluates face manipulation using MTCNN face extraction and an EfficientNet-B4 backbone trained on FaceForensics++ (C23).
- **Speech & Text Extraction**: Audio tracks are extracted via FFmpeg and transcribed using OpenAI Whisper-base; on-screen text is extracted via EasyOCR.
- **Rhetorical Propaganda Classification**: Identifies persuasion techniques based on the SemEval-2020 Task 11 taxonomy using a class-weighted RoBERTa-base classifier.
- **Hate Speech Detection**: Detects abusive language and protected-group targeting using a fine-tuned BERT-base model based on HateXplain.
- **Visual Explainability (Grad-CAM)**: Generates gradient-weighted class activation heatmaps overlaying manipulated facial boundaries.
- **Linguistic Attribution**: Highlights influential tokens and phrases driving NLP classifications.
- **Uncertainty Quantification (UQ)**: Monte Carlo Dropout ($T=20$ stochastic passes) produces predictive variance and Shannon entropy.
- **Governance & Human-in-the-Loop Policies**: Triages outputs into `CONFIDENT_PREDICTION`, `REVIEW_RECOMMENDED`, or `ANALYSIS_INCOMPLETE` fallback states.

---

## System Architecture

```text
                        Uploaded Media
              (Image / Video / Audio / Text)
                            |
             +--------------+--------------+
             |                             |
      Visual Stream                  Audio / Text Stream
             |                             |
     MTCNN Face Detector             FFmpeg Audio Demux
             |                             |
    EfficientNet-B4                 Whisper ASR / EasyOCR
     (Binary Real/Fake)                    |
             |                      Combined Transcript
             |                             |
             |              +--------------+--------------+
             |              |                             |
             |        RoBERTa-base                    BERT-base
             |     (Propaganda Techniques)          (Hate Speech)
             |              |                             |
             +--------------+--------------+--------------+
                            |
                   Multimodal Fusion
                            |
            +---------------+---------------+
            |                               |
     Explainability             Uncertainty Quantification
    (Grad-CAM / Attrib)             (MC Dropout, T=20)
            |                               |
            +---------------+---------------+
                            |
                     Governance Engine
            (CONFIDENT / REVIEW / INCOMPLETE)
                            |
                Analyst Dashboard (React UI)
```

---

## Models and Datasets

| Modality / Task | Backbone Architecture | Dataset | Training Protocol / Split Details |
| :--- | :--- | :--- | :--- |
| **Visual Deepfake** | EfficientNet-B4 | FaceForensics++ (C23) | 720 train / 140 val / 140 test video sequences; frame-balanced extraction |
| **Propaganda Detection** | RoBERTa-base | SemEval-2020 Task 11 | Article-disjoint corpus across 14 rhetorical techniques; class-weighted loss |
| **Hate Speech** | BERT-base-uncased | HateXplain | 15,383 train / 1,922 val / 1,924 test posts (3-class: normal, offensive, hatespeech) |
| **Speech-to-Text** | Whisper-base | Pre-trained (OpenAI) | 16 kHz mono acoustic feature extraction; English transcript generation |
| **Text Extraction** | EasyOCR | Pre-trained | Keyframe detection and character recognition |

---

## Results

Model evaluations were conducted using frozen checkpoints on held-out evaluation splits:

### 1. Visual Deepfake Detection (FaceForensics++ C23)
- **Evaluation Set**: $N = 700$ test videos (frame-aggregated)
- **Accuracy**: 0.8800
- **Macro F1**: 0.8321
- **Weighted F1**: 0.8859

### 2. Propaganda Technique Detection (SemEval-2020 Task 11)
- **Evaluation Set**: $N = 1,448$ article-disjoint validation spans
- **Accuracy**: 0.6519
- **Macro F1**: 0.5132
- **Weighted F1**: 0.6477

> **Dataset Split Note:** Official SemEval-2020 Task 11 test set ground-truth labels were not publicly released by competition organizers. Consequently, the reported propaganda metrics are calculated on an untouched, article-disjoint validation set and are explicitly designated as validation metrics, not test metrics.

### 3. Hate Speech Detection (HateXplain)
- **Evaluation Set**: $N = 1,924$ untouched test posts
- **Accuracy**: 0.6741
- **Macro F1**: 0.6617
- **Weighted F1**: 0.6738

---

## Explainability

DMV integrates post-hoc explainability mechanisms directly tied to model inference computations:

- **Visual Explanations (Grad-CAM)**:
  Computed on the final convolutional feature maps of EfficientNet-B4. Visual overlays localize blend seams, unnatural boundaries, and texture discontinuities.
- **Text Explanations (Token Attribution)**:
  Computed via gradient $\times$ input embeddings over RoBERTa and BERT representations. Identifies salient words directly impacting technique or toxicity classifications.

---

## Uncertainty and Governance

To avoid uncalibrated overconfidence, the platform executes Monte Carlo Dropout ($T=20$ forward stochastic passes):
- **Predictive Variance ($\sigma^2$)**: Quantifies stability across dropout variations.
- **Predictive Entropy ($H$)**: Shannon entropy of the mean predictive probability distribution.

### Governance Triage Rules
- **`CONFIDENT_PREDICTION`**: Predictive entropy $< 0.85$ bits and variance $< 0.02$. The output is displayed directly with supporting explanations.
- **`REVIEW_RECOMMENDED`**: Predictive entropy $\ge 0.85$ bits or variance $\ge 0.02$, or conflicting multi-class indicators. The dossier is flagged for human review.
- **`ANALYSIS_INCOMPLETE`**: Triggered when necessary evidence is absent (e.g., no face detected by MTCNN in an uploaded image or video).

---

## Responsible AI and Disclaimers

The DMV platform adheres to key responsible AI design constraints:
1. **Decision Support Only**: This tool is an investigative aid for human evaluators. It does not issue binding takedown orders or automated account bans.
2. **No Truth Detection Claim**: The platform analyzes observable tampering indicators and text patterns; it does not adjudicate objective factual accuracy.
3. **Epistemic Meaning of Uncertainty**: Predictive entropy reflects model ambiguity over its training manifold; it is not a calibrated probability of error.
4. **Post-Hoc Interpretability**: Explanations indicate which features influenced model decision-making; they do not represent absolute forensic proofs.
5. **Fail-Safe Fallbacks**: If no face is detected in an uploaded visual file, the system marks the analysis as `ANALYSIS_INCOMPLETE` (`NO_FACE_DETECTED`) rather than producing an ungrounded classification.

*(For detailed ethical guidelines, refer to [`docs/RESPONSIBLE_AI.md`](docs/RESPONSIBLE_AI.md)).*

---

## Curated Demo Assets (Cross-Dataset Observation)

For viva and presentation demonstrations, a set of video clips was curated to observe cross-dataset behavior:
- **Baseline Authentic Video**: FakeAVCeleb authentic celebrity interview clip (`01_real_normal.mp4`).
- **External Deepfake Video**: FakeAVCeleb FaceSwap clip (`02_external_deepfake_normal.mp4`), demonstrating out-of-domain generalization for the EfficientNet-B4 visual model.
- **Controlled Multimodal Assets**: Curated samples combining visual deepfake sequences with controlled benchmark speech (`03_external_deepfake_propaganda.mp4`, `04_external_deepfake_hate.mp4`, `05_real_propaganda.mp4`).

> **Important Note:** These video clips are curated presentation assets intended solely to illustrate pipeline mechanics and multimodal disentanglement during live demonstrations. They do not constitute benchmark evaluation results.

*(Complete asset provenance and audit data are documented in [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md)).*

---

## Project Structure

```
Digital-Media-Verification/
├── api/                       # FastAPI application, routes, schemas, and dependencies
│   ├── dependencies/          # Pipeline dependency singletons
│   ├── routes/                # Verification and telemetry endpoints
│   ├── schemas/               # Pydantic request and response schemas
│   └── utils/                 # File and serialization utilities
├── config/                    # Configuration and threshold definitions
├── data/                      # Manifests and dataset split documentation
│   ├── full_splits/           # Split manifests and summary data
│   └── mvp/                   # MVP verification manifests
├── detection/                 # Model detector wrappers
├── docs/                      # Technical documentation and guides
│   ├── ARCHITECTURE.md        # Detailed system architecture
│   ├── DEMO_GUIDE.md          # Step-by-step viva presentation guide
│   ├── RESPONSIBLE_AI.md      # Ethics, limitations, and governance policy
│   ├── SETUP.md               # Environment installation guide
│   ├── TRAINING_AND_EVALUATION.md # Benchmark evaluation protocol
│   └── research_history/      # Historical phase reports and audit documents
├── evaluation/                # Evaluation, calibration, and metrics scripts
├── explainability/            # Grad-CAM and token attribution implementations
├── extraction/                # Audio demuxing, Whisper ASR, EasyOCR, and text aggregation
├── frontend/                  # React + Vite analyst interface
├── governance/                # Decision policies and audit logger
├── models/                    # Model configurations, training routines, and README
│   └── README.md              # Checkpoint download and placement guide
├── pipeline/                  # Modular image, video, text, and multimodal pipelines
├── preprocessing/             # Face detection, frame extraction, and normalization
├── reports/                   # Technical validation summaries and Grad-CAM figures
├── scripts/                   # Evaluation runners, demo pack builders, and verification tools
├── tests/                     # Unit and integration test suite
├── uncertainty/               # Monte Carlo Dropout and variance calculation
├── .gitignore                 # Git ignore specifications
├── package.json               # Root frontend/monorepo tooling configuration (if used)
├── README.md                  # Project overview and technical summary
└── requirements.txt           # Python backend dependencies
```

---

## Setup and Installation

### 1. Backend Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Checkpoints
Large model binaries are distributed separately via project GitHub Releases. Refer to [`models/README.md`](models/README.md) for expected directory paths and placement instructions.

### 3. Frontend Setup
```bash
cd frontend
npm install
```

*(For detailed prerequisites, CUDA GPU setup, and configuration steps, see [`docs/SETUP.md`](docs/SETUP.md)).*

---

## Running the Application

### 1. Start FastAPI Backend
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Base URL: `http://localhost:8000`
- Swagger Documentation: `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/health`

### 2. Start React Frontend
```bash
cd frontend
npm run dev
```
- Analyst Dashboard: `http://localhost:5173`

---

## Running the Test Suite

```bash
# Run backend test suite (52 tests)
pytest tests/ -v

# Run frontend unit tests (12 tests)
cd frontend && npm test
```

---

## Limitations

- **Cross-Generator Generalization**: Deepfake visual detectors trained on specific facial synthesis pipelines (FaceForensics++) may experience performance degradation against novel diffusion or generative video architectures.
- **Audio-Visual Asynchrony**: The current visual pipeline does not perform dense cross-modal biometric synchronization (e.g., active lip-sync phoneme tracking).
- **Rhetorical Subjectivity**: Multi-label propaganda classification involves linguistic nuance where short phrases out of context can be ambiguous.
- **Inference Latency**: Comprehensive multimodal inference with Monte Carlo Dropout ($T=20$ passes) and keyframe extraction requires GPU acceleration for sub-10s video response times.

---

## Research Notes and Reproducibility

Phase-by-phase implementation logs, baseline benchmarking records, optimization histories, and audit sign-offs are archived in [`docs/research_history/`](docs/research_history/).
