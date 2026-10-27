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
- **Downloadable Forensic Report**: Exports verified image analyses into professional, self-contained PDF reports incorporating original/Grad-CAM overlays, prediction statistics, UQ dispersion, OCR extractions, and responsible AI governance disclaimers without rerunning model inference.

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

---

## Project Structure

```
Digital-Media-Verification/
├── api/                       # FastAPI application, routes, schemas, and utils
│   ├── routes/                # Verification, health, and PDF report endpoints
│   ├── schemas/               # Strongly-typed Pydantic request/response schemas
│   └── utils/                 # Artifact manager, report generator, and file helpers
├── config/                    # Configuration and threshold definitions
├── data/                      # Manifests and dataset split documentation
├── detection/                 # Model detector wrappers (Deepfake, NLP, Audio)
├── docs/                      # Technical guides, setup, limitations, and viva guides
│   ├── ARTIFACT_MANAGEMENT.md # Artifact retention lifecycle and safety guarantees
│   ├── DEMO_GUIDE.md          # Step-by-step viva presentation guide
│   ├── PROJECT_LIMITATIONS.md # Known failure cases and boundary constraints
│   ├── RESPONSIBLE_AI.md      # Ethics, governance, and disclaimer policies
│   ├── SETUP.md               # Environment installation guide
│   └── research_history/      # Preserved historical phase reports and audit documents
├── explainability/            # Grad-CAM and token attribution implementations
├── extraction/                # Audio demuxing, Whisper ASR, EasyOCR, and text aggregation
├── frontend/                  # React + Vite analyst interface
├── governance/                # Decision policies, human triage rules, and audit logger
├── models/                    # Model architecture definitions and README
│   └── README.md              # Checkpoint download and placement guide
├── pipeline/                  # Modular image, video, text, and multimodal pipelines
├── preprocessing/             # Face detection (MTCNN), frame extraction, normalization
├── reports/                   # Technical validation summaries and Grad-CAM figures
├── research/                  # Central scientific hub: paper-ready materials & reproducibility
│   ├── README.md              # Master research index and scientific summary
│   ├── architecture/          # End-to-end multimodal design & system specifications
│   ├── methodology/           # Datasets, split audits, evaluation protocols, corruption sweeps
│   ├── results/               # Benchmark summaries, authoritative V3 robustness report, raw data
│   │   ├── tables/            # Publication-ready CSV and Markdown data tables
│   │   └── data/              # Authoritative per-video evaluation outputs (JSON)
│   ├── figures/               # Publication figures (Vector SVG and 300 DPI PNG)
│   ├── reproducibility/       # Deterministic reproduction commands, environments, checkpoint manifest
│   └── manuscript/            # Manuscript outline, formal figure captions, BibTeX references
├── scripts/                   # Evaluation runners, figure generation, and verification tools
├── tests/                     # Comprehensive test suites (121 passed, 1 skipped)
├── uncertainty/               # Monte Carlo Dropout and Bayesian variance engine
├── requirements.txt           # Python backend dependencies
└── requirements-dev.txt       # Testing and development dependencies
```

---

## Setup and Installation

### 1. Backend Setup
```bash
# Clone the repository
git clone https://github.com/narendra8839/Digital-Media-Verification.git
cd Digital-Media-Verification

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows (PowerShell): .\venv\Scripts\Activate.ps1

# Install core runtime dependencies (includes reportlab for zero-inference PDF export)
pip install -r requirements.txt

# Install testing and development dependencies
pip install -r requirements-dev.txt
```

### 2. Model Checkpoints
- The production EfficientNet-B4 checkpoint is expected at `models/deepfake/checkpoint_full/best_model.pt`.
- Verify cryptographic integrity via SHA-256 hash:
  - Expected: `5878ae38bf5239dd9889e010fafa1d8c7da4198ac260542dc96681ede1e63234`.
- Detailed checksums, architecture manifests, and external provisioning steps are documented in [`research/reproducibility/CHECKPOINT_MANIFEST.md`](research/reproducibility/CHECKPOINT_MANIFEST.md) and [`models/README.md`](models/README.md).

### 3. Frontend Setup
> **Prerequisite:** **Node.js 18+ / 20+ / 22+ LTS** (compatible with Vite and frontend build dependencies).

```bash
cd frontend
npm install
cd ..
```

*(For detailed prerequisites, CUDA acceleration, and environment variables, see [`docs/SETUP.md`](docs/SETUP.md) and [`research/reproducibility/ENVIRONMENT_AND_DEPENDENCIES.md`](research/reproducibility/ENVIRONMENT_AND_DEPENDENCIES.md)).*

---

## Running the Application

### 1. Start FastAPI Backend
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Base URL: `http://localhost:8000`
- Interactive OpenAPI / Swagger Documentation: `http://localhost:8000/docs`
- Health Liveness Probe: `http://localhost:8000/health`
- Readiness Probe (Model Pre-warm): `http://localhost:8000/ready`

### 2. Start React Frontend
```bash
cd frontend
npm run dev
```
- Analyst Dashboard: `http://localhost:5173`

---

## Running the Verification Test Suites

```bash
# Run full backend test suite (121 passed, 1 skipped)
pytest tests/ -v

# Run frontend component and integration tests (19 passed)
cd frontend && npm test -- --run

# Verify clean production build bundle
cd frontend && npm run build
```

---

## Limitations

- **Not an Infallible Forensic Engine:** Outputs represent probabilistic statistical likelihoods; the detector is designed as an assistive triage aid, not a source of legal proof.
- **Robustness Under Extreme Degradation:** While clean accuracy reaches 88.00% on the FaceForensics++ C23 test split, heavy blurring or low-resolution downsampling ($0.25\times$) collapses real-class recall to 12.00%, as detailed in [`research/results/DEEPFAKE_ROBUSTNESS_V3_REPORT.md`](research/results/DEEPFAKE_ROBUSTNESS_V3_REPORT.md).
- **Facial Crop Dependency:** Relies on MTCNN face detection; non-frontal profiles, heavy occlusions, or faces smaller than $60\times 60$ pixels trigger human review flags.
- **Cross-Generator Generalization:** Detectors trained on GAN and face-swap manipulations may exhibit performance degradation against novel diffusion video architectures.

---

## Research Hub and Scientific Reproducibility

Comprehensive research artifacts, empirical results, and paper preparation materials are organized under the root [`research/`](research/) directory:

| Research Directory | Contents & Documentation |
| :--- | :--- |
| **[`research/README.md`](research/README.md)** | Master scientific overview and navigation index |
| **[`research/architecture/`](research/architecture/)** | [System Architecture](research/architecture/SYSTEM_ARCHITECTURE.md), [Service Design](research/architecture/SYSTEM_DESIGN.md), [Pipeline Design](research/architecture/MODEL_AND_PIPELINE_DESIGN.md) |
| **[`research/methodology/`](research/methodology/)** | [Datasets & Splits](research/methodology/DATASETS_AND_SPLITS.md), [Evaluation Protocol](research/methodology/EVALUATION_PROTOCOL.md), [Robustness Experiments](research/methodology/ROBUSTNESS_EXPERIMENTS.md) |
| **[`research/results/`](research/results/)** | [Model Evaluation Summary](research/results/MODEL_EVALUATION_SUMMARY.md), [V3 Robustness Report](research/results/DEEPFAKE_ROBUSTNESS_V3_REPORT.md), [V2 Report](research/results/DEEPFAKE_ROBUSTNESS_V2_REPORT.md), [Tables](research/results/tables/), [Data](research/results/data/) |
| **[`research/figures/`](research/figures/)** | [Figure Index](research/figures/README.md), [Vector SVGs](research/figures/svg/), [300 DPI PNGs](research/figures/png/) |
| **[`research/reproducibility/`](research/reproducibility/)** | [Reproducing Results](research/reproducibility/REPRODUCING_RESULTS.md), [Dependencies](research/reproducibility/ENVIRONMENT_AND_DEPENDENCIES.md), [Checkpoint Manifest](research/reproducibility/CHECKPOINT_MANIFEST.md) |
| **[`research/manuscript/`](research/manuscript/)** | [Paper Outline](research/manuscript/PAPER_OUTLINE.md), [Figure Captions](research/manuscript/FIGURE_CAPTIONS.md), [BibTeX References](research/manuscript/REFERENCES.md) |
| **[`docs/research_history/`](docs/research_history/)** | Preserved phase-by-phase implementation logs, baseline benchmarks, and audit sign-offs |
