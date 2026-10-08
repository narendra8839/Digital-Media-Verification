# Current Project Context for Gemini

> **Target Purpose**: This document provides an authoritative, complete, and mathematically rigorous summary of the **implemented current state** of the Digital Media Verification project on branch `swaraj`. It is designed to be provided directly to Google Gemini alongside existing slide deck (PPT) content to upgrade presentation slides from preliminary planning/TBD states to the verified, completed MVP implementation.

---

## 1. Project Identity

- **Project Title**: Digital Media Verification: Explainable and Uncertainty-Aware Deepfake, Propaganda and Hate-Speech Detection
- **Alternative Academic Title**: Explainable and Uncertainty-Aware Multimodal Digital Media Verification
- **Course**: Ethical and Responsible AI (ERA)
- **Class / Academic Level**: Third Year (TY) B.Tech, Computer Engineering
- **Academic Year**: 2025–2026
- **Team Members**:
  - **Swaraj Shedge** (Branch lead, Full-Stack Architecture, ML Pipeline & Integration)
  - **Narendra Patil** (Repository Collaborator, Dataset Curation & Model Validation)
- **Course Instructor / Guide**: Faculty Project Guide & Evaluator, Department of Computer Engineering
- **Institution**: Vishwakarma Institute of Technology (VIT Pune), Pune, India
- **Active Git Branch**: `swaraj`
- **Implementation Status**: **MVP COMPLETE & VERIFIED** (All models trained, evaluated, containerized into a FastAPI backend service, connected to a React + Vite frontend, and validated with 51 passing automated tests).

---

## 2. Problem & Research Gap

### The Problem
Digital media ecosystems are overwhelmed by manipulative multimodal misinformation where synthetic imagery, deceptive text, and manipulated speech reinforce one another:
- **Visual Forgeries**: Deepfake faces, reenactments, and identity swaps generated via autoencoders, GANs, and diffusion models erode trust in photographic and video evidence.
- **Embedded Misinformation**: Misleading slogans, manipulated headlines, and toxic claims are embedded directly into images as memes, watermarks, or banners to evade basic textual filters.
- **Acoustic Manipulation**: Synthetic voices, out-of-context speech, or incendiary audio tracks are paired with video footage to distort public perception.
- **Rhetorical Manipulation & Toxicity**: Weaponized rhetoric spans fine-grained propaganda techniques (e.g., *Appeal to Fear*, *Loaded Language*, *Name Calling*) and targeted toxic speech.

### The Research Gap
Existing academic and industrial verification solutions suffer from critical methodological and ethical deficiencies:
1. **Unimodal Silos**: Most systems isolate video, text, or audio, failing to analyze cross-modal content (e.g., verifying a video’s visual integrity while ignoring spoken audio or on-screen banners).
2. **"Black-Box" Opacity**: Conventional deep learning classifiers output scalar probability scores without explaining *why* a piece of media was flagged, hindering auditability.
3. **Overconfidence & Zero Epistemic Telemetry**: Deep neural networks are notoriously overconfident on out-of-distribution or degraded inputs, providing no measure of predictive uncertainty.
4. **Autonomous Moderation Hazards**: Automated censorship, account banning, and content deletion systems risk suppressing legitimate speech, lacking procedural justice and human-in-the-loop oversight.

### Project Contribution vs. Gap
The project resolves these gaps not by building an autonomous censorship bot, but by delivering an **explainable, uncertainty-aware human decision-support system**:
- It inspects **visual**, **on-screen text**, and **spoken audio** modalities in parallel.
- It provides **post-hoc feature attributions** (Grad-CAM for face crops, Gradient $\times$ Input for text tokens, and HateXplain rationale alignment).
- It quantifies **epistemic model uncertainty** using Monte Carlo Dropout ($T=20$) across all models.
- It establishes a **Responsible AI governance layer** that flags ambiguous or low-confidence media as `REVIEW_RECOMMENDED` or `ANALYSIS_INCOMPLETE` for human forensic evaluators.

---

## 3. Final MVP Contribution

### A. One-Sentence Contribution
> An explainable, uncertainty-aware multimodal decision-support system that detects facial deepfakes, propaganda techniques, and hate speech across images, videos, and text using fine-tuned transformer and convolutional backbones, Monte Carlo Dropout, post-hoc saliency attributions, and a React-FastAPI forensic workstation.

### B. 3–5 Sentence Technical Contribution
The system ingests images, videos, and raw text through an automated media router. Visual frames are processed through MTCNN and fine-tuned EfficientNet-B4, while on-screen text and speech are extracted via EasyOCR and OpenAI Whisper-base to drive fine-tuned RoBERTa-base (14 propaganda classes) and BERT-base (3 toxicity classes). Every inference step executes 20 Monte Carlo Dropout stochastic forward passes to compute predictive variance and Shannon entropy alongside Grad-CAM and Gradient $\times$ Input token attributions. If predictive dispersion exceeds development thresholds or required facial features are absent, the governance layer automatically triages the case as `REVIEW_RECOMMENDED` or `ANALYSIS_INCOMPLETE` for human review. The entire system is exposed via a validated FastAPI REST service and an interactive, dark-mode React 19 + Vite 8 analyst dashboard.

### C. Bullet List of Implemented Contributions
- **Multimodal Pipeline**: Parallel visual forensics, optical character recognition (OCR), automatic speech recognition (ASR), and natural language processing (NLP).
- **Facial Deepfake Detection**: Pretrained ImageNet EfficientNet-B4 fine-tuned on FaceForensics++ C23 with MTCNN face cropping.
- **Propaganda Rhetoric Detection**: Pretrained RoBERTa-base fine-tuned on SemEval-2020 Task 11 spans across 14 rhetorical categories.
- **Hate Speech & Toxicity Detection**: Pretrained BERT-base-uncased fine-tuned on the HateXplain benchmark across 3 classes.
- **Embedded Text & Speech Extraction**: EasyOCR keyframe extraction and OpenAI Whisper-base (16 kHz) speech transcription.
- **Visual Explainability (XAI)**: Grad-CAM saliency heatmaps generated on EfficientNet-B4 final convolutional activation maps ($7 \times 7 \rightarrow 224 \times 224$).
- **Textual Explainability (XAI)**: Gradient $\times$ Input token attribution ($S_i = \| \nabla_{e_i} y^c \odot e_i \|_2$) with subword reconstruction and HateXplain human rationale precision/recall/F1 scoring.
- **Epistemic Uncertainty Quantification (UQ)**: Monte Carlo Dropout ($T=20$) computing predictive mean, predictive variance, standard deviation, and Shannon entropy.
- **Responsible AI Governance Layer**: Automated triage classifying results as `CONFIDENT_PREDICTION`, `REVIEW_RECOMMENDED`, or `ANALYSIS_INCOMPLETE` (strictly non-punitive, decision-support only).
- **Production-Grade Backend Service**: Modular FastAPI REST backend featuring chunked streaming, file limits, deterministic scratch cleanup, JSON sanitization, and asynchronous video jobs.
- **Forensic Analyst Frontend**: Responsive React 19 + Vite 8 dashboard featuring interactive Grad-CAM side-by-side viewers, token saliency pills, uncertainty telemetry cards, and backend health monitoring.

---

## 4. Current Architecture

The implemented architecture is a modular, decoupled pipeline connecting media ingestion, multi-branch analysis, explainability, uncertainty quantification, governance, backend APIs, and the frontend workstation.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RAW USER INPUT (Image / Video / Text)                 │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      MEDIA ROUTER (pipeline/router.py)                          │
│        Inspects file headers, formats, dimensions; routes without full memory load│
└──────────────┬─────────────────────────┼─────────────────────────┬──────────────┘
               │ (Image)                 │ (Video)                 │ (Text)
               ▼                         ▼                         │
┌──────────────────────────────┐ ┌──────────────────────────────┐  │
│        IMAGE PIPELINE        │ │        VIDEO PIPELINE        │  │
│  - MTCNN Face Detection      │ │  - 1 FPS Frame Sampling      │  │
│  - EasyOCR on Image          │ │  - MTCNN Per-frame Crops     │  │
└──────────────┬───────────────┘ │  - EasyOCR on Keyframes      │  │
               │                 │  - FFmpeg Audio Extraction   │  │
               │                 │  - Whisper-base 16kHz ASR    │  │
               │                 └──────────────┬───────────────┘  │
               ▼                                ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             FEATURE EXTRACTION & MODELS                         │
│  ┌───────────────────────┐ ┌───────────────────────────┐ ┌───────────────────┐  │
│  │    EfficientNet-B4    │ │       RoBERTa-base        │ │ BERT-base-uncased │  │
│  │   Deepfake Detection  │ │  14 Propaganda Classes    │ │   3 Hate Classes  │  │
│  └───────────┬───────────┘ └─────────────┬─────────────┘ └─────────┬─────────┘  │
└──────────────┼───────────────────────────┼─────────────────────────┼────────────┘
               ▼                           ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         EXPLAINABLE AI (XAI) LAYER                              │
│  - Spatial Grad-CAM (7×7 -> 224×224) on final EfficientNet-B4 conv layer        │
│  - Gradient × Input Token Attribution on RoBERTa & BERT embeddings              │
│  - HateXplain Human Rationale Alignment Scoring (Precision / Recall / F1)       │
│  - Mandatory Disclaimer: Post-hoc explanatory signal, not causal proof           │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    UNCERTAINTY QUANTIFICATION (UQ) LAYER                        │
│  - Monte Carlo Dropout (T = 20 stochastic forward passes with active dropout)   │
│  - Predictive Mean, Variance (σ²), Standard Deviation (σ), Shannon Entropy (H) │
│  - Component-specific isolation (never cross-averaged across modalities)        │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   RESPONSIBLE AI GOVERNANCE & DECISION LAYER                    │
│  - Evaluates epistemic thresholds: Variance >= 0.02 OR Entropy >= 0.85          │
│  - Flags: CONFIDENT_PREDICTION vs REVIEW_RECOMMENDED vs ANALYSIS_INCOMPLETE    │
│  - Strict Non-Punitive Mandate: Decision support only; no auto-bans or deletion │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND REST SERVICE (Port 8000)                    │
│  - Endpoints: /health, /ready, /verify/text, /verify/image, /verify/video       │
│  - Shared pre-warmed Pipeline Singleton, Chunked Streaming, In-memory Async Queue│
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ (HTTP REST / JSON / FormData)
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    REACT 19 + VITE 8 FRONTEND (Port 5173)                       │
│  - Modality Workspaces, Interactive Grad-CAM Heatmap Viewer, Token Highlighting │
│  - Uncertainty Telemetry Cards, Live Backend Health Indicator, Audit Telemetry  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Characteristics
- **FastAPI + React Decoupling**: Replaces legacy monolithic/Streamlit scripts with an asynchronous REST API backend and a single-page application (SPA) client.
- **Parallel Modality Routing**: Computer vision, OCR, speech recognition, and language analysis execute along independent, decoupled pathways.
- **Resource Pre-warming**: The pipeline singleton loads model weights into memory once during FastAPI lifespan startup, preventing per-request weight reload overhead.
- **Graceful Fault Tolerance**: If a video lacks audio, Whisper is cleanly skipped (`NO_AUDIO`); if an image has no face, visual classification is bypassed (`NO_FACE_DETECTED` $\rightarrow$ `ANALYSIS_INCOMPLETE`); if media lacks text, NLP is cleanly bypassed (`NO_TEXT_AVAILABLE`).

---

## 5. Implemented Models

| Model Component | Target Purpose | Base Architecture | Training / Pretrained Dataset | Task / Taxonomy | Output Format | Explainability Method | Uncertainty Method |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Deepfake Detector** | Facial manipulation detection | `EfficientNet-B4` (`torchvision`) | ImageNet-1k init $\rightarrow$ Fine-tuned on FaceForensics++ C23 subset | Binary classification: `real` vs `fake` | Softmax probability vector $[P(\text{real}), P(\text{fake})]$ | Spatial Grad-CAM on final conv layer (`features[-1]`, $7 \times 7$) | MC Dropout ($T=20$), predictive variance & entropy |
| **Propaganda Detector** | Fine-grained rhetorical technique classification | `RoBERTa-base` (`transformers`) | Pretrained base $\rightarrow$ Fine-tuned on SemEval-2020 Task 11 spans | 14-class technique classification | 14-class probability distribution | Gradient $\times$ Input token attribution with subword cleanup | MC Dropout ($T=20$), predictive variance & entropy |
| **Hate Speech Detector** | Toxic content & hate speech classification | `BERT-base-uncased` (`transformers`) | Pretrained base $\rightarrow$ Fine-tuned on HateXplain benchmark | 3-class classification: `normal`, `offensive`, `hatespeech` | 3-class probability distribution | Gradient $\times$ Input token attribution & HateXplain rationale alignment | MC Dropout ($T=20$), predictive variance & entropy |
| **Speech Recognizer (ASR)** | Video speech-to-text transcription | `openai/whisper-base` (74M parameters) | Pretrained Whisper audio-language model (zero-shot ASR) | Automatic Speech Recognition (16 kHz mono) | Transcribed text string with timing | N/A (Transcription utility) | Container audio stream pre-probing |
| **Text Detector (OCR)** | Embedded on-screen text extraction | `EasyOCR` (CRAFT text detection + ResNet-BiLSTM-CTC) | Pretrained English OCR models | Scene & overlay text extraction | Extracted strings, bounding boxes, confidence | Character-level coordinate bounding | Confidence score filtering |
| **Face Detector** | Face localization & alignment | `MTCNN` (`facenet-pytorch`: P-Net, R-Net, O-Net) | Pretrained multi-task cascaded CNN | Face bounding box & landmark detection | $224 \times 224$ aligned RGB face crop | Landmark alignment | Landmark confidence score |

> [!NOTE]
> `Whisper-base`, `EasyOCR`, and `MTCNN` are utilized with verified pretrained weights as robust sensory extraction modules. `EfficientNet-B4`, `RoBERTa-base`, and `BERT-base-uncased` were specifically fine-tuned on task-specific research datasets.

---

## 6. Datasets

### 1. FaceForensics++ C23 (Facial Deepfakes)
- **Provenance & Academic Status**: Sourced from a public Kaggle development mirror (`xdxd003/ff-c23`) containing 3,000 verified videos across official splits. **Strictly labeled as a development/MVP engineering dataset**; official authorized TUM institutional access is being pursued.
- **Manipulation Methods Included**: 4 standard manipulation techniques: *Deepfakes*, *Face2Face*, *FaceSwap*, and *NeuralTextures* (compressed at C23 quality). Excluded: *FaceShifter* and *DeepFakeDetection*.
- **Official Metadata Splits**: Mapped strictly against official `splits/train.json`, `val.json`, and `test.json`.
- **MVP Subset Size**:
  - **Train**: 40 videos (20 Real, 20 Fake across the 4 methods; 160 extracted frames).
  - **Validation**: 18 videos (10 Real, 8 Fake; 72 extracted frames).
  - **Test**: 18 videos (10 Real, 8 Fake; 72 extracted frames).
  - **Total**: 76 videos.
- **Leakage Prevention**: Enforces strict sequence-disjoint splitting. Video ID overlap between Train, Val, and Test is exactly **0**. Frames from the same video identity never appear across multiple splits.

### 2. SemEval-2020 Task 11 (Propaganda Techniques)
- **Task**: Granular identification of rhetorical and propaganda devices.
- **Definition of One Sample**: An annotated propaganda text span extracted from news articles, provided with surrounding sentence `context`, character offsets, and `article_id`.
- **Supported Classes (14 Techniques)**:
  1. *Appeal to Authority*
  2. *Appeal to Fear-Prejudice*
  3. *Bandwagon, Reductio ad Hitlerum*
  4. *Black-and-White Fallacy*
  5. *Causal Oversimplification*
  6. *Doubt*
  7. *Exaggeration, Minimisation*
  8. *Flag-Waving*
  9. *Loaded Language*
  10. *Name Calling, Labeling*
  11. *Repetition*
  12. *Slogans*
  13. *Thought-terminating Cliches*
  14. *Whataboutism, Straw Men, Red Herring*
- **MVP Subset Size**: 700 samples (560 train, 140 validation).
- **Class Balance**: Perfectly balanced with exactly 40 train samples and 10 validation samples per technique.
- **Leakage Prevention**: Strict article-level isolation (285 unique articles in train, 72 in val). Overlap between train and val articles is exactly **0**.

### 3. HateXplain Benchmark (Hate Speech & Explainability)
- **Task**: 3-class toxicity classification paired with word-level human rationale annotations.
- **Classes**: `normal`, `offensive`, `hatespeech`.
- **MVP Subset Size**: 900 samples (600 train, 150 validation, 150 test).
- **Class Balance**: Exactly 200 train, 50 val, and 50 test samples per class (1:1:1 balanced distribution).
- **Rationale Ground Truth**: 100% of selected `hatespeech` and `offensive` samples preserve human-annotated binary rationale token masks matching sequence tokens.
- **Leakage Prevention**: Strictly adheres to official benchmark post divisions (`post_id_divisions.json`). Zero cross-split contamination.

---

## 7. Training Details

All MVP fine-tuning was executed under a fixed random seed (`42`) to guarantee reproducibility.

```
┌────────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
│ Hyperparameter / Setup │ Deepfake Detector    │ Propaganda Detector  │ Hate Speech Detector │
├────────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Architecture           │ EfficientNet-B4      │ RoBERTa-base         │ BERT-base-uncased    │
│ Pretrained Base        │ ImageNet-1k          │ RoBERTa Base LM      │ BERT Base Uncased LM │
│ Classification Head    │ Dropout(0.4) -> FC(2)│ Linear(768, 14)      │ Linear(768, 3)       │
│ Epochs                 │ 2                    │ 2                    │ 2                    │
│ Batch Size             │ 8                    │ 16                   │ 16                   │
│ Optimizer              │ AdamW                │ AdamW                │ AdamW                │
│ Learning Rate          │ 1e-4                 │ 3e-5                 │ 2e-5                 │
│ Weight Decay           │ 1e-2                 │ 0.01                 │ 0.01                 │
│ Loss Function          │ CrossEntropyLoss     │ CrossEntropyLoss     │ CrossEntropyLoss     │
│ Max Sequence Length    │ N/A (224×224 image)  │ 128 tokens           │ 128 tokens           │
│ Training Subset Size   │ 160 frames (40 vids) │ 560 text spans       │ 600 social posts     │
│ Random Seed            │ 42                   │ 42                   │ 42                   │
└────────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘
```

> [!IMPORTANT]
> **Training Status Disclaimer**: These models represent the **MVP development training configuration**, designed to prove end-to-end pipeline operability, loss convergence, gradient propagation, XAI extraction, and uncertainty estimation. They are not final hyperparameter-swept benchmark models.

---

## 8. Actual MVP Results

All reported metrics are verified empirical figures directly extracted from the validated test runs and reports on disk (`reports/mvp_model_validation.md`).

### A. Deepfake Detection Results (`EfficientNet-B4`)
- **Training Set (40 videos / 160 frames)**:
  - Frame-Level Accuracy: **95.63%** | Macro F1: **0.9562**
  - Video-Level Accuracy: **97.50%** (39/40 correct) | Macro F1: **0.9750**
  - Video Confusion Matrix: `[[19 Real, 1 Fake], [0 Real, 20 Fake]]`
- **Validation Set (18 unseen videos / 72 frames)**:
  - Frame-Level Accuracy: **54.17%** | Macro F1: **0.5394**
  - Video-Level Accuracy: **55.56%** (10/18 correct) | Macro F1: **0.5500**
  - Video Confusion Matrix: `[[4 Real, 6 Fake], [2 Real, 6 Fake]]`
- **Test Set (18 unseen videos / 72 frames)**:
  - Frame-Level Accuracy: **58.33%** | Macro F1: **0.5781**
  - Video-Level Accuracy: **50.00%** (9/18 correct) | Macro F1: **0.4857**
  - Video Confusion Matrix: `[[6 Real, 4 Fake], [5 Real, 3 Fake]]`

#### Analysis of Weak Deepfake Generalization (Must Be Preserved in PPT)
The drop from 97.50% training accuracy to 50.00% test video accuracy is an authentic, unmanipulated finding that demonstrates scientific integrity:
1. **Model Capacity vs. Sample Size**: An 19-million parameter network trained on only 160 images (representing only 40 source identities) quickly fits facial identity features rather than learning generalizable compression and manipulation artifacts.
2. **Subject Disjointness**: Because the validation and test sets enforce strict subject-disjoint isolation, the model cannot rely on memorized facial identities.
3. **Role of Uncertainty**: Crucially, this weak generalization reinforces the necessity of the project's core thesis: **because deepfake models struggle to generalize from limited data, the system must produce uncertainty telemetry and escalate ambiguous cases to human reviewers.**

### B. Propaganda Detection Results (`RoBERTa-base`)
- **Training Set (560 balanced samples across 14 classes)**:
  - Accuracy: **47.68%** | Macro F1: **0.4619**
  - *Context*: Significantly exceeds the random guessing baseline of **7.14%** ($1/14$) across 14 fine-grained classes after only 2 epochs.
- **Validation Set (140 unseen samples from disjoint articles)**:
  - Accuracy: **29.29%** | Macro F1: **0.2481** | Weighted F1: **0.2481**

### C. Hate Speech Detection Results (`BERT-base-uncased`)
- **Training Set (600 balanced samples across 3 classes)**:
  - Accuracy: **81.33%** | Macro F1: **0.8067**
  - *Context*: Substantially outperforms the random 3-class baseline of **33.33%**.
- **Validation Set (150 unseen benchmark samples)**:
  - Accuracy: **60.67%** | Macro F1: **0.5942** | Weighted F1: **0.5942**
- **Test Set (150 unseen benchmark samples)**:
  - Accuracy: **60.00%** | Macro F1: **0.5773** | Weighted F1: **0.5773**

---

## 9. Explainability (XAI)

The system implements dual-modality post-hoc explainability, providing human-interpretable evidence for every visual and textual prediction.

### Visual Saliency: Grad-CAM (Deepfakes)
- **Target Architecture**: `EfficientNet-B4` final convolutional feature extractor (`model.backbone.features[-1]`).
- **Feature Map Dimensions**: $C = 1792$ channels, spatial grid $7 \times 7$ (for $224 \times 224$ face crops).
- **Algorithm**:
  1. Computes pre-softmax class logit gradient: $\frac{\partial y^c}{\partial A^k}$.
  2. Applies Global Average Pooling to derive channel importance weights $\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$.
  3. Computes rectified linear combination: $L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$.
  4. Upsamples $7 \times 7$ map to $224 \times 224$ via bilinear interpolation and normalizes to $[0, 1]$.
  5. Blends JET colormap with the original facial crop at $\alpha = 0.5$.

### Textual Attribution: Gradient $\times$ Input (NLP)
- **Target Architectures**: `RoBERTa-base` (Propaganda) and `BERT-base-uncased` (Hate Speech).
- **Mathematical Formulation**: Measures the sensitivity of the predicted class logit $y^c$ with respect to the input word embedding vector $e_i \in \mathbb{R}^D$:
  $$S_i = \left\| \nabla_{e_i} y^c \odot e_i \right\|_2$$
- **Subword Reconstruction**: Cleans tokenizer subword artifacts (`##` WordPiece prefixes for BERT, `Ġ` byte-level BPE markers for RoBERTa) to render clean, readable token saliency chips.

### HateXplain Human Rationale Alignment
- Evaluates whether top model-attributed tokens match word-level ground truth identified by human annotators:
  - **Rationale Precision**: $\frac{|\mathcal{M} \cap \mathcal{G}|}{|\mathcal{M}|}$
  - **Rationale Recall**: $\frac{|\mathcal{M} \cap \mathcal{G}|}{|\mathcal{G}|}$
  - **Rationale F1**: $\frac{2 \cdot P_{\text{rationale}} \cdot R_{\text{rationale}}}{P_{\text{rationale}} + R_{\text{rationale}}}$
- **Empirical Demonstration**: On test sample `"do not be aggressive says the nigger mandela fuck that"` (GT: `hatespeech`), top attributed tokens matched human annotators (`nigger`), achieving **Rationale Recall = 1.0000** and **Rationale F1 = 0.6667**.

### Mandatory Responsible AI Disclaimer
Every visual and textual explanation in the UI and API includes the verbatim disclaimer:
> *"Grad-CAM and token attribution are post-hoc explanatory signals and should not be interpreted as causal proof."*

---

## 10. Uncertainty Quantification (UQ)

The system rejects single-pass softmax probabilities as measures of confidence, implementing Bayesian epistemic uncertainty estimation via **Monte Carlo (MC) Dropout**.

### Mathematical Formulation
- **Stochastic Passes**: $T = 20$ forward passes per input sample.
- **Selective Layer Activation**: Dropout layers (`nn.Dropout`) are forced into active training mode (`m.train()`), while normalization layers (`BatchNorm2d`, `LayerNorm`) remain in evaluation mode (`eval()`) with frozen running statistics.
- **Metrics Computed**:
  1. **Predictive Mean Probability**: $\bar{p}_c = \frac{1}{T} \sum_{t=1}^T p_c^{(t)}$
  2. **Predictive Variance (Epistemic Dispersion)**: $\sigma_c^2 = \frac{1}{T} \sum_{t=1}^T \left(p_c^{(t)} - \bar{p}_c\right)^2$
  3. **Predictive Standard Deviation**: $\sigma_c = \sqrt{\sigma_c^2}$
  4. **Shannon Predictive Entropy**: $H(\bar{p}) = -\sum_{c=1}^C \bar{p}_c \log_2(\bar{p}_c + \epsilon)$

### Thresholding & Review Trigger
- Configured in `config/config.yaml`:
  - `variance_threshold`: **0.02**
  - `entropy_threshold`: **0.85 bits**
- If $\max(\sigma^2) \ge 0.02$ OR $H \ge 0.85$, the decision layer flags `review_required = true` and `decision_status = "REVIEW_RECOMMENDED"`.
- **Modality Isolation**: Uncertainty is strictly evaluated per-detector (`uncertainty.deepfake`, `uncertainty.propaganda`, `uncertainty.hate_speech`) and is **never averaged across modalities**.

> [!IMPORTANT]
> **Calibration Status**: These thresholds are **development/MVP engineering baselines** chosen to demonstrate the triage workflow. They are **not formally calibrated** deployment thresholds (e.g., via temperature scaling or isotonic regression).

---

## 11. Responsible AI & Governance Safeguards

The project enforces strict Responsible AI commitments across its architecture, algorithms, and interface:

1. **Decision Support, Not Autonomous Moderation**: The system never executes automated content removal, account banning, shadowbanning, or algorithmic censorship. It operates strictly as an investigative assistant for human reviewers.
2. **Refusal of Definitive Truth Claims**: Predictions are expressed as likelihoods and dispersion metrics. The system nowhere asserts that an asset is "definitely fake" or "definitely authentic".
3. **The Faceless Media Policy (`ANALYSIS_INCOMPLETE`)**:
   - *Previous Defect*: Faceless images bypassed visual models, but because no error occurred, the system defaulted to `CONFIDENT_PREDICTION`, misleading users into believing the media was verified real.
   - *Implemented Fix*: When MTCNN finds no face, the system sets `visual.status = "NO_FACE_DETECTED"` and `governance.decision_status = "ANALYSIS_INCOMPLETE"`, displaying a safe neutral notice that visual verification could not be performed.
4. **Source Provenance Isolation**: Extracted text preserves strict provenance labels (`OCR`, `ASR`, `USER_TEXT`), preventing evaluators from confusing on-screen banner text with spoken words or user claims.
5. **Non-Binary Transparent Justifications**: Every flagged prediction provides mathematical justifications (e.g., *"Predictive variance (0.0412) exceeds threshold (0.0200)"*).
6. **Audit Trail Telemetry**: All responses record model versions, checkpoint paths, timestamp, execution latency, and error logs.

---

## 12. Multimodal Pipeline

The core orchestration engine (`pipeline/multimodal_pipeline.py`) unifies media processing across three distinct workflows:

```
IMAGE WORKFLOW:
Image Input ──► MediaRouter ──► ImagePipeline
                                  ├── MTCNN Face Crop ──► EfficientNet-B4 ──► Grad-CAM ──► MC Dropout (T=20)
                                  └── EasyOCR (Full Image) ──► TextAggregator ──► NLP Pipeline (RoBERTa + BERT)

VIDEO WORKFLOW:
Video Input ──► MediaRouter ──► VideoPipeline
                                  ├── 1 FPS Sampling (OpenCV) ──► MTCNN ──► EfficientNet-B4 (Per Frame)
                                  │     └── Mean Fake Probability ──► Representative Frame Grad-CAM & UQ
                                  ├── Keyframe Sampling (up to 4 frames) ──► EasyOCR ──► TextAggregator
                                  └── Audio Stream Probe (ffprobe)
                                        ├── Has Audio: True ──► FFmpeg (16kHz WAV) ──► Whisper-base ASR ──► TextAggregator
                                        └── Has Audio: False ──► Skip ASR (status: NO_AUDIO)
                                                                      │
                                                                      ▼
                                                              NLP Pipeline (RoBERTa + BERT)

TEXT WORKFLOW:
Direct Text Input ──► MediaRouter ──► TextPipeline ──► RoBERTa-base (Propaganda) + BERT-base (Hate Speech)
                                                         ├── Gradient × Input Token Attribution
                                                         └── MC Dropout (T=20) Epistemic UQ
```

### Key Multimodal Principle
The current MVP implements **parallel independent modality analysis and textual aggregation**. It extracts signals from visual, speech, and OCR channels independently and aggregates text for NLP models. It does **NOT** use cross-modal transformer fusion or joint multi-modal embeddings.

---

## 13. FastAPI Backend

The service layer is implemented with **FastAPI** on Python 3.14 / Uvicorn, serving as a high-performance REST API.

### Endpoints
- `GET /health`: Instant liveness check (`{"status": "ok"}`).
- `GET /ready`: Model readiness check verifying that checkpoints and pipeline detectors are loaded in memory.
- `POST /verify/text`: Analyzes text for propaganda techniques and hate speech (`application/json`).
- `POST /verify/image`: Ingests an image file and optional caption for deepfake detection, OCR, and NLP (`multipart/form-data`).
- `POST /verify/video`: Ingests video for frame sampling, deepfake analysis, OCR, and Whisper ASR. Supports synchronous execution or asynchronous background processing (`async_mode=true`).
- `GET /verify/status/{job_id}`: Polls the status and retrieves the result of an asynchronous video job.
- `POST /verify/multimodal`: Unified endpoint accepting any combination of media file and text claim.

### Backend Implementation Highlights
- **Pre-warmed Singleton Lifecycle**: Models are initialized once in memory during FastAPI startup lifespan, eliminating latency spikes.
- **Chunked File Streaming**: Uploaded media is streamed in 64 KB chunks directly to temporary files on disk, enforcing upload caps (15 MB for images, 100 MB for videos) to prevent out-of-memory crashes.
- **Guaranteed Ephemeral Cleanup**: Temporary media files are cleaned up in `finally` blocks immediately after processing.
- **Recursive JSON Sanitization**: Resolves PyTorch/NumPy serialization errors (`np.float32`, `np.bool_`, `NaN`, `Inf`) via `sanitize_for_json()`.
- **Asynchronous Task Architecture**: Offloads long video runs to FastAPI `BackgroundTasks`.
- **Known Backend Limitation**: The background job store uses an in-memory dictionary (`job_store`) and does not persist across server restarts.

---

## 14. React Frontend

The user interface is a single-page application built with **React 19** and **Vite 8**, running on `http://localhost:5173`.

### Workstation Interface Features
- **Design Philosophy**: Polished, academic dark-mode layout built with Vanilla CSS, responsive CSS grid/flexbox, `Inter` typography, and `JetBrains Mono` code telemetry.
- **4 Modality Workspaces**: Seamless switching between Image, Video, Text, and Unified Multimodal input panels.
- **Interactive Visual Saliency**: Renders Grad-CAM overlays on detected facial crops with interactive toggle controls (`Overlay` vs. `Side-by-Side` comparison with original face).
- **Token Attribution Display**: Displays interactive text saliency chips with intensity heat highlighting and HateXplain human rationale alignment metrics.
- **Uncertainty Telemetry Cards**: Renders predictive variance, standard deviation, Shannon entropy, and class probability distribution bars.
- **Human Oversight Governance Panel**: Prominently displays decision status badges (`CONFIDENT_PREDICTION`, `REVIEW_RECOMMENDED`, `ANALYSIS_INCOMPLETE`), identifying triggering components and ethics justifications.
- **Extracted Text Provenance**: Visualizes extracted text broken down by origin (`EasyOCR`, `Whisper ASR`, `User Caption`).
- **Async Video Progress Polling**: Non-blocking polling hook (`useVerification`) tracking queued video tasks.

> [!IMPORTANT]
> **Streamlit Replacement**: Any reference to Streamlit in older project planning documents is **completely obsolete**. The user interface is 100% implemented in React 19 + Vite 8.

---

## 15. Testing & Validation

The entire system has been validated across unit, integration, regression, and end-to-end performance benchmarks:

### Automated Test Suites
1. **Backend Pytest Suite** (`python -m pytest tests -v`):
   - `tests/test_api.py`: **19 / 19 PASSED**
   - `tests/test_multimodal_pipeline.py`: **11 / 11 PASSED**
   - `tests/test_explainability_uncertainty.py`: **11 / 11 PASSED**
   - **Total Backend**: **41 / 41 PASSED (100%)** in 62.59 seconds.
2. **Frontend Vitest Suite** (`cd frontend && npm test`):
   - 10 integration tests in `jsdom` testing page loads, file uploads, text verification, error handling, governance panels, Grad-CAM toggles, token pills, uncertainty cards, and async polling.
   - **Total Frontend**: **10 / 10 PASSED (100%)** in 2.27 seconds.
3. **Frontend Production Build** (`cd frontend && npm run build`):
   - Successfully compiled into optimized static assets in **2.58 seconds** with zero warnings.

### Representative Latency Benchmarks (Standard CPU Inference)
- `GET /health` / `GET /ready`: **2–4 milliseconds**
- Direct Text Verification (RoBERTa + BERT + Token Attribution + UQ $T=20$): **2.34 seconds**
- Image Verification (MTCNN + EfficientNet-B4 + Grad-CAM + EasyOCR + UQ $T=20$): **6.13 seconds**
- Video Verification (1 frame sampled + OCR + Deepfake UQ): **0.34 seconds** (sync) / **0.29 seconds** (async queue response)

---

## 16. Current Limitations

A comprehensive, factually honest overview of current system boundaries:

1. **Tiny MVP Training Subsets**: Models were trained on small development subsets (160 deepfake frames, 560 propaganda spans, 600 hate-speech posts) over 2 epochs to rapidly validate architecture.
2. **Deepfake Generalization Bottleneck**: The deepfake model achieved 97.5% training accuracy but only 50.00% test video accuracy on unseen subjects, reflecting facial identity memorization rather than robust artifact detection.
3. **Unofficial Kaggle Development Data**: FaceForensics++ data was sourced from an unofficial Kaggle development mirror rather than an authorized TUM institutional download.
4. **CPU Latency with Whisper-base & MC Dropout**: Performing 20 stochastic forward passes and Whisper-base audio decoding on standard CPU hardware requires 2–6 seconds per sample.
5. **In-Memory Background Job Store**: The asynchronous video task queue is stored in Python RAM and does not persist across server restarts.
6. **Face-Centric Visual Analysis**: Deepfake detection is strictly facial; scene-wide manipulations, background inpainting, or faceless media cannot be visually verified.
7. **English Language Scope**: OCR, Whisper ASR, RoBERTa, and BERT are configured exclusively for English-language digital media.
8. **Independent Parallel Analysis (No Cross-Modal Fusion)**: Visual, audio, and textual streams are evaluated in parallel rather than through a unified multimodal transformer with cross-attention.
9. **No Video Temporal Modeling**: Deepfakes are scored across sampled static frames without recurrent or 3D convolutional temporal coherence modeling.
10. **Uncalibrated Uncertainty Thresholds**: The variance (0.02) and entropy (0.85) thresholds are development defaults, not empirically calibrated against operational risk curves.

---

## 17. Outdated PPT Claims That Must Be Updated

When updating the presentation slide deck, Gemini must correct the following specific items:

| Slide Topic / Claim in Old PPT | Old Outdated State | Updated Implemented State (Must Use) |
| :--- | :--- | :--- |
| **Frontend / GUI** | "Planned Streamlit dashboard / GUI" | **Implemented React 19 + Vite 8 workstation** communicating with FastAPI via REST. |
| **Project Status** | "Upcoming development / Proposed architecture" | **Completed MVP implementation**; all pipelines, models, and endpoints are operational. |
| **Evaluation & Results** | "Results TBD / Future evaluation phase" | **Actual verified empirical results** (Deepfake: 97.5% train / 50.0% test; Propaganda: 47.7% train / 29.3% val; Hate Speech: 81.3% train / 60.0% test). |
| **Deepfake Model Generalization** | Overstated high accuracy or no test metrics | **Explicitly acknowledge weak test generalization (50.00%)** as an authentic finding justifying uncertainty quantification and human review. |
| **Audio ASR Model** | "Whisper-tiny" (if cited in preliminary notes) | **Upgraded to `openai/whisper-base`** (~290 MB weights) to meet project specification. |
| **Faceless Media Handling** | Unspecified or assumed real | Explicitly documented as **`NO_FACE_DETECTED` $\rightarrow$ `ANALYSIS_INCOMPLETE`** to uphold Responsible AI. |
| **Backend Architecture** | Script-based or monolithic execution | **FastAPI REST API service** with shared singleton pre-warming, async task queue, and structured JSON contracts. |
| **System Governance Goal** | "Automated fake news detection / censorship" | **Human-in-the-loop decision-support telemetry**; explicitly non-punitive. |
| **Multimodal Fusion** | "Deep cross-modal neural fusion" (if proposed) | **Parallel modality inspection with textual aggregation**; cross-modal transformer fusion remains future scope. |

---

## 18. What Should Remain Future Scope

The following items are **not implemented** in the current MVP and must be presented strictly as future research opportunities:

- **Scaling to Full Datasets**: Fine-tuning across the full 1.8-million frame FaceForensics++ benchmark, the complete SemEval corpus, and the full HateXplain dataset over extended training epochs.
- **Institutional Data Access**: Securing official authorized TUM academic access for FaceForensics++.
- **Advanced Temporal Modeling**: Integrating 3D-CNNs, Video Transformers, or Lip-Sync inconsistency detection (e.g., Wav2Lip sync error analysis) across video frames.
- **Cross-Modal Transformer Fusion**: Joint cross-attention embeddings (e.g., aligning visual facial expressions directly with vocal emotion and text sentiment).
- **Formal Uncertainty Calibration**: Calibrating MC Dropout probabilities via temperature scaling, Platt scaling, or conformal prediction against human reviewer validation benchmarks.
- **Production Infrastructure**: Replacing the in-memory job store with distributed Redis + Celery task queues, PostgreSQL result persistence, and Docker containerization.
- **Multilingual Support**: Extending OCR, ASR, and NLP models to regional and non-English languages.

---

## 19. FACTS GEMINI MUST USE

This concise, fact-dense section can be copied directly into prompts for Google Gemini to guide slide text generation:

```yaml
PROJECT_IDENTITY:
  title: "Digital Media Verification: Explainable and Uncertainty-Aware Deepfake, Propaganda and Hate-Speech Detection"
  short_title: "Explainable & Uncertainty-Aware Digital Media Verification"
  course: "Ethical and Responsible AI (ERA)"
  class: "Third Year (TY) B.Tech, Computer Engineering"
  academic_year: "2025–2026"
  institution: "Vishwakarma Institute of Technology (VIT Pune)"
  team: ["Swaraj Shedge", "Narendra Patil"]
  status: "MVP Complete and Verified"

CORE_MODELS:
  deepfake: "EfficientNet-B4 (fine-tuned on FaceForensics++ C23 subset, ImageNet-1k init)"
  propaganda: "RoBERTa-base (fine-tuned on SemEval-2020 Task 11, 14 classes)"
  hate_speech: "BERT-base-uncased (fine-tuned on HateXplain, 3 classes: normal, offensive, hatespeech)"
  asr: "openai/whisper-base (16 kHz mono speech-to-text)"
  ocr: "EasyOCR (CRAFT + ResNet-BiLSTM-CTC)"
  face_detection: "MTCNN (facenet-pytorch, aligned 224x224 crops)"

DATASETS_AND_SUBSETS:
  faceforensics_c23:
    provenance: "Development mirror (xdxd003/ff-c23); official TUM access pending"
    manipulations: ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    train: "40 videos (20 real, 20 fake; 160 frames)"
    val: "18 videos (10 real, 8 fake; 72 frames)"
    test: "18 videos (10 real, 8 fake; 72 frames)"
    leakage_prevention: "Subject-disjoint sequence splits (0 ID overlap)"
  semeval2020_task11:
    classes: 14
    train: "560 spans (40 per class, 285 articles)"
    val: "140 spans (10 per class, 72 articles)"
    leakage_prevention: "Strict article-level isolation (0 article overlap)"
  hatexplain:
    classes: 3
    train: "600 posts (200 normal, 200 offensive, 200 hatespeech)"
    val: "150 posts (50 per class)"
    test: "150 posts (50 per class)"
    rationales: "100% human-annotated ground truth masks retained"

TRAINING_SETUP:
  epochs: 2
  batch_size_cv: 8
  batch_size_nlp: 16
  optimizer: "AdamW"
  learning_rate_deepfake: 1e-4
  learning_rate_propaganda: 3e-5
  learning_rate_hate_speech: 2e-5
  seed: 42

EMPIRICAL_MVP_RESULTS:
  deepfake:
    train_video_acc: "97.50% (Macro F1: 0.9750)"
    train_frame_acc: "95.63% (Macro F1: 0.9562)"
    val_video_acc: "55.56% (Macro F1: 0.5500)"
    val_frame_acc: "54.17% (Macro F1: 0.5394)"
    test_video_acc: "50.00% (Macro F1: 0.4857)"
    test_frame_acc: "58.33% (Macro F1: 0.5781)"
    test_confusion_matrix: "[[6 Real, 4 Fake], [5 Real, 3 Fake]]"
    note: "Weak test generalization (50.0%) is an honest finding due to small MVP subset and validates the necessity of uncertainty-aware human triage."
  propaganda:
    train_acc: "47.68% (Macro F1: 0.4619)"
    val_acc: "29.29% (Macro F1: 0.2481)"
    baseline_random: "7.14%"
  hate_speech:
    train_acc: "81.33% (Macro F1: 0.8067)"
    val_acc: "60.67% (Macro F1: 0.5942)"
    test_acc: "60.00% (Macro F1: 0.5773)"
    baseline_random: "33.33%"

EXPLAINABILITY_AND_UNCERTAINTY:
  grad_cam: "EfficientNet-B4 features[-1], 7x7 activation maps -> 224x224 JET heatmap overlays"
  token_attribution: "Gradient x Input dot product norm on RoBERTa and BERT embeddings"
  rationale_alignment: "HateXplain token overlap, Precision, Recall, and F1 (F1 = 0.6667 on test sample)"
  xai_disclaimer: "Grad-CAM and token attribution are post-hoc explanatory signals and should not be interpreted as causal proof."
  mc_dropout: "T = 20 stochastic passes with active dropout; calculates mean, variance, std dev, and Shannon entropy"
  uncertainty_thresholds: "Variance >= 0.02 OR Shannon Entropy >= 0.85 bits (development/MVP thresholds, not formally calibrated)"

GOVERNANCE_AND_RESPONSIBLE_AI:
  triage_statuses: ["CONFIDENT_PREDICTION", "REVIEW_RECOMMENDED", "ANALYSIS_INCOMPLETE"]
  faceless_media_rule: "NO_FACE_DETECTED -> ANALYSIS_INCOMPLETE (system strictly refuses to make an ungrounded guess)"
  safeguards: "Human decision support only; zero automated bans, deletions, or censorship"

SYSTEM_AND_STACK:
  backend: "FastAPI, Python 3.14, Uvicorn (Port 8000)"
  frontend: "React 19, Vite 8, Vanilla CSS (Port 5173)"
  key_endpoints: ["GET /health", "GET /ready", "POST /verify/text", "POST /verify/image", "POST /verify/video", "GET /verify/status/{job_id}", "POST /verify/multimodal"]
  tests_passing:
    backend_pytest: "41 / 41 passed"
    frontend_vitest: "10 / 10 passed"
    total_tests: "51 / 51 passed"
    build_status: "Production build successful (2.58s)"
```
