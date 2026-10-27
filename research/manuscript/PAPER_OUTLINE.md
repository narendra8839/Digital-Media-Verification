# Academic Paper Manuscript Outline

**Working Title:**
*A Multimodal Framework for Digital Media Verification: Cross-Modal Explainability, Bayesian Uncertainty, and Empirical Robustness Under Realistic Degradations*

---

## Abstract
- **Context:** The proliferation of synthetic generative media (deepfakes) and coordinated computational propaganda poses existential threats to information integrity. Unimodal detectors fail when encountering composite manipulation across video, audio, and textual streams.
- **Methodology:** We propose the Digital Media Verification (DMV) framework, an end-to-end multimodal pipeline integrating forensic computer vision (EfficientNet-B4), natural language processing (RoBERTa-base for 14 propaganda techniques, BERT-base for hate speech), automated speech recognition (Whisper), and optical character recognition (EasyOCR). The system incorporates spatial Grad-CAM saliency, token attribution gradients, Bayesian epistemic uncertainty via Monte Carlo Dropout ($T=10$), and a zero-inference audit reporting engine.
- **Empirical Results:** On the 700-video FaceForensics++ (C23) test partition, our visual detector achieves 88.00% accuracy and 96.30% fake precision under a 4:1 class imbalance. On SemEval-2020 Task 11, class-balanced focal loss boosts propaganda Macro F1 from 0.4175 to 0.5132 (+0.0957). On HateXplain, BERT achieves 67.41% accuracy.
- **Robustness Findings:** Systematic evaluation under five realistic corruptions reveals extreme vulnerability to spatial low-pass filtering (blur/downsampling drops real recall to 12.0%). While experimental V3 candidates recover Gaussian noise accuracy (70.0%–71.0%), all fail the strict JPEG $Q=30$ fake recall gate (16%–34%), mandating baseline retention and rigorous human-in-the-loop governance.

---

## Section Structure

### 1. Introduction
- 1.1 The Threat Landscape of Synthetic and Manipulated Digital Media
- 1.2 Limitations of Isolated Unimodal Forensics
- 1.3 Contributions of this Work:
  1. Multimodal fusion of visual, audio, OCR, and text pathways.
  2. Integration of visual Grad-CAM and token-level attribution for interpretable verification.
  3. Epistemic uncertainty quantification using Monte Carlo Dropout to drive human triage.
  4. Exhaustive robustness stress-testing across realistic corruptions and candidate iterations.
  5. A production-ready, zero-inference audit reporting architecture.

### 2. Related Work
- 2.1 Deepfake Detection in Facial Videos (CNN backbones, spatial artifacts, temporal models).
- 2.2 Propaganda Technique Identification (SemEval shared tasks, rhetorical taxonomy).
- 2.3 Hate Speech Detection and Multi-Annotator Modeling (HateXplain).
- 2.4 Explainability and Uncertainty Quantification in Safety-Critical AI.
- 2.5 Robustness and Generalization Under Social Media Degradations.

### 3. Multimodal System Architecture
- 3.1 Visual Pathway: Face Detection (MTCNN), EfficientNet-B4 Backbone, and Grad-CAM Saliency.
- 3.2 Audio & Speech Pathway: FFmpeg Demuxing and OpenAI Whisper ASR.
- 3.3 Text & OCR Pathway: EasyOCR and Multilingual Tokenization.
- 3.4 Textual Propaganda Detection: Multi-label RoBERTa-base with Class-Balanced Focal Loss.
- 3.5 Hate Speech Detection: 3-Class BERT-base with Gradient Saliency.
- 3.6 Zero-Inference Multilingual Report Generation Architecture.

### 4. Epistemic Uncertainty & Human-in-the-Loop Governance
- 4.1 Formulating Predictive Variance via Monte Carlo Dropout ($T=10, p=0.4$).
- 4.2 Cross-Modal Conflict Detection (Visual vs. Spoken vs. Banner Discrepancies).
- 4.3 Mandatory Human Review Trigger Conditions and Workflow.

### 5. Experimental Methodology & Benchmark Splits
- 5.1 Dataset Provenance and Strict Split Hygiene (Zero Video/Sequence Leakage).
- 5.2 Video Sampling Strategy: 1.0 FPS Uniform Keyframing (max 4 frames/video).
- 5.3 Metric Definitions: Accuracy, Balanced Accuracy, Macro F1, Class Marginals.
- 5.4 Standardized Corruptions: JPEG Compression, Gaussian Noise, Gaussian Blur, Low-Res Resizing.

### 6. Benchmark Evaluation & Results
- 6.1 Cross-Modal Benchmark Summary (Table 1 / Figure A).
- 6.2 Visual Deepfake Performance and 4:1 Class Imbalance Analysis (Confusion Matrix, Figure C/D).
- 6.3 NLP Propaganda Checkpoint Optimization (14 SemEval Techniques, Figure E).
- 6.4 Hate Speech Classification and Rationale Attributions.

### 7. Robustness Stress-Testing & Candidate Audits
- 7.1 Production Baseline Degradation Curves (Figure B).
- 7.2 The Failure of Candidate V2-B: Catastrophic Gaussian Noise Collapse.
- 7.3 Candidate V3 Series (V3-A, V3-B, V3-C) Formulation and Noise Recovery.
- 7.4 Formal Evaluation Gates and Rejection: The JPEG $Q=30$ Fake Recall Bottleneck.
- 7.5 Stochastic Noise Discrepancy Reconciliation: MT19937 vs PCG64 Realizations.

### 8. Limitations & Ethical Governance
- 8.1 Assistive Decision Support vs. Forensic Proof Engine.
- 8.2 Face Occlusion and Angle Sensitivity.
- 8.3 In-the-Wild Generalization and Novel Generative Architectures (Diffusion/GANs).
- 8.4 Responsible Disclosure and Abuse Prevention.

### 9. Conclusion & Future Directions
- 9.1 Summary of Contributions.
- 9.2 Future Work: Quantitative Calibration (Reliability Diagrams), Temporal Video Transformers.
