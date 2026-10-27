# Project Limitations — Digital Media Verification System

**Document Version:** 1.0.0  
**Scope:** Verified Academic Evaluation, Implementation Constraints, and Operational Bounds  
**Target Audience:** Academic Evaluators, Research Paper Reviewers, and B.Tech Presentation Committee  
**Repository:** [https://github.com/narendra8839/Digital-Media-Verification](https://github.com/narendra8839/Digital-Media-Verification)

---

## Executive Summary

The **Digital Media Verification (DMV)** platform is an explainable, uncertainty-aware decision-support workstation designed for human-in-the-loop forensic analysis of visual and textual media. In accordance with responsible AI engineering principles, this document establishes a rigorous boundary between **empirically measured findings**, **architectural implementation constraints**, and **untested generalization hypotheses**.

The platform is strictly a **decision-support tool**; it does not claim to function as an autonomous arbiter of truth, nor should its automated predictions be used for unsupervised punitive moderation (such as automated account termination or content deletion).

---

## 1. Measured Empirical Limitations

All numerical findings in this section are directly traced to verified evaluation runs (`phase4_results.json`, `phase4_robustness.json`, and audited Phase 4/5 reports).

### 1.1 Sensitivity to Real-World Visual Distortions
*Source: `phase4_robustness.json`, `scripts/phase4_robustness.py`*

The deepfake classifier (EfficientNet-B4 trained on FaceForensics++ C23) was evaluated against five real-world visual degradation conditions on a balanced test subset ($N=100$ videos: 50 Real, 50 Fake, 4 frames/video uniform sampling):

| Input Condition | Accuracy | Macro F1 | $\Delta$ Acc vs. Baseline | Real Recall | Fake Recall | Primary Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Original (Unperturbed)** | **81.00%** | **0.8098** | — | **0.7800** | **0.8400** | Balanced baseline operating point |
| **JPEG Compression ($Q=30$)** | 73.00% | 0.7278 | -8.00% | 0.8200 | 0.6400 | Block boundary artifacts degrade fake recall |
| **Gaussian Noise ($\sigma=15$)** | 67.00% | 0.6624 | -14.00% | 0.5200 | 0.8200 | Additive high-frequency noise disrupts textures |
| **Gaussian Blur ($r=2$)** | 56.00% | 0.4544 | -25.00% | **0.1200** | **1.0000** | Real recall collapses; severe false alarm rate |
| **Low-Resolution Resize ($1/4\times$)** | 56.00% | 0.4544 | -25.00% | **0.1200** | **1.0000** | Real recall collapses; severe false alarm rate |

#### Empirical Finding
High-frequency spatial features—such as blending boundaries, subtle skin pore discrepancies, and compression seams—are eliminated by spatial blurring and low-resolution downsampling. Because the convolutional feature extractor associates missing high-frequency facial detail with synthetic artifacts, the classifier defaults almost entirely to predicting "Fake" (Fake recall = 100.0%, Real recall = 12.0%). Consequently, **uncompressed or blurred authentic videos suffer an 88% false alarm rate**, necessitating mandatory human oversight for low-quality or heavily compressed media.

---

### 1.2 Impact of Test Split Class Imbalance
*Source: `phase4_results.json`, `models/deepfake/metrics.json`*

The full FaceForensics++ C23 test split comprises $N=700$ videos with a severe 4:1 class imbalance: **140 Real videos (20.0%)** versus **560 Fake videos (80.0%)**.

```text
Full Test Split Confusion Matrix (N = 700):
                    Predicted Fake    Predicted Real      Class Recall
True Fake (560)          495                65               88.39%
True Real (140)           19               121               86.43%
Class Precision         96.30%            65.05%
```

- **Fake Detection Performance:** Precision = 96.30%, Recall = 88.39%, F1 = 0.9218 ($N=560$)
- **Real Verification Performance:** Precision = 65.05%, Recall = 86.43%, F1 = 0.7423 ($N=140$)
- **Aggregate Metrics:** Accuracy = 88.00%, Macro F1 = 0.8321, Weighted F1 = 0.8859

#### Empirical Finding
Although class recall on authentic videos is respectable (86.43%), the class precision of 65.05% reveals that **34.95% of all videos classified by the model as "Real" are actually misclassified deepfakes** (65 false negatives). In high-stakes forensic investigations, aggregate accuracy (88.00%) masks this asymmetric precision deficit.

---

### 1.3 Propaganda Model Reported on Validation Split Only
*Source: `docs/research_history/PHASE_5_MODEL_OPTIMIZATION.md`*

- **Dataset:** SemEval-2020 Task 11 (Fine-Grained Propaganda Technique Detection).
- **Split Evaluated:** Article-disjoint validation set ($N=1,448$ rhetorical spans across 71 held-out articles).
- **Reason:** Official competition test set ground-truth labels were withheld by SemEval organizers and evaluated via an external scoring portal that is no longer active.
- **Model:** RoBERTa-base with square-root inverse class weighting and linear learning rate warmup/decay (Phase 5 Exp 1).
- **Performance:** Accuracy = 65.19%, Macro F1 = 0.5132, Weighted F1 = 0.6477 across 14 fine-grained rhetorical techniques.

#### Empirical Finding
While class re-weighting resolved complete failure modes on underrepresented classes (e.g., `Whataboutism` F1 increased from 0.0000 to 0.2326; `Repetition` F1 increased from 0.1852 to 0.4790), performance on rare techniques remains constrained (`Bandwagon` F1 = 0.1212 with $N=19$; `Thought-terminating Clichés` F1 = 0.2581 with $N=17$). **No generalization claim on an external or unreleased test set is made.**

---

### 1.4 Hate Speech Performance Bounds on HateXplain
*Source: `docs/research_history/PHASE_4_EVALUATION.md`, `models/hate_speech/metrics.json`*

- **Dataset:** HateXplain benchmark test set ($N=1,924$ social media posts: 782 Normal, 548 Offensive, 594 Hate Speech).
- **Model:** BERT-base-uncased 3-class classifier.
- **Performance:** Accuracy = 67.41%, Macro F1 = 0.6617, Weighted F1 = 0.6738.
- **Class Breakdown:** Normal F1 = 0.7587, Offensive F1 = 0.5735, Hate Speech F1 = 0.6530.

#### Empirical Finding
Distinguishing offensive slang/profanity (non-directed) from targeted hate speech remains difficult even for human annotators, reflecting inherently noisy real-world crowd-sourced annotations. The automated model achieves a 0.5735 F1 on offensive language and misclassifies subtle derogatory expressions.

---

## 2. Methodological Limitations

### 2.1 Quantitative Uncertainty Calibration Not Established
The platform implements Monte Carlo Dropout ($T=20$ stochastic forward passes) across all three core models (EfficientNet-B4, RoBERTa-base, BERT-base) to measure predictive dispersion:
- Predictive variance ($\sigma^2$)
- Standard deviation ($\sigma$)
- Shannon entropy ($H$)

**Limitation:** These metrics quantify relative epistemic uncertainty (internal model disagreement under stochastic dropout), but **calibration has not been quantitatively established**.
- Expected Calibration Error (ECE), Brier Score, and reliability diagrams were not formally evaluated.
- Post-processing calibration methods (such as Platt scaling, temperature scaling, or isotonic regression) were not applied.
- The threshold triggers ($\sigma^2 > 0.010$ for visual deepfakes; $\sigma^2 > 0.020$ for propaganda; $\sigma^2 > 0.020$ for hate speech) are **heuristic development boundaries**, not mathematically calibrated confidence intervals. They flag elevated internal dispersion, not true Bayesian posterior error probabilities.

---

### 2.2 Grad-CAM Spatial Explainability Scope
- **Implementation:** Grad-CAM generates saliency heatmaps from the final convolutional layer (`features[-1]`) of EfficientNet-B4.
- **Limitation:** Grad-CAM produces a coarse $7\times 7$ feature activation map bilinearly interpolated over the face crop.
- Saliency maps highlight regions that strongly activated intermediate convolutional feature channels; **they do not constitute causal proof of manual pixel manipulation**.
- Saliency maps have not been quantitatively evaluated using bounding box localization metrics (e.g., Pointing Game, Energy-based Pointing Game, or Intersection-over-Union against pixel-level manipulation ground truth masks).
- Saliency must be interpreted strictly as an explanatory aid for human investigators, not as evidentiary truth.

---

### 2.3 NLP Token Attribution & Rationale Alignment Scope
- **Propaganda Token Attribution:** Gradient $\times$ Input saliency assigns relative importance weights to input tokens. This signal is qualitative and exploratory; no quantitative span-level IoU metric was computed against gold-standard rhetorical spans.
- **HateXplain Rationale Alignment:** The reported rationale alignment (Precision = 80.0%, Recall = 75.0%, F1 = 77.4%) is verified on a **small representative diagnostic subset** of annotated HateXplain posts containing multi-annotator rationale segmentations. It must not be cited as an aggregate full-benchmark test score across all 1,924 posts.

---

## 3. System and Architectural Constraints

### 3.1 Sparse Temporal Video Sampling Protocol
- **Sampling Rate:** The video analysis pipeline extracts frames at a sparse rate of **1.0 FPS up to a maximum of 4 frames per video** using deterministic uniform sequential sampling.
- **Limitation:** The model does not inspect dense 30 or 60 FPS video streams.
- Frame-to-frame temporal jitter, subtle blinking cadence anomalies, optical flow inconsistencies, and intra-second synthetic artifacts are not evaluated.
- Fast manipulation techniques (e.g., single-frame face inserts or brief 100ms glitches) can evade sparse 1.0 FPS sampling if they fall between sampled timestamps.

---

### 3.2 Absence of Dedicated Audio Deepfake & Lip-Sync Verification
- **Current Pipeline:** The audio subsystem utilizes OpenAI `whisper-base` strictly for speech-to-text (ASR) transcription.
- **Limitation:** The platform does not implement:
  - Acoustic voice-clone / synthetic speech detection (e.g., ASVspoof-style spectral feature classifiers or raw waveform models like AASIST).
  - Audio-visual phoneme-viseme synchronization verification (e.g., SyncNet or Wav2Lip lip-sync coherence metrics).
- Deepfakes that preserve authentic facial video but replace the audio track with a cloned voice will be classified based solely on visual facial features, ignoring the acoustic forgery.

---

### 3.3 Cascaded Error Propagation in Multimodal Workflows
- Optical Character Recognition (EasyOCR) and Automated Speech Recognition (Whisper-base) operate upstream of the NLP models.
- **Limitation:** Any transcription or character recognition error cascades directly into downstream NLP classification.
- Low-resolution video captions, motion-blurred text banners, background noise, or heavy accents that degrade OCR/ASR accuracy directly degrade RoBERTa and BERT classification accuracy.

---

### 3.4 Untested Generalization Hypotheses

| Domain / Dimension | Current Project Scope | Operational Status |
| :--- | :--- | :--- |
| **Generative AI Generalization** | Trained & evaluated exclusively on FaceForensics++ C23 (Deepfakes, Face2Face, FaceSwap, NeuralTextures). | **Untested** on commercial diffusion models (Midjourney v6, Flux, Stable Diffusion 3, Sora) or modern GAN generators (StyleGAN3). |
| **Multilingual Support** | Trained on English datasets (SemEval-2020 English articles, HateXplain English tweets/Gab posts). | **Untested** on non-English text or multilingual cross-lingual inputs. |
| **Demographic Fairness** | Evaluated on FaceForensics++ aggregate benchmark splits. | **Untested** for disaggregated subgroup performance across Fitzpatrick skin types, gender presentations, or age brackets. |

---

### 3.5 Runtime and Infrastructure Constraints
- **In-Memory Job State:** The asynchronous video verification queue stores job statuses in the Python process memory (`in-memory dict`). If the backend server process restarts, in-flight or completed background jobs are lost.
- **Artifact Accumulation:** Extracted video frame crops and Grad-CAM overlay renders are stored in the temporary filesystem directory. Automated lifecycle retention, quota management, and scheduled cache cleanup are decoupled and being implemented under a separate Antigravity task.

---

## 4. Presentation-Ready Summary (For Slides / PPT)

```text
SLIDE: Verified System Limitations & Forensic Scope

1. Visual Degradation Sensitivity:
   - High robustness to JPEG compression (F1 drop: -8%).
   - Severe vulnerability to Gaussian blur & 1/4x resize (Acc drop: -25%; Real recall drops from 78% -> 12%).
   - Missing high-frequency textures cause blurred authentic media to be misflagged as deepfakes.

2. Class Imbalance Asymmetry:
   - Evaluated on FaceForensics++ C23 test split (4:1 Fake-to-Real ratio, N = 700).
   - High Fake precision (96.3%) contrasts with Real precision (65.1%), yielding 34.9% false alarms on authentic classifications.

3. Evaluation Scope Discipline:
   - Propaganda metrics (Acc: 65.2%, Macro F1: 0.513) are VALIDATION-ONLY on SemEval-2020 Task 11 (test labels withheld).
   - Hate speech achieves 67.4% test accuracy on HateXplain; nuanced offensive slang remains challenging.

4. Methodological & System Boundaries:
   - Uncertainty (MC Dropout, T = 20) is uncalibrated: flags internal model dispersion, not true error probability.
   - Grad-CAM and token attributions are post-hoc explanatory signals, not causal ground-truth proof.
   - Video uses sparse 1.0 FPS sampling (max 4 frames); no dedicated audio deepfake or lip-sync model is deployed.

5. Ethical Alignment:
   - Designed exclusively for human-in-the-loop decision support—never autonomous censorship.
```

---

## 5. Research Paper Limitations Section (Ready for Publication)

> **Section: Limitations and Ethical Operational Bounds**  
> While the proposed multimodal verification framework achieves strong benchmark performance across deepfake detection ($88.00\%$ accuracy, $0.8859$ weighted F1 on FaceForensics++ C23), propaganda classification ($0.5132$ macro F1 across 14 SemEval-2020 validation techniques), and hate speech analysis ($0.6741$ accuracy on HateXplain), several empirical and methodological boundaries must be acknowledged. First, stress testing reveals acute sensitivity to spatial smoothing: under Gaussian blur ($r=2$) and $1/4\times$ bilinear downsampling, accuracy degrades by $25.0\%$, with authentic class recall collapsing from $78.0\%$ to $12.0\%$ as the convolutional backbone associates suppressed high-frequency facial textures with synthetic blending artifacts. Second, the $4:1$ class imbalance in the FaceForensics++ test split induces a precision disparity, where $34.95\%$ of videos predicted as authentic represent false alarms. Third, because competition ground-truth test labels were withheld by SemEval-2020 Task 11 organizers, all reported propaganda results reflect an article-disjoint validation split ($N=1,448$ spans) rather than a sequestered benchmark test set. Fourth, while Monte Carlo Dropout ($T=20$) provides meaningful signals of epistemic dispersion, calibration has not been quantitatively established via expected calibration error (ECE) or reliability diagrams; thresholded uncertainty flags remain heuristic decision-support boundaries rather than calibrated posterior probabilities. Finally, the system operates on sparse 1.0 FPS video sampling without dedicated acoustic synthetic-speech detection or audiovisual lip-sync synchronization, leaving high-frequency temporal anomalies and audio-only deepfakes outside the evaluated operational envelope. Consequently, this platform is formulated strictly as an explainable decision-support workstation for human forensic analysts, explicitly precluding autonomous, unsupervised moderation decisions.

---

## 6. Document Provenance and Evidence Map

| Claim / Topic | Supporting File / Report | Status |
| :--- | :--- | :--- |
| Distortion Degradation Values | `phase4_robustness.json`, `scripts/phase4_robustness.py` | Verified Empirical Record |
| Full Deepfake Confusion Matrix | `phase4_results.json`, `models/deepfake/metrics.json` | Verified Empirical Record |
| Propaganda Validation Metrics | `docs/research_history/PHASE_5_MODEL_OPTIMIZATION.md` | Verified Audited Report |
| Hate Speech Test Metrics | `docs/research_history/PHASE_4_EVALUATION.md`, `models/hate_speech/metrics.json` | Verified Audited Report |
| Uncertainty Calibration Status | `pipeline/uncertainty.py`, `models/deepfake/predict.py` | Verified Architectural Code |
| Grad-CAM Implementation Scope | `explainability/gradcam.py`, `frontend/src/components/VisualResultView.jsx` | Verified Architectural Code |
| Sparse Video Sampling Rate | `pipeline/video_pipeline.py`, `extraction/frame_extractor.py` | Verified Architectural Code |
| In-Memory Async Job Queue | `api/routes/verify.py` | Verified Architectural Code |
| Cross-Dataset & Demographic Scope | `docs/research_history/DEVELOPMENT_RULES.md` | Verified Untested Boundary |
