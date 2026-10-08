# Phase 4 Final Sign-Off Report

**Sign-Off Date**: October 7, 2026  
**Final Status**: **PHASE 4 COMPLETE**  
**Repository Branch**: `swaraj`  
**Execution Constraints Verified**: No model retraining performed; no frontend modified; no checkpoints changed; Phase 5 not started.

---

## 1. Executive Sign-Off Criteria Verification

| Requirement | Audited Status | Criterion Met? |
|---|---|---|
| **All 700 Deepfake Test Videos Evaluated** | Evaluated **700 / 700 videos** (140 Real, 560 Fake) from `data/full_splits/deepfake_test.json`. | **YES (COMPLETE)** |
| **Robustness Evaluated with Both Classes** | Evaluated a balanced **100-video subset** (50 Real + 50 Fake) across all 5 conditions at matching frame depth ($4$ frames/video). | **YES (COMPLETE)** |
| **Propaganda Manifest Counts Reconciled** | Reconciled against physical files on disk: **286 train articles (4,681 spans)** and **71 val articles (1,448 spans)**. | **YES (COMPLETE)** |
| **Propaganda Terminology Corrected** | Consistently designated as **article-disjoint held-out validation set** (never a test set). | **YES (COMPLETE)** |
| **Uncertainty Calibration Limitations Stated** | Explicitly declared: **"Calibration not quantitatively established."** | **YES (COMPLETE)** |
| **Explainability Scope Separated** | Partitioned qualitative visual overlays from quantitative rationale metrics. | **YES (COMPLETE)** |
| **Internal Documentation Consistency** | Synchronized across `PHASE_4_EVALUATION.md`, `PHASE_4_EVALUATION_AUDIT.md`, and this signoff. | **YES (COMPLETE)** |
| **Regression Test Suite Verified** | Executed `python -m pytest -q tests`: **41 passed** in 70.72s. | **YES (COMPLETE)** |

---

## 2. Final Evaluated Model Metrics

### 2.1. Deepfake Detection (EfficientNet-B4)
- **Checkpoint**: `models/deepfake/checkpoint_full/best_model.pt`
- **Evaluation Dataset**: `data/full_splits/deepfake_test.json` (FaceForensics++ C23)
- **Exact Test Sample Count**: **700 videos**
- **Exact Test Class Distribution**:
  - **Real**: 140 videos (20.0%)
  - **Fake**: 560 videos (80.0%)
- **Frame Sampling Policy**: Deterministic uniform sequential sampling at 1.0 FPS, **4 frames per video**, MediaPipe face crop to $224 \times 224$, standard ImageNet normalization.
- **Aggregate Test Metrics**:
  - **Accuracy**: **0.8800** (88.00%)
  - **Macro Precision**: **0.8068**
  - **Macro Recall**: **0.8741**
  - **Macro F1**: **0.8321**
  - **Weighted F1**: **0.8859**
- **Per-Class Metrics**:
  - **Real**: Precision = **0.6505** (121/186) | Recall = **0.8643** (121/140) | **F1 = 0.7423** | Support = 140
  - **Fake**: Precision = **0.9630** (495/514) | Recall = **0.8839** (495/560) | **F1 = 0.9218** | Support = 560
- **Confusion Matrix** (`labels=['fake', 'real']`):
  ```
                       Predicted Fake      Predicted Real      Total True
  True Fake (Class 0)       495                  65               560
  True Real (Class 1)        19                 121               140
  Total Predicted           514                 186               700
  ```

---

### 2.2. Balanced Deepfake Robustness Results
- **Evaluation Subset**: Deterministic balanced subset from `deepfake_test.json`: first 50 Real + first 50 Fake = **100 videos**.
- **Frame Depth**: **4 frames per video** across all 5 conditions; video frames decoded once and tested on identical frames.
- **Degradation Results Summary**:

| Condition | Evaluated (Real/Fake) | Accuracy | Macro F1 | Drop in Acc | Drop in F1 | Confusion Matrix `[[Fake], [Real]]` |
|---|---|---|---|---|---|---|
| **Original (Baseline)** | 50 Real / 50 Fake | **0.8100** | **0.8098** | - | - | `[[42, 8], [11, 39]]` |
| **JPEG Compression (Q=30)** | 50 Real / 50 Fake | **0.7300** | **0.7278** | -0.0800 | -0.0820 | `[[32, 18], [9, 41]]` |
| **Gaussian Blur ($r=2$)** | 50 Real / 50 Fake | **0.5600** | **0.4544** | -0.2500 | -0.3554 | `[[50, 0], [44, 6]]` |
| **Low-Res Resize ($1/4\times$)** | 50 Real / 50 Fake | **0.5600** | **0.4544** | -0.2500 | -0.3554 | `[[50, 0], [44, 6]]` |
| **Gaussian Noise ($\sigma=15$)** | 50 Real / 50 Fake | **0.6700** | **0.6624** | -0.1400 | -0.1474 | `[[41, 9], [24, 26]]` |

**Robustness Findings**:
The model shows good baseline performance (**81.00% accuracy** on balanced data), retaining moderate robustness under JPEG compression (73.00%) and Gaussian noise (67.00%). Under high-frequency attenuation (Blur and Resize), accuracy drops to 56.00% as the loss of fine texture causes the model to classify almost all blurred frames as fake.

---

### 2.3. Propaganda Detection (RoBERTa-base)
- **Checkpoint**: `models/propaganda/checkpoint_full/`
- **Evaluation Dataset**: `data/full_splits/propaganda_val.json`
- **Terminology**: Strictly an **article-disjoint held-out validation set** (SemEval-2020 Task 11 official test annotations are unreleased).
- **Exact Manifest Article & Span Counts**:
  - `propaganda_train.json`: **286 articles, 4,681 spans**
  - `propaganda_val.json`: **71 articles, 1,448 spans**
  - Article Overlap: **0 articles** (100% disjoint)
- **Final Validation Metrics** ($N=1,448$ spans):
  - **Accuracy**: **0.5932** (859 / 1,448 correct)
  - **Macro Precision**: **0.5238**
  - **Macro Recall**: **0.4030**
  - **Macro F1**: **0.4175**
  - **Weighted Precision**: **0.6083**
  - **Weighted Recall**: **0.5932**
  - **Weighted F1**: **0.5539**

---

### 2.4. Hate Speech Detection (BERT-base-uncased)
- **Checkpoint**: `models/hate_speech/checkpoint_full/`
- **Evaluation Dataset**: `data/full_splits/hatexplain_test.json` (HateXplain)
- **Exact Test Sample Count**: **1,924 samples** (782 Normal, 548 Offensive, 594 Hate Speech)
- **Final Test Metrics**:
  - **Accuracy**: **0.6741** (1,297 / 1,924 correct)
  - **Macro Precision**: **0.6608**
  - **Macro Recall**: **0.6685**
  - **Macro F1**: **0.6617**

---

## 3. Methodological Limitations & Governance Disclaimers

### 3.1. Uncertainty Quantification Limitation
- **Declaration**: **Calibration not quantitatively established.**
- **Details**: Monte Carlo Dropout ($T=20$) provides epistemic variance and Shannon entropy signals. However, neither Expected Calibration Error (ECE), Platt scaling, temperature scaling, nor reliability diagrams have been quantitatively computed across validation or test partitions.
- **Policy**: The triage thresholds (`variance: 0.02`, `entropy: 0.85 bits`) configured in `config/config.yaml` are heuristic MVP engineering baselines intended for human-in-the-loop review triggering, not statistically calibrated probabilities.

### 3.2. Explainability Limitation
- **Deepfake Grad-CAM**: Verified qualitatively through visual inspection of face activation heatmaps. No quantitative localization benchmark (such as bounding-box IoU or pointing-game hit rate) was evaluated.
- **Propaganda Token Attribution**: Verified qualitatively through gradient saliency chips highlighting loaded language.
- **Hate Speech Rationale Alignment**: Quantitatively verified using `evaluate_rationale_alignment()` against HateXplain annotator ground-truth masks (achieving Precision = 0.5000, Recall = 1.0000, **Rationale F1 = 0.6667** on benchmark samples).

---

## 4. Test Suite Execution
- **Command**: `python -m pytest -q tests`
- **Result**: **41 passed, 11 warnings in 70.72s (0:01:10)**
- **Integrity**: Zero test files were modified.

---

## 5. Formal Phase 4 Sign-Off Declaration

All Phase 4 deliverables, evaluations, robustness benchmarks, and documentation reconciliations have been completed according to specification without model retraining or checkpoint alteration.

**PHASE 4 IS OFFICIALLY MARKED: COMPLETE**
