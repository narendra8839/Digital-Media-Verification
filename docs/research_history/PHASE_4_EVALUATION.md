# Phase 4: Final Evaluation and Model Analysis

> [!IMPORTANT]
> **Audit Notice**: A complete evaluation consistency audit and final benchmark run were performed on October 7, 2026. See [PHASE_4_EVALUATION_AUDIT.md](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/docs/PHASE_4_EVALUATION_AUDIT.md) and [PHASE_4_FINAL_SIGNOFF.md](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/docs/PHASE_4_FINAL_SIGNOFF.md) for full audit histories, manifest reconciliations, and signoff records.

## 1. Overview

This document presents the final comprehensive evaluation of the Phase 3 candidate models on untouched benchmark test sets and held-out validation partitions.

The evaluation covers:
- **Task-Specific Metrics**: Accuracy, Precision, Recall, F1-scores, and Confusion Matrices across all evaluated classes.
- **Explainability**: Qualitative inspection of Grad-CAM (Deepfake) and Token Attribution (Propaganda); quantitative rationale alignment for Hate Speech (HateXplain).
- **Robustness (Deepfake)**: Impact of realistic image degradation (JPEG compression, blur, resize, noise) evaluated on a balanced test subset (50 Real + 50 Fake = 100 videos) at constant frame depth.
- **Uncertainty & Calibration**: Verification of stochastic variance under MC Dropout ($T=20$). *Calibration not quantitatively established.*

## 2. Datasets Evaluated
- **Deepfake**: `data/full_splits/deepfake_test.json` (FaceForensics++ C23) — **COMPLETE test set of all 700 videos** evaluated (140 Real, 560 Fake).
- **Propaganda**: `data/full_splits/propaganda_val.json` (SemEval-2020 Task 11) — strictly an **article-disjoint held-out validation set** (71 articles, 1,448 spans; official competition test labels were unreleased).
- **Hate Speech**: `data/full_splits/hatexplain_test.json` (HateXplain) — **full benchmark test split** (1,924 samples).

---

## 3. Evaluation Results

### 3.1. Deepfake Detection (EfficientNet-B4)
*Checkpoint: `models/deepfake/checkpoint_full/best_model.pt`. Evaluated on the COMPLETE FaceForensics++ C23 test split ($N=700$ videos) with deterministic uniform frame sampling at 1.0 FPS ($4$ frames per video).*

#### Aggregate Performance
- **Evaluated Count**: 700 / 700 videos (100.0% coverage)
- **Class Distribution**: 140 Real (20.0%), 560 Fake (80.0%)
- **Accuracy**: **0.8800** (88.00%)
- **Macro Precision**: **0.8068**
- **Macro Recall**: **0.8741**
- **Macro F1**: **0.8321**
- **Weighted Precision**: **0.9005**
- **Weighted Recall**: **0.8800**
- **Weighted F1**: **0.8859**

#### Class-Wise Metrics
| Class | Support | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Real** | 140 | 0.6505 (121/186) | 0.8643 (121/140) | **0.7423** |
| **Fake** | 560 | 0.9630 (495/514) | 0.8839 (495/560) | **0.9218** |
| **Macro Avg** | 700 | 0.8068 | 0.8741 | **0.8321** |
| **Weighted Avg** | 700 | 0.9005 | 0.8800 | **0.8859** |

#### Complete Confusion Matrix ($N=700$)
```
                     Predicted Fake      Predicted Real      Total True
True Fake (Class 0)       495                  65               560
True Real (Class 1)        19                 121               140
Total Predicted           514                 186               700
```
*Note: Format follows standard scikit-learn convention with labels `['fake', 'real']`: `[[495, 65], [19, 121]]`.*

#### Robustness Analysis (Balanced Subset: 50 Real + 50 Fake = 100 Videos)
Evaluated on an explicitly documented, deterministic balanced subset of 100 test videos (first 50 Real + first 50 Fake from `deepfake_test.json`) using the **exact same frames** and constant sampling depth ($4$ frames per video) across all 5 conditions:

| Degradation Condition | Accuracy | Macro F1 | Drop in Acc | Drop in F1 | Confusion Matrix `[[Fake], [Real]]` |
|---|---|---|---|---|---|
| **Original (Unmodified)** | **0.8100** | **0.8098** | - | - | `[[42, 8], [11, 39]]` |
| **JPEG Compression (Q=30)** | **0.7300** | **0.7278** | -0.0800 | -0.0820 | `[[32, 18], [9, 41]]` |
| **Gaussian Blur ($r=2$)** | **0.5600** | **0.4544** | -0.2500 | -0.3554 | `[[50, 0], [44, 6]]` |
| **Low-Res Resize ($1/4\times$)** | **0.5600** | **0.4544** | -0.2500 | -0.3554 | `[[50, 0], [44, 6]]` |
| **Gaussian Noise ($\sigma=15$)** | **0.6700** | **0.6624** | -0.1400 | -0.1474 | `[[41, 9], [24, 26]]` |

**Conclusion on Robustness**:
- Under baseline conditions, the balanced model achieves **81.00% accuracy** (Macro F1: 0.8098), correctly identifying 42/50 Fake and 39/50 Real videos.
- The model exhibits moderate degradation under JPEG compression (73.00% Acc, 8% drop) and Gaussian noise (67.00% Acc, 14% drop).
- Severe degradation occurs under high-frequency suppression (Gaussian Blur and Low-Res Resize, 56.00% Acc, 25% drop), where loss of fine textures causes the model to default heavily toward predicting "fake" (50/50 fake correct, but only 6/50 real correct).

---

### 3.2. Propaganda Detection (RoBERTa-base)
*Checkpoint: `models/propaganda/checkpoint_full/`. Evaluated on the **article-disjoint held-out validation split** (`data/full_splits/propaganda_val.json`, 71 articles, 1,448 spans).*

- **Dataset Classification**: Article-disjoint held-out validation set (NOT a test set).
- **Evaluated Count**: 1,448 spans (from 71 articles; zero article overlap with training set).
- **Accuracy**: **0.5932** (859 / 1,448 correct)
- **Macro Precision**: **0.5238**
- **Macro Recall**: **0.4030**
- **Macro F1**: **0.4175**
- **Weighted Precision**: **0.6083**
- **Weighted Recall**: **0.5932**
- **Weighted F1**: **0.5539**

---

### 3.3. Hate Speech Detection (BERT-base-uncased)
*Checkpoint: `models/hate_speech/checkpoint_full/`. Evaluated on the full benchmark test split (`data/full_splits/hatexplain_test.json`, 1,924 samples).*

- **Evaluated Count**: 1,924 test samples (782 Normal, 548 Offensive, 594 Hate Speech).
- **Accuracy**: **0.6741**
- **Macro Precision**: **0.6608**
- **Macro Recall**: **0.6685**
- **Macro F1**: **0.6617**

---

## 4. Explainability & Uncertainty

### 4.1. Deepfake (Grad-CAM & MC Dropout)
- **Grad-CAM Explainability (Qualitative Only)**: Visual heatmap inspections show activation concentrations on facial landmarks (eyes, nose, mouth boundary contours) on manipulated frames. *No quantitative localization benchmark (such as bounding-box IoU or point-game hit rate) was evaluated.*
- **Uncertainty**: MC Dropout ($T=20$) produces non-zero epistemic predictive variance.
- **Calibration Limitation**: **Calibration not quantitatively established.** No empirical Expected Calibration Error (ECE) or reliability diagrams have been generated. Thresholds (`variance: 0.02`, `entropy: 0.85 bits`) are heuristic MVP engineering baselines.

### 4.2. NLP (Token Attribution & Rationale Alignment)
- **Hate Speech (Quantitative Rationale Alignment)**: Token attribution evaluated via `evaluate_rationale_alignment()` against HateXplain human annotator ground-truth masks:
  - Overlap Tokens: `["nigger"]`
  - Sample Alignment: Precision = **0.5000**, Recall = **1.0000**, **Rationale F1 = 0.6667**.
- **Propaganda (Qualitative Only)**: Gradient $\times$ Input saliency chips highlight loaded language and persuasive rhetoric. No quantitative span-overlap benchmark was evaluated.

---

## 5. Final Decision

*Are these models acceptable for integration into the final MVP application?*

**DECISION**: **ACCEPTABLE FOR INTEGRATION (AUDIT RECONCILED)**.  
The models satisfy all baseline functional requirements for the explainable and uncertainty-aware verification pipeline. No model retraining was performed. All evaluation discrepancies, manifest distributions, and uncertainty limitations are fully verified and reconciled.
