# Phase 4 Evaluation Consistency Audit & Reconciliation

**Audit Date**: October 7, 2026  
**Status**: RECONCILED AND VERIFIED  
**Branch**: `swaraj`  
**Execution Constraints**: No model retraining; no frontend modifications; no checkpoint changes; Phase 5 not started.

---

## 1. Executive Summary of Audit Findings & Resolutions

| Audit Area | Original Discrepancy / State | Final Corrective Action & Verification | Status |
|---|---|---|---|
| **1. Deepfake Test Set Size** | Reported Acc: `0.8400`, Macro F1: `0.4565` based on only `data[:100]` (100 Real, 0 Fake). | Re-evaluated against **all 700 test videos** (140 Real, 560 Fake) using existing checkpoint. **Acc: 0.8800, Macro F1: 0.8321**. | **RESOLVED** |
| **2. Deepfake Robustness** | Reported baseline Acc: `0.6500` based on `data[:20]` (20 Real, 0 Fake) with 2 frames/video. | Re-evaluated on a **balanced 100-video subset** (50 Real + 50 Fake) with matching 4 frames/video depth. **Baseline Acc: 0.8100, Macro F1: 0.8098**. | **RESOLVED** |
| **3. Deepfake Class-Wise Metrics** | Fake class had 0 evaluated samples ($F1=0.0$), collapsing Macro F1 to 0.4565. | Complete test confusion matrix: `[[495, 65], [19, 121]]`. Real F1: 0.7423, Fake F1: 0.9218. | **RESOLVED** |
| **4. Deepfake Sampling Policy** | Frame depth differed (4 in test, 2 in robustness and val). | Standardized to **deterministic uniform sampling at 1.0 FPS, 4 frames/video** across both test evaluation and robustness. | **RESOLVED** |
| **5. SemEval Propaganda Manifests** | Documentation referenced conflicting counts (4,904 vs 4,681 spans; 285 vs 286 articles). | Audited disk manifests: Train has **286 articles, 4,681 spans**; Val has **71 articles, 1,448 spans**. Reconciled `metrics.json` ($N=1,448$). | **RESOLVED** |
| **6. Propaganda Terminology** | Inconsistent labeling as test set vs validation set. | Formally classified as **article-disjoint held-out validation set**; never designated as a test set. | **RESOLVED** |
| **7. Uncertainty Calibration** | Documented as "properly calibrated" without ECE evidence. | Explicit limitation declared: **"Calibration not quantitatively established."** | **RESOLVED** |
| **8. Explainability Separation** | Qualitative observations blurred with quantitative metrics. | Explicitly separated: Deepfake Grad-CAM & Propaganda attributions are qualitative; HateXplain is quantitative. | **RESOLVED** |

---

## 2. Detailed Findings and Verified Resolutions

### 2.1. Deepfake Test Set: Complete Re-Evaluation
- **Initial Discrepancy**: Slicing `data[:100]` evaluated only the first 100 entries of `deepfake_test.json`. Because Real videos precede Fake videos in the manifest, this subset contained 100 Real videos and 0 Fake videos.
- **Resolution Executed**:
  - Script: [`scripts/phase4_evaluate.py`](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/scripts/phase4_evaluate.py) was updated to process all entries in `data/full_splits/deepfake_test.json`.
  - Checkpoint: `models/deepfake/checkpoint_full/best_model.pt` (no retraining).
  - Hardware: NVIDIA RTX 4050 Laptop GPU (CUDA).
  - Evaluated: **700 / 700 videos** (140 Real, 560 Fake).
- **Verified Complete Metrics**:
  - **Accuracy**: **0.8800** (616 / 700 correct)
  - **Macro Precision**: **0.8068**
  - **Macro Recall**: **0.8741**
  - **Macro F1**: **0.8321**
  - **Weighted F1**: **0.8859**
  - **Real Class**: Precision = **0.6505** (121/186), Recall = **0.8643** (121/140), F1 = **0.7423**
  - **Fake Class**: Precision = **0.9630** (495/514), Recall = **0.8839** (495/560), F1 = **0.9218**
  - **Confusion Matrix** (`labels=['fake', 'real']`):
    ```
                     Predicted Fake      Predicted Real      Total True
    True Fake              495                  65               560
    True Real               19                 121               140
    Total Predicted        514                 186               700
    ```

---

### 2.2. Deepfake Robustness: Balanced Re-Evaluation
- **Initial Discrepancy**: Evaluated only `data[:20]` (20 Real, 0 Fake) with 2 frames/video, giving an anomalous baseline accuracy of 0.6500.
- **Resolution Executed**:
  - Script: [`scripts/phase4_robustness.py`](file:///c:/Users/Swaraj%20Shedge/Desktop/College/TY/ERA/Digital%20Media%20Verification/Digital-Media-Verification/scripts/phase4_robustness.py) was rewritten to select a deterministic, balanced subset: the first 50 Real videos + the first 50 Fake videos from `deepfake_test.json` ($N=100$ videos).
  - Sampling Depth: **4 frames per video** across all conditions.
  - Video Provenance: Videos are decoded once per trial, applying degradation conditions to the **exact same frames**.
- **Verified Balanced Robustness Table ($N=100$)**:

| Condition | Evaluated (Real/Fake) | Accuracy | Macro F1 | Drop in Acc | Confusion Matrix `[[Fake], [Real]]` |
|---|---|---|---|---|---|
| **Original** | 50 Real / 50 Fake | **0.8100** | **0.8098** | - | `[[42, 8], [11, 39]]` |
| **JPEG (Q=30)** | 50 Real / 50 Fake | **0.7300** | **0.7278** | -0.0800 | `[[32, 18], [9, 41]]` |
| **Blur ($r=2$)** | 50 Real / 50 Fake | **0.5600** | **0.4544** | -0.2500 | `[[50, 0], [44, 6]]` |
| **Resize ($1/4\times$)** | 50 Real / 50 Fake | **0.5600** | **0.4544** | -0.2500 | `[[50, 0], [44, 6]]` |
| **Noise ($\sigma=15$)** | 50 Real / 50 Fake | **0.6700** | **0.6624** | -0.1400 | `[[41, 9], [24, 26]]` |

---

### 2.3. SemEval Propaganda Manifest Reconciliation
- **Physical Inspection of Current Files on Disk**:
  - `data/full_splits/propaganda_train.json`:
    - Total spans: **4,681**
    - Unique articles: **286**
  - `data/full_splits/propaganda_val.json`:
    - Total spans: **1,448**
    - Unique articles: **71**
  - Combined Totals: 357 articles, 6,129 spans (matches SemEval-2020 Task 11 total annotated dataset).
  - Overlap check: **Zero article overlap** between train and validation.
- **Class Distribution Reconciled**:
  | Technique | Train Spans ($N=4,681$) | Val Spans ($N=1,448$) |
  |---|---|---|
  | Loaded_Language | 1,646 | 477 |
  | Name_Calling,Labeling | 832 | 226 |
  | Repetition | 478 | 143 |
  | Exaggeration,Minimisation | 329 | 137 |
  | Doubt | 375 | 118 |
  | Appeal_to_fear-prejudice | 212 | 82 |
  | Causal_Oversimplification | 153 | 56 |
  | Appeal_to_Authority | 95 | 49 |
  | Flag-Waving | 183 | 46 |
  | Black-and-White_Fallacy | 75 | 32 |
  | Whataboutism,Straw_Men,Red_Herring | 85 | 23 |
  | Slogans | 106 | 23 |
  | Bandwagon,Reductio_ad_hitlerum | 53 | 19 |
  | Thought-terminating_Cliches | 59 | 17 |
- **Reconciliation with Metrics**:
  - Checkpoint metrics file `models/propaganda/checkpoint_full/metrics.json` records `sample_count: 1448`.
  - Accuracy: **0.5932** (859 / 1,448)
  - Macro F1: **0.4175**
  - Weighted F1: **0.5539**
  - Explanation: The evaluated dataset was precisely `data/full_splits/propaganda_val.json` (71 articles, 1,448 spans). Earlier references mentioning 4,904 or 1,225 spans were legacy draft estimates and have been corrected.

---

### 2.4. Uncertainty and Explainability Disclaimers
1. **Uncertainty Calibration Status**:
   - **Calibration not quantitatively established.**
   - MC Dropout ($T=20$) provides epistemic variance and Shannon entropy estimates, but formal statistical calibration (e.g. Expected Calibration Error, Platt scaling, temperature scaling) was not computed.
   - Thresholds (`variance: 0.02`, `entropy: 0.85 bits`) are heuristic MVP triage rules.
2. **Explainability Breakdown**:
   - **Deepfake Grad-CAM**: Qualitative visual confirmation of feature activation. No bounding-box IoU or pointing-game metrics.
   - **Propaganda Token Attribution**: Qualitative saliency mapping on loaded phrases.
   - **Hate Speech Rationale Alignment**: Quantitative overlap against HateXplain human masks (`evaluate_rationale_alignment()`), achieving Precision = 0.5000, Recall = 1.0000, **Rationale F1 = 0.6667** on sample benchmarks.

---

## 3. Audit Conclusion

All 8 evaluation discrepancies and audit points have been fully investigated, re-evaluated, reconciled, and documented. The evaluation record is now internally consistent and verified against active checkpoints and on-disk split manifests.
