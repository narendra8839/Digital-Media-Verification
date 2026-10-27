# DMV Model Evaluation Summary

This document synthesizes the authoritative benchmark results across all three core modalities of the Digital Media Verification (DMV) framework: **Visual Deepfakes**, **Textual Propaganda**, and **Hate Speech**.

---

## 1. Cross-Modal Benchmark Performance

Evaluation was conducted under strict split isolation on peer-reviewed benchmark datasets.

| Modality / Task | Model Architecture | Evaluation Dataset & Split | Sample Count ($N$) | Accuracy | Macro F1 | Weighted F1 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Visual Deepfake** | EfficientNet-B4 | FaceForensics++ C23 (Test) | 700 videos | **0.8800** | **0.8321** | **0.8859** |
| **Propaganda Detection** | RoBERTa-base (Exp 1) | SemEval-2020 Task 11 (Val) | 1,448 spans | **0.6519** | **0.5132** | **0.6477** |
| **Hate Speech Detection** | BERT-base-uncased | HateXplain (Test) | 1,924 samples | **0.6741** | **0.6617** | **0.6738** |

*Note on Propaganda split:* Official SemEval-2020 Task 11 test set ground-truth labels were permanently withheld by competition organizers. Consequently, all propaganda evaluations strictly employ the official article-disjoint validation set and are designated as validation metrics.

---

## 2. Visual Deepfake Evaluation (FaceForensics++ C23)

### Full Test Set Confusion Matrix ($N = 700$ Videos)

The test partition contains 140 Real videos and 560 Manipulated (Fake) videos, representing a realistic $4:1$ class imbalance:

| | Predicted Fake | Predicted Real | Total Actual | Class Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Fake** | **495** ($TP$) | 65 ($FN$) | 560 | **88.39%** |
| **Actual Real** | 19 ($FP$) | **121** ($TN$) | 140 | **86.43%** |
| **Total Predicted** | 514 | 186 | 700 | — |
| **Class Precision** | **96.30%** | **65.05%** | — | **Accuracy: 88.00%** |

### Analysis of Imbalance Impact
- **High Fake Precision (96.30%):** Out of 514 videos flagged as fake, 495 are verified fakes. When the system raises a high-confidence fake alert, it is correct in $>96\%$ of cases.
- **Moderate Real Precision (65.05%):** Due to the $4:1$ negative-to-positive ratio in the evaluation set, even a low false negative rate (65 missed fakes out of 560) creates a substantial proportion of false classifications among predictions classified as Real.
- **Operational Recommendation:** Predictions marked as "Real" with elevated uncertainty ($\sigma^2 \ge 0.02$) must be triaged for secondary human inspection (`REVIEW_RECOMMENDED`).

---

## 3. NLP Propaganda Technique Classification (SemEval-2020 Task 11)

To overcome extreme class imbalance across 14 rhetorical propaganda techniques, Phase 5 introduced cost-sensitive focal loss and hyperparameter optimization (Exp 1):

| Propaganda Technique | Support ($N$) | Baseline F1 | Optimized Exp 1 F1 | $\Delta$ F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Loaded Language** | 477 | 0.7250 | **0.7754** | $+0.0504$ | Strong |
| **Name Calling** | 226 | 0.6713 | **0.7515** | $+0.0802$ | Strong |
| **Repetition** | 143 | 0.1852 | **0.4790** | **$+0.2938$** | Major Improvement |
| **Exaggeration** | 137 | 0.4851 | **0.5652** | $+0.0801$ | Moderate |
| **Doubt** | 118 | 0.5714 | **0.6639** | $+0.0925$ | Moderate |
| **Appeal to Fear-Prejudice** | 82 | 0.5081 | **0.5503** | $+0.0422$ | Moderate |
| **Causal Oversimplification** | 56 | 0.4000 | **0.5210** | $+0.1210$ | Improved |
| **Appeal to Authority** | 49 | 0.4571 | **0.5714** | $+0.1143$ | Improved |
| **Flag-Waving** | 46 | 0.6111 | **0.7333** | $+0.1222$ | Improved |
| **Black-and-White Fallacy** | 32 | 0.3922 | **0.4127** | $+0.0205$ | Minor Gain |
| **Whataboutism** | 23 | 0.0000 | **0.2326** | **$+0.2326$** | Zero to Viable |
| **Slogans** | 23 | 0.4348 | **0.5490** | $+0.1142$ | Improved |
| **Bandwagon** | 19 | 0.0833 | **0.1212** | $+0.0379$ | Low Sample |
| **Thought-terminating Clichés** | 17 | **0.3200** | 0.2581 | $-0.0619$ | Slight Tradeoff |
| **Overall Macro F1** | 1,448 | **0.4175** | **0.5132** | **$+0.0957$** | **Validated** |
| **Overall Accuracy** | 1,448 | **0.5932** | **0.6519** | **$+0.0587$** | **Validated** |
| **Overall Weighted F1** | 1,448 | **0.5539** | **0.6477** | **$+0.0938$** | **Validated** |

---

## 4. Hate Speech Detection (HateXplain)

- **Test Partition Size:** 1,924 social media posts annotated by multiple crowdsourced annotators.
- **Classes:** 3 categories (`Hate Speech`, `Offensive`, `Normal`).
- **Achieved Test Accuracy:** $67.41\%$
- **Macro F1:** $0.6617$
- **Weighted F1:** $0.6738$
- **Rationales / Explainability:** Integrated via token attribution gradients mapped to highlighted spans in the visual dashboard.

---

## 5. Summary of Limitations

1. **Not a Forensic Proof Engine:** Outputs represent probabilistic statistical likelihoods, not incontrovertible legal proof.
2. **Robustness Degeneracy:** Performance degrades significantly under strong compression ($Q \le 30$) or severe downsampling ($\le 0.25\times$), as detailed in [DEEPFAKE_ROBUSTNESS_V3_REPORT.md](DEEPFAKE_ROBUSTNESS_V3_REPORT.md).
3. **Face Quality Dependency:** Detection depends on MTCNN face localization; profile poses, heavy occlusions, or faces $<60\times 60$ pixels fail detection and require human review.
