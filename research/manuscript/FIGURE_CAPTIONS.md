# Formal Figure and Table Captions

This document provides publication-ready captions, annotations, and references for all figures and structured tables in the manuscript.

---

## 1. Figure Captions

### Figure 1 (Figure A): Task-Specific Benchmark Performance Across Modalities
> **Figure 1.** Evaluation of the Digital Media Verification (DMV) framework across core modalities: Visual Deepfakes (FaceForensics++ C23, $N=700$ test videos), Propaganda Detection (SemEval-2020 Task 11, $N=1,448$ validation spans), and Hate Speech Detection (HateXplain, $N=1,924$ test posts). Grouped bar charts depict Classification Accuracy, Macro F1, and Weighted F1. Visual deepfake detection achieves 88.00% accuracy and 0.8321 Macro F1; optimized propaganda classification reaches 0.5132 Macro F1 (+0.0957 over baseline); hate speech classification yields 67.41% accuracy and 0.6617 Macro F1.

### Figure 2 (Figure B): Deepfake Detector Performance Under Realistic Visual Distortions
> **Figure 2.** Robustness degradation profile of the production EfficientNet-B4 deepfake detector evaluated on a balanced 100-video FaceForensics++ test subset (50 Real, 50 Fake) across five conditions: Clean Baseline, JPEG Compression ($Q=30$), Gaussian Noise ($\sigma=15$), Gaussian Blur ($r=2.0$), and Low-Resolution Downsampling ($0.25\times$). While clean accuracy reaches 81.00%, blurring and downsampling cause catastrophic degradation to 56.00% accuracy, collapsing Real Recall to 12.00% due to loss of high-frequency facial texture cues.

### Figure 3 (Figure C): Complete FaceForensics++ C23 Test Confusion Matrix
> **Figure 3.** Confusion matrix for the production EfficientNet-B4 detector evaluated on the complete FaceForensics++ (C23) test split ($N=700$ videos). The dataset reflects a realistic 4:1 class imbalance (140 Real, 560 Fake). The detector correctly identifies 495 of 560 manipulated videos (88.39% Fake Recall) and 121 of 140 pristine videos (86.43% Real Recall). Class precision marginals show 96.30% precision for Fake predictions and 65.05% precision for Real predictions.

### Figure 4 (Figure D): Class-Wise Metrics and Imbalance Sensitivity
> **Figure 4.** Impact of 4:1 class imbalance on precision, recall, and F1 score for Real vs. Fake classes on FaceForensics++ C23 ($N=700$). While Fake precision remains exceptionally high (96.30%), Real precision is attenuated to 65.05% due to the prevalence of manipulated samples in the test distribution, highlighting the necessity of human review when classifying ambiguous samples as authentic.

### Figure 5 (Figure E): Per-Technique Propaganda Optimization (SemEval-2020 Task 11)
> **Figure 5.** Technique-level Macro F1 comparison between the baseline RoBERTa-base classifier and the optimized Phase 5 Exp 1 checkpoint across 14 rhetorical propaganda techniques evaluated on the article-disjoint validation split ($N=1,448$). Introduction of class-balanced focal loss yields marked improvements on rare techniques, including Repetition (+0.2938 F1), Whataboutism (+0.2326 F1), Flag-Waving (+0.1222 F1), and Causal Oversimplification (+0.1210 F1), raising overall Macro F1 from 0.4175 to 0.5132.

### Figure 6 (Figure F): Benchmark Ground-Truth Class Distributions
> **Figure 6.** Ground-truth class distributions across evaluated benchmark partitions: (a) Visual Deepfakes (FaceForensics++ C23 test split, $N=700$), showing 80% Fake and 20% Real; (b) Hate Speech (HateXplain test split, $N=1,924$), showing Normal (43.4%), Offensive (30.8%), and Hate Speech (25.8%); (c) Propaganda (SemEval-2020 Task 11 validation split, $N=1,448$), displaying the long-tail frequency distribution across 14 rhetorical classes dominated by Loaded Language and Name Calling.

---

## 2. Table Captions

### Table 1: Cross-Modal Verification Performance
> **Table 1.** Empirical evaluation of the Digital Media Verification framework across visual, propaganda, and hate speech modalities. Sample count $N$, evaluation split, Accuracy, Macro F1, and Weighted F1 are reported.

### Table 2: Deepfake Distortion Robustness Summary ($N=100$ Balanced Videos)
> **Table 2.** Performance of the production EfficientNet-B4 detector on the standardized 100-video development subset under five distortion operators. Columns report Accuracy, Macro F1, Accuracy Delta relative to clean baseline, Real Class Recall, and Fake Class Recall.

### Table 3: Deepfake Detector Test Confusion Matrix Breakdown ($N=700$)
> **Table 3.** Contingency table and marginal performance metrics for the production deepfake detector on the 700-video FaceForensics++ test partition.

### Table 4: Propaganda Technique Checkpoint Optimization (SemEval-2020 Task 11, $N=1,448$)
> **Table 4.** Per-class F1 performance and sample support for 14 propaganda techniques comparing baseline RoBERTa-base against the cost-sensitive focal loss optimized checkpoint (Exp 1).

### Table 5: Deepfake Robustness V3 Candidate Acceptance Gate Matrix
> **Table 5.** Formal acceptance gate compliance matrix comparing the production baseline against Candidate V3-A, Candidate V3-B, and Candidate V3-C across clean, Gaussian noise, and JPEG compression conditions. Demonstrates failure of all V3 candidates on the JPEG $Q=30$ fake recall gate ($<60\%$), justifying the retention of the production baseline model.
