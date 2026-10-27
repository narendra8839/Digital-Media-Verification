# Datasets, Partitions, and Split Integrity

This document outlines the dataset provenance, ground-truth distributions, split isolation protocols, and leakage prevention audits across the Digital Media Verification (DMV) framework.

---

## 1. Dataset Overview and Provenance

| Modality / Task | Benchmark Dataset | Source / Reference | Total Evaluated Split | Class Balance / Distribution |
| :--- | :--- | :--- | :--- | :--- |
| **Visual Deepfakes** | FaceForensics++ (C23) | Rössler et al. (ICCV 2019) | 700 Test Videos | 140 Real (20.0%), 560 Fake (80.0%) ($4:1$ ratio) |
| **Textual Propaganda** | SemEval-2020 Task 11 | Da San Martino et al. (2020) | 1,448 Validation Spans | 14 Rhetorical Techniques (Highly Imbalanced) |
| **Hate Speech** | HateXplain | Mathew et al. (AAAI 2021) | 1,924 Test Samples | 3 Classes: Hate Speech, Offensive, Normal |

---

## 2. FaceForensics++ (C23) Split Architecture & Leakage Audit

### 2.1 Video Partitions
The primary facial deepfake detection dataset follows standard FaceForensics++ sequence partitioning under light compression (C23 H.264):
- **Training Set (`deepfake_train.json`):** 3,600 videos (720 Real, 2,880 Fake across Deepfakes, Face2Face, FaceSwap, NeuralTextures).
- **Validation Set (`deepfake_val.json`):** 700 videos (140 Real, 560 Fake).
- **Test Set (`deepfake_test.json`):** 700 videos (140 Real, 560 Fake).

### 2.2 Leakage Prevention Audit
Strict data hygiene requires verifying zero video ID and zero sequence/actor ID overlap across splits:
1. **Video ID Isolation:** $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
2. **Actor/Sequence Identity Separation:** Manipulated sequences originate from specific actor pairs ($A \to B$). Both pristine source videos and derived synthetic manipulations are strictly constrained to their respective split partition, preventing identity leakage.
3. **Automated Verification:** Verified programmatically by `tests/test_deepfake_robustness.py::TestSplitIntegrity::test_zero_leakage_between_splits`.

---

## 3. Subsets Used in Robustness Research

Due to the computational cost of multi-distortion evaluations ($5$ corruptions $\times$ multi-scale parameter sweeps), two standardized subsets of `deepfake_test.json` were defined:

### 3.1 Standardized Development Benchmark ($N = 100$ Videos)
- **Composition:** First 50 Real videos (indices `0:50`) and first 50 Fake videos (indices `0:50`) from `deepfake_test.json`.
- **Class Balance:** Exactly 50 Real (50.0%) and 50 Fake (50.0%).
- **Usage:** Used as the primary calibration benchmark to assess degradation curves across JPEG compression, Gaussian noise, blur, and downsampling.

### 3.2 Reserved Comparison Subset ($N = 100$ Videos)
- **Composition:** Next 50 Real videos (indices `50:100`) and next 50 Fake videos (indices `50:100`) from `deepfake_test.json`.
- **Class Balance:** Exactly 50 Real (50.0%) and 50 Fake (50.0%).
- **Offset Independence:** Verified by `tests/test_deepfake_robustness.py::TestSplitIntegrity::test_offset_based_subset_isolation`.
- **Methodological Caveat:** This reserved comparison subset is **not** an untouched out-of-distribution holdout, because the production baseline model was previously evaluated on the full 700-video test split. It serves strictly as a non-overlapping secondary comparison partition within the test set to verify candidate stability.

---

## 4. NLP Benchmark Datasets

### 4.1 SemEval-2020 Task 11 (Propaganda Techniques)
- **Task Formulation:** Fine-grained persuasion technique identification on news articles.
- **Evaluation Split:** 1,448 annotated spans across an article-disjoint validation split.
- **Test Set Note:** Official competition test set ground-truth annotations were withheld by organizers; all reported propaganda metrics strictly denote the article-disjoint validation set.

### 4.2 HateXplain Benchmark
- **Task Formulation:** 3-class classification (`Hate Speech`, `Offensive`, `Normal`) with word-level rationale spans.
- **Evaluation Split:** Official test split comprising 1,924 posts annotated by 3 independent annotators, with ground truth resolved by majority vote.
