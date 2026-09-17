# Project Datasets: Digital Media Verification

This directory contains the dataset resources for the Ethical and Responsible AI course project:
**Digital Media Verification: Explainable and Uncertainty-Aware Deepfake, Propaganda and Hate-Speech Detection**.

All large video, image, and raw dataset files are strictly excluded from version control via `.gitignore`.

---

## 1. FaceForensics++ (Deepfake Detection)

### Purpose
Facial manipulation detection across real and tampered video sequences.

### Required Subset
**C23 (HQ / Light-Compression Subset)**, encoded with H.264 at constant rate factor 23, as specified in the project synopsis.

### Location
`data/faceforensics/C23/`

### Expected Directory Structure
```text
data/faceforensics/
└── C23/
    ├── conversion_dict.json
    ├── splits/
    │   ├── train.json   # 360 sequence pairs (720 videos)
    │   ├── val.json     # 70 sequence pairs (140 videos)
    │   └── test.json    # 70 sequence pairs (140 videos)
    ├── original_sequences/
    │   └── youtube/     # Pristine source video sequences
    └── manipulated_sequences/
        ├── Deepfakes/        # Deepfakes manipulation
        ├── Face2Face/        # Face2Face reenactment
        ├── FaceSwap/         # FaceSwap replacement
        └── NeuralTextures/   # NeuralTextures rendering
```

### Access Restrictions & Acquisition Process
- **Official Host:** Technical University of Munich (TUM) Computer Vision Group / Stanford University.
- **Access Procedure:** Due to copyright and human subject protections on the underlying YouTube videos, access to video downloads requires completing the official academic request form:  
  [FaceForensics Access Request Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform).
- **Official Script:** Upon acceptance, the authors provide the authorized download script (`download-FaceForensics_v3.py`) and temporary authentication tokens.
- **Repository Setup:** The official sequence mapping (`conversion_dict.json`) and the official train/validation/test sequence splits (`train.json`, `val.json`, `test.json`) are pre-populated in `data/faceforensics/C23/splits/`.
- **Mirror Policy:** Unofficial mirrors are strictly avoided to ensure academic integrity and data fidelity.

---

## 2. SemEval-2020 Task 11 (Propaganda Technique Detection)

### Purpose
Fine-grained detection and classification of propaganda techniques in news text.

### Location
`data/semeval2020_task11/`

### Source & Acquisition
- **Corpus:** Propaganda Techniques Corpus (PTC) Version 2, prepared by Qatar Computing Research Institute (QCRI) / Da San Martino et al. (EMNLP 2019 / SemEval-2020).
- **Subtasks:**
  - **Task 1 (SI):** Span Identification (character-level start and end offsets of propaganda fragments).
  - **Task 2 (TC):** Technique Classification (identifying which of the 14 techniques is applied).

### Available Splits & Size
- **Train Partition:** 371 articles with matching `.task1-SI.labels` and `.task2-TC.labels`.
- **Development Partition:** 75 articles (`dev-articles/`) with evaluation template (`dev-task-TC-template.out`).
- **Total Articles:** 446 articles (~350,000 tokens).

### 14 Supported Propaganda Techniques
1. `Appeal_to_Authority`
2. `Appeal_to_fear-prejudice`
3. `Bandwagon,Reductio_ad_hitlerum`
4. `Black-and-White_Fallacy`
5. `Causal_Oversimplification`
6. `Doubt`
7. `Exaggeration,Minimisation`
8. `Flag-Waving`
9. `Loaded_Language`
10. `Name_Calling,Labeling`
11. `Repetition`
12. `Slogans`
13. `Thought-terminating_Cliches`
14. `Whataboutism,Straw_Men,Red_Herring`

### Expected Directory Structure
```text
data/semeval2020_task11/
├── README.md
├── dev-articles/
├── dev-task-TC-template.out
├── train-articles/
├── train-labels-task1-span-identification/
├── train-labels-task2-technique-classification/
├── train-task1-SI.labels
└── train-task2-TC.labels
```

---

## 3. HateXplain (Hate-Speech & Rationale Analysis)

### Purpose
Three-class hate speech detection (Hate, Offensive, Normal) with human-annotated rationale masks for explainability.

### Location
`data/hatexplain/`

### Source & Acquisition
- **Publication:** Mathew et al., *"HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection"*, AAAI 2021.
- **Repository:** `https://github.com/punyajoy/HateXplain`

### Available Splits & Size
- **Total Posts:** 20,148 annotated social media posts (Twitter and Gab).
- **Train Split:** 15,383 posts (~76.3%).
- **Validation Split:** 1,922 posts (~9.5%).
- **Test Split:** 1,924 posts (~9.5%).
- **Data Size:** `dataset.json` (~12.2 MB), `post_id_divisions.json` (~592 KB).

### Label Structure & Rationale Information
- **Classification Classes (3 distinct classes):**
  1. `hatespeech`
  2. `offensive`
  3. `normal`
- **Annotators:** 3 independent annotations per post detailing target identity groups.
- **Rationales:** Binary token-level masks (`1` for tokens indicating hate/offensive rationale, `0` otherwise) allowing transparent explainability evaluation.

### Expected Directory Structure
```text
data/hatexplain/
├── README.md
├── classes.npy
├── classes_two.npy
├── dataset.json
└── post_id_divisions.json
```

---

## 4. Git and Storage Policy

1. All raw and processed dataset files in `data/faceforensics/`, `data/semeval2020_task11/`, and `data/hatexplain/` are excluded from version control via `.gitignore`.
2. Only documentation (`data/README.md`) and structural placeholders (`.gitkeep`) are committed.
3. Model training or pipeline modification must only reference data in these designated directories.
