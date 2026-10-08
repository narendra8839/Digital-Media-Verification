# Phase 2 — Data and Training Pipeline

## 1. Git Status
- **Current Branch:** `swaraj` (Verified via `git branch --show-current`).
- **Working Tree:** All uncommitted user modifications in `frontend/src/` (`App.css`, `Header.jsx`, `ResultDashboard.jsx`, `TokenAttributionView.jsx`, `index.css`) have been strictly preserved.
- **Commit Baseline:** `79a0767 Implement MVP digital media verification system with FastAPI, React frontend, XAI, UQ, and governance`. Zero changes or commits made on `main`.

---

## 2. Dataset Inventory

### FaceForensics++ (C23)
- **Physical Files Verified on Disk:**
  - `original_sequences/youtube/c23/videos/`: **1,000 mp4s** (Real video sequences)
  - `manipulated_sequences/Deepfakes/c23/videos/`: **1,000 mp4s** (Deepfakes)
  - `manipulated_sequences/Face2Face/c23/videos/`: **1,000 mp4s** (Face2Face)
  - `manipulated_sequences/FaceSwap/c23/videos/`: **1,000 mp4s** (FaceSwap)
  - `manipulated_sequences/NeuralTextures/c23/videos/`: **1,000 mp4s** (NeuralTextures)
  - **Total Physical Videos on Disk:** **5,000 MP4 files**.
- **Split Counts & Class Distribution:**
  - **Train Split (`train.json` - 360 pairs = 720 unique sequence IDs):**
    - Real: **720 videos**
    - Fake: **2,880 videos** (720 per manipulation method × 4 methods)
    - Total: **3,600 videos**
  - **Validation Split (`val.json` - 70 pairs = 140 unique sequence IDs):**
    - Real: **140 videos**
    - Fake: **560 videos** (140 per manipulation method × 4 methods)
    - Total: **700 videos**
  - **Test Split (`test.json` - 70 pairs = 140 unique sequence IDs):**
    - Real: **140 videos**
    - Fake: **560 videos** (140 per manipulation method × 4 methods)
    - Total: **700 videos** (Strictly held out for final Phase 4 evaluation; never used in training).
  - **Grand Total:** 3,600 + 700 + 700 = **5,000 physical videos**.
- **Verified Inconsistencies Resolved:**
  - *Previous documentation anomaly:* Earlier MVP reports stated the dataset comprised 3,000 videos. Factual inspection revealed all 5 directories exist on disk, each containing exactly 1,000 videos, totaling 5,000 videos. Every video maps 100% to the official splits.
- **Split Integrity:**
  - Unique sequence IDs: Train (720), Val (140), Test (140). Total = 1,000 unique IDs (`000` to `999`).
  - Train $\cap$ Val: $\emptyset$ (0 overlap)
  - Train $\cap$ Test: $\emptyset$ (0 overlap)
  - Val $\cap$ Test: $\emptyset$ (0 overlap)
  - **Verification:** 100% subject/sequence-disjoint. Zero frame or video leakage.

### HateXplain
- **Total Posts in Dataset:** **20,148 posts** in `data/hatexplain/dataset.json`.
- **Split Counts & Class Distribution (from `post_id_divisions.json`):**
  - **Train:** **15,383 posts**
    - `normal`: 6,251 (40.6%)
    - `hatespeech`: 4,748 (30.9%)
    - `offensive`: 4,384 (28.5%)
    - Annotated rationales present: 9,130 posts
  - **Validation:** **1,922 posts**
    - `normal`: 781 (40.6%)
    - `hatespeech`: 593 (30.9%)
    - `offensive`: 548 (28.5%)
    - Annotated rationales present: 1,141 posts
  - **Test:** **1,924 posts** (Held out untouched for Phase 4)
    - `normal`: 782 (40.6%)
    - `hatespeech`: 594 (30.9%)
    - `offensive`: 548 (28.5%)
    - Annotated rationales present: 1,142 posts
  - *Note on remaining 919 posts:* 20,148 total posts minus 19,229 partitioned posts leaves 919 unassigned posts in raw collection, which are omitted from official divisions by benchmark authors.
- **Split Integrity:**
  - Train $\cap$ Val: $\emptyset$ (0 post ID overlap)
  - Train $\cap$ Test: $\emptyset$ (0 post ID overlap)
  - Val $\cap$ Test: $\emptyset$ (0 post ID overlap)
  - **Verification:** 100% disjoint; zero leakage.

### SemEval-2020 Task 11 (Propaganda Techniques)
- **Source Files & Inventory:**
  - Raw articles: 371 text files in `data/semeval2020_task11/train-articles/*.txt`.
  - Annotated articles: 357 articles contain technique spans (14 articles are propaganda-free).
  - Annotations: 6,129 labeled spans in `data/semeval2020_task11/train-task2-TC.labels`.
  - Technique categories: 14 distinct rhetorical techniques.
  - Official Dev template: `dev-task-TC-template.out` contains masked labels (`?`) intended for competitive evaluation servers.
- **Audited Article-Disjoint Partition (Seed 42):**
  - **Train Split (286 articles):**
    - Total labeled spans: **4,681 spans** across all 14 classes.
  - **Validation Split (71 articles):**
    - Total labeled spans: **1,448 spans** across all 14 classes.
  - **Class Distribution (All 14 Classes Preserved):**
    - `Loaded_Language`: Train 1,607 | Val 516
    - `Name_Calling,Labeling`: Train 817 | Val 241
    - `Repetition`: Train 466 | Val 155
    - `Doubt`: Train 384 | Val 109
    - `Exaggeration,Minimisation`: Train 354 | Val 112
    - `Appeal_to_fear-prejudice`: Train 232 | Val 62
    - `Flag-Waving`: Train 177 | Val 52
    - `Causal_Oversimplification`: Train 154 | Val 55
    - `Appeal_to_Authority`: Train 109 | Val 35
    - `Slogans`: Train 104 | Val 25
    - `Whataboutism,Straw_Men,Red_Herring`: Train 83 | Val 25
    - `Black-and-White_Fallacy`: Train 78 | Val 29
    - `Thought-terminating_Cliches`: Train 59 | Val 17
    - `Bandwagon,Reductio_ad_hitlerum`: Train 57 | Val 15
- **Split Integrity:**
  - Train articles $\cap$ Val articles: $\emptyset$ (0 article overlap).
  - **Verification:** 100% article-disjoint. Zero span or article leakage.

---

## 3. Data Quality
- **Missing Files:** 0. Every single video file (5,000), article file (371), and post ID (19,229) referenced in the generated full split manifests was verified to exist physically on disk.
- **Duplicates:** 0 duplicate sequence IDs, post IDs, or span records.
- **Invalid Records:** 0 unmappable labels or corrupt entries.
- **Overlap Findings:** Strictly zero overlap across all train, validation, and test partitions across all three datasets.

---

## 4. Deepfake Training Pipeline
- **Architecture:** EfficientNet-B4 binary classifier (`Real` vs `Fake`).
- **Input Specifications:** 224×224 RGB image crops, ImageNet normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).
- **Data Flow:**
  - Reads `data/full_splits/deepfake_train.json` (3,600 videos) and `data/full_splits/deepfake_val.json` (700 videos).
  - Video-to-frame sampling: Configurable FPS sampling (`sample_fps=1.0`, `max_frames_per_video=4`).
  - Face Detection: MTCNN face detection and cropping to 224×224.
  - Video-level Grouping: Frame predictions are grouped by source video identifier for video-level aggregation.
- **Robustness Augmentation (`models/deepfake/augmentation.py`):**
  - Applied to **training data only**:
    1. In-memory JPEG compression ($Q \in [50, 95]$, $p=0.5$) simulating social media compression.
    2. Mild Gaussian blur ($kernel=3, \sigma \in [0.1, 1.2], p=0.3$) simulating transmission smoothing.
    3. Additive Gaussian noise ($\sigma \in [0.005, 0.03], p=0.3$) simulating sensor noise.
    4. Mild color jitter ($brightness=0.1, contrast=0.1, p=0.5$) and horizontal flips ($p=0.5$).
  - **Validation & Test Pipeline:** Strictly deterministic (Resize $\to$ ToTensor $\to$ Normalize). Verified via test assertions.
- **Class Balancing:**
  - FaceForensics++ has a 1:4 Real:Fake ratio (720 Real vs 2,880 Fake).
  - Implemented inverse class frequency weighting in `nn.CrossEntropyLoss`:
    $w_{real} = \frac{N}{2 \cdot N_{real}} = 2.50$, $w_{fake} = \frac{N}{2 \cdot N_{fake}} = 0.625$.
  - Prevents classifier bias toward the majority "fake" class.

---

## 5. Propaganda Training Pipeline
- **Architecture:** RoBERTa-base sequence classification.
- **Input Specifications:** Max sequence length = 128 tokens, padded/truncated.
- **Dataset:** `data/full_splits/propaganda_train.json` (4,681 spans) and `data/full_splits/propaganda_val.json` (1,448 spans).
- **Label Mapping:** 14 canonical SemEval-2020 Task 11 technique classes mapped to integer IDs ($0 \dots 13$).
- **Checkpoint Selection:** Strictly based on Validation Macro F1.
- **Test Isolation:** No test split is referenced or evaluated during training.

---

## 6. Hate-Speech Training Pipeline
- **Architecture:** BERT-base-uncased sequence classification.
- **Input Specifications:** Max sequence length = 128 tokens, padded/truncated.
- **Dataset:** `data/full_splits/hatexplain_train.json` (15,383 posts) and `data/full_splits/hatexplain_val.json` (1,922 posts).
- **Label Mapping:** 3 classes (`normal`: 0, `offensive`: 1, `hatespeech`: 2).
- **Checkpoint Selection:** Strictly based on Validation Macro F1.
- **Test Isolation:** The test split (`data/full_splits/hatexplain_test.json`, 1,924 posts) is completely removed from the training pipeline and held out untouched for Phase 4.

---

## 7. Reproducibility
- **Deterministic Seeding:** Explicit `seed` parameter (default 42) controlling `torch.manual_seed`, `np.random.seed`, and `torch.cuda.manual_seed_all`.
- **Configuration Persistence:** `training_config.json` containing all hyperparameters (seed, batch size, learning rate, max length, manifest paths, sample counts, device) is automatically saved alongside the model weights.
- **Metrics Persistence:** Validation metrics are saved in `metrics.json`.
- **Checkpoint Safety:** New runs write to isolated directories (`checkpoint_full` or user-defined `output_dir`), guaranteeing that existing MVP checkpoints (`models/deepfake/checkpoint/best_model.pt`, etc.) are NEVER overwritten.
- **Structured Logging:** All pipelines log sample counts, class distributions, per-epoch train loss, validation loss, validation accuracy/macro F1, and checkpoint save events.

---

## 8. Device Readiness
- **Configurable Device Selection:** All training pipelines utilize dynamic device detection:
  `device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))`.
- **Current Environment State:**
  - Physical NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) is present and functional via `nvidia-smi`.
  - Active Python 3.14 installation currently has `torch-2.14.0+cpu` installed (`torch.cuda.is_available() == False`).
  - All training scripts run seamlessly on CPU for smoke testing and development.
- **Why Full GPU Training is Deferred to Phase 3:**
  - Upgrading to a CUDA-enabled PyTorch build requires environment adjustments that belong strictly to Phase 3.
  - Phase 2 focuses exclusively on data preparation, split integrity, pipeline architecture, robustness augmentation, and smoke tests.

---

## 9. Smoke Tests
- **Test Script:** `scripts/smoke_test_training_pipelines.py`.
- **Execution Mode:** 1 epoch, batch size 4, tiny sample sizes (8–16 samples), testing end-to-end forward pass, loss calculation, backward pass, optimizer step, validation evaluation, and checkpoint/config saving.
- **Results:**
  1. **Deepfake Pipeline:**
     - 4 train videos (8 frames), 2 val videos (4 frames).
     - Augmentation verified; class-weighted loss verified; checkpoint (`best_model.pt`), `training_config.json`, and `metrics.json` saved.
     - **Status: PASSED**
  2. **Propaganda Pipeline:**
     - 16 train spans, 8 val spans across technique classes.
     - RoBERTa tokenization and forward/backward pass verified; checkpoint (`model.safetensors`), `tokenizer/`, and `training_config.json` saved.
     - **Status: PASSED**
  3. **Hate Speech Pipeline:**
     - 16 train posts, 8 val posts across 3 classes.
     - BERT tokenization and forward/backward pass verified; checkpoint (`model.safetensors`), `tokenizer/`, and `training_config.json` saved.
     - **Status: PASSED**
- **Existing Baseline Test Suite:**
  - Command: `python -m pytest -q tests`
  - Result: **41 passed, 0 failed** in 79.86s (100% baseline preserved, zero regressions).
  - Frontend Vitest suite: **10 passed, 0 failed** in 2.88s.

---

## 10. Files Changed

### Created:
1. `data/full_splits/deepfake_train.json` (3,600 videos)
2. `data/full_splits/deepfake_val.json` (700 videos)
3. `data/full_splits/deepfake_test.json` (700 videos)
4. `data/full_splits/hatexplain_train.json` (15,383 posts)
5. `data/full_splits/hatexplain_val.json` (1,922 posts)
6. `data/full_splits/hatexplain_test.json` (1,924 posts)
7. `data/full_splits/propaganda_train.json` (4,681 spans)
8. `data/full_splits/propaganda_val.json` (1,448 spans)
9. `data/full_splits/manifest_summary.json` (Complete audit statistics)
10. `models/deepfake/augmentation.py` (Configurable JPEG compression, noise, blur robustness transforms)
11. `scripts/prepare_full_datasets.py` (Manifest generator and split integrity auditor)
12. `scripts/smoke_test_training_pipelines.py` (Automated smoke test suite for all 3 pipelines)
13. `docs/PHASE_2_DATA_AND_TRAINING.md` (This document)

### Revised:
1. `models/deepfake/train.py` (Full dataset loader, augmentation integration, class balance loss, validation-only checkpointing, device readiness)
2. `models/propaganda/train.py` (Full SemEval dataset support, article-disjoint split, validation F1 checkpointing, isolated output path)
3. `models/hate_speech/train.py` (Full HateXplain dataset support, complete test split isolation, validation F1 checkpointing, isolated output path)

---

## 11. Problems Discovered

1. **PyTorch CPU-Only Environment**
   - *Status:* **VERIFIED** | **NEEDS PHASE 3**
   - Active Python environment has `torch-2.14.0+cpu`. PyTorch CUDA enablement will be performed in Phase 3 prior to full training.
2. **FaceForensics++ 1:4 Class Imbalance**
   - *Status:* **VERIFIED** | **NEEDS PHASE 3**
   - Addressed in pipeline with inverse class frequency loss weighting ($w_{real}=2.50, w_{fake}=0.625$). Will be monitored during full Phase 3 training.
3. **SemEval-2020 Official Dev Labels Masked**
   - *Status:* **VERIFIED** | **NEEDS PHASE 3**
   - The competition dev set has masked labels. Successfully resolved by establishing a rigorous article-disjoint 80/20 partition (seed 42) on the 357 annotated articles.
4. **Transient Video Job Store in Backend**
   - *Status:* **VERIFIED** | **NEEDS FUTURE SCOPE**
   - In-memory job store in `api/routes/verify.py` is transient across process restarts. Safe for MVP/academic demonstration.

---

## 12. Phase 2 Status
**COMPLETE**
All Phase 2 objectives (full dataset auditing, leakage-free split manifests, robustness augmentation, reproducible training pipelines, test split isolation, small smoke tests, and baseline test preservation) have been genuinely completed and verified.
