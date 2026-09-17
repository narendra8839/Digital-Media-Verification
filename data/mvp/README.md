# MVP Dataset Subsets & Data Quality Report

This directory contains the inspected, validated, and balanced MVP subsets prepared for the model fine-tuning stage.

---

## 1. FaceForensics++ (Deepfake Detection)

- **Source & Provenance:**
  - Videos sourced from Kaggle development mirror (`xdxd003/ff-c23`), used **strictly for local engineering, debugging, and MVP demonstration**.
  - **Official TUM academic access is still pending.** Final reported research benchmarks will use the authorized dataset once author credentials are received.
- **Mapping Architecture:**
  - Transparent NTFS junctions map video directories into standard project structure at zero additional disk usage:
    - `data/faceforensics/C23/original_sequences/youtube/c23/videos/` (1,000 real videos)
    - `data/faceforensics/C23/manipulated_sequences/Deepfakes/c23/videos/` (1,000 fake videos)
    - `data/faceforensics/C23/manipulated_sequences/Face2Face/c23/videos/` (1,000 fake videos)
    - `data/faceforensics/C23/manipulated_sequences/FaceSwap/c23/videos/` (1,000 fake videos)
    - `data/faceforensics/C23/manipulated_sequences/NeuralTextures/c23/videos/` (1,000 fake videos)
  - `FaceShifter` and `DeepFakeDetection` are strictly excluded from this mapping.
- **Official Split Validation Results:**
  - **Train Split:** 360 sequence pairs. Expected originals: 720 (Found: 720, Missing: 0). Expected manipulated (4 methods): 1,440 (Found: 1,440, Missing: 0).
  - **Validation Split:** 70 sequence pairs. Expected originals: 140 (Found: 140, Missing: 0). Expected manipulated: 280 (Found: 280, Missing: 0).
  - **Test Split:** 70 sequence pairs. Expected originals: 140 (Found: 140, Missing: 0). Expected manipulated: 280 (Found: 280, Missing: 0).
  - **Total Corrupt / Unreadable:** 0.
- **Data Leakage Prevention:**
  - Strict video/source sequence level splitting from `splits/train.json`, `val.json`, and `test.json`.
  - Video ID overlap: Train/Val = 0, Train/Test = 0, Val/Test = 0.
  - Video frames from the same sequence never appear across multiple splits.
- **MVP Deepfake Subset:**
  - Sampled strictly from `train.json` for the training subset: 40 videos (20 real, 20 fake).
  - Validation subset (18 videos) and test subset (18 videos) are strictly sampled from `val.json` and `test.json` respectively.

---

## 2. SemEval-2020 Task 11 (Propaganda Technique Detection)

- **Manifest:** `data/mvp/semeval2020_task11/manifest.json`
- **Total Source Corpus:** 371 train articles, 75 dev articles, 6,129 annotated spans.
- **MVP Subset Size:** 700 samples (560 train, 140 val).
- **Supported Classes (14 Techniques):**
  - Exactly 40 train samples and 10 val samples per technique.
- **Data Leakage Prevention:**
  - Article-level partition (285 articles train, 72 articles validation).
  - No spans from validation articles ever appear in the training split.

---

## 3. HateXplain (Hate Speech & Explainability)

- **Manifest:** `data/mvp/hatexplain/manifest.json`
- **Total Source Corpus:** 20,148 annotated posts.
- **MVP Subset Size:** 900 samples (600 train, 150 val, 150 test).
- **Classes:** Strictly separated 3 classes (`hatespeech`, `normal`, `offensive`).
- **Class Balance:** Exactly 200 train, 50 val, and 50 test samples per class (1:1:1 balanced distribution).
- **Rationale Availability:**
  - 100% of selected `hatespeech` and `offensive` samples retain ground-truth token-level rationale binary masks.
- **Data Leakage Prevention:**
  - Strictly follows official benchmark post divisions (`post_id_divisions.json`). Zero cross-split contamination.

---

## 4. Configuration

All subset counts are controlled in `config/config.yaml` under `mvp_subsets`:
```yaml
mvp_subsets:
  random_seed: 42
  semeval:
    samples_per_class_train: 40
    samples_per_class_val: 10
  hatexplain:
    samples_per_class_train: 200
    samples_per_class_val: 50
    samples_per_class_test: 50
  deepfake:
    train_real_videos: 20
    train_fake_videos_per_method: 5
    val_real_videos: 10
    val_fake_videos_per_method: 2
    test_real_videos: 10
    test_fake_videos_per_method: 2
```
