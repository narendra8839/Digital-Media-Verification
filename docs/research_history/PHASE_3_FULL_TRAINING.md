# Phase 3 — GPU Environment & Full-Data Model Training Report

## 1. Executive Summary & Status

- **Phase Status:** **COMPLETE**
- **Git Branch:** `swaraj` (Exclusively worked on `swaraj`; zero changes on `main`).
- **User Frontend Edits:** Untouched (`frontend/src/` uncommitted changes preserved intact).
- **Test Set Isolation:** Guaranteed. Neither `data/full_splits/deepfake_test.json` nor `data/full_splits/hatexplain_test.json` was loaded, evaluated, or touched during Phase 3.
- **Baseline Checkpoints:** Strictly preserved in `models/*/checkpoint/`. All Phase 3 candidates saved exclusively to `models/*/checkpoint_full/`.
- **Regression Suite:** **41 / 41 passed** (100% pass rate on full regression test suite).

---

## 2. Environment Architecture & WSL2 GPU Passthrough

### Problem Resolution
On Windows 11 Home, Smart App Control (SAC) blocks unsigned native PyTorch CUDA DLLs with `WinError 4551`. Without modifying Windows Code Integrity policies or disabling Smart App Control, Option B (**WSL2 + Ubuntu + NVIDIA DirectX GPU-P passthrough**) was successfully configured:
- **Host GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB GDDR6 VRAM)
- **Host Windows NVIDIA Driver:** 592.82 (CUDA API 13.1)
- **WSL Distribution:** Ubuntu 26.04 LTS on WSL2 (Kernel 6.6.x)
- **Linux GPU Interface:** `/dev/dxg` passthrough (No Linux display drivers installed inside WSL; host driver passthrough leveraged).
- **Security Posture:** Windows Smart App Control remains fully enabled in enforcement mode.

### Reusable Device-Level Environment (`~/venvs/ml-gpu`)
Created a standalone, reusable virtual environment for future ML workloads on this laptop:
- **Location:** `/home/swaraj_shedge/venvs/ml-gpu`
- **Python Version:** 3.14.4
- **Pip Version:** 26.2.1
- **PyTorch Installed:** `torch==2.14.0+cu130`, `torchvision==0.29.0+cu130`, `triton==3.8.0`
- **CUDA Runtime:** 13.0
- **Verification Script:** `~/gpu_verify.py`
- **Real CUDA Matrix Multiplication Benchmark:** Passed. 4096×4096 tensor matrix multiplication on `cuda:0` allocated 200.12 MB VRAM and synchronized successfully.
- **System Documentation:** Detailed instructions recorded in `~/ML_GPU_ENVIRONMENT.md`.

### Project-Specific Environment (`~/venvs/dmv-gpu`)
Isolated project environment configured for Digital Media Verification:
- **Location:** `/home/swaraj_shedge/venvs/dmv-gpu`
- **Core Packages:** `torch==2.14.0+cu130`, `torchvision==0.29.0+cu130`, `transformers==5.18.0`, `scikit-learn==1.9.1`, `opencv-python-headless==5.0.0.93`, `facenet-pytorch==2.5.3`, `easyocr==1.7.2`, `albumentations==2.0.8`, `fastapi==0.142.2`, `pytest`
- **Model GPU Smoke Test:** Loaded MVP BERT checkpoint onto `cuda:0`; verified parameters and tensors on `cuda:0`; forward pass produced expected logits shape `[1, 3]`.

---

## 3. Strict Safety Enforcements Across Training Pipelines

Every training script was updated to ensure strict reproducibility and hardware compliance:
1. **Strict CUDA Requirement:** Each pipeline raises a `RuntimeError` if `selected_device.type != 'cuda'`, strictly prohibiting silent fallback to CPU.
2. **Startup Verification Logging:** Formally prints model architecture, dataset manifests, batch size, learning rate, epoch count, sample counts, seed, and parameter device prior to training.
3. **Batch Tensor Device Assertion:** The first training batch asserts `batch_tensor.device.type == 'cuda'`.
4. **Isolated Checkpoint Directories:** Outputs are saved to versioned directories (`checkpoint_full/`), keeping MVP baseline weights untouched.

---

## 4. Full Candidate Model Training Results

### Candidate 1: Hate Speech Classification (BERT-base-uncased)

- **Architecture:** `bert-base-uncased` (Sequence Classification, 3 Classes: `normal`, `offensive`, `hatespeech`)
- **Dataset Manifests:**
  - Train: `data/full_splits/hatexplain_train.json` (15,383 posts)
  - Validation: `data/full_splits/hatexplain_val.json` (1,922 posts)
- **Training Hyperparameters:** `epochs=3`, `batch_size=8`, `lr=3e-5`, `max_length=128`, `seed=42`, `optimizer=AdamW`
- **Hardware Telemetry:**
  - Device: `cuda:0` (NVIDIA GeForce RTX 4050 Laptop GPU)
  - GPU Compute Utilization: **96% – 98%**
  - VRAM Allocated: **2,421 MiB** (48W – 53W power draw)
  - Training Duration: ~21 minutes
- **Validation Progression:**
  - Epoch 1: Train Loss: `0.8019` | Val Accuracy: `0.6727` | Val Macro F1: `0.6376`
  - Epoch 2: Train Loss: `0.6364` | Val Accuracy: `0.6842` | Val Macro F1: `0.6756` (Selected Best Checkpoint)
  - Epoch 3: Train Loss: `0.4432` | Val Accuracy: `0.6779` | Val Macro F1: `0.6744`
- **Final Validation Metrics (Best Checkpoint):**
  - Accuracy: **0.6842** (Baseline MVP: 0.6035 -> **+8.07%**)
  - Macro Precision: **0.6740**
  - Macro Recall: **0.6809**
  - Macro F1: **0.6756** (Baseline MVP: 0.5898 -> **+8.58%**)
  - Weighted F1: **0.6828**
- **Checkpoint Location:** `models/hate_speech/checkpoint_full/` (`model.safetensors`, `config.json`, `tokenizer/`, `metrics.json`, `training_config.json`, `label_mapping.json`)

---

### Candidate 2: Propaganda Technique Classification (RoBERTa-base)

- **Architecture:** `roberta-base` (Sequence Classification, 14 SemEval-2020 Task 11 Classes)
- **Dataset Manifests:**
  - Train: `data/full_splits/propaganda_train.json` (4,681 spans across disjoint articles)
  - Validation: `data/full_splits/propaganda_val.json` (1,448 spans across disjoint articles)
- **Training Hyperparameters:** `epochs=3`, `batch_size=8`, `lr=3e-5`, `max_length=128`, `seed=42`, `optimizer=AdamW`
- **Hardware Telemetry:**
  - Device: `cuda:0` (NVIDIA GeForce RTX 4050 Laptop GPU)
  - GPU Compute Utilization: **97% – 99%**
  - VRAM Allocated: **2,731 MiB** (49W – 52W power draw)
  - Training Duration: ~6 minutes
- **Validation Progression:**
  - Epoch 1: Train Loss: `1.5835` | Val Accuracy: `0.5463` | Val Macro F1: `0.2427`
  - Epoch 2: Train Loss: `1.0825` | Val Accuracy: `0.5939` | Val Macro F1: `0.3602`
  - Epoch 3: Train Loss: `0.7926` | Val Accuracy: `0.5932` | Val Macro F1: `0.4175` (Selected Best Checkpoint)
- **Final Validation Metrics (Best Checkpoint):**
  - Accuracy: **0.5932** (Baseline MVP: 0.3540 -> **+23.92%**)
  - Macro Precision: **0.5238**
  - Macro Recall: **0.4030**
  - Macro F1: **0.4175** (Baseline MVP: 0.2882 -> **+12.93%**)
  - Weighted F1: **0.5539**
- **Checkpoint Location:** `models/propaganda/checkpoint_full/` (`model.safetensors`, `config.json`, `tokenizer/`, `metrics.json`, `training_config.json`, `label_mapping.json`)

---

### Candidate 3: Deepfake Video Detection (EfficientNet-B4)

- **Architecture:** `EfficientNet-B4` binary classification with MTCNN face crops & robust augmentations (JPEG compression, Gaussian blur, Gaussian noise, color jitter)
- **Dataset Manifests:**
  - Train: `data/full_splits/deepfake_train.json` (3,600 videos -> 7,200 sampled face frames)
  - Validation: `data/full_splits/deepfake_val.json` (700 videos -> 1,400 sampled face frames)
- **Training Hyperparameters:** `epochs=3`, `batch_size=8`, `lr=1e-4`, `frames_per_video=2`, `class_weights=[2.500, 0.625]`, `seed=42`, `optimizer=AdamW`
- **Hardware Telemetry:**
  - Frame Extraction: Cached to native ext4 (`~/cache_deepfake_frames/`) to avoid 9P filesystem seek constraints.
  - Device: `cuda:0` (NVIDIA GeForce RTX 4050 Laptop GPU)
  - GPU Compute Utilization: **41% – 47%** (sustained forward/backward passes on RTX 4050)
  - VRAM Allocated: **2,255 MiB** (34W – 38W power draw)
  - Training Duration: ~23 minutes (Extraction + 3 epochs GPU training)
- **Validation Progression:**
  - Epoch 1: Train Loss: `0.6674` | Val Frame Acc: `0.6850` | Val Video Acc: `0.7086`
  - Epoch 2: Train Loss: `0.5784` | Val Frame Acc: `0.8386` | Val Video Acc: `0.8714`
  - Epoch 3: Train Loss: `0.5023` | Val Frame Acc: `0.8557` | Val Video Acc: `0.8829` (Selected Best Checkpoint)
- **Final Validation Metrics (Best Checkpoint):**
  - Frame-Level Accuracy: **0.8557** (Baseline MVP: 0.7714 -> **+8.43%**)
  - Frame-Level Macro F1: **0.8030**
  - Video-Level Accuracy: **0.8829** (Baseline MVP: 0.8143 -> **+6.86%**)
  - Video-Level Macro Precision: **0.8106**
  - Video-Level Macro Recall: **0.8839**
  - Video-Level Macro F1: **0.8374**
  - Video-Level Weighted F1: **0.8890**
- **Checkpoint Location:** `models/deepfake/checkpoint_full/` (`best_model.pt`, `metrics.json`, `training_config.json`, `label_mapping.json`)

---

## 5. Comparative Validation Summary: Baseline MVP vs. Phase 3 Candidate

| Task & Architecture | Dataset & Samples Evaluated | MVP Baseline Metric | Phase 3 Full Candidate Metric | Absolute Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Hate Speech** (BERT-base) | HateXplain (1,922 val posts) | Acc: `0.6035`, Macro F1: `0.5898` | Acc: **0.6842**, Macro F1: **0.6756** | **+8.58% F1** |
| **Propaganda** (RoBERTa-base) | SemEval-2020 (1,448 val spans) | Acc: `0.3540`, Macro F1: `0.2882` | Acc: **0.5932**, Macro F1: **0.4175** | **+12.93% F1** |
| **Deepfake** (EfficientNet-B4) | FF++ C23 (700 val videos) | Video Acc: `0.8143` | Video Acc: **0.8829**, Video F1: **0.8374** | **+6.86% Acc** |

---

## 6. Regression Testing Verification

Following completion of all candidate trainings, the full project test suite was executed on the Windows host environment:
```powershell
python -m pytest -q tests
```
- **Test Result:** `41 passed, 11 warnings in 81.65s`
- **Integrity Confirmed:** All multimodal API endpoints, routers, explanation services, uncertainty engines, and baseline models remain fully functional without regressions.

---

## 7. Phase 3 Completion Sign-Off

All objectives of Phase 3 have been fully achieved:
1. Reusable WSL2 + PyTorch CUDA environment established on RTX 4050 GPU without compromising Windows Smart App Control.
2. Verified real tensor compute on CUDA.
3. Successfully trained all three full-dataset candidate models on `cuda:0` with strict CPU fallback prohibition.
4. Checkpoints, metrics, and configurations recorded in `checkpoint_full/` directories.
5. Zero leakage: Test splits held out completely for Phase 4.
6. Baseline MVP checkpoints and user frontend modifications strictly preserved.
