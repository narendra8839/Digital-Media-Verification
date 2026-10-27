# Model Checkpoint Manifest and Cryptographic Provenance

This document provides a verified cryptographic manifest of all neural network model weights in the Digital Media Verification (DMV) framework.

---

## 1. Production Model Checkpoints

The production checkpoint evaluated across the full benchmark and deployed in the DMV FastAPI backend is the **Baseline Full-Trained EfficientNet-B4**:

| Model Name | Role | Expected Relative Path | File Size (Bytes) | Cryptographic SHA-256 Hash |
| :--- | :--- | :--- | :---: | :--- |
| **Production Baseline** | Active Deepfake Detector | `models/deepfake/checkpoint_full/best_model.pt` | 70,979,219 | `5878ae38bf5239dd9889e010fafa1d8c7da4198ac260542dc96681ede1e63234` |

> [!IMPORTANT]
> The production checkpoint has remained strictly byte-for-byte unchanged across all research audits and optimization phases. No checkpoint replacement or fine-tuning was performed.

### Associated NLP & Auxiliary Models
- **Propaganda Detector**: Fine-tuned RoBERTa-base (`roberta-base`) on SemEval-2020 Task 11 (14-technique multi-label span classifier). Checkpoint directory: `models/propaganda/checkpoint/` or dynamically provisioned via Hugging Face Hub cache.
- **Hate Speech Detector**: Fine-tuned BERT-base-uncased (`bert-base-uncased`) on HateXplain (3-class: hate speech, offensive language, normal). Checkpoint directory: `models/hate_speech/checkpoint/`.
- **Audio ASR**: OpenAI Whisper `base` model (~140 MB), provisioned via `openai-whisper` runtime caching in `~/.cache/whisper`.
- **OCR Engine**: EasyOCR English CRAFT detector and recognition networks, cached in `~/.EasyOCR/model`.
- **Face Detection**: MTCNN (`facenet-pytorch`) cascaded multi-task convolutional networks (P-Net, R-Net, O-Net), loaded in-memory.

---

## 2. Experimental Robustness Checkpoints (Archived)

During the Deepfake Robustness research phases (V1, V2, and V3), experimental candidates were trained to evaluate resilience against real-world corruptions (JPEG compression, Gaussian noise, blurring, downsampling).

These candidates are archived in the experimental robustness worktree (`.worktrees/robustness`) on branch `feat/deepfake-robustness`. Because they did not satisfy all production acceptance criteria (notably failing the strict JPEG $Q=30$ fake recall gate), they are preserved solely for experimental reproducibility and are **not** deployed to production:

| Candidate ID | Model Description | Relative Path in Worktree | Size (Bytes) | Cryptographic SHA-256 Hash | Status |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Candidate V1** | Balanced fine-tuning with basic augmentation | `models/deepfake/checkpoint_robust_candidate/best_model.pt` | 70,979,219 | `c071ff9ac178e02a479d91acff5ac4566b06a5f042946b36308b75d02f441dd0` | Archived (Experimental) |
| **Candidate V2-A** | Heavy noise & blur augmentations | `models/deepfake/checkpoint_robust_v2a/best_model.pt` | 70,979,219 | `96e178f3d755790eb66a15199d9cc3cfcae7ec1d925f9adfcd990f14e195b0f6` | Archived (Experimental) |
| **Candidate V2-B** | Frequency residual & multi-distortion training | `models/deepfake/checkpoint_robust_v2b/best_model.pt` | 70,979,219 | `f8b8c0ba272b37b353fbcaa2b3148844a0d9b0d4278c52b73a3259f9d5d277c8` | Archived (Experimental, 6% noise recall collapse) |
| **Candidate V3-A** | Stochastic noise regularized ($\sigma \in [5, 15]$) | `models/deepfake/checkpoint_robust_v3a/best_model.pt` | 70,979,219 | `5f8057f3d54759c2c71163e6d4c60cf6bf303dae611739491e02e1ae8452c41d` | Archived (Experimental, failed JPEG Q30 fake recall) |
| **Candidate V3-B** | Multi-scale downsampling & mild blur | `models/deepfake/checkpoint_robust_v3b/best_model.pt` | 70,979,219 | `1b53f93e70d2301fb819f987d2f0c0bd6c6c37ac6560142d933d8542f9470032` | Archived (Experimental, failed JPEG Q30 fake recall) |
| **Candidate V3-C** | Frequency preservation loss & high dropout | `models/deepfake/checkpoint_robust_v3c/best_model.pt` | 70,979,219 | `12b552738dcfcf598b5b18b2ce65820087063703346ac76b24c7b6382efefd34` | Archived (Experimental, failed JPEG Q30 & noise gate) |

---

## 3. Verification Commands

To verify the integrity of the production checkpoint:

### PowerShell
```powershell
Get-FileHash -Algorithm SHA256 "models/deepfake/checkpoint_full/best_model.pt"
```

### Linux / WSL Bash
```bash
sha256sum models/deepfake/checkpoint_full/best_model.pt
```

Expected output:
```
5878ae38bf5239dd9889e010fafa1d8c7da4198ac260542dc96681ede1e63234
```
