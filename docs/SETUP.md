# Environment Setup and Installation Guide

This document provides a step-by-step guide for cloning, installing dependencies, downloading model weights, configuring GPU/CPU acceleration, and running the Digital Media Verification (DMV) platform.

---

## 1. Prerequisites

Before setting up the project, ensure your host system has the following prerequisites:

- **Operating System:** Linux (Ubuntu 20.04+ recommended), macOS, or Windows 10/11 (PowerShell or WSL 2)
- **Python:** **Python 3.10, 3.11, or 3.12** (Tested & Recommended). Python 3.14 is also supported for core execution, but 3.10–3.12 is recommended for optimal binary wheel compatibility.
- **Node.js:** **Node.js 22 LTS (version 22.12 or newer)** recommended (compatible with Vite 8 and frontend build dependencies; Node.js 18 is not supported). Tested with Node.js v25.6.0 and npm 11.x.
- **FFmpeg:** System binary required for video audio demuxing and extraction (`audio_demux.py`).
- **CUDA Toolkit & cuDNN (Optional):** NVIDIA GPU (e.g., RTX 30xx/40xx) with CUDA 12.x for accelerated inference. CUDA is strictly optional; CPU inference is fully supported.

### Verifying Prerequisites

Run the following commands in your terminal:
```bash
python --version   # or python3 --version
node --version
npm --version
ffmpeg -version
```

If **FFmpeg** is not installed:
- **Windows:** `winget install Gyan.FFmpeg` or `choco install ffmpeg` (ensure ffmpeg is added to system `PATH`).
- **Ubuntu/Debian:** `sudo apt update && sudo apt install -y ffmpeg`
- **macOS:** `brew install ffmpeg`

---

## 2. Backend Setup

### A. Clone the Repository
```bash
git clone https://github.com/narendra8839/Digital-Media-Verification.git
cd Digital-Media-Verification
```

### B. Create and Activate Virtual Environment
```bash
# On Linux / macOS / WSL:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### C. Install Python Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> **PDF Report Generation Dependency:**
> The image analysis report export feature uses `reportlab>=4.0.0` (included in `requirements.txt`). It performs pure Python in-memory document compilation without requiring external binaries like wkhtmltopdf, LaTeX, or headless browsers.

If you plan to run tests or perform development tasks, also install development dependencies:
```bash
pip install -r requirements-dev.txt
```

### D. PyTorch Acceleration (CPU vs GPU)

By default, standard PyTorch will be installed. To explicitly choose your target acceleration:

- **CPU-Only Mode (Universal, default):**
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

- **CUDA 12.x GPU Mode (Optional, NVIDIA GPUs):**
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```

Verify PyTorch device detection:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

> **Tip:** You can force CPU execution at any time by setting the environment variable:
> - PowerShell: `$env:DMV_DEVICE = "cpu"`
> - Linux/macOS: `export DMV_DEVICE=cpu`

---

## 3. Model Checkpoints Installation

The repository tracks configurations, architectures, and tokenizer configurations. Due to Git file size limits, production model binaries will be published under release tag `v1.0.0` once finalized by the repository maintainer.

> [!WARNING]
> **Release Availability Notice:** GitHub Release `v1.0.0` has **not yet been published** on GitHub (`https://github.com/narendra8839/Digital-Media-Verification/releases/tag/v1.0.0` currently returns HTTP 404). Do not attempt to download archives from this URL until published.
>
> **Actual Provisioning Process:**
> - If setting up in an environment where models have not yet been extracted, obtain the three `.tar.gz` archives from local storage (e.g., `D:\DMV_Data\DMV_Model_Release\` or team backups) and place them in the workspace root.
> - If working with an already initialized repository checkout, verify that `models/deepfake/checkpoint_full/best_model.pt` exists and matches the verified SHA-256 hash.

> **Important:** The application includes weight-validation guards. If model weights are missing, the loaders will raise a clear `FileNotFoundError` and reject inference requests rather than producing invalid predictions from uninitialized weights.

### Release Checkpoint Archives

| Model | Archive Name | Size | SHA-256 Checksum | Extraction Target |
| :--- | :--- | :--- | :--- | :--- |
| **Visual Deepfake** (EfficientNet-B4) | `deepfake_efficientnet_b4_checkpoint.tar.gz` | ~63 MB | `7df27eb2dfe1c39759508778a7c37eb056a1ac4c88265a3ce2e861fe225d609b` | `models/deepfake/` |
| **Hate Speech** (BERT 3-class) | `hate_speech_bert_checkpoint.tar.gz` | ~388 MB | `fd4988f1492d7a6a37fed8fa1891db17d31982fcf2252c92479d4d1b6b6fa822` | `models/hate_speech/` |
| **Propaganda** (RoBERTa 14-class) | `propaganda_roberta_exp1_checkpoint.tar.gz` | ~409 MB | `f4cc85cafcfc16a9f6cb3ffb55e06bbc2f1bdc0cba516a4e3567964b630f3cb0` | `models/propaganda/optimization/` |

### Extraction Instructions (from Local Storage or Published Release)

1. Place the three `.tar.gz` archives into the repository root directory.

2. (Optional but recommended) Verify archive checksums:
   - **Linux/macOS:**
     ```bash
     sha256sum deepfake_efficientnet_b4_checkpoint.tar.gz
     sha256sum hate_speech_bert_checkpoint.tar.gz
     sha256sum propaganda_roberta_exp1_checkpoint.tar.gz
     ```
   - **Windows (PowerShell):**
     ```powershell
     Get-FileHash -Algorithm SHA256 .\deepfake_efficientnet_b4_checkpoint.tar.gz
     Get-FileHash -Algorithm SHA256 .\hate_speech_bert_checkpoint.tar.gz
     Get-FileHash -Algorithm SHA256 .\propaganda_roberta_exp1_checkpoint.tar.gz
     ```

3. Extract each archive from the repository root:
   ```bash
   # Deepfake detector weights -> models/deepfake/checkpoint_full/best_model.pt
   tar -xzf deepfake_efficientnet_b4_checkpoint.tar.gz -C models/deepfake/

   # Hate speech detector weights -> models/hate_speech/checkpoint_full/model.safetensors
   tar -xzf hate_speech_bert_checkpoint.tar.gz -C models/hate_speech/

   # Propaganda detector weights -> models/propaganda/optimization/exp1_sqrt_weighted_scheduled/model.safetensors
   mkdir -p models/propaganda/optimization
   tar -xzf propaganda_roberta_exp1_checkpoint.tar.gz -C models/propaganda/optimization/
   ```

4. Confirm expected files exist:
   - `models/deepfake/checkpoint_full/best_model.pt`
   - `models/hate_speech/checkpoint_full/model.safetensors`
   - `models/propaganda/optimization/exp1_sqrt_weighted_scheduled/model.safetensors`

---

## 4. Frontend Setup

In a separate terminal window:
```bash
cd frontend
npm install
```

### Environment Configuration
The frontend uses Vite environment variables. A minimal template is provided at `frontend/.env.example`:
```bash
# If .env does not exist, copy from .env.example
cp .env.example .env
```
Default `frontend/.env` content:
```env
VITE_API_BASE_URL=http://localhost:8000
```

---

## 5. Running the Application

Launch the backend API and frontend dev server in **separate terminal windows**:

### Terminal 1: FastAPI Backend
From the repository root (with virtual environment active):
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- **API URL:** `http://localhost:8000`
- **Interactive Swagger Docs:** `http://localhost:8000/docs`
- **Liveness Health Check:** `http://localhost:8000/health`
  - Returns `{"status": "healthy", "service": "digital-media-verification"}`
- **Readiness Check (Model Loading):** `http://localhost:8000/ready`
  - Returns `{"status": "ready", ...}` when all detector models are pre-warmed.

### Terminal 2: React Frontend
From the `frontend/` directory:
```bash
cd frontend
npm run dev
```
- **Web Dashboard:** `http://localhost:5173`
- Open `http://localhost:5173` in your browser to interact with the full decision-support UI.

---

## 6. Running Tests

### Backend Test Suite (Pytest)
Run the complete backend integration and unit test suite:
```bash
pytest tests/ -v
```
All backend suites should pass (121 passed, 1 skipped across API contracts, report generation, artifact retention, explainability, uncertainty, multimodal pipelines, label mapping, and robustness suites).

### Frontend Test Suite (Vitest)
From the `frontend/` directory:
```bash
cd frontend
npm test -- --run
```
All 19 component and integration tests across `App.test.jsx` and `ReportDownload.test.jsx` should pass.

### Production Build Verification
Verify that the frontend builds cleanly for production:
```bash
cd frontend
npm run build
```

---

## 7. Common Setup Errors and Troubleshooting

### 1. `FileNotFoundError: Deepfake detector checkpoint not found...`
- **Cause:** The checkpoint `.tar.gz` archives have not been extracted into their designated paths.
- **Solution:** Follow Section 3 above and ensure `best_model.pt` and `model.safetensors` are placed in `models/deepfake/checkpoint_full/`, `models/hate_speech/checkpoint_full/`, and `models/propaganda/optimization/exp1_sqrt_weighted_scheduled/`.

### 2. `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`
- **Cause:** FFmpeg is not installed or not in your system's `PATH`.
- **Solution:** Install FFmpeg using `winget install Gyan.FFmpeg` (Windows) or `sudo apt install ffmpeg` (Linux) and reopen your terminal.

### 3. Port 8000 or 5173 Already in Use
- **Cause:** Another process or previous server instance is occupying the port.
- **Solution:**
  - Backend: Run on a different port: `uvicorn api.main:app --port 8001 --reload` and update `VITE_API_BASE_URL` in `frontend/.env`.
  - Or terminate the existing process using the port.

### 4. PyTorch CUDA Out of Memory (OOM)
- **Cause:** Insufficient GPU VRAM when running multiple models concurrently.
- **Solution:** Set `DMV_DEVICE=cpu` to force CPU inference:
  ```bash
  export DMV_DEVICE=cpu  # Linux/macOS
  $env:DMV_DEVICE="cpu"  # Windows PowerShell
  ```

---

## 8. Artifact Retention & Storage Safety Configuration

The backend manages spatial explainability files (e.g., Grad-CAM heatmaps) under the managed artifact directory (`artifacts/`). To prevent disk accumulation over continuous usage, an automated background cleanup worker prunes expired artifacts.

### Configuration Environment Variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `DMV_ARTIFACT_TTL_HOURS` | `24.0` | Maximum lifetime of generated explanation artifacts in hours before eligible for cleanup. |
| `DMV_ARTIFACT_CLEANUP_INTERVAL_MINUTES` | `60.0` | Execution frequency of the background pruning worker in minutes. |
| `DMV_ARTIFACT_DIR` | `artifacts` | Relative or absolute path to the managed artifact storage directory. |
| `DMV_ARTIFACT_CLEANUP_ENABLED` | `1` (true) | Set to `0` or `false` to disable automatic background cleanup. |

### Safety Guarantees
- **Managed Targets Only:** Only explanation overlays matching application naming patterns (`gradcam_*.png`, `*_gradcam.png`) are eligible for deletion.
- **Protected Files:** `.gitkeep`, `.gitignore`, `README.md`, `manifest.json`, models in `models/`, reports in `reports/`, and subdirectories are strictly protected from deletion.
- **Active-Write Buffer:** Files modified within the last 120 seconds or registered in active inference pipelines are protected from premature pruning.
- **Traversal & Symlink Defenses:** Resolves paths against the canonical root to prevent directory traversal escapes and ignores symlinks pointing outside the storage root.
- **Graceful UI Handling:** If an artifact expires while viewed on the frontend, an informative expiration alert is displayed advising the user to rerun verification.
