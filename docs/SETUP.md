# Environment Setup and Installation Guide

This document outlines the step-by-step procedure for setting up, installing dependencies, configuring GPU acceleration, downloading model weights, and running the Digital Media Verification (DMV) platform.

---

## 1. Prerequisites

Ensure the following prerequisites are installed on your host system:

- **Operating System:** Linux, macOS, or Windows 10/11 (with WSL 2 recommended for CUDA GPU execution on Windows)
- **Python:** Version 3.10, 3.11, 3.12, 3.13, or 3.14
- **Node.js:** Version 18.x or newer (with `npm`)
- **FFmpeg:** System binary required for audio extraction from video containers
- **CUDA Toolkit & cuDNN (Optional, recommended for GPU inference):** CUDA 12.x compatible GPU (e.g., NVIDIA GeForce RTX 4050 Laptop GPU)

Verify prerequisites from your terminal:
```bash
python3 --version
node --version
npm --version
ffmpeg -version
```

---

## 2. Backend Setup

### A. Clone and Virtual Environment

Clone the repository and initialize a Python virtual environment:
```bash
git clone https://github.com/<owner>/Digital-Media-Verification.git
cd Digital-Media-Verification

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux / WSL / macOS:
source venv/bin/activate
# On Windows (PowerShell):
# .\venv\Scripts\Activate.ps1
```

### B. Install Python Dependencies

Install the core dependencies:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If GPU acceleration is desired, ensure PyTorch with CUDA support is installed:
```bash
# Example for CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU availability:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 3. Model Weights Installation

The repository tracks configurations, architectures, and tokenizer configurations. Production weights must be placed in their expected paths before starting the API server:

1. Download the release archives from the GitHub Releases tab:
   - `deepfake_efficientnet_b4_checkpoint.tar.gz`
   - `hate_speech_bert_checkpoint.tar.gz`
   - `propaganda_roberta_exp1_checkpoint.tar.gz`

2. Extract archives into the local `models/` hierarchy:
   ```bash
   # Deepfake detector
   tar -xzf deepfake_efficientnet_b4_checkpoint.tar.gz -C models/deepfake/

   # Hate speech detector
   tar -xzf hate_speech_bert_checkpoint.tar.gz -C models/hate_speech/

   # Propaganda detector
   mkdir -p models/propaganda/optimization
   tar -xzf propaganda_roberta_exp1_checkpoint.tar.gz -C models/propaganda/optimization/
   ```

Verify expected weight file paths:
- `models/deepfake/checkpoint_full/best_model.pt`
- `models/hate_speech/checkpoint_full/model.safetensors`
- `models/propaganda/optimization/exp1_sqrt_weighted_scheduled/model.safetensors`

*(Refer to `models/README.md` for complete architecture and parameter documentation).*

---

## 4. Frontend Setup

In a separate terminal, install the web application dependencies:
```bash
cd frontend
npm install
```

Configure local environment variables if necessary:
```bash
cp .env.example .env
```
Default `.env` configuration:
```env
VITE_API_BASE_URL=http://localhost:8000
```

---

## 5. Running the Application

### Step 1: Launch Backend API
From the repository root (with virtual environment active):
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- API server listens on: `http://localhost:8000`
- Interactive OpenAPI / Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Step 2: Launch Frontend Development Server
From the `frontend/` directory:
```bash
cd frontend
npm run dev
```
- Web dashboard accessible at: `http://localhost:5173`

---

## 6. Running Tests

### Backend Test Suite (52 Tests)
Run pytest across all modules:
```bash
pytest tests/ -v
```

### Frontend Test Suite (12 Tests)
Run vitest across all components:
```bash
cd frontend
npm test
```

### Production Build Verification
Test the frontend production build bundle:
```bash
cd frontend
npm run build
```
