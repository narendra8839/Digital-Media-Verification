# Environment and Dependencies

This document provides complete system specifications, hardware recommendations, and package requirements for reproducing the Digital Media Verification (DMV) framework.

---

## 1. System Requirements

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Operating System** | Ubuntu 22.04 LTS / Windows 10/11 (x64) | Ubuntu 22.04 LTS or Windows 11 with WSL2 |
| **CPU** | 4-Core x86_64 Processor | 8+ Core modern x86_64 CPU (Intel i7/i9 or AMD Ryzen 7/9) |
| **RAM** | 16 GB | 32 GB DDR4/DDR5 |
| **GPU (Optional)** | CPU execution supported | NVIDIA GPU with $\ge 8$ GB VRAM (RTX 3070 / 4070 or better, CUDA 12.x) |
| **Disk Space** | 10 GB free space | 50 GB free space (for caching datasets and models) |

---

## 2. External System Tooling

### 2.1 FFmpeg & FFprobe
Required for demuxing audio tracks and decoding container video formats:
- **Ubuntu / Debian:** `sudo apt-get update && sudo apt-get install -y ffmpeg`
- **Windows:** Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) or install via Chocolatey (`choco install ffmpeg`) / Winget (`winget install Gyan.FFmpeg`). Ensure `ffmpeg` and `ffprobe` are on system `PATH`.

### 2.2 System Typography for PDF Generation
The zero-inference report generator (`api/utils/report_generator.py`) formats multilingual text using TrueType fonts:
- **Windows:** Built-in fonts used automatically (`C:\Windows\Fonts\arial.ttf`, `Nirmala.ttc`).
- **Linux / Ubuntu:** Install standard fonts:
  ```bash
  sudo apt-get install -y fonts-dejavu-core fonts-freefont-ttf
  ```

---

## 3. Python Runtime & Core Libraries

- **Python Version:** 3.10, 3.11, 3.12, 3.13, or 3.14. (Evaluated on Python 3.14.4 x86_64).
- **Core Dependencies (`requirements.txt`):**
  - `torch`, `torchvision`: Neural network modeling and tensor operations.
  - `facenet-pytorch`: MTCNN facial detection and landmark extraction.
  - `transformers`: Hugging Face transformers for RoBERTa and BERT models.
  - `openai-whisper`: Automated speech recognition for audio streams.
  - `easyocr`: Optical character recognition for video keyframes and banner text.
  - `reportlab` ($\ge 4.0$): Zero-inference PDF report rendering engine.
  - `fastapi`, `uvicorn`: Asynchronous web server and REST gateway.
  - `pydantic` ($\ge 2.0$): Strongly-typed request and response schema validation.
  - `scikit-learn`: Metrics computation (F1, balanced accuracy, confusion matrices).
  - `matplotlib`, `seaborn`: Figure generation and plotting.
  - `pillow`, `opencv-python-headless`: Image preprocessing and corruptions.
- **Development & Testing (`requirements-dev.txt`):**
  - `pytest` ($\ge 8.0$): Test suite runner.
  - `httpx`: FastAPI TestClient HTTP transport.

---

## 4. Frontend Runtime & Tooling

- **Node.js:** Version 18.x, 20.x, or 22.x LTS.
- **Package Manager:** `npm` ($\ge 9.0$).
- **Core Frontend Libraries (`frontend/package.json`):**
  - `react`, `react-dom` ($\ge 18.2$): Client-side reactive UI.
  - `lucide-react`: UI icon set.
  - `tailwindcss`, `postcss`, `autoprefixer`: Utility styling.
- **Build & Test Tooling:**
  - `vite` ($\ge 5.0$ / 6.0): Rapid local development and rollup production bundling.
  - `vitest`, `@testing-library/react`: Frontend unit testing.

---

## 5. Environment Variables Configuration

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DMV_ARTIFACT_DIR` | `artifacts/` | Path where transient Grad-CAM visualization heatmaps are cached |
| `DMV_DEVICE` | `auto` | Execution device: `cuda`, `cpu`, or `auto` (detects GPU availability) |
| `DMV_PORT` | `8000` | Port on which the FastAPI gateway listens |
| `DMV_HOST` | `127.0.0.1` | Host address for binding the API service |
| `VITE_API_URL` | `http://127.0.0.1:8000` | Backend API base URL consumed by the React client |
