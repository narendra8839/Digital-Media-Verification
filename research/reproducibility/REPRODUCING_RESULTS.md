# Reproducing Results and Empirical Audits

This guide provides exact, deterministic commands to reproduce all evaluations, regression suites, figure generation, and sanity audits in the Digital Media Verification (DMV) framework.

---

## 1. Prerequisites and Environment Setup

Ensure your local or virtual environment has Python 3.10+ (with PyTorch and CUDA if GPU acceleration is available) and Node.js 18+:

```bash
# Clone the repository
git clone https://github.com/narendra8839/Digital-Media-Verification.git
cd Digital-Media-Verification

# Set up Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows PowerShell: .\venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install frontend dependencies
cd frontend
npm ci
cd ..
```

---

## 2. Verify Model Checkpoint Integrity

Before running evaluations, verify that the production EfficientNet-B4 checkpoint is authentic:

### Linux / WSL Bash
```bash
sha256sum models/deepfake/checkpoint_full/best_model.pt
```

### Windows PowerShell
```powershell
Get-FileHash -Algorithm SHA256 "models/deepfake/checkpoint_full/best_model.pt"
```

**Expected Hash:**
```
5878ae38bf5239dd9889e010fafa1d8c7da4198ac260542dc96681ede1e63234
```

---

## 3. Running Automated Test Suites

### 3.1 Full Backend Test Suite
Executes all unit, integration, artifact retention, report generation, and robustness tests:

```bash
pytest tests/ -v
```

**Expected Result:**
```
121 passed, 1 skipped in ~110s
```
*(The single skipped test gracefully skips missing experimental candidate weights archived in the robustness worktree).*

### 3.2 Frontend Unit and Integration Suite
Executes Vitest tests covering UI components, drag-and-drop ingestion, result badges, and PDF download triggers:

```bash
cd frontend
npm test -- --run
cd ..
```

**Expected Result:**
```
Test Files  2 passed (2)
Tests       19 passed (19)
```

### 3.3 Frontend Production Build Validation
Validates client-side bundling, TypeScript/JSX syntax, and asset hashing:

```bash
cd frontend
npm run build
cd ..
```

**Expected Result:**
```
✓ built in ~500ms (dist/ generated without errors)
```

---

## 4. Regenerating Publication Figures

To regenerate all publication-quality vector (`svg/`) and raster (`png/`) figures and export structured data tables:

```bash
python scripts/generate_research_figures.py
```

- **Grounding / Data Source:** Embedded verified empirical benchmark metrics and distortion matrices (from `phase4_results.json`, `baseline_robustness.json`, `docs/research_history/PHASE_4_EVALUATION.md`, and `docs/research_history/PHASE_5_MODEL_OPTIMIZATION.md`).
- **Outputs Generated:**
  - **Canonical Research Artifacts:**
    - `research/figures/png/fig_[a-f]_*.png` (300 DPI rasters for manuscript drafts)
    - `research/figures/svg/fig_[a-f]_*.svg` (Scalable vector graphics)
    - `research/results/tables/research_figures_data.csv` (Raw tabular data)
    - `research/results/tables/research_figures_data.md` (Structured markdown documentation)
  - **Engineering Compatibility Artifacts:**
    - `reports/figures/fig_[a-f]_*.{png,svg}` (Flat directory for engineering reports)
    - `reports/research_figures_data.{csv,md}`

---

## 5. Running Robustness Evaluations (Inference-Only)

To re-evaluate the baseline or candidate models under the 5 standardized corruptions (Clean, JPEG Q30, Gaussian Noise, Blur, Low-Res Downsampling):

### 5.1 Standardized Development Benchmark ($N=100$)
```bash
python scripts/evaluate_robustness.py \
  --checkpoint models/deepfake/checkpoint_full/best_model.pt \
  --split data/full_splits/deepfake_test.json \
  --offset-real 0 \
  --offset-fake 0 \
  --num-samples 50 \
  --output research/results/data/baseline_robustness.json
```

### 5.2 Reserved Comparison Subset ($N=100$)
```bash
python scripts/evaluate_robustness.py \
  --checkpoint models/deepfake/checkpoint_full/best_model.pt \
  --split data/full_splits/deepfake_test.json \
  --offset-real 50 \
  --offset-fake 50 \
  --num-samples 50 \
  --output research/results/data/reserved_baseline_robustness.json
```

### 5.3 Multi-Severity Distortion Sweep
```bash
python scripts/evaluate_severity_curve.py \
  --checkpoint models/deepfake/checkpoint_full/best_model.pt \
  --output research/results/data/severity_curve_baseline.json
```

---

## 6. Mathematical Consistency Verification

To verify that all per-video predictions, confusion matrices, balanced accuracy scores, and macro F1 metrics match stored JSON outputs bit-for-bit:

```bash
pytest tests/test_deepfake_robustness.py -v
```

**Expected Result:**
```
27 passed in ~8s
```
