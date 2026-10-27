# Research Figures Catalog

This directory contains publication-ready figures in both vector (`svg/`) and raster (`png/`, 300 DPI) formats, generated from empirical benchmark and robustness evaluations of the Digital Media Verification (DMV) framework.

---

## 1. Figure Inventory and Descriptions

| Figure ID | File Base | Title | Topic / Modality | Data Source | Generating Script |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Figure A** | `fig_a_benchmark_metrics` | Task-Specific Benchmark Performance | Visual Deepfakes, Propaganda, Hate Speech across benchmark splits | `research/results/tables/research_figures_data.csv` (Table A) | `scripts/generate_research_figures.py` |
| **Figure B** | `fig_b_deepfake_robustness` | Deepfake Detector Distortion Robustness | Visual Deepfakes under 5 realistic corruptions ($N=100$) | `research/results/data/baseline_robustness.json`, `research/results/tables/research_figures_data.csv` | `scripts/generate_research_figures.py` |
| **Figure C** | `fig_c_deepfake_confusion_matrix` | Deepfake Full Test Confusion Matrix | Visual Deepfakes on 700-video FaceForensics++ C23 test split | `research/results/tables/research_figures_data.csv` (Table C) | `scripts/generate_research_figures.py` |
| **Figure D** | `fig_d_deepfake_class_metrics` | Class-Wise Metrics & Imbalance Impact | Visual Deepfakes class-level Precision, Recall, F1 ($4:1$ ratio) | `research/results/tables/research_figures_data.csv` (Table C/D) | `scripts/generate_research_figures.py` |
| **Figure E** | `fig_e_propaganda_optimization` | Propaganda Technique Optimization | NLP RoBERTa-base optimization across 14 SemEval techniques | `research/results/tables/research_figures_data.csv` (Table D) | `scripts/generate_research_figures.py` |
| **Figure F** | `fig_f_dataset_distributions` | Benchmark Ground-Truth Class Distributions | Full evaluated benchmark splits across all 3 modalities | `research/results/tables/research_figures_data.csv` (Table F) | `scripts/generate_research_figures.py` |

---

## 2. Directory Structure

```
research/figures/
├── README.md                          # This catalog
├── png/                               # Publication-quality 300 DPI rasters
│   ├── fig_a_benchmark_metrics.png
│   ├── fig_b_deepfake_robustness.png
│   ├── fig_c_deepfake_confusion_matrix.png
│   ├── fig_d_deepfake_class_metrics.png
│   ├── fig_e_propaganda_optimization.png
│   └── fig_f_dataset_distributions.png
└── svg/                               # Resolution-independent vector graphics
    ├── fig_a_benchmark_metrics.svg
    ├── fig_b_deepfake_robustness.svg
    ├── fig_c_deepfake_confusion_matrix.svg
    ├── fig_d_deepfake_class_metrics.svg
    ├── fig_e_propaganda_optimization.svg
    └── fig_f_dataset_distributions.svg
```

---

## 3. Reproduction Instructions

To regenerate all figures from source data:

```bash
python scripts/generate_research_figures.py
```

The script parses `research/results/tables/research_figures_data.csv` and outputs matching PNG and SVG files. Matplotlib styling uses custom typography and high-contrast color palettes suitable for peer-reviewed IEEE/ACM manuscript submissions.
