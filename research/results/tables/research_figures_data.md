# Research Figures Data and Provenance Guide

This document accompanies `research_figures_data.csv` (available canonically at `research/results/tables/research_figures_data.csv` and in `reports/`) and provides complete provenance, source file traceability, reproduction commands, and structured tables ready for manuscript and Python plotting.

---

## 1. Generated Figures Overview

| Figure ID | File Name | Research Topic | Source File & Type | Sample Count & Split |
| :--- | :--- | :--- | :--- | :--- |
| **Figure A** | `fig_a_benchmark_metrics.png` | Task-Specific Benchmark Metrics across modalities | `phase4_results.json`, `PHASE_4_EVALUATION.md`, `PHASE_5_MODEL_OPTIMIZATION.md` | Deepfake ($N=700$, Test), Propaganda ($N=1,448$, Val), Hate Speech ($N=1,924$, Test) |
| **Figure B** | `fig_b_deepfake_robustness.png` | Deepfake Robustness under 5 Realistic Distortions | `phase4_robustness.json` (Raw Output) | $N=100$ videos (50 Real, 50 Fake balanced test subset) |
| **Figure C** | `fig_c_deepfake_confusion_matrix.png` | Complete Deepfake Confusion Matrix with Class Support | `phase4_results.json` (Raw Output) | $N=700$ videos (140 Real, 560 Fake test split) |
| **Figure D** | `fig_d_deepfake_class_metrics.png` | Class-Wise Precision, Recall, F1 and Imbalance Impact | `phase4_results.json` (Raw Output) | $N=700$ videos (140 Real, 560 Fake) |
| **Figure E** | `fig_e_propaganda_optimization.png` | Propaganda Classifier Checkpoint Optimization | `PHASE_5_MODEL_OPTIMIZATION.md` (Audited Report) | $N=1,448$ spans (Article-disjoint validation split) |
| **Figure F** | `fig_f_dataset_distributions.png` | Class Distribution & Imbalance Across Benchmark Splits | Manifests & Final Evaluation Reports | All 3 modalities ($N=700$, $N=1,924$, $N=1,448$) |

---

## 2. Figures That Were Intentionally NOT Generated & Why

1. **Quantitative Calibration Curve / Reliability Diagram**:
   - **Reason**: Calibration has **not been quantitatively established** in this project. Generating a synthetic curve or presenting heuristic Monte Carlo Dropout variance thresholds as calibrated probabilities would violate academic research integrity guidelines. Documented as future work.
2. **Dense Video Temporal Alignment Curve**:
   - **Reason**: The model samples frames sparsely at 1.0 FPS ($4$ frames per video) using deterministic uniform sequential sampling. Dense frame-by-frame temporal sync was not part of the evaluated architecture.
3. **Propaganda Test Split Curve**:
   - **Reason**: Official SemEval-2020 Task 11 test labels were withheld by competition organizers. All propaganda metrics strictly reflect an article-disjoint validation set and are explicitly designated as validation metrics.

---

## 3. Structured Data Tables for GPT Plotting

### Table A: Benchmark Metrics across Modalities
```csv
Task,Model,Dataset,Split,Sample Count,Accuracy,Macro F1,Weighted F1
Visual Deepfake,EfficientNet-B4,FaceForensics++ C23,Test,700,0.8800,0.8321,0.8859
Propaganda,RoBERTa-base (Exp 1),SemEval-2020 Task 11,Validation,1448,0.6519,0.5132,0.6477
Hate Speech,BERT-base-uncased,HateXplain,Test,1924,0.6741,0.6617,0.6738
```

### Table B: Deepfake Distortion Robustness ($N=100$ Balanced Videos)
```csv
Condition,Accuracy,Macro F1,Accuracy Drop,Real Recall,Fake Recall
Original (Baseline),0.8100,0.8098,0.0000,0.7800,0.8400
JPEG Compression (Q=30),0.7300,0.7278,-0.0800,0.8200,0.6400
Gaussian Noise (sigma=15),0.6700,0.6624,-0.1400,0.5200,0.8200
Gaussian Blur (r=2),0.5600,0.4544,-0.2500,0.1200,1.0000
Low-Res Resize (1/4x),0.5600,0.4544,-0.2500,0.1200,1.0000
```

### Table C: Complete Deepfake Confusion Matrix ($N=700$)
```csv
True Class,Predicted Fake,Predicted Real,Row Total,Class Recall
True Fake,495,65,560,88.39%
True Real,19,121,140,86.43%
Col Total,514,186,700,-
Class Precision,96.30%,65.05%,-,88.00%
```

### Table D: Propaganda Checkpoint Optimization (SemEval-2020 Val, $N=1,448$)
```csv
Technique,Baseline F1,Optimized Exp 1 F1,Delta F1,Support
Whataboutism,0.0000,0.2326,+0.2326,23
Repetition,0.1852,0.4790,+0.2938,143
Flag-Waving,0.6111,0.7333,+0.1222,46
Causal Oversimplification,0.4000,0.5210,+0.1210,56
Appeal to Authority,0.4571,0.5714,+0.1143,49
Doubt,0.5714,0.6639,+0.0925,118
Name Calling,0.6713,0.7515,+0.0802,226
Exaggeration,0.4851,0.5652,+0.0801,137
Loaded Language,0.7250,0.7754,+0.0504,477
Slogans,0.4348,0.5490,+0.1142,23
Appeal to Fear-Prejudice,0.5081,0.5503,+0.0422,82
Black-and-White Fallacy,0.3922,0.4127,+0.0205,32
Bandwagon,0.0833,0.1212,+0.0379,19
Thought-terminating Clichés,0.3200,0.2581,-0.0619,17
Overall Macro F1,0.4175,0.5132,+0.0957,1448
Overall Accuracy,0.5932,0.6519,+0.0587,1448
Overall Weighted F1,0.5539,0.6477,+0.0938,1448
```

---

## 4. Regeneration Command

To re-execute and generate all figures, run:
```bash
python scripts/generate_research_figures.py
```
Outputs are written canonically to `research/figures/png/` (300 DPI rasters) and `research/figures/svg/` (vector graphics), with synchronized engineering copies in `reports/figures/`.
