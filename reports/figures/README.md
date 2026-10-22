# Research Figures Directory

This directory contains publication-quality plots and vector graphics generated from verified evaluation data:

1. **`fig_a_benchmark_metrics.{png,svg}`**: Task-specific benchmark metrics (Accuracy, Macro F1, Weighted F1) for Visual Deepfakes, Propaganda, and Hate Speech.
2. **`fig_b_deepfake_robustness.{png,svg}`**: Deepfake detector performance under 5 real-world visual distortions (Original, JPEG, Noise, Blur, Low-Res) on a balanced 100-video test subset.
3. **`fig_c_deepfake_confusion_matrix.{png,svg}`**: Full confusion matrix on the 700-video FaceForensics++ C23 test split with precision/recall marginals.
4. **`fig_d_deepfake_class_metrics.{png,svg}`**: Class-wise precision, recall, and F1 illustrating the impact of 4:1 class imbalance.
5. **`fig_e_propaganda_optimization.{png,svg}`**: Baseline vs. Phase 5 Exp 1 checkpoint comparison across 14 SemEval techniques on the article-disjoint validation set.
6. **`fig_f_dataset_distributions.{png,svg}`**: Ground-truth class distributions across evaluated benchmark splits.

### Provenance and Reproducibility
- Data source: `reports/research_figures_data.csv`
- Documentation: `reports/research_figures_data.md`
- Script: `scripts/generate_research_figures.py`
