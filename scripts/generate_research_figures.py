"""Generate publication-quality research and presentation figures from verified project evaluation data.

Outputs:
- reports/figures/fig_a_benchmark_metrics.{png,svg}
- reports/figures/fig_b_deepfake_robustness.{png,svg}
- reports/figures/fig_c_deepfake_confusion_matrix.{png,svg}
- reports/figures/fig_d_deepfake_class_metrics.{png,svg}
- reports/figures/fig_e_propaganda_optimization.{png,svg}
- reports/figures/fig_f_dataset_distributions.{png,svg}
- reports/research_figures_data.csv
- reports/research_figures_data.md
- reports/figures/README.md

All figures are grounded exclusively in genuine project evaluation outputs:
- phase4_results.json
- phase4_robustness.json
- docs/research_history/PHASE_4_EVALUATION.md
- docs/research_history/PHASE_5_MODEL_OPTIMIZATION.md
"""

import os
import sys
import tempfile
import uuid
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Publication styling tokens
COLOR_PRIMARY = "#1e3a8a"       # Deep Navy
COLOR_SECONDARY = "#f54e00"     # Restrained Orange
COLOR_ACCENT = "#0d9488"        # Slate Teal
COLOR_MUTED = "#64748b"         # Slate Grey
COLOR_LIGHT_BG = "#f8fafc"      # Subtle Canvas
COLOR_BORDER = "#cbd5e1"        # Hairline Border
COLOR_ALERT = "#b91c1c"         # Crimson
COLOR_SUCCESS = "#15803d"       # Forest Green

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "medium",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.titleweight": "bold",
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.8,
    "grid.color": "#f1f5f9",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff"
})


def _atomic_export_figure(target_path, save_func):
    """
    Safely export a figure to target_path using a temporary file and atomic replacement.

    Guarantees:
    1. Writes to a temporary file in the destination directory (matching filesystem).
    2. Validates that the export completed and produced a non-empty file.
    3. Atomically replaces target_path with the validated temporary file.
    4. Cleans up temporary files if any error occurs, keeping the previous file intact.
    5. Explicitly reports Windows file-locking or permission errors without deleting
       the existing output.
    """
    target_path = os.path.abspath(target_path)
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    ext = os.path.splitext(target_path)[1]

    # Create temporary file in destination directory to allow atomic os.replace
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=f".tmp_{uuid.uuid4().hex[:6]}_", suffix=ext, dir=target_dir)
    os.close(tmp_fd)

    try:
        # Perform export
        save_func(tmp_path)

        # Validate existence and non-zero size
        if not os.path.exists(tmp_path):
            raise FileNotFoundError(f"Export did not produce temporary file: {tmp_path}")

        file_size = os.path.getsize(tmp_path)
        if file_size == 0:
            raise ValueError(f"Export produced an empty file (0 bytes) for destination: {target_path}")

        # Atomic replacement: replaces destination only after verified success
        os.replace(tmp_path, target_path)

    except PermissionError as pe:
        sys.stderr.write(
            f"ERROR: Permission or file-locking conflict updating '{target_path}'. "
            f"Existing output was preserved intact. Details: {pe}\n"
        )
        raise
    except OSError as oe:
        sys.stderr.write(
            f"ERROR: OS error while updating '{target_path}'. "
            f"Existing output was preserved intact. Details: {oe}\n"
        )
        raise
    except Exception as err:
        sys.stderr.write(
            f"ERROR: Unexpected error while exporting '{target_path}'. "
            f"Existing output was preserved intact. Details: {err}\n"
        )
        raise
    finally:
        # Clean up temporary file if it was not renamed or if an error occurred
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def safe_savefig(png_path=None, svg_path=None, fig=None, dpi=300, **kwargs):
    """
    Safely export figure to PNG and/or SVG formats with atomic replacement and validation.

    Preserves existing files if export or replacement fails, validates non-empty output,
    and cleans up temporary files on failure.
    """
    saver = fig.savefig if fig is not None else plt.savefig
    exported = []

    try:
        if png_path:
            _atomic_export_figure(png_path, lambda tmp: saver(tmp, dpi=dpi, format="png", **kwargs))
            exported.append(png_path)

        if svg_path:
            _atomic_export_figure(svg_path, lambda tmp: saver(tmp, format="svg", **kwargs))
            exported.append(svg_path)

        print(f"Generated: {' & '.join(exported)}")
    finally:
        if fig is not None:
            plt.close(fig)
        else:
            plt.close()


# -----------------------------------------------------------------------------
# FIGURE A: Final Task-Specific Benchmark Metrics
# -----------------------------------------------------------------------------
def generate_figure_a():
    """Figure A: Final Task-Specific Benchmark Metrics across separate tasks."""
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.4), sharey=True)
    fig.suptitle(
        "Figure A: Task-Specific Benchmark Evaluation Metrics (Per Modality & Dataset)",
        fontsize=13, fontweight="bold", y=0.98
    )

    tasks = [
        {
            "name": "Visual Deepfake Detection",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split": "Test Split (N=700 videos)",
            "metrics": {"Accuracy": 0.8800, "Macro F1": 0.8321, "Weighted F1": 0.8859},
            "color": "#1e3a8a"
        },
        {
            "name": "Propaganda Technique Detection",
            "model": "RoBERTa-base (Exp 1)",
            "dataset": "SemEval-2020 Task 11",
            "split": "Held-Out Val (N=1,448 spans)",
            "metrics": {"Accuracy": 0.6519, "Macro F1": 0.5132, "Weighted F1": 0.6477},
            "color": "#f54e00"
        },
        {
            "name": "Hate Speech Classification",
            "model": "BERT-base-uncased",
            "dataset": "HateXplain",
            "split": "Test Split (N=1,924 posts)",
            "metrics": {"Accuracy": 0.6741, "Macro F1": 0.6617, "Weighted F1": 0.6738},
            "color": "#0d9488"
        }
    ]

    metric_names = ["Accuracy", "Macro F1", "Weighted F1"]

    for idx, (ax, task) in enumerate(zip(axes, tasks)):
        vals = [task["metrics"][m] for m in metric_names]
        bars = ax.bar(metric_names, vals, color=task["color"], width=0.55, edgecolor="#0f172a", linewidth=0.7, alpha=0.9)
        ax.set_ylim(0, 1.14)  # Headroom ensures bars (max 0.8859) never touch subtitle tags
        ax.set_title(f"{task['name']}\n({task['model']})", pad=8, fontsize=11)
        ax.grid(axis="y", alpha=0.7)

        # Subtitle tags with safe positioning
        ax.text(0.5, 0.95, f"{task['dataset']}", transform=ax.transAxes,
                ha="center", fontsize=8.5, fontweight="bold", color="#334155")
        ax.text(0.5, 0.88, f"Split: {task['split']}", transform=ax.transAxes,
                ha="center", fontsize=8, color="#64748b",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8fafc", edgecolor="#e2e8f0", linewidth=0.6))

        # Value labels above bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.4f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        if idx == 0:
            ax.set_ylabel("Metric Score [0.0 - 1.0]", fontsize=10)

    fig.text(0.5, 0.02,
             "Note: Tasks represent heterogeneous modalities, benchmark protocols, and split structures. Propaganda uses an article-disjoint validation split (official test unreleased).",
             ha="center", fontsize=8, style="italic", color="#475569")

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    png_path = os.path.join(FIGURES_DIR, "fig_a_benchmark_metrics.png")
    svg_path = os.path.join(FIGURES_DIR, "fig_a_benchmark_metrics.svg")
    safe_savefig(png_path, svg_path)


# -----------------------------------------------------------------------------
# FIGURE B: Deepfake Robustness Evaluation
# -----------------------------------------------------------------------------
def generate_figure_b():
    """Figure B: Deepfake Model Degradation under Real-World Visual Distortions."""
    # Data from phase4_robustness.json
    conditions = [
        "Original\n(Baseline)",
        "JPEG\nCompression\n(Q=30)",
        "Gaussian\nNoise\n(σ=15)",
        "Gaussian\nBlur\n(r=2)",
        "Low-Res\nResize\n(1/4×)"
    ]
    accuracies = [0.8100, 0.7300, 0.6700, 0.5600, 0.5600]
    macro_f1s = [0.8098, 0.7278, 0.6624, 0.4544, 0.4544]
    drops = [0.0, -0.0800, -0.1400, -0.2500, -0.2500]

    x = np.arange(len(conditions))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2.2, 1]})
    fig.suptitle(
        "Figure B: Deepfake Detector Robustness under Realistic Visual Perturbations",
        fontsize=13, y=0.98
    )

    # Left plot: Accuracy & Macro F1
    rects1 = ax1.bar(x - width/2, accuracies, width, label="Accuracy", color="#1e3a8a", edgecolor="#0f172a", linewidth=0.7)
    rects2 = ax1.bar(x + width/2, macro_f1s, width, label="Macro F1", color="#f54e00", edgecolor="#0f172a", linewidth=0.7)

    ax1.set_ylabel("Score [0.0 - 1.0]")
    ax1.set_title("Performance across Distortion Conditions (N=100 Balanced Videos: 50 Real, 50 Fake)", pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions)
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc="lower left", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.7)

    for bar in rects1:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in rects2:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Right plot: Drop from Baseline Accuracy
    colors_drop = ["#64748b", "#ea580c", "#dc2626", "#991b1b", "#991b1b"]
    bars_drop = ax2.barh(conditions[::-1], [abs(d) * 100 for d in drops[::-1]], color=colors_drop[::-1],
                         edgecolor="#0f172a", linewidth=0.7, height=0.55)
    ax2.set_xlabel("Accuracy Degradation (% Points Drop)")
    ax2.set_title("Absolute Accuracy Drop", pad=8)
    ax2.set_xlim(0, 35)
    ax2.grid(axis="x", alpha=0.7)

    for bar in bars_drop:
        w = bar.get_width()
        ax2.annotate(f"-{w:.1f}%", xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5, fontweight="bold")

    fig.text(0.5, 0.02,
             "Evaluated on a deterministic balanced subset of 100 test videos (50 Real + 50 Fake from FaceForensics++ C23 test split) sampled at 4 frames/video.",
             ha="center", fontsize=8, style="italic", color="#475569")

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    png_path = os.path.join(FIGURES_DIR, "fig_b_deepfake_robustness.png")
    svg_path = os.path.join(FIGURES_DIR, "fig_b_deepfake_robustness.svg")
    safe_savefig(png_path, svg_path)


# -----------------------------------------------------------------------------
# FIGURE C: Deepfake Confusion Matrix
# -----------------------------------------------------------------------------
def generate_figure_c():
    """Figure C: Complete Deepfake Test Confusion Matrix (N=700 Videos)."""
    # Raw matrix from phase4_results.json:
    # labels: ['fake', 'real']
    # [[495, 65], [19, 121]]
    # row 0 = True Fake (560), row 1 = True Real (140)
    cm = np.array([[495, 65], [19, 121]])
    labels = ["Fake", "Real"]
    total = np.sum(cm)

    fig = plt.figure(figsize=(7.2, 7.2))
    fig.suptitle(
        "Figure C: Deepfake Detection Confusion Matrix\n(FaceForensics++ C23 Complete Test Split, N=700)",
        fontsize=12, fontweight="bold", y=0.96
    )

    gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.2], hspace=0.32,
                          top=0.89, bottom=0.05, left=0.15, right=0.88)
    ax = fig.add_subplot(gs[0])
    ax_metrics = fig.add_subplot(gs[1])
    ax_metrics.axis("off")

    # Heatmap rendering
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Video Sequence Count", rotation=270, labelpad=14, fontsize=9.5)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10.5, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=10.5, fontweight="bold")

    ax.set_xlabel("Predicted Class", fontsize=10.5, labelpad=8)
    ax.set_ylabel("True Ground-Truth Class", fontsize=10.5, labelpad=8)

    # Add text inside cells
    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            pct_row = (val / np.sum(cm[i, :])) * 100
            pct_total = (val / total) * 100
            txt_color = "white" if val > thresh else "black"
            cell_text = f"{val}\n({pct_row:.1f}% of row)\n[{pct_total:.1f}% total]"
            ax.text(j, i, cell_text, ha="center", va="center", color=txt_color,
                    fontsize=10, fontweight="bold" if i == j else "normal")

    # Structured metrics annotation card positioned comfortably in lower subplot
    metrics_text = (
        "Dataset: FaceForensics++ C23 Test Split  |  Total Sequences: 700 (560 Fake, 140 Real)\n"
        "Overall Accuracy: 88.00% (616 / 700)  |  Macro F1: 0.8321  |  Weighted F1: 0.8859\n"
        "Fake Class (N=560): Recall = 88.39% (495/560), Precision = 96.30% (495/514), F1 = 0.9218\n"
        "Real Class (N=140): Recall = 86.43% (121/140), Precision = 65.05% (121/186), F1 = 0.7423"
    )
    ax_metrics.text(
        0.5, 0.5, metrics_text,
        transform=ax_metrics.transAxes,
        ha="center", va="center",
        fontsize=8.5, color="#1e293b", linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.8)
    )

    png_path = os.path.join(FIGURES_DIR, "fig_c_deepfake_confusion_matrix.png")
    svg_path = os.path.join(FIGURES_DIR, "fig_c_deepfake_confusion_matrix.svg")
    safe_savefig(png_path, svg_path)


# -----------------------------------------------------------------------------
# FIGURE D: Class-Wise Deepfake Metrics & Imbalance Impact
# -----------------------------------------------------------------------------
def generate_figure_d():
    """Figure D: Class-Wise Precision, Recall, F1 and Support Distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [2, 1.1]})
    fig.suptitle(
        "Figure D: Class-Wise Deepfake Performance and Class Imbalance Impact",
        fontsize=13, y=0.98
    )

    classes = ["Real Class\n(Support: 140)", "Fake Class\n(Support: 560)", "Macro Avg\n(N=700)", "Weighted Avg\n(N=700)"]
    precisions = [0.6505, 0.9630, 0.8068, 0.9005]
    recalls = [0.8643, 0.8839, 0.8741, 0.8800]
    f1s = [0.7423, 0.9218, 0.8321, 0.8859]

    x = np.arange(len(classes))
    width = 0.25

    rects1 = ax1.bar(x - width, precisions, width, label="Precision", color="#0284c7", edgecolor="#0f172a", linewidth=0.7)
    rects2 = ax1.bar(x, recalls, width, label="Recall", color="#16a34a", edgecolor="#0f172a", linewidth=0.7)
    rects3 = ax1.bar(x + width, f1s, width, label="F1-Score", color="#f54e00", edgecolor="#0f172a", linewidth=0.7)

    ax1.set_ylabel("Score [0.0 - 1.0]")
    ax1.set_title("Precision, Recall, and F1 by Class vs Aggregates", pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, fontsize=9)
    ax1.set_ylim(0, 1.08)
    ax1.legend(loc="lower left", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.7)

    for bar in rects1 + rects2 + rects3:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    # Right plot: Class Distribution Pie/Donut Chart
    sizes = [140, 560]
    pie_labels = ["Real Videos\n(140, 20.0%)", "Fake Videos\n(560, 80.0%)"]
    colors = ["#38bdf8", "#1e3a8a"]

    wedges, texts, autotexts = ax2.pie(
        sizes, labels=pie_labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.7,
        wedgeprops=dict(width=0.45, edgecolor="#ffffff", linewidth=2)
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(9)

    ax2.set_title("Test Set Class Imbalance\n(FaceForensics++ C23, N=700)", pad=8)

    fig.text(0.5, 0.02,
             "Imbalance finding: 4:1 Fake-to-Real prevalence causes 65.05% Real precision (34.95% false alarms on Real) despite 86.43% Real recall.",
             ha="center", fontsize=8, style="italic", color="#475569")

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    png_path = os.path.join(FIGURES_DIR, "fig_d_deepfake_class_metrics.png")
    svg_path = os.path.join(FIGURES_DIR, "fig_d_deepfake_class_metrics.svg")
    safe_savefig(png_path, svg_path)


# -----------------------------------------------------------------------------
# FIGURE E: Propaganda Checkpoint Optimization (Baseline vs Exp 1 Deployed)
# -----------------------------------------------------------------------------
def generate_figure_e():
    """Figure E: Propaganda Technique Detection Checkpoint Optimization (Validation Set)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1, 2]})
    fig.suptitle(
        "Figure E: Propaganda Classifier Optimization: Phase 3 Baseline vs. Phase 5 Exp 1 Checkpoint",
        fontsize=13, y=0.98
    )

    # Subplot 1: Macro & Aggregate Metrics
    agg_metrics = ["Accuracy", "Macro F1", "Macro Recall", "Weighted F1"]
    baseline_agg = [0.5932, 0.4175, 0.4030, 0.5539]
    exp1_agg = [0.6519, 0.5132, 0.5165, 0.6477]

    x1 = np.arange(len(agg_metrics))
    w1 = 0.35

    b1 = ax1.bar(x1 - w1/2, baseline_agg, w1, label="Phase 3 Baseline (Unweighted)", color="#94a3b8", edgecolor="#0f172a", linewidth=0.7)
    b2 = ax1.bar(x1 + w1/2, exp1_agg, w1, label="Phase 5 Exp 1 (Sqrt-Weighted)", color="#f54e00", edgecolor="#0f172a", linewidth=0.7)

    ax1.set_ylabel("Validation Score [0.0 - 1.0]")
    ax1.set_title("Aggregate Validation Gains (N=1,448 Spans)", pad=8)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(agg_metrics, rotation=20, ha="right", fontsize=9)
    ax1.set_ylim(0, 0.8)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax1.grid(axis="y", alpha=0.7)

    for bar, base, opt in zip(b2, baseline_agg, exp1_agg):
        diff = (opt - base) * 100
        ax1.annotate(f"+{diff:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#c2410c")

    # Subplot 2: Per-Class F1 Deltas across all 14 SemEval techniques
    techniques = [
        "Whataboutism",
        "Repetition",
        "Flag-Waving",
        "Causal Oversimpl.",
        "Appeal to Auth.",
        "Doubt",
        "Name Calling",
        "Exaggeration",
        "Loaded Language",
        "Slogans",
        "Fear-Prejudice",
        "Black-&-White",
        "Bandwagon",
        "Thought Clichés"
    ]
    baseline_class_f1 = [0.0000, 0.1852, 0.6111, 0.4000, 0.4571, 0.5714, 0.6713, 0.4851, 0.7250, 0.4348, 0.5081, 0.3922, 0.0833, 0.3200]
    exp1_class_f1 =     [0.2326, 0.4790, 0.7333, 0.5210, 0.5714, 0.6639, 0.7515, 0.5652, 0.7754, 0.5490, 0.5503, 0.4127, 0.1212, 0.2581]

    y2 = np.arange(len(techniques))
    w2 = 0.38

    ax2.barh(y2 + w2/2, baseline_class_f1[::-1], w2, label="Baseline F1", color="#94a3b8", edgecolor="#0f172a", linewidth=0.6)
    ax2.barh(y2 - w2/2, exp1_class_f1[::-1], w2, label="Optimized Exp 1 F1", color="#f54e00", edgecolor="#0f172a", linewidth=0.6)

    ax2.set_xlabel("F1-Score on Validation Set")
    ax2.set_title("Per-Class F1 Gains across 14 SemEval Techniques (Ordered by F1 Delta)", pad=8)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(techniques[::-1], fontsize=8.5)
    ax2.set_xlim(0, 0.9)
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax2.grid(axis="x", alpha=0.7)

    fig.text(0.5, 0.02,
             "Validation protocol notice: Results reflect an article-disjoint held-out validation set (71 articles, 1,448 spans). Sqrt-inverse class weighting produced a +9.57% Macro F1 gain.",
             ha="center", fontsize=8, style="italic", color="#475569")

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    png_path = os.path.join(FIGURES_DIR, "fig_e_propaganda_optimization.png")
    svg_path = os.path.join(FIGURES_DIR, "fig_e_propaganda_optimization.svg")
    safe_savefig(png_path, svg_path)


# -----------------------------------------------------------------------------
# FIGURE F: Dataset Split Distributions
# -----------------------------------------------------------------------------
def generate_figure_f():
    """Figure F: Dataset Class Distributions across all 3 evaluated splits."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    fig.suptitle(
        "Figure F: Ground-Truth Class Distribution in Evaluated Benchmark Splits",
        fontsize=13, fontweight="bold", y=0.97
    )

    # 1. Deepfake test (N=700)
    df_classes = ["Real", "Fake"]
    df_counts = [140, 560]
    axes[0].bar(df_classes, df_counts, color=["#38bdf8", "#1e3a8a"], edgecolor="#0f172a", linewidth=0.7, width=0.48)
    axes[0].set_title("Deepfake Test Set\n(FaceForensics++ C23, N=700)", fontsize=11, pad=12)
    axes[0].set_ylabel("Sample Count", fontsize=10)
    axes[0].set_ylim(0, 700)  # Generous headroom so 560 (80.0%) annotation never touches title
    axes[0].grid(axis="y", alpha=0.7)
    for i, c in enumerate(df_counts):
        axes[0].annotate(f"{c}\n({c/700*100:.1f}%)", xy=(i, c), xytext=(0, 5),
                         textcoords="offset points", ha="center", va="bottom",
                         fontsize=8.5, fontweight="bold")

    # 2. Hate speech test (N=1,924)
    hs_classes = ["Normal", "Offensive", "Hatespeech"]
    hs_counts = [782, 548, 594]
    axes[1].bar(hs_classes, hs_counts, color=["#10b981", "#f59e0b", "#ef4444"], edgecolor="#0f172a", linewidth=0.7, width=0.52)
    axes[1].set_title("Hate Speech Test Set\n(HateXplain, N=1,924)", fontsize=11, pad=12)
    axes[1].set_ylabel("Post Count", fontsize=10)
    axes[1].set_ylim(0, 960)  # Generous headroom so 782 (40.6%) annotation fits comfortably
    axes[1].grid(axis="y", alpha=0.7)
    for i, c in enumerate(hs_counts):
        axes[1].annotate(f"{c}\n({c/1924*100:.1f}%)", xy=(i, c), xytext=(0, 5),
                         textcoords="offset points", ha="center", va="bottom",
                         fontsize=8.5, fontweight="bold")

    # 3. Propaganda validation (top 8 + other, N=1,448)
    prop_top_labels = ["Loaded Lang.", "Name Calling", "Repetition", "Exaggeration", "Doubt", "Fear-Prej.", "Causal Over.", "Other (7 Tech.)"]
    prop_top_counts = [477, 226, 143, 137, 118, 82, 56, 49+46+32+23+23+19+17] # 209 in other
    axes[2].barh(prop_top_labels[::-1], prop_top_counts[::-1], color="#f54e00", edgecolor="#0f172a", linewidth=0.7, height=0.55)
    axes[2].set_title("Propaganda Validation Set\n(SemEval-2020 Task 11, N=1,448 Spans)", fontsize=11, pad=12)
    axes[2].set_xlabel("Span Count", fontsize=10)
    axes[2].set_xlim(0, 620)  # Generous horizontal room so 477 (32.9%) text never touches border
    axes[2].grid(axis="x", alpha=0.7)
    for i, c in enumerate(prop_top_counts[::-1]):
        axes[2].annotate(f" {c} ({c/1448*100:.1f}%)", xy=(c, i), va="center",
                         fontsize=8, fontweight="bold" if c == max(prop_top_counts) else "normal")

    fig.text(0.5, 0.02,
             "Dataset distribution context: Demonstrates severe natural class imbalance in visual deepfakes (80.0% Fake) and rhetorical propaganda spans (Loaded Language represents 33.0%).",
             ha="center", fontsize=8, style="italic", color="#475569")

    plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.93], w_pad=2.2)
    png_path = os.path.join(FIGURES_DIR, "fig_f_dataset_distributions.png")
    svg_path = os.path.join(FIGURES_DIR, "fig_f_dataset_distributions.svg")
    safe_savefig(png_path, svg_path)


# -----------------------------------------------------------------------------
# STRUCTURED DATA EXPORT: CSV and Markdown
# -----------------------------------------------------------------------------
def export_research_data_csv_and_md():
    """Export structured, source-traceable tabular data for research transparency and plotting."""
    csv_path = os.path.join(REPORTS_DIR, "research_figures_data.csv")
    md_path = os.path.join(REPORTS_DIR, "research_figures_data.md")
    readme_path = os.path.join(FIGURES_DIR, "README.md")

    rows = [
        # Figure A: Benchmark metrics
        {
            "figure_id": "Fig_A",
            "task": "Visual Deepfake Detection",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split_type": "test",
            "sample_count": 700,
            "class_or_condition": "Overall",
            "metric": "Accuracy",
            "value": 0.8800,
            "source_file": "phase4_results.json",
            "source_type": "raw_evaluation_output",
            "notes": "Complete 700 video test split; 4 frames/video uniform sampling"
        },
        {
            "figure_id": "Fig_A",
            "task": "Visual Deepfake Detection",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split_type": "test",
            "sample_count": 700,
            "class_or_condition": "Overall",
            "metric": "Macro F1",
            "value": 0.8321,
            "source_file": "phase4_results.json",
            "source_type": "raw_evaluation_output",
            "notes": "Unweighted arithmetic mean of Real and Fake F1"
        },
        {
            "figure_id": "Fig_A",
            "task": "Visual Deepfake Detection",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split_type": "test",
            "sample_count": 700,
            "class_or_condition": "Overall",
            "metric": "Weighted F1",
            "value": 0.8859,
            "source_file": "phase4_results.json",
            "source_type": "raw_evaluation_output",
            "notes": "Support-weighted F1 across 140 Real and 560 Fake"
        },
        {
            "figure_id": "Fig_A",
            "task": "Propaganda Technique Detection",
            "model": "RoBERTa-base (Exp 1)",
            "dataset": "SemEval-2020 Task 11",
            "split_type": "validation",
            "sample_count": 1448,
            "class_or_condition": "Overall",
            "metric": "Accuracy",
            "value": 0.6519,
            "source_file": "docs/research_history/PHASE_5_MODEL_OPTIMIZATION.md",
            "source_type": "audited_report",
            "notes": "Article-disjoint held-out validation set (71 articles; competition test unreleased)"
        },
        {
            "figure_id": "Fig_A",
            "task": "Propaganda Technique Detection",
            "model": "RoBERTa-base (Exp 1)",
            "dataset": "SemEval-2020 Task 11",
            "split_type": "validation",
            "sample_count": 1448,
            "class_or_condition": "Overall",
            "metric": "Macro F1",
            "value": 0.5132,
            "source_file": "docs/research_history/PHASE_5_MODEL_OPTIMIZATION.md",
            "source_type": "audited_report",
            "notes": "Macro average across 14 SemEval rhetorical techniques"
        },
        {
            "figure_id": "Fig_A",
            "task": "Propaganda Technique Detection",
            "model": "RoBERTa-base (Exp 1)",
            "dataset": "SemEval-2020 Task 11",
            "split_type": "validation",
            "sample_count": 1448,
            "class_or_condition": "Overall",
            "metric": "Weighted F1",
            "value": 0.6477,
            "source_file": "docs/research_history/PHASE_5_MODEL_OPTIMIZATION.md",
            "source_type": "audited_report",
            "notes": "Support-weighted F1 across 14 SemEval rhetorical techniques"
        },
        {
            "figure_id": "Fig_A",
            "task": "Hate Speech Classification",
            "model": "BERT-base-uncased",
            "dataset": "HateXplain",
            "split_type": "test",
            "sample_count": 1924,
            "class_or_condition": "Overall",
            "metric": "Accuracy",
            "value": 0.6741,
            "source_file": "docs/research_history/PHASE_4_EVALUATION.md",
            "source_type": "audited_report",
            "notes": "Full benchmark test split (782 Normal, 548 Offensive, 594 Hate Speech)"
        },
        {
            "figure_id": "Fig_A",
            "task": "Hate Speech Classification",
            "model": "BERT-base-uncased",
            "dataset": "HateXplain",
            "split_type": "test",
            "sample_count": 1924,
            "class_or_condition": "Overall",
            "metric": "Macro F1",
            "value": 0.6617,
            "source_file": "docs/research_history/PHASE_4_EVALUATION.md",
            "source_type": "audited_report",
            "notes": "Unweighted arithmetic mean across Normal, Offensive, Hatespeech"
        },
        {
            "figure_id": "Fig_A",
            "task": "Hate Speech Classification",
            "model": "BERT-base-uncased",
            "dataset": "HateXplain",
            "split_type": "test",
            "sample_count": 1924,
            "class_or_condition": "Overall",
            "metric": "Weighted F1",
            "value": 0.6738,
            "source_file": "docs/research_history/PHASE_4_EVALUATION.md",
            "source_type": "audited_report",
            "notes": "Support-weighted F1 across 3 classes"
        },

        # Figure B: Deepfake Robustness
        {
            "figure_id": "Fig_B",
            "task": "Deepfake Robustness",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split_type": "test_balanced_subset",
            "sample_count": 100,
            "class_or_condition": "original",
            "metric": "Accuracy",
            "value": 0.8100,
            "source_file": "phase4_robustness.json",
            "source_type": "raw_evaluation_output",
            "notes": "50 Real, 50 Fake balanced subset, 4 frames/video"
        },
        {
            "figure_id": "Fig_B",
            "task": "Deepfake Robustness",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split_type": "test_balanced_subset",
            "sample_count": 100,
            "class_or_condition": "original",
            "metric": "Macro F1",
            "value": 0.8098,
            "source_file": "phase4_robustness.json",
            "source_type": "raw_evaluation_output",
            "notes": "Real F1: 0.8041, Fake F1: 0.8155"
        },
        {
            "figure_id": "Fig_B",
            "task": "Deepfake Robustness",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split_type": "test_balanced_subset",
            "sample_count": 100,
            "class_or_condition": "jpeg_compression_Q30",
            "metric": "Accuracy",
            "value": 0.7300,
            "source_file": "phase4_robustness.json",
            "source_type": "raw_evaluation_output",
            "notes": "-8.00% drop from baseline; Real F1: 0.7523, Fake F1: 0.7033"
        },
        {
            "figure_id": "Fig_B",
            "task": "Deepfake Robustness",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split_type": "test_balanced_subset",
            "sample_count": 100,
            "class_or_condition": "gaussian_noise_sigma15",
            "metric": "Accuracy",
            "value": 0.6700,
            "source_file": "phase4_robustness.json",
            "source_type": "raw_evaluation_output",
            "notes": "-14.00% drop from baseline; Real F1: 0.6118, Fake F1: 0.7130"
        },
        {
            "figure_id": "Fig_B",
            "task": "Deepfake Robustness",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split_type": "test_balanced_subset",
            "sample_count": 100,
            "class_or_condition": "gaussian_blur_r2",
            "metric": "Accuracy",
            "value": 0.5600,
            "source_file": "phase4_robustness.json",
            "source_type": "raw_evaluation_output",
            "notes": "-25.00% drop; Severe high-freq loss; Real recall collapses to 0.12"
        },
        {
            "figure_id": "Fig_B",
            "task": "Deepfake Robustness",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split_type": "test_balanced_subset",
            "sample_count": 100,
            "class_or_condition": "lowres_resize_quarter",
            "metric": "Accuracy",
            "value": 0.5600,
            "source_file": "phase4_robustness.json",
            "source_type": "raw_evaluation_output",
            "notes": "-25.00% drop; 1/4 bilinear downsampling suppresses fine boundary seams"
        },

        # Figure C & D: Deepfake Class-wise metrics and confusion matrix
        {
            "figure_id": "Fig_C_D",
            "task": "Visual Deepfake Detection",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split_type": "test",
            "sample_count": 700,
            "class_or_condition": "Real",
            "metric": "Support",
            "value": 140,
            "source_file": "phase4_results.json",
            "source_type": "raw_evaluation_output",
            "notes": "20.0% of test dataset"
        },
        {
            "figure_id": "Fig_C_D",
            "task": "Visual Deepfake Detection",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split_type": "test",
            "sample_count": 700,
            "class_or_condition": "Fake",
            "metric": "Support",
            "value": 560,
            "source_file": "phase4_results.json",
            "source_type": "raw_evaluation_output",
            "notes": "80.0% of test dataset"
        },
        {
            "figure_id": "Fig_C_D",
            "task": "Visual Deepfake Detection",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split_type": "test",
            "sample_count": 700,
            "class_or_condition": "Real",
            "metric": "Precision",
            "value": 0.6505,
            "source_file": "phase4_results.json",
            "source_type": "raw_evaluation_output",
            "notes": "121 True Real / 186 Total Predicted Real"
        },
        {
            "figure_id": "Fig_C_D",
            "task": "Visual Deepfake Detection",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split_type": "test",
            "sample_count": 700,
            "class_or_condition": "Real",
            "metric": "Recall",
            "value": 0.8643,
            "source_file": "phase4_results.json",
            "source_type": "raw_evaluation_output",
            "notes": "121 True Real / 140 Total Ground-Truth Real"
        },
        {
            "figure_id": "Fig_C_D",
            "task": "Visual Deepfake Detection",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split_type": "test",
            "sample_count": 700,
            "class_or_condition": "Real",
            "metric": "F1-Score",
            "value": 0.7423,
            "source_file": "phase4_results.json",
            "source_type": "raw_evaluation_output",
            "notes": "Harmonic mean of precision 0.6505 and recall 0.8643"
        },
        {
            "figure_id": "Fig_C_D",
            "task": "Visual Deepfake Detection",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split_type": "test",
            "sample_count": 700,
            "class_or_condition": "Fake",
            "metric": "Precision",
            "value": 0.9630,
            "source_file": "phase4_results.json",
            "source_type": "raw_evaluation_output",
            "notes": "495 True Fake / 514 Total Predicted Fake"
        },
        {
            "figure_id": "Fig_C_D",
            "task": "Visual Deepfake Detection",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split_type": "test",
            "sample_count": 700,
            "class_or_condition": "Fake",
            "metric": "Recall",
            "value": 0.8839,
            "source_file": "phase4_results.json",
            "source_type": "raw_evaluation_output",
            "notes": "495 True Fake / 560 Total Ground-Truth Fake"
        },
        {
            "figure_id": "Fig_C_D",
            "task": "Visual Deepfake Detection",
            "model": "EfficientNet-B4",
            "dataset": "FaceForensics++ C23",
            "split_type": "test",
            "sample_count": 700,
            "class_or_condition": "Fake",
            "metric": "F1-Score",
            "value": 0.9218,
            "source_file": "phase4_results.json",
            "source_type": "raw_evaluation_output",
            "notes": "Harmonic mean of precision 0.9630 and recall 0.8839"
        },

        # Figure E: Propaganda optimization comparison
        {
            "figure_id": "Fig_E",
            "task": "Propaganda Optimization",
            "model": "RoBERTa-base (Phase 3 Baseline)",
            "dataset": "SemEval-2020 Task 11",
            "split_type": "validation",
            "sample_count": 1448,
            "class_or_condition": "Overall",
            "metric": "Macro F1",
            "value": 0.4175,
            "source_file": "docs/research_history/PHASE_5_MODEL_OPTIMIZATION.md",
            "source_type": "audited_report",
            "notes": "Unweighted cross-entropy loss, constant learning rate 3e-5"
        },
        {
            "figure_id": "Fig_E",
            "task": "Propaganda Optimization",
            "model": "RoBERTa-base (Phase 5 Exp 1)",
            "dataset": "SemEval-2020 Task 11",
            "split_type": "validation",
            "sample_count": 1448,
            "class_or_condition": "Overall",
            "metric": "Macro F1",
            "value": 0.5132,
            "source_file": "docs/research_history/PHASE_5_MODEL_OPTIMIZATION.md",
            "source_type": "audited_report",
            "notes": "Square-root inverse class weighting + linear warmup and decay schedule (+9.57% gain)"
        }
    ]

    # Write CSV
    fieldnames = [
        "figure_id", "task", "model", "dataset", "split_type",
        "sample_count", "class_or_condition", "metric", "value",
        "source_file", "source_type", "notes"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Generated: {csv_path} ({len(rows)} rows)")

    # Write Markdown data guide for GPT plotting
    md_content = """# Research Figures Data and Provenance Guide

This document accompanies `reports/research_figures_data.csv` and provides complete provenance, source file traceability, reproduction commands, and structured tables ready for GPT/Python plotting.

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
Outputs are written to `reports/figures/*.png` (300 DPI) and `reports/figures/*.svg`.
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Generated: {md_path}")

    # Write reports/figures/README.md
    fig_readme_content = """# Research Figures Directory

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
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(fig_readme_content)
    print(f"Generated: {readme_path}")


if __name__ == "__main__":
    print("Generating publication research figures...")
    generate_figure_a()
    generate_figure_b()
    generate_figure_c()
    generate_figure_d()
    generate_figure_e()
    generate_figure_f()
    export_research_data_csv_and_md()
    print("All figures and datasets successfully generated!")
