"""Extracts and prints formatted Markdown tables comparing Baseline, V2B, and V3A/B/C."""

import os
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_json(rel_path):
    p = os.path.join(PROJECT_ROOT, rel_path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def format_benchmark(title, files_map):
    print(f"\n### {title}\n")
    print("| Condition | Baseline (Full) | Candidate V2-B | Candidate V3-A | Candidate V3-B | Candidate V3-C |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")

    conds = ["original", "jpeg", "blur", "resize", "noise"]
    loaded = {name: load_json(path) for name, path in files_map.items()}

    for c in conds:
        row = [c.capitalize()]
        for name in ["Baseline", "V2-B", "V3-A", "V3-B", "V3-C"]:
            d = loaded[name]["conditions"][c]
            acc = d["accuracy"] * 100
            f1 = d["macro_f1"]
            cm = d["confusion_matrix"]
            # cm is [[TP_fake, FN_fake], [FP_fake, TN_fake]]
            tp_fake, fn_fake = cm[0]
            fp_fake, tn_fake = cm[1]
            fake_rec = (tp_fake / (tp_fake + fn_fake)) * 100 if (tp_fake + fn_fake) > 0 else 0
            real_rec = (tn_fake / (fp_fake + tn_fake)) * 100 if (fp_fake + tn_fake) > 0 else 0
            row.append(f"**{acc:.1f}%** (F1: {f1:.3f})<br>F-Rec: {fake_rec:.0f}% \| R-Rec: {real_rec:.0f}%")
        print("| " + " | ".join(row) + " |")

def format_severity():
    print("\n### Multi-Severity Distortion Degradation Curve: Baseline vs Candidate V3-A\n")
    d = load_json("eval_results/severity_curve_v3.json")
    base = d["baseline"]
    cand = d["candidate"]

    print("| Distortion Type | Severity / Level | Baseline Acc | Candidate V3-A Acc | Delta Acc | Baseline F1 | Candidate V3-A F1 | V3-A Fake Rec | V3-A Real Rec |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for k in base:
        b = base[k]
        c = cand[k]
        dist, val = k.split("_")
        desc = b["description"]
        d_acc = (c["accuracy"] - b["accuracy"]) * 100
        b_acc = b["accuracy"] * 100
        c_acc = c["accuracy"] * 100
        b_f1 = b["macro_f1"]
        c_f1 = c["macro_f1"]
        c_f_rec = c["per_class"]["fake"]["recall"] * 100
        c_r_rec = c["per_class"]["real"]["recall"] * 100
        print(f"| {dist.capitalize()} | {desc} | {b_acc:.1f}% | {c_acc:.1f}% | {d_acc:+.1f}% | {b_f1:.4f} | {c_f1:.4f} | {c_f_rec:.1f}% | {c_r_rec:.1f}% |")

if __name__ == "__main__":
    format_benchmark("Development Benchmark (N=100, Offset 0)", {
        "Baseline": "eval_results/baseline_robustness.json",
        "V2-B": "eval_results/candidate_v2b_robustness.json",
        "V3-A": "eval_results/candidate_v3a_robustness.json",
        "V3-B": "eval_results/candidate_v3b_robustness.json",
        "V3-C": "eval_results/candidate_v3c_robustness.json",
    })

    format_benchmark("Reserved Comparison Subset (N=100, Offset 50)", {
        "Baseline": "eval_results/reserved_baseline_robustness.json",
        "V2-B": "eval_results/reserved_candidate_v2b_robustness.json",
        "V3-A": "eval_results/reserved_candidate_v3a_robustness.json",
        "V3-B": "eval_results/reserved_candidate_v3b_robustness.json",
        "V3-C": "eval_results/reserved_candidate_v3c_robustness.json",
    })

    format_severity()
