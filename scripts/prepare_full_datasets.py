"""Audited preparation and manifest generation for full intended datasets.

Covers:
1. FaceForensics++ C23 (5,000 physical videos across 5 methods; 3,600 train / 700 val / 700 test)
2. HateXplain (19,229 posts; 15,383 train / 1,922 val / 1,924 test)
3. SemEval-2020 Task 11 (6,129 spans; article-disjoint 80/20 train/val partition)

Strictly enforces train != val != test separation with zero sample or sequence leakage.
"""

import os
import json
import glob
import random
from collections import Counter
from typing import Dict, Any, List, Set


def prepare_faceforensics(base_dir: str = "data/faceforensics/C23",
                          output_dir: str = "data/full_splits") -> Dict[str, Any]:
    """Audit and prepare FaceForensics++ C23 splits."""
    print("--- Auditing FaceForensics++ C23 ---")
    splits_dir = os.path.join(base_dir, "splits")
    methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

    results = {}
    split_id_sets: Dict[str, Set[str]] = {}

    for split in ["train", "val", "test"]:
        split_file = os.path.join(splits_dir, f"{split}.json")
        with open(split_file, "r") as f:
            pairs = json.load(f)

        split_ids = set([x for p in pairs for x in p])
        split_id_sets[split] = split_ids
        records: List[Dict[str, Any]] = []

        # Real videos: original_sequences/youtube/c23/videos/{id}.mp4
        for sid in sorted(list(split_ids)):
            rel_path = f"original_sequences/youtube/c23/videos/{sid}.mp4"
            abs_path = os.path.join(base_dir, rel_path)
            if os.path.exists(abs_path):
                records.append({
                    "video_id": f"youtube_{sid}",
                    "relative_path": rel_path,
                    "label": "real",
                    "label_id": 0,
                    "method": "youtube",
                    "sequence_ids": [sid]
                })

        # Fake videos: manipulated_sequences/{method}/c23/videos/{src}_{tgt}.mp4
        for p in pairs:
            for m in methods:
                candidates = [f"{p[0]}_{p[1]}.mp4", f"{p[1]}_{p[0]}.mp4"]
                for cand in candidates:
                    rel_path = f"manipulated_sequences/{m}/c23/videos/{cand}"
                    abs_path = os.path.join(base_dir, rel_path)
                    if os.path.exists(abs_path):
                        records.append({
                            "video_id": f"{m}_{cand.replace('.mp4', '')}",
                            "relative_path": rel_path,
                            "label": "fake",
                            "label_id": 1,
                            "method": m,
                            "sequence_ids": [p[0], p[1]]
                        })

        out_path = os.path.join(output_dir, f"deepfake_{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        real_count = sum(1 for r in records if r["label_id"] == 0)
        fake_count = sum(1 for r in records if r["label_id"] == 1)
        results[split] = {
            "total_videos": len(records),
            "real_videos": real_count,
            "fake_videos": fake_count,
            "unique_sequence_ids": len(split_ids),
            "output_file": out_path
        }
        print(f"  [{split.upper()}] Total: {len(records)} | Real: {real_count} | Fake: {fake_count}")

    # Leakage and overlap checks
    overlap_train_val = split_id_sets["train"] & split_id_sets["val"]
    overlap_train_test = split_id_sets["train"] & split_id_sets["test"]
    overlap_val_test = split_id_sets["val"] & split_id_sets["test"]

    assert len(overlap_train_val) == 0, f"Train-Val sequence overlap detected: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"Train-Test sequence overlap detected: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Val-Test sequence overlap detected: {overlap_val_test}"

    print("  [INTEGRITY] Zero sequence overlap between Train, Val, and Test verified.")
    results["leakage_check"] = "PASSED (Zero sequence overlap)"
    return results


def prepare_hatexplain(base_dir: str = "data/hatexplain",
                       output_dir: str = "data/full_splits") -> Dict[str, Any]:
    """Audit and prepare HateXplain splits."""
    print("--- Auditing HateXplain ---")
    with open(os.path.join(base_dir, "post_id_divisions.json"), "r") as f:
        divisions = json.load(f)
    with open(os.path.join(base_dir, "dataset.json"), "r") as f:
        dataset = json.load(f)

    label_map = {"normal": 0, "offensive": 1, "hatespeech": 2}
    results = {}
    div_id_sets = {k: set(v) for k, v in divisions.items()}

    for split in ["train", "val", "test"]:
        records: List[Dict[str, Any]] = []
        for pid in divisions[split]:
            if pid not in dataset:
                continue
            item = dataset[pid]
            annotator_labels = [a["label"] for a in item["annotators"]]
            c = Counter(annotator_labels)
            maj_label = c.most_common(1)[0][0]
            tokens = item["post_tokens"]
            text = " ".join(tokens)
            rationales = item.get("rationales", [])

            records.append({
                "post_id": pid,
                "text": text,
                "label": maj_label,
                "label_id": label_map[maj_label],
                "tokens": tokens,
                "rationales": rationales
            })

        out_path = os.path.join(output_dir, f"hatexplain_{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        counts = Counter(r["label"] for r in records)
        results[split] = {
            "total_samples": len(records),
            "class_distribution": dict(counts),
            "has_rationales_count": sum(1 for r in records if any(any(x) for x in r["rationales"])),
            "output_file": out_path
        }
        print(f"  [{split.upper()}] Total: {len(records)} | Classes: {dict(counts)}")

    # Leakage check
    overlap_train_val = div_id_sets["train"] & div_id_sets["val"]
    overlap_train_test = div_id_sets["train"] & div_id_sets["test"]
    overlap_val_test = div_id_sets["val"] & div_id_sets["test"]

    assert len(overlap_train_val) == 0, f"Train-Val overlap in HateXplain: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"Train-Test overlap in HateXplain: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Val-Test overlap in HateXplain: {overlap_val_test}"

    print("  [INTEGRITY] Zero post_id overlap between Train, Val, and Test verified.")
    results["leakage_check"] = "PASSED (Zero post_id overlap)"
    return results


def prepare_semeval(base_dir: str = "data/semeval2020_task11",
                    output_dir: str = "data/full_splits",
                    seed: int = 42,
                    val_article_ratio: float = 0.20) -> Dict[str, Any]:
    """Audit and prepare SemEval-2020 Task 11 splits."""
    print("--- Auditing SemEval-2020 Task 11 ---")
    # 1. Read all article text files
    articles: Dict[str, str] = {}
    article_files = glob.glob(os.path.join(base_dir, "train-articles/*.txt"))
    for f in article_files:
        art_id = os.path.splitext(os.path.basename(f))[0].replace("article", "")
        with open(f, "r", encoding="utf-8") as af:
            articles[art_id] = af.read()

    # 2. Read annotations
    tc_labels_file = os.path.join(base_dir, "train-task2-TC.labels")
    all_annotations: List[Dict[str, Any]] = []
    with open(tc_labels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                art_id, tech, s_str, e_str = parts[0], parts[1], parts[2], parts[3]
                s, e = int(s_str), int(e_str)
                text_span = articles[art_id][s:e]
                all_annotations.append({
                    "article_id": art_id,
                    "technique": tech,
                    "start_offset": s,
                    "end_offset": e,
                    "text_span": text_span
                })

    # 3. Article-level disjoint partition (seed=42)
    random.seed(seed)
    annotated_art_ids = sorted(list(set(a["article_id"] for a in all_annotations)))
    shuffled_art_ids = list(annotated_art_ids)
    random.shuffle(shuffled_art_ids)

    num_val = int(len(shuffled_art_ids) * val_article_ratio)
    val_art_ids = set(shuffled_art_ids[:num_val])
    train_art_ids = set(shuffled_art_ids[num_val:])

    train_records = [a for a in all_annotations if a["article_id"] in train_art_ids]
    val_records = [a for a in all_annotations if a["article_id"] in val_art_ids]

    train_out = os.path.join(output_dir, "propaganda_train.json")
    val_out = os.path.join(output_dir, "propaganda_val.json")

    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2)
    with open(val_out, "w", encoding="utf-8") as f:
        json.dump(val_records, f, indent=2)

    train_techs = Counter(r["technique"] for r in train_records)
    val_techs = Counter(r["technique"] for r in val_records)

    print(f"  [TRAIN] Total: {len(train_records)} spans across {len(train_art_ids)} articles | Classes: {len(train_techs)}")
    print(f"  [VAL] Total: {len(val_records)} spans across {len(val_art_ids)} articles | Classes: {len(val_techs)}")

    # Leakage check
    overlap_articles = train_art_ids & val_art_ids
    assert len(overlap_articles) == 0, f"Article overlap detected in SemEval: {overlap_articles}"
    print("  [INTEGRITY] Zero article overlap between Train and Val verified.")

    return {
        "total_annotations": len(all_annotations),
        "total_annotated_articles": len(annotated_art_ids),
        "train": {
            "num_articles": len(train_art_ids),
            "num_spans": len(train_records),
            "class_distribution": dict(train_techs),
            "output_file": train_out
        },
        "val": {
            "num_articles": len(val_art_ids),
            "num_spans": len(val_records),
            "class_distribution": dict(val_techs),
            "output_file": val_out
        },
        "leakage_check": "PASSED (Zero article overlap)"
    }


def main():
    output_dir = "data/full_splits"
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "faceforensics": prepare_faceforensics(output_dir=output_dir),
        "hatexplain": prepare_hatexplain(output_dir=output_dir),
        "semeval2020_task11": prepare_semeval(output_dir=output_dir)
    }

    summary_file = os.path.join(output_dir, "manifest_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSuccessfully generated full dataset manifests in '{output_dir}'.")
    print(f"Summary written to '{summary_file}'.")


if __name__ == "__main__":
    main()
