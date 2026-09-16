"""Comprehensive leakage validation check for all MVP subsets."""

import json
import os

def check_deepfake_leakage():
    print("Checking Deepfake subsets for leakage...")
    with open("data/mvp/deepfake/train_subset.json", "r", encoding="utf-8") as f:
        train = json.load(f)
    with open("data/mvp/deepfake/val_subset.json", "r", encoding="utf-8") as f:
        val = json.load(f)
    with open("data/mvp/deepfake/test_subset.json", "r", encoding="utf-8") as f:
        test = json.load(f)

    train_ids = set(x["video_id"] for x in train)
    val_ids = set(x["video_id"] for x in val)
    test_ids = set(x["video_id"] for x in test)

    train_val_ovlp = train_ids.intersection(val_ids)
    train_test_ovlp = train_ids.intersection(test_ids)
    val_test_ovlp = val_ids.intersection(test_ids)

    print(f"Deepfake video count: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"Train/Val overlap: {train_val_ovlp}")
    print(f"Train/Test overlap: {train_test_ovlp}")
    print(f"Val/Test overlap: {val_test_ovlp}")
    assert len(train_val_ovlp) == 0 and len(train_test_ovlp) == 0 and len(val_test_ovlp) == 0, "Deepfake leakage detected!"
    print("--> Deepfake leakage check: ZERO LEAKAGE (DISJOINT SEQUENCES)\n")


def check_semeval_leakage():
    print("Checking SemEval subsets for article-level leakage...")
    with open("data/mvp/semeval2020_task11/train_subset.json", "r", encoding="utf-8") as f:
        train = json.load(f)
    with open("data/mvp/semeval2020_task11/val_subset.json", "r", encoding="utf-8") as f:
        val = json.load(f)

    train_articles = set(x["article_id"] for x in train)
    val_articles = set(x["article_id"] for x in val)

    overlap = train_articles.intersection(val_articles)
    print(f"SemEval articles: train={len(train_articles)}, val={len(val_articles)}")
    print(f"Article overlap: {overlap}")
    assert len(overlap) == 0, "SemEval article leakage detected!"
    print("--> SemEval leakage check: ZERO LEAKAGE (ARTICLE-LEVEL DISJOINT)\n")


def check_hatexplain_leakage():
    print("Checking HateXplain subsets for split boundary integrity...")
    with open("data/mvp/hatexplain/train_subset.json", "r", encoding="utf-8") as f:
        train = json.load(f)
    with open("data/mvp/hatexplain/val_subset.json", "r", encoding="utf-8") as f:
        val = json.load(f)
    with open("data/mvp/hatexplain/test_subset.json", "r", encoding="utf-8") as f:
        test = json.load(f)

    train_ids = set(x["post_id"] for x in train)
    val_ids = set(x["post_id"] for x in val)
    test_ids = set(x["post_id"] for x in test)

    train_val_ovlp = train_ids.intersection(val_ids)
    train_test_ovlp = train_ids.intersection(test_ids)
    val_test_ovlp = val_ids.intersection(test_ids)

    print(f"HateXplain post count: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"Train/Val overlap: {train_val_ovlp}")
    print(f"Train/Test overlap: {train_test_ovlp}")
    print(f"Val/Test overlap: {val_test_ovlp}")
    assert len(train_val_ovlp) == 0 and len(train_test_ovlp) == 0 and len(val_test_ovlp) == 0, "HateXplain leakage detected!"

    with open("data/hatexplain/post_id_divisions.json", "r", encoding="utf-8") as f:
        official_splits = json.load(f)

    official_train = set(official_splits["train"])
    official_val = set(official_splits["val"])
    official_test = set(official_splits["test"])

    assert train_ids.issubset(official_train), "Train samples not subset of official train!"
    assert val_ids.issubset(official_val), "Val samples not subset of official val!"
    assert test_ids.issubset(official_test), "Test samples not subset of official test!"

    print("--> HateXplain leakage check: ZERO LEAKAGE (OFFICIAL BENCHMARK SPLIT PRESERVED)\n")


if __name__ == "__main__":
    check_deepfake_leakage()
    check_semeval_leakage()
    check_hatexplain_leakage()
    print("ALL LEAKAGE CHECKS CONFIRMED CLEAN!")
