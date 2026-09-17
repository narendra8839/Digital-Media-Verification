"""Demo test script for token attribution and rationale alignment."""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
from explainability.text_attention import TokenAttribution, evaluate_rationale_alignment
from models.hate_speech.predict import load_hate_speech_model
from models.propaganda.predict import load_propaganda_model

def test_propaganda_attribution():
    print("=" * 60)
    print("TEST: PROPAGANDA TOKEN ATTRIBUTION (RoBERTa 14-class)")
    print("=" * 60)
    model, tokenizer, label_mapping, device = load_propaganda_model()
    explainer = TokenAttribution(model, tokenizer, device=device)

    with open("data/mvp/semeval2020_task11/val_subset.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)

    sample = val_data[0]
    text = sample["text_span"]
    gt = sample["technique"]
    print(f"Sample: '{text}' (GT: {gt})")

    res = explainer.attribute(text, label_mapping=label_mapping)
    print(f"Prediction: {res['prediction']} (conf: {res['confidence']})")
    print("Top Influential Tokens:")
    for t in res["top_tokens"]:
        print(f"  {t['clean_token']:15s}: {t['score']:.4f}")
    assert len(res["top_tokens"]) > 0, "No top tokens extracted!"
    print("--> Propaganda token attribution PASSED.\n")


def test_hate_speech_attribution_and_rationale():
    print("=" * 60)
    print("TEST: HATE SPEECH ATTRIBUTION & RATIONALE ALIGNMENT (BERT 3-class)")
    print("=" * 60)
    model, tokenizer, label_mapping, device = load_hate_speech_model()
    explainer = TokenAttribution(model, tokenizer, device=device)

    with open("data/mvp/hatexplain/test_subset.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Pick a sample that has rationales (e.g. hatespeech)
    sample = [x for x in test_data if x["label"] == "hatespeech"][0]
    text = sample["text"]
    gt = sample["label"]
    post_tokens = sample["post_tokens"]
    rationales = sample.get("rationales", [])

    print(f"Sample: '{text}' (GT: {gt})")
    print(f"Post Tokens: {post_tokens}")

    res = explainer.attribute(text, label_mapping=label_mapping)
    print(f"Prediction: {res['prediction']} (conf: {res['confidence']})")
    print("Top Influential Tokens:")
    for t in res["top_tokens"]:
        print(f"  {t['clean_token']:15s}: {t['score']:.4f}")

    # Evaluate rationale alignment
    alignment = evaluate_rationale_alignment(
        attributed_tokens=res["token_scores"],
        ground_truth_rationales=rationales,
        post_tokens=post_tokens,
        top_k=3
    )

    print("\nRationale Alignment Evaluation:")
    print(f"  Ground Truth Count: {alignment['ground_truth_count']}")
    print(f"  Model Selected Count: {alignment['model_selected_count']}")
    print(f"  Overlap Count: {alignment['overlap_count']}")
    print(f"  Overlap Tokens: {alignment['overlap_tokens']}")
    print(f"  Precision: {alignment['precision']}")
    print(f"  Recall: {alignment['recall']}")
    print(f"  F1: {alignment['f1']}")

    assert alignment["overlap_count"] > 0, "Expected overlap on slurs in hatespeech sample!"
    print("--> Hate Speech attribution and rationale alignment PASSED.\n")


if __name__ == "__main__":
    test_propaganda_attribution()
    test_hate_speech_attribution_and_rationale()
    print("ALL TOKEN ATTRIBUTION TESTS PASSED SUCCESSFULLY!")
