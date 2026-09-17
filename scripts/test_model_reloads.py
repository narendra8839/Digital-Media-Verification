"""Independent verification script to reload and test all 3 trained MVP checkpoints.

Tests:
1. Deepfake Model (EfficientNet-B4)
2. Propaganda Model (RoBERTa 14-class)
3. Hate Speech Model (BERT 3-class)
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import torch

def test_deepfake_reload():
    print("=" * 60)
    print("TEST 1: DEEPFAKE MODEL RELOAD (EfficientNet-B4)")
    print("=" * 60)
    from models.deepfake.predict import load_deepfake_model, predict_deepfake_video

    model, device = load_deepfake_model(
        checkpoint_path="models/deepfake/checkpoint/best_model.pt"
    )
    print(f"Loaded Deepfake model on device: {device}")

    with open("data/mvp/deepfake/val_subset.json", "r", encoding="utf-8") as f:
        val_subset = json.load(f)

    test_video = os.path.join("data/faceforensics/C23", val_subset[0]["relative_path"])
    gt = val_subset[0]["label"]
    print(f"Testing inference on unseen validation video: {test_video} (GT: {gt})")

    result = predict_deepfake_video(model, test_video, device=device, max_frames=4)
    print("Inference Result:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Aggregated Probabilities: {result['probabilities']}")
    print(f"  Frames analyzed: {result['frames_analyzed']}")
    assert "prediction" in result and "confidence" in result, "Deepfake reload test failed format validation"
    print("--> Deepfake reload test PASSED.\n")


def test_propaganda_reload():
    print("=" * 60)
    print("TEST 2: PROPAGANDA MODEL RELOAD (RoBERTa 14-class)")
    print("=" * 60)
    from models.propaganda.predict import load_propaganda_model, predict_propaganda

    model, tokenizer, label_mapping, device = load_propaganda_model(
        checkpoint_dir="models/propaganda/checkpoint",
        tokenizer_dir="models/propaganda/tokenizer"
    )
    print(f"Loaded Propaganda model & tokenizer on device: {device}")
    print(f"Number of classes in mapping: {len(label_mapping)}")

    with open("data/mvp/semeval2020_task11/val_subset.json", "r", encoding="utf-8") as f:
        val_subset = json.load(f)

    for i in range(min(3, len(val_subset))):
        sample = val_subset[i]
        text = sample["text_span"]
        gt = sample["technique"]
        res = predict_propaganda(model, tokenizer, label_mapping, text, device=device)
        print(f"Sample {i+1}:")
        print(f"  Text: {text[:80]}...")
        print(f"  Ground Truth: {gt}")
        print(f"  Prediction: {res['prediction']} (conf: {res['confidence']:.4f})")
    assert "prediction" in res, "Propaganda reload test failed format validation"
    print("--> Propaganda reload test PASSED.\n")


def test_hate_speech_reload():
    print("=" * 60)
    print("TEST 3: HATE SPEECH MODEL RELOAD (BERT 3-class)")
    print("=" * 60)
    from models.hate_speech.predict import load_hate_speech_model, predict_hate_speech

    model, tokenizer, label_mapping, device = load_hate_speech_model(
        checkpoint_dir="models/hate_speech/checkpoint",
        tokenizer_dir="models/hate_speech/tokenizer"
    )
    print(f"Loaded Hate Speech model & tokenizer on device: {device}")
    print(f"Label mapping: {label_mapping}")

    with open("data/mvp/hatexplain/test_subset.json", "r", encoding="utf-8") as f:
        test_subset = json.load(f)

    for i in range(min(3, len(test_subset))):
        sample = test_subset[i]
        text = sample["text"]
        gt = sample["label"]
        res = predict_hate_speech(model, tokenizer, label_mapping, text, device=device)
        print(f"Sample {i+1}:")
        print(f"  Text: {text[:80]}...")
        print(f"  Ground Truth: {gt}")
        print(f"  Prediction: {res['prediction']} (conf: {res['confidence']:.4f})")
        print(f"  Probabilities: {res['probabilities']}")
    assert "prediction" in res, "Hate Speech reload test failed format validation"
    print("--> Hate Speech reload test PASSED.\n")


if __name__ == "__main__":
    test_deepfake_reload()
    test_propaganda_reload()
    test_hate_speech_reload()
    print("=" * 60)
    print("ALL THREE MODEL RELOAD TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
