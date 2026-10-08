import json
import torch
from detection.propaganda_detector import PropagandaDetector
from detection.hate_speech_detector import HateSpeechDetector

print("Loading detectors...")
prop_det = PropagandaDetector(device="cuda:0")
hate_det = HateSpeechDetector(device="cuda:0")

# 1. Propaganda search
print("\n--- Testing Propaganda Sentences from SemEval-2020 ---")
with open("data/full_splits/propaganda_val.json", "r", encoding="utf-8") as f:
    prop_val = json.load(f)

candidates = []
for item in prop_val:
    tech = item.get("technique", "")
    span = item.get("text_span", "").strip()
    if 30 <= len(span) <= 120 and tech in ["Loaded_Language", "Appeal_to_fear-prejudice", "Doubt", "Flag-Waving", "Causal_Oversimplification"]:
        candidates.append((tech, span))

print(f"Total candidate spans: {len(candidates)}")
best_prop = None
best_prop_conf = 0.0

for tech, span in candidates:
    res = prop_det.detect_text(span)
    pred = res.get("prediction")
    conf = res.get("confidence", 0.0)
    if pred not in [None, "non_propaganda"] and conf > 0.80:
        print(f"Propaganda Sample: [{tech}] -> Pred: {pred} ({conf:.2%}) | Review: {res.get('review_required')}")
        print(f"  Text: \"{span}\"")
        best_prop = (tech, span, pred, conf)
        if conf > 0.90:
            break

# 2. Hate speech search
print("\n--- Testing HateXplain Sentences ---")
with open("data/full_splits/hatexplain_val.json", "r", encoding="utf-8") as f:
    hate_val = json.load(f)

print(f"HateXplain val entries: {len(hate_val)}")
best_hate = None

for k, item in list(hate_val.items()) if isinstance(hate_val, dict) else enumerate(hate_val):
    entry = item if isinstance(item, dict) else hate_val[k]
    tokens = entry.get("post_tokens", [])
    text = " ".join(tokens)
    if 30 <= len(text) <= 120:
        res = hate_det.detect_text(text)
        pred = res.get("prediction")
        conf = res.get("confidence", 0.0)
        if pred == "hatespeech" and conf > 0.75:
            print(f"Hate Speech Sample: Pred: {pred} ({conf:.2%}) | Review: {res.get('review_required')}")
            print(f"  Text: \"{text}\"")
            best_hate = (text, pred, conf)
            break
        elif pred == "offensive" and conf > 0.85 and best_hate is None:
            best_hate = (text, pred, conf)

if best_hate and not best_hate[1] == "hatespeech":
    print(f"Selected Offensive/Hate Candidate: Pred: {best_hate[1]} ({best_hate[2]:.2%})\n  Text: \"{best_hate[0]}\"")
