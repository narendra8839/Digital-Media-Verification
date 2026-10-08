import json
import os
from detection.propaganda_detector import PropagandaDetector

val_path = "data/full_splits/propaganda_val.json"
with open(val_path, "r", encoding="utf-8") as f:
    val_data = json.load(f)

print(f"Loaded {len(val_data)} validation samples.")
detector = PropagandaDetector(device="cuda:0")

# Test a few samples with clear rhetorical techniques
sample_candidates = []
for item in val_data:
    label = item.get("label", "")
    text = item.get("text", "").strip()
    if label not in ["non_propaganda", "None", "0"] and 40 < len(text) < 180:
        sample_candidates.append((label, text))

print(f"Found {len(sample_candidates)} candidate propaganda texts.")
for label, text in sample_candidates[:10]:
    pred = detector.predict(text)
    print(f"\nGround Truth: {label}")
    print(f"Text: \"{text}\"")
    print(f"Model Prediction: {pred['prediction']} (Conf: {pred['confidence']:.2%})")
