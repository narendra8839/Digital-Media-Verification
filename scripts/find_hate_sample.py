import json
from detection.hate_speech_detector import HateSpeechDetector

hate_det = HateSpeechDetector(device="cuda:0")

with open("data/full_splits/hatexplain_val.json", "r", encoding="utf-8") as f:
    hate_val = json.load(f)

found = 0
for k, item in (hate_val.items() if isinstance(hate_val, dict) else enumerate(hate_val)):
    tokens = item.get("post_tokens", [])
    text = " ".join(tokens)
    # Check majority label in annotators
    annotators = item.get("annotators", [])
    labels = [a.get("label") for a in annotators]
    if labels.count(0) >= 2 or item.get("label") in ["hatespeech", 0]: # 0 = hatespeech in HateXplain
        res = hate_det.detect_text(text)
        if res["prediction"] == "hatespeech" and res["confidence"] > 0.80:
            print(f"\nHate Sample: Conf: {res['confidence']:.2%}")
            print(f"Text: \"{text}\"")
            found += 1
            if found >= 3:
                break
