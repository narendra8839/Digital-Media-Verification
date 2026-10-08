import json
from detection.hate_speech_detector import HateSpeechDetector

hate_det = HateSpeechDetector(device="cuda:0")

with open("data/full_splits/hatexplain_val.json", "r", encoding="utf-8") as f:
    d = json.load(f)

hates = [x for x in d if x.get("label") == "hatespeech"]
print(f"Total hatespeech samples: {len(hates)}")

for x in hates:
    txt = x.get("text", "")
    # Filter for clean speech suitable for presentation (no extreme slurs, clear bias)
    if 40 <= len(txt) <= 120 and "<number>" not in txt and not any(w in txt for w in ["fuck", "cunt", "nigger", "kike", "faggot", "shithole"]):
        res = hate_det.detect_text(txt)
        if res.get("prediction") == "hatespeech" and res.get("confidence", 0) > 0.80:
            print(f"\nTarget Text: \"{txt}\"")
            print(f"Prediction: {res.get('prediction')} ({res.get('confidence'):.2%}) | Review: {res.get('review_required')}")
            break
