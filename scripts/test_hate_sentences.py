from detection.hate_speech_detector import HateSpeechDetector

d = HateSpeechDetector(device="cuda:0")
sentences = [
    "all muslims are dangerous terrorists who must be expelled from our nation",
    "these jewish bankers are evil leeches controlling our government and media",
    "illegal aliens are subhuman scum taking over our cities and spreading disease",
    "those savage migrants are violent beasts who do not belong in civilized society"
]

for s in sentences:
    res = d.detect_text(s)
    print(f"Pred: {res['prediction']} ({res['confidence']:.2%}) | Status: {res['decision_status']} | Text: \"{s}\"")
