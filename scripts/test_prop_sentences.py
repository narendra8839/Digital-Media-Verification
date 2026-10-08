from detection.propaganda_detector import PropagandaDetector

d = PropagandaDetector(device="cuda:0")
sentences = [
    "This throws their entire narrative into question and makes one wonder what else is being hidden from the public?",
    "If we do not act immediately, these dangerous extremists will destroy our freedom and our children's future.",
    "The corrupt media and their puppet politicians are spreading poisonous lies to deceive the honest working people.",
    "Every patriotic citizen must stand united against the sinister forces that threaten our homeland."
]

for s in sentences:
    res = d.detect_text(s)
    print(f"Pred: {res['prediction']} ({res['confidence']:.2%}) | Status: {res['decision_status']} | Text: \"{s}\"")
