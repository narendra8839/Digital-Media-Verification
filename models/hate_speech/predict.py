"""Inference and checkpoint loading for BERT 3-class hate speech classifier."""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any


def load_hate_speech_model(checkpoint_dir: str = "models/hate_speech/checkpoint",
                           tokenizer_dir: str = "models/hate_speech/tokenizer",
                           device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir if os.path.exists(tokenizer_dir) else "bert-base-uncased")
    model.eval()

    mapping_file = "models/hate_speech/label_mapping.json"
    if os.path.exists(mapping_file):
        with open(mapping_file, "r") as f:
            label_mapping = json.load(f)
    else:
        label_mapping = {"0": "normal", "1": "offensive", "2": "hatespeech"}

    return model, tokenizer, label_mapping, device


def predict_hate_speech(model, tokenizer, label_mapping: Dict[str, str], text: str, device: str) -> Dict[str, Any]:
    """Run inference on input text string."""
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = int(torch.argmax(outputs.logits, dim=1).cpu().item())
    top_label = label_mapping.get(str(top_idx), f"class_{top_idx}")
    top_confidence = float(probs[top_idx])

    prob_dict = {
        label_mapping.get("0", "normal"): round(float(probs[0]), 4),
        label_mapping.get("1", "offensive"): round(float(probs[1]), 4),
        label_mapping.get("2", "hatespeech"): round(float(probs[2]), 4)
    }

    return {
        "prediction": top_label,
        "confidence": round(top_confidence, 4),
        "class_index": top_idx,
        "probabilities": prob_dict
    }
