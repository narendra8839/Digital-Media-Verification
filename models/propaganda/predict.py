"""Inference and checkpoint loading for RoBERTa propaganda technique classifier."""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any


def load_propaganda_model(checkpoint_dir: str = "models/propaganda/optimization/exp1_sqrt_weighted_scheduled",
                          tokenizer_dir: str = None,
                          device: str = None):
    if device is None:
        try:
            from api.utils.device import get_optimal_device
            device = get_optimal_device()
        except ImportError:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Fallback to checkpoint_full or checkpoint if specified path not found
    if not os.path.exists(checkpoint_dir):
        if os.path.exists("models/propaganda/checkpoint_full"):
            checkpoint_dir = "models/propaganda/checkpoint_full"
        elif os.path.exists("models/propaganda/checkpoint"):
            checkpoint_dir = "models/propaganda/checkpoint"

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)

    # Resolve tokenizer directory
    if tokenizer_dir is None or not os.path.exists(tokenizer_dir):
        chk_tok = os.path.join(checkpoint_dir, "tokenizer")
        if os.path.exists(chk_tok):
            tokenizer_dir = chk_tok
        elif os.path.exists("models/propaganda/tokenizer"):
            tokenizer_dir = "models/propaganda/tokenizer"
        else:
            tokenizer_dir = "roberta-base"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model.eval()

    # Load label mapping
    mapping_file = os.path.join(checkpoint_dir, "label_mapping.json")
    if not os.path.exists(mapping_file) and os.path.exists("models/propaganda/label_mapping.json"):
        mapping_file = "models/propaganda/label_mapping.json"

    if os.path.exists(mapping_file):
        with open(mapping_file, "r") as f:
            label_mapping = json.load(f)
    else:
        label_mapping = {str(k): v for k, v in model.config.id2label.items()}

    return model, tokenizer, label_mapping, device


def predict_propaganda(model, tokenizer, label_mapping: Dict[str, str], text: str, device: str) -> Dict[str, Any]:
    """Run inference on input text string."""
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = int(torch.argmax(outputs.logits, dim=1).cpu().item())
    top_label = label_mapping.get(str(top_idx), f"class_{top_idx}")
    top_confidence = float(probs[top_idx])

    prob_dict = {label_mapping.get(str(i), f"class_{i}"): round(float(p), 4) for i, p in enumerate(probs)}

    return {
        "prediction": top_label,
        "technique": top_label,
        "confidence": round(top_confidence, 4),
        "class_index": top_idx,
        "probabilities": prob_dict
    }
