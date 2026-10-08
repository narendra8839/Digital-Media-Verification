import os
import sys
import json
import logging
from collections import Counter

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from models.propaganda.train import PropagandaDataset, evaluate_propaganda

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("finalize_propaganda")

def finalize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required"
    checkpoint_dir = "models/propaganda/checkpoint_full"
    train_manifest = "data/full_splits/propaganda_train.json"
    val_manifest = "data/full_splits/propaganda_val.json"

    with open(train_manifest, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_manifest, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    all_techs = sorted(list(set(x["technique"] for x in train_data) | set(x["technique"] for x in val_data)))
    label2id = {tech: i for i, tech in enumerate(all_techs)}
    id2label = {str(i): tech for i, tech in enumerate(all_techs)}
    train_class_counts = dict(Counter(x["technique"] for x in train_data))
    val_class_counts = dict(Counter(x["technique"] for x in val_data))

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(checkpoint_dir, "tokenizer"))
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)

    val_texts = [x["text_span"] for x in val_data]
    val_labels = [label2id[x["technique"]] for x in val_data]
    val_dataset = PropagandaDataset(val_texts, val_labels, tokenizer, max_length=128)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    logger.info(f"Evaluating best saved checkpoint on {len(val_data)} validation spans on {device}...")
    metrics = evaluate_propaganda(model, val_loader, device)
    logger.info(f"Final Validation Metrics: Acc={metrics['accuracy']:.4f}, Macro F1={metrics['macro']['f1']:.4f}")

    training_config = {
        "model": "roberta-base",
        "num_classes": len(all_techs),
        "label2id": label2id,
        "id2label": id2label,
        "epochs": 3,
        "best_epoch": 3,
        "batch_size": 8,
        "learning_rate": 3e-5,
        "max_length": 128,
        "random_seed": 42,
        "device": str(device),
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "train_samples_count": len(train_data),
        "val_samples_count": len(val_data),
        "class_distribution_train": train_class_counts,
        "class_distribution_val": val_class_counts,
        "checkpoint_path": checkpoint_dir
    }

    with open(os.path.join(checkpoint_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)
    with open(os.path.join(checkpoint_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(checkpoint_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2)

    logger.info("Propaganda candidate metadata and metrics saved successfully.")

if __name__ == "__main__":
    finalize()
