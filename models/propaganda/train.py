"""Reproducible training pipeline for RoBERTa 14-class propaganda classifier.

Key features:
1. Operates on full audited SemEval-2020 Task 11 dataset (or configurable subsets for smoke tests).
2. Article-disjoint train/validation partition strictly preventing cross-article leakage.
3. Checkpoint selection strictly based on validation Macro F1.
4. Preserves existing MVP baseline checkpoints by writing new runs to an isolated directory.
5. Device-ready: auto-detects CUDA when available while supporting explicit CPU/CUDA configuration.
6. Saves complete training metadata, label mappings, and validation metrics alongside weights.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List
from collections import Counter

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from evaluation.metrics import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("propaganda_train")


class PropagandaDataset(Dataset):
    """Dataset serving tokenized text spans and technique labels."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def train_propaganda_model(
    train_manifest: str = "data/full_splits/propaganda_train.json",
    val_manifest: str = "data/full_splits/propaganda_val.json",
    output_dir: str = "models/propaganda/checkpoint_full",
    pretrained_model_name: str = "roberta-base",
    max_length: int = 128,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 3e-5,
    seed: int = 42,
    device: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    save_checkpoint: bool = True
) -> Dict[str, Any]:
    """Train RoBERTa 14-class propaganda technique classifier."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    selected_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected_device.type != "cuda":
        raise RuntimeError("Phase 3 full training requires CUDA (cuda:0). Silent fallback to CPU is strictly prohibited.")
    logger.info(f"Initializing propaganda training on device: {selected_device}")

    # 1. Load full audited data
    with open(train_manifest, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_manifest, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    if max_train_samples is not None:
        train_data = train_data[:max_train_samples]
    if max_val_samples is not None:
        val_data = val_data[:max_val_samples]

    # Establish canonical 14 technique label mapping
    all_techs = sorted(list(set(x["technique"] for x in train_data) | set(x["technique"] for x in val_data)))
    label2id = {tech: i for i, tech in enumerate(all_techs)}
    id2label = {str(i): tech for i, tech in enumerate(all_techs)}

    logger.info(f"Loaded {len(train_data)} train spans and {len(val_data)} val spans across {len(all_techs)} classes.")

    train_class_counts = dict(Counter(x["technique"] for x in train_data))
    val_class_counts = dict(Counter(x["technique"] for x in val_data))

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=len(all_techs),
        id2label=id2label,
        label2id=label2id
    ).to(selected_device)

    # Detailed Startup Verification Logging
    gpu_name = torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else "N/A"
    model_param_device = next(model.parameters()).device
    logger.info("=" * 60)
    logger.info(f"model: {pretrained_model_name}")
    logger.info(f"dataset: {train_manifest} (train) | {val_manifest} (val)")
    logger.info(f"device: {selected_device}")
    logger.info(f"GPU name: {gpu_name}")
    logger.info(f"batch size: {batch_size}")
    logger.info(f"learning rate: {lr}")
    logger.info(f"epochs: {epochs}")
    logger.info(f"train sample count: {len(train_data)}")
    logger.info(f"validation sample count: {len(val_data)}")
    logger.info(f"seed: {seed}")
    logger.info(f"model.parameters().device: {model_param_device}")
    logger.info("=" * 60)
    assert model_param_device.type == "cuda", f"Model parameters must be on CUDA, got {model_param_device}"

    train_texts = [x["text_span"] for x in train_data]
    train_labels = [label2id[x["technique"]] for x in train_data]

    val_texts = [x["text_span"] for x in val_data]
    val_labels = [label2id[x["technique"]] for x in val_data]

    train_dataset = PropagandaDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    val_dataset = PropagandaDataset(val_texts, val_labels, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    checkpoint_dir = output_dir
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    if save_checkpoint:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(tokenizer_dir, exist_ok=True)

    best_val_f1 = -1.0
    best_epoch = -1
    first_batch_verified = False

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(selected_device)
            attention_mask = batch["attention_mask"].to(selected_device)
            labels = batch["labels"].to(selected_device)

            if not first_batch_verified:
                logger.info(f"batch tensor device: {input_ids.device}")
                assert input_ids.device.type == "cuda", f"Batch tensors must be on CUDA, got {input_ids.device}"
                first_batch_verified = True

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(input_ids)

        epoch_loss = running_loss / max(1, len(train_dataset))
        val_metrics = evaluate_propaganda(model, val_loader, selected_device)

        macro_f1 = val_metrics["macro"]["f1"]
        logger.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val Macro F1: {macro_f1:.4f}"
        )

        if macro_f1 >= best_val_f1:
            best_val_f1 = macro_f1
            best_epoch = epoch + 1
            if save_checkpoint:
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(tokenizer_dir)
                logger.info(f"Saved best model (Macro F1: {best_val_f1:.4f}) to {checkpoint_dir}")

    # Load and evaluate best model on validation split (TEST SPLIT IS NEVER TOUCHED)
    if save_checkpoint and os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
        best_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(selected_device)
        final_val_metrics = evaluate_propaganda(best_model, val_loader, selected_device)
    else:
        final_val_metrics = evaluate_propaganda(model, val_loader, selected_device)

    training_config = {
        "model": pretrained_model_name,
        "num_classes": len(all_techs),
        "label2id": label2id,
        "id2label": id2label,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "learning_rate": lr,
        "max_length": max_length,
        "random_seed": seed,
        "device": str(selected_device),
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "train_samples_count": len(train_data),
        "val_samples_count": len(val_data),
        "class_distribution_train": train_class_counts,
        "class_distribution_val": val_class_counts,
        "checkpoint_path": checkpoint_dir if save_checkpoint else None
    }

    if save_checkpoint:
        with open(os.path.join(checkpoint_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(training_config, f, indent=2)
        with open(os.path.join(checkpoint_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(final_val_metrics, f, indent=2)
        with open(os.path.join(checkpoint_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(id2label, f, indent=2)

    logger.info("Propaganda training process complete. Results and configuration recorded.")
    return {
        "config": training_config,
        "metrics": final_val_metrics,
        "best_val_f1": best_val_f1
    }


def evaluate_propaganda(model, data_loader, device) -> Dict[str, Any]:
    """Evaluate propaganda classification metrics on validation split."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return compute_classification_metrics(all_labels, all_preds)


if __name__ == "__main__":
    train_propaganda_model()
