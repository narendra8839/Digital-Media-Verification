"""Training script for RoBERTa 14-class propaganda classifier on SemEval MVP subset."""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from evaluation.metrics import compute_classification_metrics


class PropagandaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def train_propaganda_model(epochs: int = 2, batch_size: int = 16, lr: float = 3e-5, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Propaganda] Initializing RoBERTa training on device: {device}")

    # Load MVP subsets
    with open("data/mvp/semeval2020_task11/train_subset.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open("data/mvp/semeval2020_task11/val_subset.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)

    # Establish exact 14 technique label mapping
    all_techs = sorted(list(set(x["technique"] for x in train_data)))
    label2id = {tech: i for i, tech in enumerate(all_techs)}
    id2label = {str(i): tech for i, tech in enumerate(all_techs)}

    print(f"[Propaganda] Loaded {len(train_data)} train samples and {len(val_data)} val samples across {len(all_techs)} classes.")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=len(all_techs),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    train_texts = [x["text_span"] for x in train_data]
    train_labels = [label2id[x["technique"]] for x in train_data]

    val_texts = [x["text_span"] for x in val_data]
    val_labels = [label2id[x["technique"]] for x in val_data]

    train_dataset = PropagandaDataset(train_texts, train_labels, tokenizer)
    val_dataset = PropagandaDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    checkpoint_dir = "models/propaganda/checkpoint"
    tokenizer_dir = "models/propaganda/tokenizer"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    best_val_f1 = -1.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(input_ids)

        epoch_loss = running_loss / len(train_dataset)
        val_metrics = evaluate_propaganda(model, val_loader, device)

        print(f"[Propaganda] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val Macro F1: {val_metrics['macro']['f1']:.4f}")

        if val_metrics["macro"]["f1"] >= best_val_f1:
            best_val_f1 = val_metrics["macro"]["f1"]
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(tokenizer_dir)
            print(f"[Propaganda] Saved best model to {checkpoint_dir}")

    # Load best model for final evaluation
    best_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
    final_val_metrics = evaluate_propaganda(best_model, val_loader, device)

    training_config = {
        "model": "roberta-base",
        "num_classes": len(all_techs),
        "classes": all_techs,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "random_seed": seed,
        "device": str(device),
        "dataset_provenance": "SemEval-2020 Task 11 (Propaganda Techniques Corpus) article-level split"
    }

    metrics = {
        "validation": final_val_metrics
    }

    with open("models/propaganda/training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    with open("models/propaganda/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("models/propaganda/label_mapping.json", "w") as f:
        json.dump(id2label, f, indent=2)

    print("[Propaganda] Training complete! Metrics and configuration saved.")
    return metrics


def evaluate_propaganda(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    return compute_classification_metrics(y_true, y_pred)


if __name__ == "__main__":
    train_propaganda_model()
