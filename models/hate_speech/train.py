"""Training script for BERT 3-class hate speech classifier on HateXplain MVP subset."""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from evaluation.metrics import compute_classification_metrics


class HateXplainDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def train_hate_speech_model(epochs: int = 2, batch_size: int = 16, lr: float = 3e-5, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HateSpeech] Initializing BERT training on device: {device}")

    # Load MVP subsets
    with open("data/mvp/hatexplain/train_subset.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open("data/mvp/hatexplain/val_subset.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open("data/mvp/hatexplain/test_subset.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Strictly separated 3 classes
    label2id = {"normal": 0, "offensive": 1, "hatespeech": 2}
    id2label = {"0": "normal", "1": "offensive", "2": "hatespeech"}

    print(f"[HateSpeech] Loaded Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}.")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    train_texts = [x["text"] for x in train_data]
    train_labels = [label2id[x["label"]] for x in train_data]

    val_texts = [x["text"] for x in val_data]
    val_labels = [label2id[x["label"]] for x in val_data]

    test_texts = [x["text"] for x in test_data]
    test_labels = [label2id[x["label"]] for x in test_data]

    train_loader = DataLoader(HateXplainDataset(train_texts, train_labels, tokenizer), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(HateXplainDataset(val_texts, val_labels, tokenizer), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(HateXplainDataset(test_texts, test_labels, tokenizer), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    checkpoint_dir = "models/hate_speech/checkpoint"
    tokenizer_dir = "models/hate_speech/tokenizer"
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

        epoch_loss = running_loss / len(train_texts)
        val_metrics = evaluate_hate_speech(model, val_loader, device)

        print(f"[HateSpeech] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val Macro F1: {val_metrics['macro']['f1']:.4f}")

        if val_metrics["macro"]["f1"] >= best_val_f1:
            best_val_f1 = val_metrics["macro"]["f1"]
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(tokenizer_dir)
            print(f"[HateSpeech] Saved best model to {checkpoint_dir}")

    # Final evaluation using best model
    best_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
    final_val_metrics = evaluate_hate_speech(best_model, val_loader, device)
    final_test_metrics = evaluate_hate_speech(best_model, test_loader, device)

    training_config = {
        "model": "bert-base-uncased",
        "num_classes": 3,
        "classes": ["normal", "offensive", "hatespeech"],
        "label_mapping": id2label,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "random_seed": seed,
        "device": str(device),
        "dataset_provenance": "HateXplain official benchmark splits (Mathew et al., AAAI 2021)"
    }

    metrics = {
        "validation": final_val_metrics,
        "test": final_test_metrics
    }

    with open("models/hate_speech/training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    with open("models/hate_speech/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("models/hate_speech/label_mapping.json", "w") as f:
        json.dump(id2label, f, indent=2)

    print("[HateSpeech] Training complete! Metrics and configuration saved.")
    return metrics


def evaluate_hate_speech(model, data_loader, device):
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
    train_hate_speech_model()
