import os
import sys
import json
import logging
import time
from typing import Dict, Any, List
from collections import Counter
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("optimize_propaganda")

class PropagandaDataset(Dataset):
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

def compute_metrics(y_true, y_pred, id2label):
    acc = accuracy_score(y_true, y_pred)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    num_classes = len(id2label)
    c_prec, c_rec, c_f1, c_supp = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    
    per_class = {}
    for i in range(num_classes):
        label_name = id2label[str(i)]
        per_class[label_name] = {
            "precision": float(c_prec[i]),
            "recall": float(c_rec[i]),
            "f1": float(c_f1[i]),
            "support": int(c_supp[i])
        }
        
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
    
    return {
        "accuracy": float(acc),
        "macro": {
            "precision": float(macro_prec),
            "recall": float(macro_rec),
            "f1": float(macro_f1)
        },
        "weighted": {
            "precision": float(weighted_prec),
            "recall": float(weighted_rec),
            "f1": float(weighted_f1)
        },
        "per_class": per_class,
        "confusion_matrix": cm,
        "sample_count": len(y_true)
    }

def evaluate_model(model, data_loader, device, id2label):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.numpy().tolist())
            
    return compute_metrics(all_targets, all_preds, id2label)

def run_experiment(
    exp_name: str,
    output_dir: str,
    epochs: int = 4,
    lr: float = 3e-5,
    weight_type: str = "sqrt_inv",  # 'none', 'sqrt_inv', 'inv_freq'
    max_weight_clip: float = 4.0,
    use_scheduler: bool = True,
    warmup_ratio: float = 0.1,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required"
    
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)

    train_manifest = "data/full_splits/propaganda_train.json"
    val_manifest = "data/full_splits/propaganda_val.json"

    with open(train_manifest, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_manifest, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    all_techs = sorted(list(set(x["technique"] for x in train_data) | set(x["technique"] for x in val_data)))
    label2id = {tech: i for i, tech in enumerate(all_techs)}
    id2label = {str(i): tech for i, tech in enumerate(all_techs)}
    num_classes = len(all_techs)

    train_class_counts = dict(Counter(x["technique"] for x in train_data))
    val_class_counts = dict(Counter(x["technique"] for x in val_data))

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    train_texts = [x["text_span"] for x in train_data]
    train_labels = [label2id[x["technique"]] for x in train_data]
    val_texts = [x["text_span"] for x in val_data]
    val_labels = [label2id[x["technique"]] for x in val_data]

    train_dataset = PropagandaDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = PropagandaDataset(val_texts, val_labels, tokenizer, max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Class weighting calculation
    if weight_type == "none":
        criterion = nn.CrossEntropyLoss()
        class_weights_dict = {tech: 1.0 for tech in all_techs}
    elif weight_type == "sqrt_inv":
        # w_c = (total / count)^0.5, normalized to mean = 1.0
        counts = np.array([train_class_counts[id2label[str(i)]] for i in range(num_classes)], dtype=np.float32)
        raw_weights = np.sqrt(len(train_data) / counts)
        weights = raw_weights / np.mean(raw_weights)
        if max_weight_clip:
            weights = np.clip(weights, 0.2, max_weight_clip)
            weights = weights / np.mean(weights)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        class_weights_dict = {id2label[str(i)]: float(weights[i]) for i in range(num_classes)}
    elif weight_type == "inv_freq":
        # standard balanced inverse frequency
        counts = np.array([train_class_counts[id2label[str(i)]] for i in range(num_classes)], dtype=np.float32)
        raw_weights = len(train_data) / (num_classes * counts)
        if max_weight_clip:
            raw_weights = np.clip(raw_weights, 0.2, max_weight_clip)
        weights = raw_weights / np.mean(raw_weights)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        class_weights_dict = {id2label[str(i)]: float(weights[i]) for i in range(num_classes)}
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio) if use_scheduler else 0

    if use_scheduler:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    else:
        scheduler = None

    logger.info("=" * 60)
    logger.info(f"STARTING EXPERIMENT: {exp_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Epochs: {epochs} | LR: {lr} | Weight Type: {weight_type} | Scheduler: {use_scheduler}")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    logger.info("=" * 60)

    best_val_f1 = -1.0
    best_epoch = -1
    history = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item() * len(input_ids)

        epoch_loss = running_loss / len(train_dataset)
        val_metrics = evaluate_model(model, val_loader, device, id2label)
        macro_f1 = val_metrics["macro"]["f1"]
        acc = val_metrics["accuracy"]
        macro_rec = val_metrics["macro"]["recall"]
        weighted_f1 = val_metrics["weighted"]["f1"]

        current_lr = scheduler.get_last_lr()[0] if scheduler else lr
        logger.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | "
            f"Val Acc: {acc:.4f} | Val Macro F1: {macro_f1:.4f} | "
            f"Val Macro Rec: {macro_rec:.4f} | LR: {current_lr:.2e}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(epoch_loss),
            "val_accuracy": float(acc),
            "val_macro_f1": float(macro_f1),
            "val_macro_recall": float(macro_rec),
            "val_weighted_f1": float(weighted_f1)
        })

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_epoch = epoch + 1
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(tokenizer_dir)
            logger.info(f"--> Saved new BEST checkpoint (Macro F1: {best_val_f1:.4f}) at epoch {best_epoch}")

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s. Best epoch: {best_epoch} with Val Macro F1: {best_val_f1:.4f}")

    # Reload best model and evaluate
    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    final_metrics = evaluate_model(best_model, val_loader, device, id2label)

    config_record = {
        "experiment_name": exp_name,
        "model": "roberta-base",
        "num_classes": num_classes,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "learning_rate": lr,
        "weight_type": weight_type,
        "class_weights": class_weights_dict,
        "scheduler_enabled": use_scheduler,
        "warmup_steps": warmup_steps,
        "random_seed": seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "train_samples_count": len(train_data),
        "val_samples_count": len(val_data),
        "training_time_seconds": round(elapsed, 2),
        "checkpoint_path": output_dir
    }

    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    with open(os.path.join(output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2)
    with open(os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "exp_name": exp_name,
        "best_epoch": best_epoch,
        "best_val_f1": float(final_metrics["macro"]["f1"]),
        "accuracy": float(final_metrics["accuracy"]),
        "macro_precision": float(final_metrics["macro"]["precision"]),
        "macro_recall": float(final_metrics["macro"]["recall"]),
        "weighted_f1": float(final_metrics["weighted"]["f1"]),
        "elapsed_seconds": round(elapsed, 2)
    }

def main():
    exp_id = sys.argv[1] if len(sys.argv) > 1 else "exp1"
    
    if exp_id == "exp1":
        # Candidate A: Sqrt-inverse class weighting + Linear warmup scheduler (4 epochs, lr 3e-5)
        res = run_experiment(
            exp_name="exp1_sqrt_weighted_scheduled",
            output_dir="models/propaganda/optimization/exp1_sqrt_weighted_scheduled",
            epochs=4,
            lr=3e-5,
            weight_type="sqrt_inv",
            max_weight_clip=3.5,
            use_scheduler=True,
            warmup_ratio=0.1
        )
    elif exp_id == "exp2":
        # Candidate B: Inverse-frequency class weighting (clipped) + Linear warmup scheduler (4 epochs, lr 2.5e-5)
        res = run_experiment(
            exp_name="exp2_inv_freq_weighted",
            output_dir="models/propaganda/optimization/exp2_inv_freq_weighted",
            epochs=4,
            lr=2.5e-5,
            weight_type="inv_freq",
            max_weight_clip=4.0,
            use_scheduler=True,
            warmup_ratio=0.1
        )
    elif exp_id == "exp3":
        # Candidate C: Unweighted with Linear warmup scheduler + LR 2e-5 (4 epochs) to isolate scheduler effect
        res = run_experiment(
            exp_name="exp3_scheduler_lr2e5",
            output_dir="models/propaganda/optimization/exp3_scheduler_lr2e5",
            epochs=4,
            lr=2e-5,
            weight_type="none",
            use_scheduler=True,
            warmup_ratio=0.1
        )
    else:
        print(f"Unknown experiment: {exp_id}")
        return

    print("\n" + "=" * 50)
    print("EXPERIMENT RESULT SUMMARY:")
    print(json.dumps(res, indent=2))
    print("=" * 50)

if __name__ == "__main__":
    main()
