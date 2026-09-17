"""Training script for EfficientNet-B4 binary deepfake classifier on MVP subset."""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

from models.deepfake.efficientnet_b4 import EfficientNetB4Deepfake
from preprocessing.frame_sampling import sample_video_frames
from preprocessing.face_detection import FaceDetector
from evaluation.metrics import compute_classification_metrics


class DeepfakeFrameDataset(Dataset):
    def __init__(self, frame_records: List[Dict[str, Any]], transform=None):
        self.records = frame_records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = rec["face_crop"]
        if self.transform:
            img = self.transform(img)
        label = rec["label_id"]
        return img, label, rec["video_id"]


def extract_frames_from_subset(subset_path: str, detector: FaceDetector, max_frames_per_video: int = 4) -> List[Dict[str, Any]]:
    with open(subset_path, "r") as f:
        videos = json.load(f)

    frame_records = []
    base_data = "data/faceforensics/C23"

    for vid in videos:
        rel_path = vid["relative_path"]
        abs_path = os.path.join(base_data, rel_path)
        if not os.path.exists(abs_path):
            continue

        sampled = sample_video_frames(abs_path, target_fps=1.0, max_frames=max_frames_per_video)
        for ts, frame_rgb in sampled:
            crop, is_detected = detector.detect_and_crop(frame_rgb, target_size=(224, 224))
            frame_records.append({
                "video_id": vid["video_id"],
                "label_id": vid["label_id"],
                "label_name": vid["label"],
                "timestamp": ts,
                "face_crop": crop,
                "face_detected": is_detected
            })
    return frame_records


def train_deepfake_model(epochs: int = 2, batch_size: int = 8, lr: float = 1e-4, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Deepfake] Initializing training on device: {device}")

    detector = FaceDetector(device="cpu")

    print("[Deepfake] Preprocessing frames from MVP subsets...")
    train_frames = extract_frames_from_subset("data/mvp/deepfake/train_subset.json", detector, max_frames_per_video=4)
    val_frames = extract_frames_from_subset("data/mvp/deepfake/val_subset.json", detector, max_frames_per_video=4)
    test_frames = extract_frames_from_subset("data/mvp/deepfake/test_subset.json", detector, max_frames_per_video=4)

    print(f"[Deepfake] Extracted frames: Train={len(train_frames)}, Val={len(val_frames)}, Test={len(test_frames)}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(DeepfakeFrameDataset(train_frames, transform=transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(DeepfakeFrameDataset(val_frames, transform=transform), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(DeepfakeFrameDataset(test_frames, transform=transform), batch_size=batch_size, shuffle=False)

    model = EfficientNetB4Deepfake(dropout_rate=0.4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    best_val_acc = -1.0
    best_weights_path = "models/deepfake/checkpoint/best_model.pt"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / max(1, len(train_frames))

        # Validation evaluation
        val_metrics, val_vid_metrics = evaluate(model, val_loader, device)
        print(f"[Deepfake] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} | Val Frame Acc: {val_metrics['accuracy']:.4f} | Val Video Acc: {val_vid_metrics['accuracy']:.4f}")

        if val_vid_metrics["accuracy"] >= best_val_acc:
            best_val_acc = val_vid_metrics["accuracy"]
            torch.save(model.state_dict(), best_weights_path)
            print(f"[Deepfake] Saved best checkpoint to {best_weights_path}")

    # Final evaluation using best model
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    val_frame_m, val_video_m = evaluate(model, val_loader, device)
    test_frame_m, test_video_m = evaluate(model, test_loader, device)

    label_mapping = {"0": "real", "1": "fake"}
    training_config = {
        "model": "efficientnet_b4",
        "num_classes": 2,
        "label_mapping": label_mapping,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "random_seed": seed,
        "device": str(device),
        "frame_resolution": [224, 224],
        "frames_per_video_train": 4,
        "dataset_provenance": "Kaggle development mirror (xdxd003/ff-c23). Strictly for MVP development; official TUM FF++ access pending."
    }

    metrics = {
        "validation": {
            "frame_level": val_frame_m,
            "video_level": val_video_m
        },
        "test": {
            "frame_level": test_frame_m,
            "video_level": test_video_m
        }
    }

    with open("models/deepfake/training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    with open("models/deepfake/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("models/deepfake/label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)

    print("[Deepfake] Training complete! Metrics and configuration saved.")
    return metrics


def evaluate(model, data_loader, device):
    model.eval()
    y_true_frames = []
    y_pred_frames = []
    video_preds = defaultdict(list)
    video_trues = {}

    with torch.no_grad():
        for imgs, labels, vids in data_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            labels_np = labels.numpy()
            y_true_frames.extend(labels_np.tolist())
            y_pred_frames.extend(preds.tolist())

            for vid, prob, true_label in zip(vids, probs, labels_np):
                video_preds[vid].append(prob)
                video_trues[vid] = true_label

    frame_metrics = compute_classification_metrics(y_true_frames, y_pred_frames)

    # Video-level aggregation: mean fake probability >= 0.5 -> fake
    y_true_videos = []
    y_pred_videos = []
    for vid, prob_list in video_preds.items():
        avg_fake_prob = float(np.mean(prob_list))
        pred_label = 1 if avg_fake_prob >= 0.5 else 0
        y_true_videos.append(video_trues[vid])
        y_pred_videos.append(pred_label)

    video_metrics = compute_classification_metrics(y_true_videos, y_pred_videos)
    return frame_metrics, video_metrics


if __name__ == "__main__":
    train_deepfake_model()
