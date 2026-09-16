"""Inference and checkpoint loading for EfficientNet-B4 deepfake classifier."""

import os
import json
import torch
from torchvision import transforms
from PIL import Image
from typing import Dict, Any, Union
import numpy as np

from models.deepfake.efficientnet_b4 import EfficientNetB4Deepfake


def load_deepfake_model(checkpoint_path: str = "models/deepfake/checkpoint/best_model.pt", device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EfficientNetB4Deepfake(dropout_rate=0.4, pretrained=False)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def predict_deepfake_crop(model, image_crop: Union[Image.Image, np.ndarray], device: str) -> Dict[str, Any]:
    """Run inference on a single 224x224 face crop."""
    if isinstance(image_crop, np.ndarray):
        image_crop = Image.fromarray(image_crop)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(image_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    fake_prob = float(probs[1])
    real_prob = float(probs[0])
    pred_label = "fake" if fake_prob >= 0.5 else "real"

    return {
        "prediction": pred_label,
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4),
        "raw_logits": [round(float(x), 4) for x in logits.squeeze(0).cpu().numpy()]
    }


def predict_deepfake_video(model, video_path: str, device: str, max_frames: int = 4) -> Dict[str, Any]:
    """Sample frames from video, detect face crops, predict per frame, and aggregate at video level."""
    from preprocessing.frame_sampling import sample_video_frames
    from preprocessing.face_detection import FaceDetector

    detector = FaceDetector(device=device)
    sampled = sample_video_frames(video_path, target_fps=1.0, max_frames=max_frames)

    if not sampled:
        return {
            "prediction": "unknown",
            "confidence": 0.0,
            "probabilities": {"real": 0.5, "fake": 0.5},
            "frames_analyzed": 0,
            "error": "No frames could be sampled from video"
        }

    frame_results = []
    fake_probs = []

    for ts, frame_rgb in sampled:
        crop, is_detected = detector.detect_and_crop(frame_rgb, target_size=(224, 224))
        res = predict_deepfake_crop(model, crop, device)
        res["timestamp"] = ts
        res["face_detected"] = is_detected
        frame_results.append(res)
        fake_probs.append(res["fake_probability"])

    mean_fake_prob = float(np.mean(fake_probs))
    mean_real_prob = 1.0 - mean_fake_prob
    pred_label = "fake" if mean_fake_prob >= 0.5 else "real"
    confidence = mean_fake_prob if pred_label == "fake" else mean_real_prob

    return {
        "prediction": pred_label,
        "confidence": round(confidence, 4),
        "probabilities": {
            "real": round(mean_real_prob, 4),
            "fake": round(mean_fake_prob, 4)
        },
        "frames_analyzed": len(frame_results),
        "frame_details": frame_results
    }

