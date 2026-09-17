"""Reusable Monte Carlo Dropout uncertainty estimation interface."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Callable, Union
from PIL import Image
from torchvision import transforms

from uncertainty.variance import summarize_uncertainty


def enable_mc_dropout(model: nn.Module) -> int:
    """Enable only Dropout modules in the model while keeping BatchNorm/LayerNorm in eval mode.

    Returns:
        Count of dropout layers activated.
    """
    model.eval()
    dropout_count = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()
            dropout_count += 1
    return dropout_count


class MCDropoutEstimator:
    """Generic Monte Carlo Dropout estimator for PyTorch models."""

    def __init__(self, mc_samples: int = 20, device: str = "cpu"):
        self.mc_samples = mc_samples
        self.device = device

    def estimate_uncertainty(self, model: nn.Module,
                             forward_fn: Callable[[], torch.Tensor],
                             label_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Perform MC Dropout stochastic sampling by repeatedly invoking forward_fn.

        Args:
            model: PyTorch model
            forward_fn: Zero-argument callable that performs one forward pass and returns raw logits
            label_mapping: Dict mapping string class indices to label names

        Returns:
            Dictionary with prediction, mean probability, variance, std, entropy, and distributions.
        """
        enable_mc_dropout(model)

        probs_list = []
        for _ in range(self.mc_samples):
            with torch.no_grad():
                logits = forward_fn()
                probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
                probs_list.append(probs)

        probs_arr = np.array(probs_list) # Shape: (T, C)
        uncertainty = summarize_uncertainty(probs_arr)

        mean_p = np.array(uncertainty["mean_probability"])
        top_idx = int(np.argmax(mean_p))
        confidence = float(mean_p[top_idx])

        top_label = str(top_idx)
        class_probs = {str(i): round(float(p), 4) for i, p in enumerate(mean_p)}
        class_var = {str(i): uncertainty["variance"][i] for i in range(len(mean_p))}

        if label_mapping is not None:
            top_label = label_mapping.get(str(top_idx), str(top_idx))
            class_probs = {label_mapping.get(str(i), f"class_{i}"): round(float(p), 4) for i, p in enumerate(mean_p)}
            class_var = {label_mapping.get(str(i), f"class_{i}"): uncertainty["variance"][i] for i in range(len(mean_p))}

        # Verification of stochasticity
        is_stochastic = uncertainty["max_variance"] > 1e-8

        return {
            "prediction": top_label,
            "class_index": top_idx,
            "confidence": round(confidence, 4),
            "probabilities": class_probs,
            "class_variances": class_var,
            "mean_probability": uncertainty["mean_probability"],
            "variance": uncertainty["variance"],
            "std": uncertainty["std"],
            "entropy": uncertainty["entropy"],
            "max_variance": uncertainty["max_variance"],
            "mean_variance": uncertainty["mean_variance"],
            "mc_samples": self.mc_samples,
            "is_stochastic": is_stochastic,
            "disclaimer": "MC Dropout uncertainty is a model uncertainty signal and has not been formally calibrated for deployment."
        }

    def estimate_deepfake_crop(self, model: nn.Module, image_crop: Union[Image.Image, np.ndarray],
                               label_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Run MC Dropout uncertainty on a deepfake face crop."""
        if isinstance(image_crop, np.ndarray):
            pil_crop = Image.fromarray(image_crop).convert("RGB")
        else:
            pil_crop = image_crop.convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(pil_crop).unsqueeze(0).to(self.device)

        if label_mapping is None:
            label_mapping = {"0": "real", "1": "fake"}

        return self.estimate_uncertainty(
            model=model,
            forward_fn=lambda: model(tensor),
            label_mapping=label_mapping
        )

    def estimate_text(self, model: nn.Module, tokenizer: Any, text: str,
                      label_mapping: Optional[Dict[str, str]] = None,
                      max_length: int = 128) -> Dict[str, Any]:
        """Run MC Dropout uncertainty on NLP model input text."""
        inputs = tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(self.device)

        return self.estimate_uncertainty(
            model=model,
            forward_fn=lambda: model(**inputs).logits,
            label_mapping=label_mapping
        )
