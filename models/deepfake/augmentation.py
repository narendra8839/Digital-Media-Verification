"""Robustness augmentation pipeline for deepfake training.

Applies domain-specific transformations to training data only:
1. JPEG compression artifacts (simulating social media compression)
2. Mild Gaussian blur (simulating transmission/rescaling softening)
3. Mild additive Gaussian noise (simulating sensor noise)
4. Mild color jitter and random horizontal flips
5. Optional resolution downscale/upscale (simulating low-res social media video)

Validation and test transformations remain strictly deterministic.
"""

import io
import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageFilter
from typing import Optional, Dict, Any, Tuple


class JPEGCompression:
    """Applies in-memory JPEG compression to simulate social media compression."""

    def __init__(self, quality_range: Tuple[int, int] = (50, 95), p: float = 0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomDownsampleResize:
    """Simulates social media transmission resolution reduction and restoration."""

    def __init__(self, scale_range: Tuple[float, float] = (0.35, 0.70), p: float = 0.35):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        down_w = max(16, int(w * scale))
        down_h = max(16, int(h * scale))
        down = img.resize((down_w, down_h), Image.BILINEAR)
        return down.resize((w, h), Image.BILINEAR)


class AddGaussianNoise:
    """Adds mild zero-mean Gaussian noise to a float tensor in [0, 1]."""

    def __init__(self, mean: float = 0.0, std_range: Tuple[float, float] = (0.01, 0.04), p: float = 0.3):
        self.mean = mean
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        std = random.uniform(self.std_range[0], self.std_range[1])
        noise = torch.randn_like(tensor) * std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


def get_deepfake_transforms(
    is_train: bool = True,
    target_size: Tuple[int, int] = (224, 224),
    config: Optional[Dict[str, Any]] = None
) -> transforms.Compose:
    """Construct data transformation pipeline.

    Args:
        is_train: If True, applies stochastic robustness augmentations.
                  If False, strictly applies deterministic resize and normalization.
        target_size: Target image dimensions (height, width).
        config: Optional configuration dictionary to tune augmentation probabilities.

    Returns:
        torchvision transforms.Compose pipeline.
    """
    cfg = config or {}
    mean = cfg.get("normalize_mean", [0.485, 0.456, 0.406])
    std = cfg.get("normalize_std", [0.229, 0.224, 0.225])

    if not is_train or not cfg.get("augmentation_enabled", True):
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    # Robustness training transforms
    jpeg_p = cfg.get("jpeg_p", 0.5)
    jpeg_min_q = cfg.get("jpeg_min_quality", 50)
    jpeg_max_q = cfg.get("jpeg_max_quality", 95)
    noise_p = cfg.get("noise_p", 0.3)
    noise_std_max = cfg.get("noise_std_max", 0.03)
    blur_p = cfg.get("blur_p", 0.3)
    blur_kernel_size = cfg.get("blur_kernel_size", 3)
    blur_sigma_range = cfg.get("blur_sigma_range", (0.1, 1.2))
    resize_p = cfg.get("resize_p", 0.0)
    flip_p = cfg.get("flip_p", 0.5)

    train_ops = [
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(p=flip_p),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        JPEGCompression(quality_range=(jpeg_min_q, jpeg_max_q), p=jpeg_p),
    ]

    if resize_p > 0:
        train_ops.append(RandomDownsampleResize(p=resize_p))

    if blur_p > 0:
        train_ops.append(
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma_range)], p=blur_p)
        )

    train_ops.extend([
        transforms.ToTensor(),
        AddGaussianNoise(std_range=(0.005, noise_std_max), p=noise_p),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transforms.Compose(train_ops)
