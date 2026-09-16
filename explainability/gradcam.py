"""Grad-CAM heatmap generation and overlay for EfficientNet-B4 deepfake classifier."""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple, Union
from torchvision import transforms


class GradCAM:
    """Computes Grad-CAM saliency maps for convolutional neural networks."""

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        if target_layer is None:
            # For EfficientNetB4Deepfake, features[-1] is the final Conv2dNormActivation block
            if hasattr(model, "backbone") and hasattr(model.backbone, "features"):
                self.target_layer = model.backbone.features[-1]
            else:
                raise ValueError("Target layer could not be automatically determined. Please specify target_layer.")
        else:
            self.target_layer = target_layer

        self.activations = None
        self.gradients = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks.extend([h1, h2])

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, int, float, list]:
        """Generate normalized 2D Grad-CAM heatmap for the given input tensor.

        Args:
            input_tensor: Tensor of shape (1, 3, H, W)
            target_class: Target class index. If None, uses predicted class.

        Returns:
            (heatmap_2d, pred_idx, confidence, probs)
        """
        input_tensor = input_tensor.to(self.device)
        self.model.zero_grad()

        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1).squeeze(0).detach().cpu().numpy().tolist()
        pred_idx = int(torch.argmax(output, dim=1).item())

        if target_class is None:
            target_class = pred_idx

        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured during Grad-CAM backward pass.")

        # Channel weights via Global Average Pooling
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().detach().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            heatmap = (cam - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam)

        confidence = probs[pred_idx]
        return heatmap, pred_idx, confidence, probs

    def overlay_heatmap(self, image: Union[Image.Image, np.ndarray], heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Resize heatmap to image size, apply JET colormap, and blend with RGB image.

        Args:
            image: PIL.Image or uint8 numpy RGB array
            heatmap: 2D float numpy array with values in [0, 1]
            alpha: Heatmap blend weight (0.0 to 1.0)

        Returns:
            RGB numpy uint8 array of the overlay
        """
        if isinstance(image, Image.Image):
            img_rgb = np.array(image.convert("RGB"))
        else:
            img_rgb = image.copy()
            if img_rgb.dtype != np.uint8:
                img_rgb = (img_rgb * 255).astype(np.uint8)

        h, w = img_rgb.shape[:2]
        # Resize heatmap to match image dimensions
        resized_heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * resized_heatmap)

        # Apply colormap (cv2 outputs BGR)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = cv2.addWeighted(img_rgb, 1.0 - alpha, color_heatmap_rgb, alpha, 0)
        return overlay

    def explain(self, image_crop: Union[Image.Image, np.ndarray], target_class: Optional[int] = None,
                output_path: Optional[str] = None, label_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Full Grad-CAM explanation on a face crop.

        Args:
            image_crop: PIL Image or numpy array
            target_class: Target class for explanation (default: predicted class)
            output_path: Optional file path to save the overlay image
            label_mapping: Dict mapping string class indices to label names

        Returns:
            Explanation dictionary with prediction, confidence, heatmap, and overlay metadata.
        """
        if label_mapping is None:
            label_mapping = {"0": "real", "1": "fake"}

        if isinstance(image_crop, np.ndarray):
            pil_crop = Image.fromarray(image_crop).convert("RGB")
        else:
            pil_crop = image_crop.convert("RGB")

        # Standard preprocessing for EfficientNet-B4
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(pil_crop).unsqueeze(0).to(self.device)
        heatmap, pred_idx, confidence, probs = self.generate_heatmap(input_tensor, target_class=target_class)

        # Generate overlay
        img_224 = pil_crop.resize((224, 224), Image.BILINEAR)
        overlay_rgb = self.overlay_heatmap(img_224, heatmap, alpha=0.5)

        saved_path = None
        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            Image.fromarray(overlay_rgb).save(output_path)
            saved_path = output_path

        pred_label = label_mapping.get(str(pred_idx), f"class_{pred_idx}")
        prob_dict = {label_mapping.get(str(i), f"class_{i}"): round(p, 4) for i, p in enumerate(probs)}

        return {
            "prediction": pred_label,
            "confidence": round(float(confidence), 4),
            "class_index": pred_idx,
            "probabilities": prob_dict,
            "heatmap": heatmap.tolist(),
            "heatmap_shape": list(heatmap.shape),
            "overlay_path": saved_path,
            "overlay_array": overlay_rgb,
            "disclaimer": "Grad-CAM is a post-hoc feature attribution method and should not be interpreted as causal proof."
        }
