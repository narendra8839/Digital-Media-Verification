"""Deepfake detector module integrating EfficientNet-B4, Grad-CAM, and MC Dropout."""

import os
from typing import Dict, Any, Optional, Union
from PIL import Image
import numpy as np
import torch

from models.deepfake.predict import load_deepfake_model
from preprocessing.face_detection import FaceDetector
from explainability.gradcam import GradCAM
from uncertainty.uncertainty_engine import UncertaintyEngine


class DeepfakeDetector:
    """Orchestrates face detection, deepfake classification, Grad-CAM explainability, and uncertainty estimation."""

    def __init__(self, checkpoint_path: str = "models/deepfake/checkpoint/best_model.pt",
                 device: Optional[str] = None,
                 variance_threshold: float = 0.02,
                 entropy_threshold: float = 0.85):
        self.model, self.device = load_deepfake_model(checkpoint_path=checkpoint_path, device=device)
        self.face_detector = FaceDetector(device=self.device)
        self.gradcam = GradCAM(model=self.model, device=self.device)
        self.uncertainty_engine = UncertaintyEngine(
            mc_samples=20,
            variance_threshold=variance_threshold,
            entropy_threshold=entropy_threshold,
            device=self.device
        )
        self.label_mapping = {"0": "real", "1": "fake"}

    def detect_image(self, image: Union[Image.Image, np.ndarray, str],
                     output_dir: Optional[str] = None,
                     generate_gradcam: bool = True) -> Dict[str, Any]:
        """Process a single image for deepfake detection.

        Args:
            image: PIL Image, numpy array, or file path.
            output_dir: Optional directory to save Grad-CAM overlay image.
            generate_gradcam: Whether to generate Grad-CAM visualization.

        Returns:
            Dictionary containing detection prediction, confidence, explanation, uncertainty, and review status.
        """
        if isinstance(image, str):
            if not os.path.exists(image):
                return {"status": "FILE_NOT_FOUND", "error": f"Image file not found: {image}"}
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            return {"status": "INVALID_INPUT", "error": f"Unsupported image type: {type(image)}"}

        # 1. Face detection
        rgb_np = np.array(pil_img)
        crop, face_detected = self.face_detector.detect_and_crop(rgb_np, target_size=(224, 224))

        if not face_detected or crop is None:
            return {
                "status": "NO_FACE_DETECTED",
                "face_detected": False,
                "prediction": None,
                "confidence": 0.0,
                "note": "No human face was detected in the input image. Automated deepfake verification skipped.",
                "explanation": None,
                "uncertainty": None,
                "review_required": False,
                "decision_status": "NOT_APPLICABLE"
            }

        # 2. Uncertainty estimation & inference via MC Dropout (T=20)
        uq_result = self.uncertainty_engine.process_and_evaluate(
            model=self.model,
            input_data=crop,
            modality="deepfake",
            label_mapping=self.label_mapping
        )

        # 3. Grad-CAM explanation
        explanation = None
        if generate_gradcam:
            save_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, "gradcam_overlay.png")
            explanation = self.gradcam.explain(
                image_crop=crop,
                output_path=save_path,
                label_mapping=self.label_mapping
            )

        return {
            "status": "SUCCESS",
            "prediction": uq_result["prediction"],
            "confidence": uq_result["confidence"],
            "probabilities": uq_result["probabilities"],
            "face_detected": True,
            "explanation": explanation,
            "uncertainty": uq_result["uncertainty"],
            "review_required": uq_result["review_required"],
            "decision_status": uq_result["decision_status"],
            "review_trigger_component": uq_result.get("review_trigger_component"),
            "review_justification": uq_result["review_justification"]
        }
