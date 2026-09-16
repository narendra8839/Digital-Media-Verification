"""MTCNN-based face detection and cropping interface."""

import numpy as np
from PIL import Image
from typing import Optional, Tuple
from facenet_pytorch import MTCNN


class FaceDetector:
    def __init__(self, device: str = "cpu", margin: float = 0.2):
        self.device = device
        self.margin = margin
        self.detector = MTCNN(keep_all=False, select_largest=True, device=device)

    def detect_and_crop(self, image_rgb: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> Tuple[Optional[Image.Image], bool]:
        """Detect face in RGB numpy array, crop and resize.

        Returns (PIL.Image of face or center crop, face_detected_bool).
        """
        pil_img = Image.fromarray(image_rgb)
        w, h = pil_img.size

        try:
            boxes, probs = self.detector.detect(pil_img)
            if boxes is not None and len(boxes) > 0 and probs[0] is not None and probs[0] > 0.8:
                box = boxes[0]
                bx1, by1, bx2, by2 = box
                bw, bh = bx2 - bx1, by2 - by1
                x1 = max(0, int(bx1 - self.margin * bw))
                y1 = max(0, int(by1 - self.margin * bh))
                x2 = min(w, int(bx2 + self.margin * bw))
                y2 = min(h, int(by2 + self.margin * bh))
                crop = pil_img.crop((x1, y1, x2, y2)).resize(target_size, Image.BILINEAR)
                return crop, True
        except Exception:
            pass

        # Fallback: Center crop
        min_dim = min(w, h)
        cx1 = (w - min_dim) // 2
        cy1 = (h - min_dim) // 2
        crop = pil_img.crop((cx1, cy1, cx1 + min_dim, cy1 + min_dim)).resize(target_size, Image.BILINEAR)
        return crop, False
