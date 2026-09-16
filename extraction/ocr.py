"""Optical Character Recognition (OCR) interface using EasyOCR."""

import os
from typing import Dict, Any, List, Union, Optional, Tuple
import numpy as np
from PIL import Image
import easyocr


class EasyOCRReader:
    """Singleton/shared EasyOCR reader instance."""

    _instance = None
    _reader = None

    def __new__(cls, languages: Optional[List[str]] = None, gpu: bool = False):
        if cls._instance is None:
            cls._instance = super(EasyOCRReader, cls).__new__(cls)
            cls._instance.languages = languages or ["en"]
            cls._instance.gpu = gpu
            cls._instance._reader = None
        return cls._instance

    def _get_reader(self):
        if self._reader is None:
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def extract_text(self, image: Union[Image.Image, np.ndarray, str]) -> Dict[str, Any]:
        """Extract text from an image.

        Args:
            image: PIL Image, RGB numpy array, or file path.

        Returns:
            Dictionary with extracted text, mean confidence, detections, and status.
        """
        if isinstance(image, str):
            if not os.path.exists(image):
                return {"text": "", "source": "OCR", "confidence": 0.0, "status": "FILE_NOT_FOUND", "detections": []}
            img_np = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            img_np = image.copy()
        else:
            return {"text": "", "source": "OCR", "confidence": 0.0, "status": "INVALID_INPUT", "detections": []}

        try:
            reader = self._get_reader()
            raw_results = reader.readtext(img_np)

            if not raw_results:
                return {
                    "text": "",
                    "source": "OCR",
                    "confidence": 0.0,
                    "status": "NO_TEXT",
                    "detections": []
                }

            detections = []
            texts = []
            confs = []

            for bbox, text, conf in raw_results:
                clean_t = str(text).strip()
                if clean_t:
                    texts.append(clean_t)
                    confs.append(float(conf))
                    detections.append({
                        "text": clean_t,
                        "confidence": round(float(conf), 4),
                        "bbox": [[int(pt[0]), int(pt[1])] for pt in bbox]
                    })

            combined = " ".join(texts)
            mean_conf = float(np.mean(confs)) if confs else 0.0

            return {
                "text": combined,
                "source": "OCR",
                "confidence": round(mean_conf, 4),
                "status": "SUCCESS" if combined else "NO_TEXT",
                "detections": detections
            }
        except Exception as e:
            return {
                "text": "",
                "source": "OCR",
                "confidence": 0.0,
                "status": "ERROR",
                "error": str(e),
                "detections": []
            }

    def extract_from_frames(self, sampled_frames: List[Tuple[float, np.ndarray]], max_frames: int = 4) -> List[Dict[str, Any]]:
        """Extract text from representative sampled video frames and deduplicate.

        Args:
            sampled_frames: List of (timestamp, rgb_frame) tuples.
            max_frames: Max number of evenly spaced frames to inspect.

        Returns:
            List of frame OCR result dictionaries with frame_index and timestamp.
        """
        if not sampled_frames:
            return []

        n = len(sampled_frames)
        step = max(1, n // max_frames)
        selected_indices = list(range(0, n, step))[:max_frames]

        results = []
        seen_texts = set()

        for idx in selected_indices:
            ts, frame_rgb = sampled_frames[idx]
            ocr_res = self.extract_text(frame_rgb)
            txt = ocr_res.get("text", "").strip()

            if txt and txt.lower() not in seen_texts:
                seen_texts.add(txt.lower())
                ocr_res["frame_index"] = idx
                ocr_res["timestamp"] = round(float(ts), 2)
                results.append(ocr_res)

        return results


_default_ocr: Optional[EasyOCRReader] = None


def extract_ocr_text(image: Union[Image.Image, np.ndarray, str], gpu: bool = False) -> Dict[str, Any]:
    """Convenience function to extract OCR text using shared EasyOCR reader."""
    global _default_ocr
    if _default_ocr is None:
        _default_ocr = EasyOCRReader(gpu=gpu)
    return _default_ocr.extract_text(image)
