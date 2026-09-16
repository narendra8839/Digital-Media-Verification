"""Media ingestion and input routing module."""

import os
from typing import Dict, Any, Union
from PIL import Image
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


class MediaRouter:
    """Detects media type and inspects input streams safely without heavy memory loading."""

    @staticmethod
    def inspect_input(media_input: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Determine input media type and metadata.

        Args:
            media_input: File path, text string, PIL Image, or numpy array.

        Returns:
            Structured metadata dictionary with input_type, filename, size_bytes, and status.
        """
        # Case 1: In-memory image object
        if isinstance(media_input, Image.Image):
            return {
                "input_type": "image",
                "filename": "<in-memory PIL.Image>",
                "size_bytes": media_input.width * media_input.height * 3,
                "dimensions": [media_input.width, media_input.height],
                "status": "VALID"
            }
        elif isinstance(media_input, np.ndarray):
            return {
                "input_type": "image",
                "filename": "<in-memory numpy array>",
                "size_bytes": int(media_input.nbytes),
                "dimensions": list(media_input.shape[:2][::-1]),
                "status": "VALID"
            }

        # Case 2: String input (either file path or raw text)
        if isinstance(media_input, str):
            # Check if it points to an existing file on disk
            if os.path.isfile(media_input):
                abs_path = os.path.abspath(media_input)
                ext = os.path.splitext(media_input)[1].lower()
                size_bytes = os.path.getsize(abs_path)
                filename = os.path.basename(abs_path)

                if ext in IMAGE_EXTENSIONS:
                    return {
                        "input_type": "image",
                        "filename": filename,
                        "file_path": abs_path,
                        "size_bytes": size_bytes,
                        "status": "VALID"
                    }
                elif ext in VIDEO_EXTENSIONS:
                    return {
                        "input_type": "video",
                        "filename": filename,
                        "file_path": abs_path,
                        "size_bytes": size_bytes,
                        "status": "VALID"
                    }
                else:
                    return {
                        "input_type": "unknown",
                        "filename": filename,
                        "file_path": abs_path,
                        "size_bytes": size_bytes,
                        "status": f"UNSUPPORTED_EXTENSION: {ext}"
                    }
            else:
                # Raw text string
                text_bytes = len(media_input.encode("utf-8"))
                return {
                    "input_type": "text",
                    "filename": "<raw text>",
                    "text_content": media_input,
                    "size_bytes": text_bytes,
                    "status": "VALID"
                }

        return {
            "input_type": "unknown",
            "filename": "<unknown>",
            "size_bytes": 0,
            "status": f"INVALID_TYPE: {type(media_input).__name__}"
        }


def detect_media_type(media_input: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
    """Convenience function for MediaRouter.inspect_input."""
    return MediaRouter.inspect_input(media_input)
