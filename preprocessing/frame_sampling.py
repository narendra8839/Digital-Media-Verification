"""Video frame sampling interface at configurable FPS."""

import cv2
from typing import List, Tuple
import numpy as np


def sample_video_frames(video_path: str, target_fps: float = 1.0, max_frames: int = 8) -> List[Tuple[float, np.ndarray]]:
    """Sample video frames at target_fps, returning list of (timestamp_sec, frame_rgb)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(round(native_fps / target_fps)))

    sampled = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = frame_idx / native_fps
            sampled.append((timestamp, frame_rgb))
            if max_frames and len(sampled) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    return sampled
