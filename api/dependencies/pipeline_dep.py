"""Pipeline dependency and singleton lifecycle manager."""

import logging
from typing import Optional
from pipeline.multimodal_pipeline import MultimodalPipeline

logger = logging.getLogger("dmv_api.dependencies")

_pipeline_instance: Optional[MultimodalPipeline] = None


def init_pipeline(device: str = "cpu") -> MultimodalPipeline:
    """Initialize the shared MultimodalPipeline singleton once at application startup."""
    global _pipeline_instance
    if _pipeline_instance is None:
        logger.info("Initializing shared MultimodalPipeline singleton...")
        _pipeline_instance = MultimodalPipeline(device=device)
        logger.info("MultimodalPipeline singleton successfully initialized.")
    return _pipeline_instance


def get_pipeline() -> MultimodalPipeline:
    """FastAPI dependency provider returning the pre-warmed MultimodalPipeline singleton."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = init_pipeline(device="cpu")
    return _pipeline_instance


def is_pipeline_ready() -> bool:
    """Check whether the underlying models and sub-pipelines are loaded and ready."""
    global _pipeline_instance
    if _pipeline_instance is None:
        return False
    # Validate component readiness
    has_df = _pipeline_instance.deepfake_detector is not None
    has_pr = _pipeline_instance.propaganda_detector is not None
    has_hs = _pipeline_instance.hate_speech_detector is not None
    return has_df and has_pr and has_hs
