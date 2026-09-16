"""Health and readiness monitoring endpoints."""

from fastapi import APIRouter
from api.schemas.responses import HealthResponse, ReadyResponse
from api.dependencies.pipeline_dep import is_pipeline_ready, get_pipeline

router = APIRouter(tags=["Health & Monitoring"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Lightweight liveness probe indicating service is running. Does NOT load or reload model weights."
)
async def health_check() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="healthy", service="digital-media-verification")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness Probe",
    description="Readiness probe verifying that underlying AI models and components are pre-warmed and ready to serve traffic."
)
async def readiness_check() -> ReadyResponse:
    """Readiness probe checking model initialization."""
    ready = is_pipeline_ready()
    status_str = "ready" if ready else "initializing"

    models_info = {
        "deepfake": "EfficientNet-B4 (models/deepfake/checkpoint/best_model.pt)",
        "propaganda": "RoBERTa-base (models/propaganda/checkpoint)",
        "hate_speech": "BERT-base-uncased (models/hate_speech/checkpoint)",
        "asr": "Whisper-base (openai/whisper-base)",
        "ocr": "EasyOCR (English CRAFT detector)"
    }

    components_info = {
        "face_detector": "MTCNN",
        "uncertainty_engine": "MCDropoutEstimator (T=20)",
        "explainability": "Grad-CAM + Token Attribution",
        "audio_probe": "imageio-ffmpeg"
    }

    return ReadyResponse(
        status=status_str,
        service="digital-media-verification",
        components=components_info,
        models=models_info
    )
