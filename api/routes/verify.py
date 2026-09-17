"""Verification endpoints for Image, Video, Text, and Multimodal inputs."""

import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    UploadFile,
    File,
    Form,
    HTTPException,
    BackgroundTasks
)

from api.schemas.requests import TextVerificationRequest
from api.schemas.responses import AsyncJobResponse, JobStatusResponse
from api.dependencies.pipeline_dep import get_pipeline
from api.utils import file_utils
from api.utils.serialization import sanitize_for_json
from pipeline.multimodal_pipeline import MultimodalPipeline

logger = logging.getLogger("dmv_api.verify")

router = APIRouter(prefix="/verify", tags=["Verification"])

# In-memory background job tracking store for async video verification
# NOTE: In-memory store is designed for MVP. Jobs do not persist across process restarts.
job_store: Dict[str, Dict[str, Any]] = {}


def _run_video_background(job_id: str,
                           temp_path: str,
                           caption: Optional[str],
                           sample_fps: float,
                           max_frames: int,
                           pipeline: MultimodalPipeline) -> None:
    """Worker function executed inside FastAPI BackgroundTasks for video analysis."""
    logger.info(f"[Job {job_id}] Started background processing for video: {temp_path}")
    if job_id in job_store:
        job_store[job_id]["status"] = "PROCESSING"

    try:
        raw_result = pipeline.verify(
            media_input=temp_path,
            caption=caption,
            sample_fps=sample_fps,
            max_frames=max_frames
        )
        sanitized = sanitize_for_json(raw_result)
        if job_id in job_store:
            job_store[job_id]["status"] = "COMPLETED"
            job_store[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            job_store[job_id]["result"] = sanitized
            logger.info(f"[Job {job_id}] Successfully completed background processing.")
    except Exception as exc:
        logger.error(f"[Job {job_id}] Background processing failed: {str(exc)}", exc_info=True)
        if job_id in job_store:
            job_store[job_id]["status"] = "FAILED"
            job_store[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            job_store[job_id]["error"] = f"Video analysis failed: {str(exc)}"
    finally:
        file_utils.cleanup_temp_file(temp_path)


@router.post(
    "/text",
    summary="Verify Text",
    description="Verify textual content for propaganda techniques (RoBERTa 14-class) and hate speech (BERT 3-class) with token attribution and MC Dropout."
)
async def verify_text(
    payload: TextVerificationRequest,
    pipeline: MultimodalPipeline = Depends(get_pipeline)
) -> Dict[str, Any]:
    """Execute text verification."""
    logger.info(f"Received text verification request ({len(payload.text)} characters)")
    try:
        raw_result = pipeline.verify(payload.text)
        return sanitize_for_json(raw_result)
    except Exception as exc:
        logger.error(f"Error during text verification: {str(exc)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during text verification.")


@router.post(
    "/image",
    summary="Verify Image",
    description="Verify still image for facial deepfakes (EfficientNet-B4 + Grad-CAM) and text content via EasyOCR + NLP routing."
)
async def verify_image(
    file: UploadFile = File(..., description="Uploaded image file (JPEG, PNG, WebP, etc.)"),
    caption: Optional[str] = Form(None, description="Optional caption accompanying the image"),
    pipeline: MultimodalPipeline = Depends(get_pipeline)
) -> Dict[str, Any]:
    """Execute image verification."""
    logger.info(f"Received image upload: {file.filename}")
    temp_path, clean_name, size = await file_utils.save_upload_to_temp(
        upload_file=file,
        allowed_extensions=file_utils.SUPPORTED_IMAGE_EXTENSIONS,
        max_size_bytes=file_utils.MAX_IMAGE_SIZE_BYTES
    )

    try:
        raw_result = pipeline.verify(temp_path, caption=caption)
        return sanitize_for_json(raw_result)
    except Exception as exc:
        logger.error(f"Error during image verification: {str(exc)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error during image verification: {str(exc)}")
    finally:
        file_utils.cleanup_temp_file(temp_path)


@router.post(
    "/video",
    summary="Verify Video",
    description="Verify video file for facial deepfakes, keyframe OCR, and audio/Whisper speech transcription. Supports synchronous or background job processing."
)
async def verify_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Uploaded video file (MP4, AVI, MOV, MKV, WebM)"),
    caption: Optional[str] = Form(None, description="Optional caption accompanying the video"),
    async_mode: bool = Form(False, description="Set to true to process video asynchronously in background"),
    sample_fps: float = Form(1.0, description="Sampling rate in frames per second"),
    max_frames: int = Form(16, description="Maximum number of frames to sample"),
    pipeline: MultimodalPipeline = Depends(get_pipeline)
) -> Any:
    """Execute video verification either directly or via background job."""
    logger.info(f"Received video upload: {file.filename} (async_mode={async_mode})")
    temp_path, clean_name, size = await file_utils.save_upload_to_temp(
        upload_file=file,
        allowed_extensions=file_utils.SUPPORTED_VIDEO_EXTENSIONS,
        max_size_bytes=file_utils.MAX_VIDEO_SIZE_BYTES
    )

    if async_mode:
        job_id = str(uuid.uuid4())
        job_store[job_id] = {
            "job_id": job_id,
            "status": "QUEUED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "result": None,
            "error": None
        }
        background_tasks.add_task(
            _run_video_background,
            job_id=job_id,
            temp_path=temp_path,
            caption=caption,
            sample_fps=sample_fps,
            max_frames=max_frames,
            pipeline=pipeline
        )
        return AsyncJobResponse(
            job_id=job_id,
            status="QUEUED",
            message="Video verification job successfully queued for background processing.",
            poll_url=f"/verify/status/{job_id}"
        )
    else:
        try:
            raw_result = pipeline.verify(
                temp_path,
                caption=caption,
                sample_fps=sample_fps,
                max_frames=max_frames
            )
            return sanitize_for_json(raw_result)
        except Exception as exc:
            logger.error(f"Error during video verification: {str(exc)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal error during video verification: {str(exc)}")
        finally:
            file_utils.cleanup_temp_file(temp_path)


@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Check Asynchronous Job Status",
    description="Retrieve the execution state or final verification result of an asynchronous background video verification job."
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Retrieve status or result for background job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' was not found.")
    return JobStatusResponse(**job_store[job_id])


@router.post(
    "/multimodal",
    summary="Verify Multimodal Media",
    description="Unified endpoint accepting any combination of uploaded media (image or video) and accompanying text, preserving provenance across all channels."
)
async def verify_multimodal(
    file: Optional[UploadFile] = File(None, description="Optional image or video file"),
    text: Optional[str] = Form(None, description="Optional raw text or user-provided caption"),
    pipeline: MultimodalPipeline = Depends(get_pipeline)
) -> Dict[str, Any]:
    """Execute multimodal verification across available channels."""
    clean_text = text.strip() if text else None
    if file is None and not clean_text:
        raise HTTPException(
            status_code=400,
            detail="At least one media input ('file' or 'text') must be provided."
        )

    # Case 1: Only text provided
    if file is None and clean_text:
        logger.info(f"Processing multimodal text-only request ({len(clean_text)} chars)")
        try:
            raw_result = pipeline.verify(clean_text)
            return sanitize_for_json(raw_result)
        except Exception as exc:
            logger.error(f"Error during multimodal text processing: {str(exc)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error during multimodal text verification.")

    # Case 2: File provided (optionally with text)
    original_filename = file.filename or "upload"
    ext = os.path.splitext(original_filename)[1].lower()

    if ext in file_utils.SUPPORTED_IMAGE_EXTENSIONS:
        allowed = file_utils.SUPPORTED_IMAGE_EXTENSIONS
        max_size = file_utils.MAX_IMAGE_SIZE_BYTES
    elif ext in file_utils.SUPPORTED_VIDEO_EXTENSIONS:
        allowed = file_utils.SUPPORTED_VIDEO_EXTENSIONS
        max_size = file_utils.MAX_VIDEO_SIZE_BYTES
    else:
        all_exts = ", ".join(sorted(file_utils.SUPPORTED_IMAGE_EXTENSIONS | file_utils.SUPPORTED_VIDEO_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported media extension '{ext}'. Allowed extensions: {all_exts}"
        )

    temp_path, clean_name, size = await file_utils.save_upload_to_temp(
        upload_file=file,
        allowed_extensions=allowed,
        max_size_bytes=max_size
    )

    try:
        raw_result = pipeline.verify(temp_path, caption=clean_text)
        return sanitize_for_json(raw_result)
    except Exception as exc:
        logger.error(f"Error during multimodal verification: {str(exc)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error during multimodal verification: {str(exc)}")
    finally:
        file_utils.cleanup_temp_file(temp_path)
