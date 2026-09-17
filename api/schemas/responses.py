"""Response schemas for API endpoints."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field("healthy", description="Service health status")
    service: str = Field("digital-media-verification", description="Service identifier")


class ReadyResponse(BaseModel):
    """Readiness probe response."""
    status: str = Field(..., description="Readiness status ('ready' or 'initializing')")
    service: str = Field("digital-media-verification", description="Service identifier")
    components: Dict[str, str] = Field(default_factory=dict, description="Component status map")
    models: Dict[str, str] = Field(default_factory=dict, description="Active model checkpoints")


class AsyncJobResponse(BaseModel):
    """Initial response when video verification is submitted in asynchronous background mode."""
    job_id: str = Field(..., description="Unique UUID tracking the verification job")
    status: str = Field(..., description="Current job status (QUEUED, PROCESSING, COMPLETED, FAILED)")
    message: str = Field(..., description="Human-readable job status description")
    poll_url: str = Field(..., description="URL endpoint to poll for results")


class JobStatusResponse(BaseModel):
    """Status query response for asynchronous background jobs."""
    job_id: str = Field(..., description="Unique UUID tracking the verification job")
    status: str = Field(..., description="Current job status (QUEUED, PROCESSING, COMPLETED, FAILED)")
    created_at: str = Field(..., description="ISO 8601 timestamp of job submission")
    completed_at: Optional[str] = Field(None, description="ISO 8601 timestamp of completion")
    result: Optional[Dict[str, Any]] = Field(None, description="Unified verification result payload upon completion")
    error: Optional[str] = Field(None, description="Error message if processing failed")


class ErrorResponse(BaseModel):
    """Structured error payload returned on 4xx/5xx responses."""
    error: str = Field(..., description="High-level error classification")
    detail: str = Field(..., description="Specific explanation of the failure")
    status_code: int = Field(..., description="HTTP status code")
