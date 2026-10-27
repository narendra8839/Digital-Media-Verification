from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class TextVerificationRequest(BaseModel):
    """Payload for text verification endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text content to verify for propaganda techniques and hate speech."
    )

    @field_validator("text")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Text content cannot be empty or only whitespace.")
        return stripped


class ImageReportRequest(BaseModel):
    """Payload for downloading an image analysis PDF report."""

    result: Dict[str, Any] = Field(
        ...,
        description="Unified image analysis result payload previously returned by /verify/image."
    )
    filename: Optional[str] = Field(
        default="image_analysis.jpg",
        description="Original name of the analyzed image file."
    )
    image_preview: Optional[str] = Field(
        default=None,
        description="Optional base64 data URI of the analyzed image for thumbnail rendering."
    )
