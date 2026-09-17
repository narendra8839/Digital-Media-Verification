"""Request schemas for verification endpoints."""

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
