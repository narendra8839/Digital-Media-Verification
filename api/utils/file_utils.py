"""Temporary file management, format validation, and secure upload streaming."""

import os
import re
import tempfile
from typing import Set, Tuple
from fastapi import UploadFile, HTTPException

SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
SUPPORTED_VIDEO_EXTENSIONS: Set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# File size limits: 15 MB for image, 100 MB for video
MAX_IMAGE_SIZE_BYTES: int = 15 * 1024 * 1024
MAX_VIDEO_SIZE_BYTES: int = 100 * 1024 * 1024
CHUNK_SIZE: int = 64 * 1024 # 64 KB


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename to prevent directory traversal attacks."""
    basename = os.path.basename(filename or "uploaded_media")
    # Replace any potentially dangerous characters
    clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', basename)
    return clean_name or "uploaded_media"


def validate_file_extension(filename: str, allowed_extensions: Set[str]) -> str:
    """Validate file extension against allowed set and return normalized lowercase extension."""
    ext = os.path.splitext(filename)[1].lower()
    if not ext or ext not in allowed_extensions:
        allowed_str = ", ".join(sorted(allowed_extensions))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {allowed_str}"
        )
    return ext


async def save_upload_to_temp(upload_file: UploadFile,
                              allowed_extensions: Set[str],
                              max_size_bytes: int) -> Tuple[str, str, int]:
    """Stream uploaded file to temporary disk location with strict size enforcement.

    Args:
        upload_file: FastAPI UploadFile object.
        allowed_extensions: Set of permitted file extensions.
        max_size_bytes: Maximum permitted size in bytes.

    Returns:
        Tuple of (temp_file_path, sanitized_filename, total_bytes_written).

    Raises:
        HTTPException(400) if file is empty or extension invalid.
        HTTPException(413) if file exceeds size limit.
    """
    original_filename = upload_file.filename or "upload"
    clean_filename = sanitize_filename(original_filename)
    ext = validate_file_extension(clean_filename, allowed_extensions)

    # Create temporary file with preserved extension
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix="dmv_upload_")
    temp_path = temp_file.name
    bytes_written = 0

    try:
        while chunk := await upload_file.read(CHUNK_SIZE):
            bytes_written += len(chunk)
            if bytes_written > max_size_bytes:
                max_mb = max_size_bytes / (1024 * 1024)
                temp_file.close()
                cleanup_temp_file(temp_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload size exceeded maximum allowed limit of {max_mb:.1f} MB."
                )
            temp_file.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        temp_file.close()
        cleanup_temp_file(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to stream upload: {str(exc)}")
    finally:
        temp_file.close()

    if bytes_written == 0:
        cleanup_temp_file(temp_path)
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    return temp_path, clean_filename, bytes_written


def cleanup_temp_file(file_path: str) -> None:
    """Safely delete temporary file without raising exceptions."""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass
