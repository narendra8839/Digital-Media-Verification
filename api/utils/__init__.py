"""API utility modules for serialization and file streaming."""

from api.utils.serialization import sanitize_for_json
from api.utils.file_utils import (
    save_upload_to_temp,
    cleanup_temp_file,
    sanitize_filename,
    validate_file_extension,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    MAX_IMAGE_SIZE_BYTES,
    MAX_VIDEO_SIZE_BYTES
)

from api.utils.artifact_manager import (
    ArtifactRetentionConfig,
    CleanupSummary,
    ArtifactCleanupRunner,
    cleanup_expired_artifacts,
    protect_active_artifact,
    is_managed_artifact
)

__all__ = [
    "sanitize_for_json",
    "save_upload_to_temp",
    "cleanup_temp_file",
    "sanitize_filename",
    "validate_file_extension",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "MAX_IMAGE_SIZE_BYTES",
    "MAX_VIDEO_SIZE_BYTES",
    "ArtifactRetentionConfig",
    "CleanupSummary",
    "ArtifactCleanupRunner",
    "cleanup_expired_artifacts",
    "protect_active_artifact",
    "is_managed_artifact"
]
