"""Artifact lifecycle management, configurable age-based retention, and secure cleanup.

Provides automatic pruning of application-managed spatial explanation artifacts (Grad-CAM)
with strict path traversal guards, symlink protections, active-write buffers,
and non-destructive error handling.
"""

import os
import re
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from contextlib import contextmanager

logger = logging.getLogger("dmv_api.artifact_manager")

# Default configuration constants
DEFAULT_TTL_HOURS: float = 24.0
DEFAULT_CLEANUP_INTERVAL_MINUTES: float = 60.0
DEFAULT_ARTIFACT_DIR: str = "artifacts"
DEFAULT_CLEANUP_ENABLED: bool = True

# Safety buffer: Protect artifacts modified within the last N seconds (active write buffer)
ACTIVE_WRITE_BUFFER_SECONDS: float = 120.0

# Managed artifact extensions and regex patterns
MANAGED_ARTIFACT_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".webp"}
MANAGED_ARTIFACT_PATTERNS = [
    re.compile(r"^gradcam_[a-zA-Z0-9_-]+\.(png|jpg|jpeg|webp)$", re.IGNORECASE),
    re.compile(r"^.+_gradcam\.(png|jpg|jpeg|webp)$", re.IGNORECASE),
    re.compile(r"^.+_frame\d+_gradcam\.(png|jpg|jpeg|webp)$", re.IGNORECASE),
]

# Explicitly protected filenames that must never be deleted
EXCLUDED_FILENAMES: Set[str] = {
    ".gitkeep",
    ".gitignore",
    "README.md",
    "manifest.json"
}

# In-memory registry of actively generating/writing artifact paths
_ACTIVE_ARTIFACTS: Set[str] = set()


@dataclass
class ArtifactRetentionConfig:
    """Configuration for artifact lifecycle and retention policy."""
    ttl_hours: float = DEFAULT_TTL_HOURS
    cleanup_interval_minutes: float = DEFAULT_CLEANUP_INTERVAL_MINUTES
    artifact_dir: str = DEFAULT_ARTIFACT_DIR
    enabled: bool = DEFAULT_CLEANUP_ENABLED

    @classmethod
    def from_env(cls) -> "ArtifactRetentionConfig":
        """Read configuration from environment variables with rigorous validation and fallback defaults."""
        ttl_raw = os.environ.get("DMV_ARTIFACT_TTL_HOURS", "").strip()
        interval_raw = os.environ.get("DMV_ARTIFACT_CLEANUP_INTERVAL_MINUTES", "").strip()
        dir_raw = os.environ.get("DMV_ARTIFACT_DIR", "").strip()
        enabled_raw = os.environ.get("DMV_ARTIFACT_CLEANUP_ENABLED", "").strip().lower()

        # Parse TTL
        ttl = DEFAULT_TTL_HOURS
        if ttl_raw:
            try:
                val = float(ttl_raw)
                if val <= 0:
                    logger.warning(
                        f"Invalid non-positive DMV_ARTIFACT_TTL_HOURS='{ttl_raw}'. "
                        f"Falling back to default ({DEFAULT_TTL_HOURS}h)."
                    )
                else:
                    ttl = val
            except ValueError:
                logger.warning(
                    f"Invalid non-numeric DMV_ARTIFACT_TTL_HOURS='{ttl_raw}'. "
                    f"Falling back to default ({DEFAULT_TTL_HOURS}h)."
                )

        # Parse Cleanup Interval
        interval = DEFAULT_CLEANUP_INTERVAL_MINUTES
        if interval_raw:
            try:
                val = float(interval_raw)
                if val <= 0:
                    logger.warning(
                        f"Invalid non-positive DMV_ARTIFACT_CLEANUP_INTERVAL_MINUTES='{interval_raw}'. "
                        f"Falling back to default ({DEFAULT_CLEANUP_INTERVAL_MINUTES}m)."
                    )
                else:
                    interval = val
            except ValueError:
                logger.warning(
                    f"Invalid non-numeric DMV_ARTIFACT_CLEANUP_INTERVAL_MINUTES='{interval_raw}'. "
                    f"Falling back to default ({DEFAULT_CLEANUP_INTERVAL_MINUTES}m)."
                )

        # Parse Artifact Dir
        artifact_dir = dir_raw if dir_raw else DEFAULT_ARTIFACT_DIR

        # Parse Enabled flag
        enabled = DEFAULT_CLEANUP_ENABLED
        if enabled_raw in ("0", "false", "no", "off", "disable", "disabled"):
            enabled = False
        elif enabled_raw in ("1", "true", "yes", "on", "enable", "enabled"):
            enabled = True

        return cls(
            ttl_hours=ttl,
            cleanup_interval_minutes=interval,
            artifact_dir=artifact_dir,
            enabled=enabled
        )


@dataclass
class CleanupSummary:
    """Summary metrics of an executed artifact cleanup cycle."""
    scanned: int = 0
    deleted: int = 0
    preserved_recent: int = 0
    preserved_active: int = 0
    skipped_unmanaged: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scanned": self.scanned,
            "deleted": self.deleted,
            "preserved_recent": self.preserved_recent,
            "preserved_active": self.preserved_active,
            "skipped_unmanaged": self.skipped_unmanaged,
            "errors": list(self.errors)
        }


def is_managed_artifact(filename: str) -> bool:
    """Determine whether a file is an application-managed explanation artifact eligible for lifecycle cleanup."""
    base = os.path.basename(filename)
    if not base or base in EXCLUDED_FILENAMES or base.startswith("."):
        return False
    ext = os.path.splitext(base)[1].lower()
    if ext not in MANAGED_ARTIFACT_EXTENSIONS:
        return False
    return any(pattern.match(base) for pattern in MANAGED_ARTIFACT_PATTERNS)


def register_active_artifact(path: Optional[str]) -> None:
    """Mark an artifact path as actively in-progress or being written."""
    if path:
        try:
            _ACTIVE_ARTIFACTS.add(os.path.realpath(path))
        except Exception:
            _ACTIVE_ARTIFACTS.add(os.path.abspath(path))


def unregister_active_artifact(path: Optional[str]) -> None:
    """Unmark an active artifact path upon completion or failure."""
    if path:
        try:
            _ACTIVE_ARTIFACTS.discard(os.path.realpath(path))
        except Exception:
            _ACTIVE_ARTIFACTS.discard(os.path.abspath(path))


@contextmanager
def protect_active_artifact(path: Optional[str]):
    """Context manager ensuring an artifact path is registered as active while in scope."""
    if path:
        register_active_artifact(path)
    try:
        yield path
    finally:
        if path:
            unregister_active_artifact(path)


def is_artifact_active(resolved_path: str, mtime: float, now: float) -> bool:
    """Check whether a file is actively registered or within the active write buffer."""
    if resolved_path in _ACTIVE_ARTIFACTS:
        return True
    if (now - mtime) < ACTIVE_WRITE_BUFFER_SECONDS:
        return True
    return False


def cleanup_expired_artifacts(
    artifact_dir: Optional[str] = None,
    ttl_hours: Optional[float] = None,
    config: Optional[ArtifactRetentionConfig] = None,
    now: Optional[float] = None,
    dry_run: bool = False
) -> CleanupSummary:
    """Safely inspect and prune expired application-managed explanation artifacts.

    Enforces:
    1. Only managed artifacts matching approved patterns (gradcam_*.png, etc.) are deleted.
    2. Explicitly protects .gitkeep, subdirectories, and non-managed files.
    3. Traversal protection: confirms resolved path is strictly within the resolved artifact root.
    4. Symlink protection: skips symlink entries without following them.
    5. Active protection: protects files modified within the write buffer or registered active.
    6. Non-crashing: captures permission/lock errors gracefully.

    Args:
        artifact_dir: Optional override for the target artifact directory.
        ttl_hours: Optional override for the expiration duration in hours.
        config: Optional pre-constructed ArtifactRetentionConfig.
        now: Optional epoch timestamp for deterministic testing.
        dry_run: When True, simulates deletion without removing files.

    Returns:
        CleanupSummary containing metrics and recoverable error messages.
    """
    if config is None:
        config = ArtifactRetentionConfig.from_env()

    effective_dir = artifact_dir if artifact_dir is not None else config.artifact_dir
    effective_ttl_hours = ttl_hours if ttl_hours is not None else config.ttl_hours
    ttl_seconds = effective_ttl_hours * 3600.0
    current_time = now if now is not None else time.time()

    summary = CleanupSummary()

    # Guard: if directory does not exist or is not a directory, exit safely
    if not os.path.exists(effective_dir):
        logger.debug(f"Artifact directory '{effective_dir}' does not exist. Skipping cleanup.")
        return summary

    if not os.path.isdir(effective_dir):
        logger.warning(f"Configured artifact path '{effective_dir}' is not a directory. Skipping cleanup.")
        summary.errors.append(f"Target path '{effective_dir}' is not a directory.")
        return summary

    # Canonicalize artifact root for strict traversal bounds check
    resolved_root = os.path.realpath(effective_dir)

    try:
        entries = os.listdir(effective_dir)
    except OSError as err:
        logger.warning(f"Failed to list artifact directory '{effective_dir}': {err}")
        summary.errors.append(f"Directory access error: {err}")
        return summary

    for entry_name in entries:
        raw_path = os.path.join(effective_dir, entry_name)
        summary.scanned += 1

        # 1. Symlink Safety: Never follow symlinks
        if os.path.islink(raw_path):
            logger.debug(f"Skipping symlink entry: {raw_path}")
            summary.skipped_unmanaged += 1
            continue

        # 2. Never delete subdirectories
        if os.path.isdir(raw_path):
            summary.skipped_unmanaged += 1
            continue

        # 3. Target only managed artifacts (gradcam_*.png, etc.), skip .gitkeep and others
        if not is_managed_artifact(entry_name):
            summary.skipped_unmanaged += 1
            continue

        # 4. Canonical path traversal prevention: must resolve inside resolved_root
        resolved_file = os.path.realpath(raw_path)
        try:
            common = os.path.commonpath([resolved_file, resolved_root])
        except ValueError:
            summary.skipped_unmanaged += 1
            summary.errors.append(f"Path traversal detected across drives: {raw_path}")
            continue

        if common != resolved_root or resolved_file == resolved_root:
            logger.warning(
                f"Path traversal blocked: '{raw_path}' resolves to '{resolved_file}' outside '{resolved_root}'"
            )
            summary.skipped_unmanaged += 1
            summary.errors.append(f"Path traversal violation: {raw_path}")
            continue

        # 5. Check file modification time and active status
        try:
            stat_res = os.stat(resolved_file)
            mtime = stat_res.st_mtime
        except OSError as err:
            logger.warning(f"Could not stat artifact '{resolved_file}': {err}")
            summary.errors.append(f"Stat error on {entry_name}: {err}")
            continue

        # Check if actively being written or registered in active buffer
        if is_artifact_active(resolved_file, mtime, current_time):
            summary.preserved_active += 1
            continue

        # Check age against TTL
        age_seconds = current_time - mtime
        if age_seconds < ttl_seconds:
            summary.preserved_recent += 1
            continue

        # File is expired and eligible for deletion
        if dry_run:
            summary.deleted += 1
            continue

        try:
            os.remove(resolved_file)
            summary.deleted += 1
            logger.info(f"Pruned expired explanation artifact: {entry_name} (age: {age_seconds / 3600.0:.1f}h)")
        except (PermissionError, OSError) as err:
            logger.warning(f"Could not delete expired artifact '{resolved_file}': {err}")
            summary.errors.append(f"Deletion error on {entry_name}: {err}")

    logger.info(
        f"Artifact cleanup completed for '{effective_dir}': "
        f"scanned={summary.scanned}, deleted={summary.deleted}, "
        f"preserved_recent={summary.preserved_recent}, preserved_active={summary.preserved_active}, "
        f"skipped_unmanaged={summary.skipped_unmanaged}, errors={len(summary.errors)}"
    )
    return summary


class ArtifactCleanupRunner:
    """Manages periodic execution of artifact cleanup in an asynchronous FastAPI lifespan."""

    def __init__(self, config: Optional[ArtifactRetentionConfig] = None):
        self.config = config or ArtifactRetentionConfig.from_env()
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start the background cleanup loop if not already running."""
        if not self.config.enabled:
            logger.info("Artifact automatic retention is disabled via configuration (DMV_ARTIFACT_CLEANUP_ENABLED=0).")
            return

        if self.is_running:
            logger.warning("Artifact cleanup runner is already active; ignoring duplicate start request.")
            return

        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="dmv_artifact_cleanup_worker")
        logger.info(
            f"Started background artifact cleanup worker: interval={self.config.cleanup_interval_minutes}m, "
            f"ttl={self.config.ttl_hours}h, dir='{self.config.artifact_dir}'"
        )

    async def _run_loop(self) -> None:
        interval_seconds = max(1.0, self.config.cleanup_interval_minutes * 60.0)
        while not self._stop_event.is_set():
            try:
                # Wait for next scheduled run or stop signal.
                # Runs periodically rather than deleting all artifacts immediately at startup.
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval_seconds)
                break
            except asyncio.TimeoutError:
                # Execute cleanup in a background thread to prevent blocking the asyncio event loop
                try:
                    await asyncio.to_thread(cleanup_expired_artifacts, config=self.config)
                except Exception as exc:
                    logger.error(f"Unexpected error in background artifact cleanup: {exc}", exc_info=True)

    async def stop(self) -> None:
        """Gracefully stop the background cleanup worker during application shutdown."""
        if not self.is_running:
            return

        logger.info("Stopping background artifact cleanup worker...")
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning(f"Exception while awaiting cleanup task cancellation: {exc}")
        self._task = None
        logger.info("Background artifact cleanup worker stopped successfully.")
