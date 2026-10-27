import sys
import os
import time
import asyncio
import tempfile
import pytest
from unittest.mock import patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.utils.artifact_manager import (
    ArtifactRetentionConfig,
    CleanupSummary,
    ArtifactCleanupRunner,
    cleanup_expired_artifacts,
    protect_active_artifact,
    register_active_artifact,
    unregister_active_artifact,
    is_managed_artifact,
    DEFAULT_TTL_HOURS,
    DEFAULT_CLEANUP_INTERVAL_MINUTES
)


class TestArtifactRetentionConfig:
    """Tests for configuration parsing, environment variables, validation, and defaults."""

    def test_default_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            config = ArtifactRetentionConfig.from_env()
            assert config.ttl_hours == DEFAULT_TTL_HOURS
            assert config.cleanup_interval_minutes == DEFAULT_CLEANUP_INTERVAL_MINUTES
            assert config.artifact_dir == "artifacts"
            assert config.enabled is True

    def test_valid_environment_overrides(self):
        env = {
            "DMV_ARTIFACT_TTL_HOURS": "48.5",
            "DMV_ARTIFACT_CLEANUP_INTERVAL_MINUTES": "30",
            "DMV_ARTIFACT_DIR": "custom_artifacts",
            "DMV_ARTIFACT_CLEANUP_ENABLED": "1"
        }
        with patch.dict(os.environ, env, clear=True):
            config = ArtifactRetentionConfig.from_env()
            assert config.ttl_hours == 48.5
            assert config.cleanup_interval_minutes == 30.0
            assert config.artifact_dir == "custom_artifacts"
            assert config.enabled is True

    def test_disabled_cleanup_flag(self):
        for flag in ["0", "false", "no", "off", "disable", "disabled"]:
            with patch.dict(os.environ, {"DMV_ARTIFACT_CLEANUP_ENABLED": flag}, clear=True):
                config = ArtifactRetentionConfig.from_env()
                assert config.enabled is False

    def test_invalid_values_fallback_to_defaults(self):
        env = {
            "DMV_ARTIFACT_TTL_HOURS": "not-a-number",
            "DMV_ARTIFACT_CLEANUP_INTERVAL_MINUTES": "-10",
        }
        with patch.dict(os.environ, env, clear=True):
            config = ArtifactRetentionConfig.from_env()
            assert config.ttl_hours == DEFAULT_TTL_HOURS
            assert config.cleanup_interval_minutes == DEFAULT_CLEANUP_INTERVAL_MINUTES


class TestManagedArtifactMatching:
    """Tests identifying managed explanation artifacts vs excluded files."""

    def test_managed_artifact_patterns(self):
        assert is_managed_artifact("gradcam_03a5b700.png") is True
        assert is_managed_artifact("gradcam_abc123.jpg") is True
        assert is_managed_artifact("video_upload_frame0_gradcam.png") is True
        assert is_managed_artifact("sample_face_gradcam.webp") is True

    def test_unmanaged_and_excluded_files(self):
        assert is_managed_artifact(".gitkeep") is False
        assert is_managed_artifact(".gitignore") is False
        assert is_managed_artifact("README.md") is False
        assert is_managed_artifact("manifest.json") is False
        assert is_managed_artifact("model_weights.pt") is False
        assert is_managed_artifact("user_upload.mp4") is False
        assert is_managed_artifact("other_image.png") is False


class TestRetentionAndCleanupBehavior:
    """Tests age-based artifact expiration, boundary conditions, and active write protection."""

    def test_stale_artifact_deleted_and_recent_preserved(self, tmp_path):
        artifact_dir = str(tmp_path / "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)

        now = 100000.0
        ttl_hours = 24.0
        ttl_seconds = ttl_hours * 3600.0

        # Create a stale file (age = 30 hours)
        stale_file = os.path.join(artifact_dir, "gradcam_stale.png")
        with open(stale_file, "wb") as f:
            f.write(b"fake_stale_image_bytes")
        stale_mtime = now - (30.0 * 3600.0)
        os.utime(stale_file, (stale_mtime, stale_mtime))

        # Create a recent file (age = 2 hours)
        recent_file = os.path.join(artifact_dir, "gradcam_recent.png")
        with open(recent_file, "wb") as f:
            f.write(b"fake_recent_image_bytes")
        recent_mtime = now - (2.0 * 3600.0)
        os.utime(recent_file, (recent_mtime, recent_mtime))

        # Run cleanup
        summary = cleanup_expired_artifacts(artifact_dir=artifact_dir, ttl_hours=ttl_hours, now=now)

        assert summary.deleted == 1
        assert summary.preserved_recent == 1
        assert not os.path.exists(stale_file)
        assert os.path.exists(recent_file)

    def test_boundary_condition(self, tmp_path):
        """Test boundary condition: file exactly at TTL boundary is considered expired (age >= ttl)."""
        artifact_dir = str(tmp_path / "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)

        now = 100000.0
        ttl_hours = 24.0
        ttl_seconds = ttl_hours * 3600.0

        # Exactly at boundary: age == ttl_seconds (not younger than TTL)
        boundary_file = os.path.join(artifact_dir, "gradcam_boundary.png")
        with open(boundary_file, "wb") as f:
            f.write(b"boundary_content")
        os.utime(boundary_file, (now - ttl_seconds, now - ttl_seconds))

        # Just inside boundary: 1 second younger than TTL
        younger_file = os.path.join(artifact_dir, "gradcam_younger.png")
        with open(younger_file, "wb") as f:
            f.write(b"younger_content")
        os.utime(younger_file, (now - ttl_seconds + 1.0, now - ttl_seconds + 1.0))

        summary = cleanup_expired_artifacts(artifact_dir=artifact_dir, ttl_hours=ttl_hours, now=now)

        assert not os.path.exists(boundary_file)
        assert os.path.exists(younger_file)

    def test_active_write_buffer_protection(self, tmp_path):
        """Files modified within the active write buffer (< 120s) are protected even if ttl is very small."""
        artifact_dir = str(tmp_path / "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)

        now = 100000.0
        # Active file modified 30 seconds ago
        active_file = os.path.join(artifact_dir, "gradcam_active_write.png")
        with open(active_file, "wb") as f:
            f.write(b"in_flight_bytes")
        os.utime(active_file, (now - 30.0, now - 30.0))

        # Even with tiny TTL of 0.001 hours (3.6s), the active buffer protects it
        summary = cleanup_expired_artifacts(artifact_dir=artifact_dir, ttl_hours=0.001, now=now)

        assert summary.preserved_active == 1
        assert summary.deleted == 0
        assert os.path.exists(active_file)

    def test_explicit_active_artifact_registry(self, tmp_path):
        """Files registered in active registry are protected from deletion regardless of age."""
        artifact_dir = str(tmp_path / "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)

        now = 100000.0
        in_flight_file = os.path.join(artifact_dir, "gradcam_registered.png")
        with open(in_flight_file, "wb") as f:
            f.write(b"generating_bytes")
        # Older mtime
        os.utime(in_flight_file, (now - 200000.0, now - 200000.0))

        with protect_active_artifact(in_flight_file):
            summary = cleanup_expired_artifacts(artifact_dir=artifact_dir, ttl_hours=1.0, now=now)
            assert os.path.exists(in_flight_file)
            assert summary.preserved_active == 1
            assert summary.deleted == 0

        # After exiting context, cleanup now expires it
        summary2 = cleanup_expired_artifacts(artifact_dir=artifact_dir, ttl_hours=1.0, now=now)
        assert not os.path.exists(in_flight_file)
        assert summary2.deleted == 1

    def test_video_representative_frame_artifact_protected_during_generation(self, tmp_path):
        """Representative-frame Grad-CAM generated during video pipeline is shielded by protect_active_artifact."""
        artifact_dir = str(tmp_path / "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)

        now = 500000.0
        rep_frame_artifact = os.path.join(artifact_dir, "my_video.mp4_frame3_gradcam.png")
        with open(rep_frame_artifact, "wb") as f:
            f.write(b"video_gradcam_bytes")
        # Set artificial old mtime to ensure it would expire without protection
        os.utime(rep_frame_artifact, (now - 300000.0, now - 300000.0))

        with protect_active_artifact(rep_frame_artifact):
            summary = cleanup_expired_artifacts(artifact_dir=artifact_dir, ttl_hours=1.0, now=now)
            assert os.path.exists(rep_frame_artifact)
            assert summary.preserved_active == 1
            assert summary.deleted == 0

        # After releasing protection, expired cleanup evicts it
        summary_after = cleanup_expired_artifacts(artifact_dir=artifact_dir, ttl_hours=1.0, now=now)
        assert not os.path.exists(rep_frame_artifact)
        assert summary_after.deleted == 1

    def test_dry_run_simulation(self, tmp_path):
        """Dry run identifies expired files without deleting them."""
        artifact_dir = str(tmp_path / "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)

        now = 100000.0
        stale_file = os.path.join(artifact_dir, "gradcam_test_dry.png")
        with open(stale_file, "wb") as f:
            f.write(b"dry_run_data")
        os.utime(stale_file, (now - 100000.0, now - 100000.0))

        summary = cleanup_expired_artifacts(artifact_dir=artifact_dir, ttl_hours=1.0, now=now, dry_run=True)
        assert summary.deleted == 1
        assert os.path.exists(stale_file)  # Preserved because dry_run=True


class TestFilesystemSafety:
    """Tests ensuring files outside the managed root and non-managed files are protected."""

    def test_preserves_gitkeep_and_unmanaged_files(self, tmp_path):
        artifact_dir = str(tmp_path / "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)

        gitkeep = os.path.join(artifact_dir, ".gitkeep")
        with open(gitkeep, "w") as f:
            f.write("")

        readme = os.path.join(artifact_dir, "README.md")
        with open(readme, "w") as f:
            f.write("# Artifacts directory")

        unrelated = os.path.join(artifact_dir, "checkpoint.pt")
        with open(unrelated, "wb") as f:
            f.write(b"model_weights")

        # Set very old mtimes
        old_time = 1000.0
        now = 1000000.0
        for p in [gitkeep, readme, unrelated]:
            os.utime(p, (old_time, old_time))

        summary = cleanup_expired_artifacts(artifact_dir=artifact_dir, ttl_hours=1.0, now=now)

        assert os.path.exists(gitkeep)
        assert os.path.exists(readme)
        assert os.path.exists(unrelated)
        assert summary.deleted == 0
        assert summary.skipped_unmanaged == 3

    def test_never_deletes_subdirectories(self, tmp_path):
        artifact_dir = str(tmp_path / "artifacts")
        subdir = os.path.join(artifact_dir, "sub_directory")
        os.makedirs(subdir, exist_ok=True)

        summary = cleanup_expired_artifacts(artifact_dir=artifact_dir, ttl_hours=1.0)
        assert os.path.exists(subdir)
        assert summary.deleted == 0

    def test_missing_and_empty_directories_handled_gracefully(self, tmp_path):
        missing_dir = str(tmp_path / "non_existent_artifacts")
        summary1 = cleanup_expired_artifacts(artifact_dir=missing_dir)
        assert summary1.scanned == 0
        assert summary1.deleted == 0

        empty_dir = str(tmp_path / "empty_artifacts")
        os.makedirs(empty_dir, exist_ok=True)
        summary2 = cleanup_expired_artifacts(artifact_dir=empty_dir)
        assert summary2.scanned == 0
        assert summary2.deleted == 0

    def test_symlink_safety(self, tmp_path):
        """Symlinks pointing to outside files are skipped and never followed or deleted."""
        artifact_dir = str(tmp_path / "artifacts")
        external_dir = str(tmp_path / "external")
        os.makedirs(artifact_dir, exist_ok=True)
        os.makedirs(external_dir, exist_ok=True)

        # Create external file (e.g. valuable checkpoint)
        external_file = os.path.join(external_dir, "valuable_checkpoint.png")
        with open(external_file, "wb") as f:
            f.write(b"important_data")

        # Create symlink inside artifact directory if supported on this OS
        link_path = os.path.join(artifact_dir, "gradcam_symlink.png")
        try:
            os.symlink(external_file, link_path)
        except (OSError, NotImplementedError):
            pytest.skip("Symlink creation not permitted or supported on this system.")

        summary = cleanup_expired_artifacts(artifact_dir=artifact_dir, ttl_hours=0.01)

        # Symlink is skipped without affecting the external target
        assert summary.skipped_unmanaged >= 1
        assert os.path.exists(external_file)

    def test_locked_file_handled_without_crashing(self, tmp_path):
        """Locked or permission-denied files are logged as errors without crashing cleanup."""
        artifact_dir = str(tmp_path / "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)

        now = 100000.0
        locked_file = os.path.join(artifact_dir, "gradcam_locked.png")
        with open(locked_file, "wb") as f:
            f.write(b"locked_content")
        os.utime(locked_file, (now - 100000.0, now - 100000.0))

        # Open file exclusively (simulating locked handle on Windows)
        with open(locked_file, "r+") as handle:
            # Patch os.remove to raise PermissionError simulating locked handle
            with patch("os.remove", side_effect=PermissionError("Simulated locked file on Windows")):
                summary = cleanup_expired_artifacts(artifact_dir=artifact_dir, ttl_hours=1.0, now=now)
                assert summary.deleted == 0
                assert len(summary.errors) == 1
                assert "Deletion error" in summary.errors[0]

        # File is still present after failed deletion attempt
        assert os.path.exists(locked_file)


class TestArtifactCleanupRunner:
    """Tests for the asynchronous lifespan runner."""

    def test_runner_start_stop_lifecycle(self, tmp_path):
        async def _run():
            config = ArtifactRetentionConfig(
                ttl_hours=1.0,
                cleanup_interval_minutes=0.001,  # Fast loop for testing
                artifact_dir=str(tmp_path),
                enabled=True
            )
            runner = ArtifactCleanupRunner(config=config)
            assert runner.is_running is False

            runner.start()
            assert runner.is_running is True

            # Duplicate start should be safely ignored
            runner.start()
            assert runner.is_running is True

            await asyncio.sleep(0.05)
            await runner.stop()
            assert runner.is_running is False

        asyncio.run(_run())

    def test_runner_disabled_flag(self):
        config = ArtifactRetentionConfig(enabled=False)
        runner = ArtifactCleanupRunner(config=config)
        runner.start()
        assert runner.is_running is False
