# Grad-CAM Artifact Lifecycle, Retention & Storage Safety

This document details the lifecycle, retention policies, storage security, and monitoring of generated Explainable AI (Grad-CAM) artifacts in the **Digital Media Verification** system.

---

## 1. Artifact Lifecycle Overview

During visual media analysis (single images or sampled video keyframes), the deepfake detection subsystem generates spatial heatmaps using **Grad-CAM** on the final convolutional layer of EfficientNet-B4 (`features[-1]`).

1. **Generation:**
   - Single Image: `DeepfakeDetector.detect_image()` generates `gradcam_<uuid8>.png`.
   - Video Keyframes: `VideoPipeline.process_video()` generates `<filename>_frame<rep_idx>_gradcam.png` for the single representative frame.
   - The file is written to the managed directory (default: `artifacts/`).
2. **Registration & Serving:**
   - Active files are temporarily protected in an in-memory registry while generation is underway.
   - The backend mounts `artifacts/` statically at `/artifacts` (`StaticFiles`).
   - The API response payload returns relative URLs (e.g. `/artifacts/gradcam_xxxx.png`) and browser-accessible base64 data URIs.
3. **Retention & Pruning:**
   - An asynchronous lifecycle runner (`ArtifactCleanupRunner`) runs in the background of the FastAPI application.
   - Files exceeding the Time-To-Live (default: **24 hours**) are automatically pruned.
   - If an artifact expires while viewed on the frontend, the UI gracefully renders an expiration notice without broken image icons or application crashes.

---

## 2. Configuration & Defaults

All retention behavior is configurable through standard environment variables with verified fallback defaults:

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DMV_ARTIFACT_TTL_HOURS` | `24.0` | Expiration threshold in hours. Artifacts older than this age are eligible for cleanup. |
| `DMV_ARTIFACT_CLEANUP_INTERVAL_MINUTES` | `60.0` | Sleep interval between periodic pruning sweeps. |
| `DMV_ARTIFACT_DIR` | `artifacts` | Target storage directory for managed artifacts. |
| `DMV_ARTIFACT_CLEANUP_ENABLED` | `1` (true) | Master toggle. Set to `0`, `false`, `no`, or `disabled` to deactivate cleanup. |

### Configuration Validation
If non-numeric or non-positive values are passed (e.g., `DMV_ARTIFACT_TTL_HOURS="abc"` or `"-5"`), the system logs a descriptive warning and falls back to the safe default (24h TTL, 60m interval).

---

## 3. Storage Safety & Security Guarantees

The cleanup implementation in [`api/utils/artifact_manager.py`](file:///api/utils/artifact_manager.py) enforces strict filesystem safety constraints:

1. **Managed Artifact Matching:**
   - Deletion targets only application-owned files matching:
     - `^gradcam_[a-zA-Z0-9_-]+\.(png|jpg|jpeg|webp)$`
     - `^.+_gradcam\.(png|jpg|jpeg|webp)$`
     - `^.+_frame\d+_gradcam\.(png|jpg|jpeg|webp)$`
2. **Explicit Exclusions:**
   - Preserves repository control files such as `.gitkeep`, `.gitignore`, `README.md`, and `manifest.json`.
   - Never deletes subdirectories or parent directories.
   - Never touches model weights (`models/`), raw datasets (`data/`), reports (`reports/`), or user upload streams.
3. **Active-Write Protection:**
   - **Active Write Buffer:** Any file modified within the last **120 seconds** (`ACTIVE_WRITE_BUFFER_SECONDS`) is skipped, preventing deletion of files currently being written to disk.
   - **In-Memory Registry:** Inference routines register the target path via `protect_active_artifact(path)` during generation.
4. **Path Traversal & Symlink Escapes:**
   - Canonicalizes directory and file paths using `os.path.realpath()`.
   - Rejects any candidate whose common path does not strictly match the approved artifact root.
   - Symlinks (`os.path.islink()`) are explicitly detected and skipped without following.
5. **Non-Crashing Error Handling:**
   - Catches `PermissionError` and `OSError` (e.g. Windows file locking when a file is open in a viewer).
   - Skips locked files, logs the path and error details, and continues pruning without terminating the server.

---

## 4. Inspecting Cleanup Logs

The cleanup worker logs directly to the application logger (`dmv_api.artifact_manager`).

### Successful Cleanup Log Example:
```
2026-10-22 13:00:00 [INFO] dmv_api.artifact_manager: Pruned expired explanation artifact: gradcam_03a5b700.png (age: 26.3h)
2026-10-22 13:00:00 [INFO] dmv_api.artifact_manager: Artifact cleanup completed for 'artifacts': scanned=23, deleted=1, preserved_recent=21, preserved_active=0, skipped_unmanaged=1, errors=0
```

### Diagnosing Locked or Inaccessible Files:
If an operating system process or media indexer locks a file during a cleanup run:
```
2026-10-22 13:00:00 [WARNING] dmv_api.artifact_manager: Could not delete expired artifact 'C:\...\artifacts\gradcam_locked.png': [WinError 32] The process cannot access the file because it is being used by another process.
```
- The locked file is preserved without corrupting the artifact directory.
- The failure is recorded in `summary.errors`, and the worker continues servicing subsequent files.

---

## 5. Frontend Expiration Experience

When an explanation URL returns HTTP 404 due to TTL expiration:
- The `<img>` element triggers an `onError` event.
- The component transitions to an explanation expiration alert banner:
  > **Explanation Overlay Expired**
  > *The spatial Grad-CAM overlay for this analysis has reached its retention limit (24 hours) or is no longer available on the server. Please rerun verification to regenerate spatial explainability overlays.*
- When the user runs a new analysis, the error state resets automatically.
