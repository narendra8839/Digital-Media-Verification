"""Audio extraction interface using bundled FFmpeg from imageio-ffmpeg."""

import os
import subprocess
import tempfile
from typing import Optional
import imageio_ffmpeg


def get_ffmpeg_path() -> str:
    """Return the absolute path to the bundled FFmpeg executable."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def has_audio_stream(video_path: str) -> bool:
    """Check if the video file contains an audio stream.

    Args:
        video_path: Path to the video file.

    Returns:
        True if an audio stream is detected, False otherwise.
    """
    if not os.path.exists(video_path):
        return False

    ffmpeg_bin = get_ffmpeg_path()
    cmd = [ffmpeg_bin, "-i", video_path]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
        # FFmpeg outputs file stream information to stderr
        return "Audio:" in proc.stderr
    except Exception:
        return False


def extract_audio_to_wav(video_path: str, output_wav_path: Optional[str] = None) -> Optional[str]:
    """Extract audio track from video file into 16kHz mono 16-bit PCM WAV.

    Args:
        video_path: Path to input video file.
        output_wav_path: Optional destination path for the extracted WAV file.

    Returns:
        Path to the extracted WAV file, or None if no audio stream exists or extraction failed.
    """
    if not os.path.exists(video_path):
        return None

    if not has_audio_stream(video_path):
        return None

    if output_wav_path is None:
        fd, output_wav_path = tempfile.mkstemp(suffix=".wav", prefix="extracted_audio_")
        os.close(fd)

    ffmpeg_bin = get_ffmpeg_path()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav_path
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if res.returncode == 0 and os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0:
            return output_wav_path
        return None
    except Exception:
        if os.path.exists(output_wav_path):
            try:
                os.remove(output_wav_path)
            except OSError:
                pass
        return None
