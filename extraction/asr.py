"""Whisper speech-to-text extraction interface."""

import os
from typing import Dict, Any, Optional
from transformers import pipeline


def get_default_whisper_model(config_path: str = "config/config.yaml") -> str:
    """Retrieve configured whisper model or fallback to openai/whisper-base."""
    default_model = "openai/whisper-base"
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if cfg and "models" in cfg and "whisper" in cfg["models"]:
                return str(cfg["models"]["whisper"])
        except Exception:
            pass
    return default_model


class WhisperASR:
    """Singleton/reusable Whisper ASR pipeline."""

    _instance = None
    _pipe = None

    def __new__(cls, model_id: Optional[str] = None, device: str = "cpu"):
        effective_model_id = model_id or get_default_whisper_model()
        if cls._instance is None:
            cls._instance = super(WhisperASR, cls).__new__(cls)
            cls._instance.model_id = effective_model_id
            cls._instance.device = device
            cls._instance._pipe = None
        elif model_id is not None and model_id != cls._instance.model_id:
            cls._instance.model_id = effective_model_id
            cls._instance.device = device
            cls._instance._pipe = None
        return cls._instance

    def _get_pipeline(self):
        if self._pipe is None:
            self._pipe = pipeline("automatic-speech-recognition", model=self.model_id, device=self.device)
        return self._pipe

    def transcribe(self, audio_path: Optional[str]) -> Dict[str, Any]:
        """Transcribe audio track from file path.

        Args:
            audio_path: Path to WAV/MP3 audio file, or None if no audio track exists.

        Returns:
            Dictionary with transcript text, source provenance, and execution status.
        """
        if audio_path is None or not os.path.exists(audio_path):
            return {
                "text": "",
                "source": "ASR",
                "status": "NO_AUDIO"
            }

        try:
            pipe = self._get_pipeline()
            # Read WAV directly into float32 array to avoid HF pipeline requiring ffmpeg in system PATH
            audio_input = audio_path
            try:
                import wave
                import numpy as np
                with wave.open(audio_path, 'rb') as wf:
                    sr = wf.getframerate()
                    n_frames = wf.getnframes()
                    frames = wf.readframes(n_frames)
                    audio_arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    audio_input = {"raw": audio_arr, "sampling_rate": sr}
            except Exception:
                # Fall back to passing filename if wave reading fails
                audio_input = audio_path

            result = pipe(audio_input)
            transcript = result.get("text", "").strip()

            return {
                "text": transcript,
                "source": "ASR",
                "status": "SUCCESS" if transcript else "NO_SPEECH_DETECTED"
            }
        except Exception as e:
            return {
                "text": "",
                "source": "ASR",
                "status": "ERROR",
                "error": str(e)
            }


_default_asr: Optional[WhisperASR] = None


def transcribe_audio(audio_path: Optional[str], device: str = "cpu") -> Dict[str, Any]:
    """Convenience function to transcribe audio using shared Whisper instance."""
    global _default_asr
    if _default_asr is None:
        _default_asr = WhisperASR(device=device)
    return _default_asr.transcribe(audio_path)
