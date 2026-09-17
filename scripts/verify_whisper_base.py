"""Verify Whisper-base integration and real multimodal pipeline execution."""

import os
import sys
import pprint

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from extraction.asr import WhisperASR, get_default_whisper_model
from extraction.audio_extraction import has_audio_stream, extract_audio_to_wav
from pipeline.multimodal_pipeline import MultimodalPipeline


def main():
    print("=" * 60)
    print("STEP 1: Verify Whisper Model Configuration")
    print("=" * 60)
    configured_model = get_default_whisper_model()
    print(f"Configured model: {configured_model}")
    assert configured_model == "openai/whisper-base", f"Expected openai/whisper-base, got {configured_model}"

    # Reset any singleton
    WhisperASR._instance = None
    asr = WhisperASR()
    print(f"WhisperASR instance model_id: {asr.model_id}")
    assert asr.model_id == "openai/whisper-base", f"Expected openai/whisper-base, got {asr.model_id}"

    print("\n" + "=" * 60)
    print("STEP 2: Verify Real Audio Video Processing")
    print("=" * 60)
    audio_video = os.path.join(PROJECT_ROOT, "tests", "assets", "audio_video.mp4")
    assert os.path.exists(audio_video), f"Missing test video: {audio_video}"

    # Check audio stream probing
    audio_detected = has_audio_stream(audio_video)
    print(f"Audio stream detected in {audio_video}: {audio_detected}")
    assert audio_detected is True, "Audio stream should be detected in audio_video.mp4"

    # Check audio extraction
    wav_path = extract_audio_to_wav(audio_video)
    print(f"Extracted audio to WAV: {wav_path} (exists: {os.path.exists(wav_path)})")
    assert wav_path is not None and os.path.exists(wav_path), "WAV extraction failed"

    # Check Whisper-base transcription
    transcription = asr.transcribe(wav_path)
    print(f"Direct Whisper-base transcription result: {transcription}")
    assert transcription["source"] == "ASR", "Provenance source must be ASR"
    assert transcription["status"] in ("SUCCESS", "NO_SPEECH_DETECTED"), f"Unexpected status: {transcription['status']}"

    # Cleanup temp wav
    if os.path.exists(wav_path):
        os.remove(wav_path)

    print("\n" + "=" * 60)
    print("STEP 3: Run Full MultimodalPipeline on Video with Audio")
    print("=" * 60)
    pipeline = MultimodalPipeline(device="cpu")
    res_audio = pipeline.verify(audio_video, sample_fps=1.0, max_frames=2)

    print("Input Metadata:")
    pprint.pprint(res_audio["input"])
    print("\nText Summary:")
    pprint.pprint(res_audio["text"])
    print("\nPropaganda Result:")
    pprint.pprint(res_audio["propaganda"])
    print("\nHate Speech Result:")
    pprint.pprint(res_audio["hate_speech"])
    print("\nAudit:")
    pprint.pprint(res_audio["audit"])

    # Verification assertions
    assert res_audio["audit"]["has_audio"] is True
    assert res_audio["audit"]["asr_executed"] is True
    assert "Whisper-base (openai/whisper-base)" in res_audio["audit"]["checkpoint_versions"]["asr"]

    asr_segments = [s for s in res_audio["text"]["segments"] if s["source"] == "ASR"]
    assert len(asr_segments) == 1, f"Expected 1 ASR segment, got {len(asr_segments)}"
    assert asr_segments[0]["source"] == "ASR", "Provenance must be ASR"
    print("\n--> Audio Video Pipeline successfully passed all checks!")

    print("\n" + "=" * 60)
    print("STEP 4: Run MultimodalPipeline on Silent Video (No Audio)")
    print("=" * 60)
    silent_video = os.path.join(PROJECT_ROOT, "tests", "assets", "silent_video.mp4")
    assert os.path.exists(silent_video), f"Missing silent video: {silent_video}"

    silent_detected = has_audio_stream(silent_video)
    print(f"Audio stream detected in {silent_video}: {silent_detected}")
    assert silent_detected is False, "Silent video must not have audio stream"

    res_silent = pipeline.verify(silent_video, sample_fps=1.0, max_frames=2)
    print("Silent Video Audit:")
    pprint.pprint(res_silent["audit"])

    assert res_silent["audit"]["has_audio"] is False
    assert res_silent["audit"]["asr_executed"] is False
    asr_silent_segments = [s for s in res_silent["text"]["segments"] if s["source"] == "ASR"]
    assert len(asr_silent_segments) == 0 or asr_silent_segments[0]["text"] == ""
    print("\n--> Silent Video Pipeline successfully passed: Whisper was NOT executed!")

    print("\n" + "=" * 60)
    print("ALL REAL EXECUTION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
