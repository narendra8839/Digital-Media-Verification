"""Live verification of FastAPI service layer using real media assets and checkpoints."""

import os
import sys
import pprint
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app
from api.dependencies.pipeline_dep import get_pipeline


def main():
    print("=" * 70)
    print("FASTAPI SERVICE LAYER: REAL EXECUTION VERIFICATION")
    print("=" * 70)

    # Pre-warm pipeline
    print("\n[1/5] Pre-warming pipeline singleton...")
    get_pipeline()

    client = TestClient(app)

    # 1. Health & Readiness
    print("\n[2/5] Testing GET /health and GET /ready...")
    res_health = client.get("/health")
    assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
    print(f"GET /health -> HTTP {res_health.status_code}: {res_health.json()}")

    res_ready = client.get("/ready")
    assert res_ready.status_code == 200, f"Readiness check failed: {res_ready.text}"
    print(f"GET /ready  -> HTTP {res_ready.status_code}: {res_ready.json()['status']}")

    # 2. Real Text Verification
    print("\n[3/5] Testing POST /verify/text with real NLP inference...")
    sample_text = "The government is systematically hiding the true election ballots."
    res_text = client.post("/verify/text", json={"text": sample_text})
    assert res_text.status_code == 200, f"Text verification failed: {res_text.text}"
    text_data = res_text.json()
    print(f"POST /verify/text -> HTTP {res_text.status_code}")
    print("  Propaganda Prediction:", text_data["propaganda"]["prediction"], f"(Confidence: {text_data['propaganda']['confidence']})")
    print("  Hate Speech Prediction:", text_data["hate_speech"]["prediction"], f"(Confidence: {text_data['hate_speech']['confidence']})")
    print("  Decision Status:", text_data["governance"]["decision_status"])
    print("  Audit Models:", text_data["audit"]["models_executed"])

    # 3. Real Image Verification
    print("\n[4/5] Testing POST /verify/image with real face image...")
    face_path = os.path.join(PROJECT_ROOT, "tests", "assets", "face_image.jpg")
    assert os.path.exists(face_path), f"Missing face image: {face_path}"

    with open(face_path, "rb") as f:
        res_img = client.post(
            "/verify/image",
            files={"file": ("face_image.jpg", f, "image/jpeg")},
            data={"caption": "Official statement photo"}
        )
    assert res_img.status_code == 200, f"Image verification failed: {res_img.text}"
    img_data = res_img.json()
    print(f"POST /verify/image -> HTTP {res_img.status_code}")
    print("  Visual Status:", img_data["visual"]["status"])
    print("  Face Detected:", img_data["visual"]["face_detected"])
    print("  Deepfake Prediction:", img_data["visual"]["prediction"], f"(Confidence: {img_data['visual']['confidence']})")
    print("  Grad-CAM Saliency Shape:", img_data["visual"]["explanation"]["heatmap_shape"])
    print("  MC Dropout Variance:", img_data["uncertainty"]["deepfake"]["mean_variance"])

    # 4. Real Video Verification
    print("\n[5/5] Testing POST /verify/video with real video (audio stream + keyframes)...")
    video_path = os.path.join(PROJECT_ROOT, "tests", "assets", "audio_video.mp4")
    assert os.path.exists(video_path), f"Missing video: {video_path}"

    with open(video_path, "rb") as f:
        res_vid = client.post(
            "/verify/video",
            files={"file": ("audio_video.mp4", f, "video/mp4")},
            data={"async_mode": "false", "sample_fps": "1.0", "max_frames": "2"}
        )
    assert res_vid.status_code == 200, f"Video verification failed: {res_vid.text}"
    vid_data = res_vid.json()
    print(f"POST /verify/video -> HTTP {res_vid.status_code}")
    print("  Input File:", vid_data["input"]["filename"])
    print("  Has Audio:", vid_data["audit"]["has_audio"])
    print("  ASR Executed:", vid_data["audit"]["asr_executed"])
    print("  ASR Checkpoint:", vid_data["audit"]["checkpoint_versions"]["asr"])
    print("  Text Sources:", vid_data["text"]["sources"])
    print("  Combined Text:", vid_data["text"]["combined_text"])
    print("  Decision Status:", vid_data["governance"]["decision_status"])

    print("\n" + "=" * 70)
    print("ALL REAL FASTAPI REQUESTS EXECUTED AND VALIDATED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
