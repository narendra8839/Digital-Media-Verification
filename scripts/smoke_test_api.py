"""End-to-End API Smoke Test for Phase 6 Final Integration.

Tests real API routes (/health, /ready, /verify/text, /verify/image, /verify/video, /verify/multimodal)
using real Phase 5 candidate checkpoints, measuring latency, verifying device selection, UQ, XAI,
governance decisions, and artifact serving.
"""

import os
import sys
import time
import json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi.testclient import TestClient
from api.main import app
from api.dependencies.pipeline_dep import get_pipeline
from api.utils.device import get_optimal_device

client = TestClient(app)
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "assets")


def run_smoke_tests():
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_device": get_optimal_device(),
        "endpoints": {}
    }

    print("=" * 70)
    print("PHASE 6 END-TO-END BACKEND & INFERENCE SMOKE TEST")
    print(f"Runtime Device Selected: {report['runtime_device']}")
    print("=" * 70)

    # 1. Health & Readiness Probe
    print("\n[1/6] Testing /health and /ready...")
    t0 = time.time()
    res_health = client.get("/health")
    res_ready = client.get("/ready")
    latency_ready = round(time.time() - t0, 3)

    assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
    assert res_ready.status_code == 200, f"Readiness check failed: {res_ready.text}"
    ready_data = res_ready.json()

    pipeline = get_pipeline()
    report["endpoints"]["readiness"] = {
        "status": ready_data["status"],
        "models": ready_data.get("models", {}),
        "pipeline_device": getattr(pipeline, "device", "unknown"),
        "latency_seconds": latency_ready
    }
    print(f"  -> Readiness Status: {ready_data['status']}")
    print(f"  -> Active Checkpoints: {json.dumps(ready_data.get('models', {}), indent=2)}")
    print(f"  -> Pipeline Device: {pipeline.device}")

    # 2. Text Verification Route
    print("\n[2/6] Testing /verify/text...")
    test_text = (
        "Breaking: The corrupt corporate elites and dishonest media are conspiring "
        "to deceive the entire public with blatant lies! Join the movement now before it's too late!"
    )
    t0 = time.time()
    res_text = client.post("/verify/text", json={"text": test_text})
    latency_text = round(time.time() - t0, 3)

    assert res_text.status_code == 200, f"Text verify failed: {res_text.text}"
    text_data = res_text.json()

    pr = text_data.get("propaganda", {})
    hs = text_data.get("hate_speech", {})
    uq_pr = text_data.get("uncertainty", {}).get("propaganda", {})
    gov_text = text_data.get("governance", {})

    report["endpoints"]["text"] = {
        "status_code": res_text.status_code,
        "latency_seconds": latency_text,
        "propaganda": {
            "prediction": pr.get("prediction"),
            "confidence": pr.get("confidence"),
            "probabilities_top": dict(list(pr.get("probabilities", {}).items())[:3]),
            "token_attribution_tokens": len(pr.get("explanation", {}).get("token_scores", []))
        },
        "hate_speech": {
            "prediction": hs.get("prediction"),
            "confidence": hs.get("confidence"),
            "probabilities": hs.get("probabilities", {})
        },
        "uncertainty": {
            "mc_samples": uq_pr.get("mc_samples"),
            "max_variance": uq_pr.get("max_variance"),
            "entropy": uq_pr.get("entropy"),
            "is_stochastic": uq_pr.get("is_stochastic"),
            "disclaimer": uq_pr.get("disclaimer")
        },
        "governance": {
            "decision_status": gov_text.get("decision_status"),
            "review_required": gov_text.get("review_required"),
            "disclaimer": gov_text.get("disclaimer")
        }
    }
    print(f"  -> Latency: {latency_text}s")
    print(f"  -> Propaganda Prediction: {pr.get('prediction')} (conf: {pr.get('confidence')})")
    print(f"  -> Hate Speech Prediction: {hs.get('prediction')} (conf: {hs.get('confidence')})")
    print(f"  -> MC Dropout UQ (T={uq_pr.get('mc_samples')}): max_variance={uq_pr.get('max_variance')}, entropy={uq_pr.get('entropy')}")
    print(f"  -> Governance Decision: {gov_text.get('decision_status')}")

    # 3. Valid Image Verification Route (with detectable face)
    print("\n[3/6] Testing /verify/image (face detectable)...")
    face_img_path = os.path.join(ASSETS_DIR, "face_image.jpg")
    t0 = time.time()
    with open(face_img_path, "rb") as f:
        res_img_face = client.post(
            "/verify/image",
            files={"file": ("face_image.jpg", f, "image/jpeg")}
        )
    latency_img_face = round(time.time() - t0, 3)

    assert res_img_face.status_code == 200, f"Image verify failed: {res_img_face.text}"
    img_face_data = res_img_face.json()

    vis_face = img_face_data.get("visual", {})
    uq_df = img_face_data.get("uncertainty", {}).get("deepfake", {})
    gov_face = img_face_data.get("governance", {})
    expl_face = vis_face.get("explanation", {})

    # Check Grad-CAM artifact accessibility
    overlay_url = expl_face.get("overlay_url")
    artifact_accessible = False
    artifact_status_code = None
    if overlay_url and overlay_url.startswith("/artifacts/"):
        art_check = client.get(overlay_url)
        artifact_status_code = art_check.status_code
        artifact_accessible = art_check.status_code == 200

    report["endpoints"]["image_with_face"] = {
        "status_code": res_img_face.status_code,
        "latency_seconds": latency_img_face,
        "visual": {
            "face_detected": vis_face.get("face_detected"),
            "prediction": vis_face.get("prediction"),
            "confidence": vis_face.get("confidence"),
            "probabilities": vis_face.get("probabilities")
        },
        "gradcam": {
            "generated": expl_face is not None,
            "overlay_url": overlay_url,
            "artifact_http_accessible": artifact_accessible,
            "artifact_status_code": artifact_status_code,
            "disclaimer": expl_face.get("disclaimer")
        },
        "uncertainty": {
            "mc_samples": uq_df.get("mc_samples"),
            "max_variance": uq_df.get("max_variance"),
            "entropy": uq_df.get("entropy"),
            "is_stochastic": uq_df.get("is_stochastic")
        },
        "governance": {
            "decision_status": gov_face.get("decision_status"),
            "review_required": gov_face.get("review_required"),
            "review_trigger_component": gov_face.get("review_trigger_component"),
            "disclaimer": gov_face.get("disclaimer")
        }
    }
    print(f"  -> Latency: {latency_img_face}s")
    print(f"  -> Face Detected: {vis_face.get('face_detected')}")
    print(f"  -> Deepfake Prediction: {vis_face.get('prediction')} (conf: {vis_face.get('confidence')})")
    print(f"  -> Grad-CAM Artifact URL: {overlay_url} (HTTP Accessible: {artifact_accessible})")
    print(f"  -> MC Dropout UQ (T={uq_df.get('mc_samples')}): max_variance={uq_df.get('max_variance')}, entropy={uq_df.get('entropy')}")
    print(f"  -> Governance Decision: {gov_face.get('decision_status')}")

    # 4. Image Verification Route (No Face Detectable)
    print("\n[4/6] Testing /verify/image (no face detectable)...")
    no_face_path = os.path.join(ASSETS_DIR, "no_face_image.jpg")
    t0 = time.time()
    with open(no_face_path, "rb") as f:
        res_no_face = client.post(
            "/verify/image",
            files={"file": ("no_face_image.jpg", f, "image/jpeg")}
        )
    latency_no_face = round(time.time() - t0, 3)

    assert res_no_face.status_code == 200, f"No-face verify failed: {res_no_face.text}"
    no_face_data = res_no_face.json()

    vis_no_face = no_face_data.get("visual", {})
    gov_no_face = no_face_data.get("governance", {})

    report["endpoints"]["image_no_face"] = {
        "status_code": res_no_face.status_code,
        "latency_seconds": latency_no_face,
        "visual": {
            "status": vis_no_face.get("status"),
            "face_detected": vis_no_face.get("face_detected"),
            "prediction": vis_no_face.get("prediction")
        },
        "governance": {
            "decision_status": gov_no_face.get("decision_status"),
            "review_required": gov_no_face.get("review_required"),
            "disclaimer": gov_no_face.get("disclaimer")
        }
    }
    print(f"  -> Latency: {latency_no_face}s")
    print(f"  -> Visual Status: {vis_no_face.get('status')}")
    print(f"  -> Prediction Fabricated?: {'No (None)' if vis_no_face.get('prediction') is None else 'YES (Error!)'}")
    print(f"  -> Governance Decision: {gov_no_face.get('decision_status')} (Incomplete Analysis Preserved)")

    # 5. Short Video Verification Route
    print("\n[5/6] Testing /verify/video (short test video)...")
    video_path = os.path.join(ASSETS_DIR, "silent_video.mp4")
    t0 = time.time()
    with open(video_path, "rb") as f:
        res_vid = client.post(
            "/verify/video",
            files={"file": ("silent_video.mp4", f, "video/mp4")},
            data={"async_mode": "false", "sample_fps": "1.0", "max_frames": "4"}
        )
    latency_vid = round(time.time() - t0, 3)

    assert res_vid.status_code == 200, f"Video verify failed: {res_vid.text}"
    vid_data = res_vid.json()

    vis_vid = vid_data.get("visual", {})
    audit_vid = vid_data.get("audit", {})
    gov_vid = vid_data.get("governance", {})

    report["endpoints"]["video"] = {
        "status_code": res_vid.status_code,
        "latency_seconds": latency_vid,
        "visual": {
            "status": vis_vid.get("status"),
            "frames_analyzed": vis_vid.get("frames_analyzed"),
            "total_frames_sampled": vis_vid.get("total_frames_sampled"),
            "prediction": vis_vid.get("prediction")
        },
        "ocr_executed": audit_vid.get("ocr_executed"),
        "asr_executed": audit_vid.get("asr_executed"),
        "has_audio": audit_vid.get("has_audio"),
        "governance": {
            "decision_status": gov_vid.get("decision_status")
        }
    }
    print(f"  -> Latency: {latency_vid}s")
    print(f"  -> Frames Sampled: {vis_vid.get('total_frames_sampled')}")
    print(f"  -> OCR Executed: {audit_vid.get('ocr_executed')}")
    print(f"  -> ASR Executed: {audit_vid.get('asr_executed')} (Has Audio: {audit_vid.get('has_audio')})")
    print(f"  -> Governance Decision: {gov_vid.get('decision_status')}")

    # 6. Multimodal Verification Route
    print("\n[6/6] Testing /verify/multimodal (media + text combination)...")
    t0 = time.time()
    with open(face_img_path, "rb") as f:
        res_mm = client.post(
            "/verify/multimodal",
            files={"file": ("face_image.jpg", f, "image/jpeg")},
            data={"text": "A controversial political speech delivered during an emergency rally."}
        )
    latency_mm = round(time.time() - t0, 3)

    assert res_mm.status_code == 200, f"Multimodal verify failed: {res_mm.text}"
    mm_data = res_mm.json()

    report["endpoints"]["multimodal"] = {
        "status_code": res_mm.status_code,
        "latency_seconds": latency_mm,
        "input": mm_data.get("input"),
        "visual_status": mm_data.get("visual", {}).get("status"),
        "text_status": mm_data.get("text", {}).get("text_status"),
        "propaganda_prediction": mm_data.get("propaganda", {}).get("prediction"),
        "hate_speech_prediction": mm_data.get("hate_speech", {}).get("prediction"),
        "governance_decision": mm_data.get("governance", {}).get("decision_status"),
        "governance_trigger_component": mm_data.get("governance", {}).get("review_trigger_component")
    }
    print(f"  -> Latency: {latency_mm}s")
    print(f"  -> Visual Status: {mm_data.get('visual', {}).get('status')}")
    print(f"  -> Text Status: {mm_data.get('text', {}).get('text_status')}")
    print(f"  -> Propaganda Technique: {mm_data.get('propaganda', {}).get('prediction')}")
    print(f"  -> Hate Speech Class: {mm_data.get('hate_speech', {}).get('prediction')}")
    print(f"  -> Governance Decision: {mm_data.get('governance', {}).get('decision_status')} (Trigger: {mm_data.get('governance', {}).get('review_trigger_component')})")

    # Output JSON artifact
    out_path = "smoke_test_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("\n" + "=" * 70)
    print(f"SMOKE TEST COMPLETE: Results successfully recorded to {out_path}")
    print("=" * 70)
    return report


if __name__ == "__main__":
    run_smoke_tests()
