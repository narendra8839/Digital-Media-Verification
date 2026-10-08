import os
import time
import json
import requests
import subprocess
import cv2

API_URL = "http://127.0.0.1:8000/verify/video"
DEMO_DIR = "/mnt/d/DMV_Data/DMV_Demo_Assets/final_video_demo"

demo_files = [
    {
        "filename": "01_real_normal.mp4",
        "dataset": "FakeAVCeleb",
        "source_id": "FakeAVCeleb_CatA_id01215",
        "visual_ground_truth": "REAL",
        "audio_provenance": "Original YouTube celebrity interview recording (FakeAVCeleb Category A)",
        "audio_type": "Natural",
        "purpose": "Baseline demonstration: Verify that authentic media passes through the pipeline as REAL with low uncertainty and normal speech classifications."
    },
    {
        "filename": "02_external_deepfake_normal.mp4",
        "dataset": "FakeAVCeleb",
        "source_id": "FakeAVCeleb_CatC_id01215",
        "visual_ground_truth": "FAKE",
        "audio_provenance": "Original YouTube celebrity interview recording (FakeAVCeleb Category C)",
        "audio_type": "Natural",
        "purpose": "Visual-only threat: Demonstrate that external deepfake face manipulation (FaceSwap) is successfully detected by EfficientNet-B4, with Grad-CAM highlighting facial boundary artifacts while speech remains normal."
    },
    {
        "filename": "03_external_deepfake_propaganda.mp4",
        "dataset": "FakeAVCeleb + SemEval-2020 Task 11",
        "source_id": "FakeAVCeleb_CatC_id01215_Propaganda",
        "visual_ground_truth": "FAKE",
        "audio_provenance": "Controlled speech audio synthesized from SemEval-2020 Task 11 benchmark technique text",
        "audio_type": "Controlled",
        "purpose": "Multimodal compound threat: Demonstrate concurrent detection of visual manipulation (FaceSwap) AND linguistic persuasion (Appeal to fear/prejudice) extracted via Whisper ASR."
    },
    {
        "filename": "04_external_deepfake_hate.mp4",
        "dataset": "FakeAVCeleb + HateXplain",
        "source_id": "FakeAVCeleb_CatC_id01215_HateSpeech",
        "visual_ground_truth": "FAKE",
        "audio_provenance": "Controlled speech audio synthesized from HateXplain benchmark target text",
        "audio_type": "Controlled",
        "purpose": "Multimodal severe threat: Demonstrate detection of visual manipulation paired with explicit hate speech, triggering governance REVIEW_RECOMMENDED status and token attribution explanations."
    },
    {
        "filename": "05_real_propaganda.mp4",
        "dataset": "FakeAVCeleb + SemEval-2020 Task 11",
        "source_id": "FakeAVCeleb_CatA_id01215_Propaganda",
        "visual_ground_truth": "REAL",
        "audio_provenance": "Controlled speech audio synthesized from SemEval-2020 Task 11 benchmark technique text",
        "audio_type": "Controlled",
        "purpose": "Orthogonal disentanglement demo: Demonstrate that visual authenticity (REAL face) and textual manipulation (PROPAGANDA speech) are treated as distinct independent analytical dimensions by DMV."
    }
]

print("=== FINAL VERIFICATION OF DEMO ASSETS VIA POST /verify/video ===\n")
manifest_records = []

for item in demo_files:
    fname = item["filename"]
    fpath = os.path.join(DEMO_DIR, fname)
    assert os.path.exists(fpath), f"File {fpath} does not exist!"
    
    # 1. Inspect media properties
    cap = cv2.VideoCapture(fpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frames / fps if fps > 0 else 0
    cap.release()
    
    probe_cmd = f"ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,sample_rate,channels -of default=noprint_wrappers=1 '{fpath}'"
    r_probe = subprocess.run(probe_cmd, shell=True, capture_output=True, text=True)
    audio_info = r_probe.stdout.strip().replace("\n", ", ")
    
    print(f"Testing {fname}...")
    print(f"  Media: {w}x{h} @ {fps:.1f}fps, {duration:.2f}s ({frames} frames) | Audio: {audio_info}")
    
    # 2. Call real POST /verify/video
    start_time = time.time()
    with open(fpath, "rb") as f:
        resp = requests.post(
            API_URL,
            files={"file": (fname, f, "video/mp4")},
            data={"caption": ""},
            timeout=120
        )
    latency = time.time() - start_time
    print(f"  HTTP: {resp.status_code} | Latency: {latency:.2f}s")
    assert resp.status_code == 200, f"Error calling API: {resp.text}"
    
    res = resp.json()
    vis = res.get("visual", {})
    txt = res.get("text", {})
    prop = res.get("propaganda", {})
    hate = res.get("hate_speech", {})
    uq = res.get("uncertainty", {})
    gov = res.get("governance", {})
    
    rep_frame = vis.get("representative_frame", {})
    gradcam_obj = rep_frame.get("explanation", {})
    has_gradcam = bool(gradcam_obj and (gradcam_obj.get("overlay_url") or gradcam_obj.get("overlay_path")))
    
    df_uq = uq.get("deepfake", {})
    prop_uq = uq.get("propaganda", {})
    hate_uq = uq.get("hate_speech", {})
    
    record = {
        "filename": fname,
        "source_dataset": item["dataset"],
        "source_id": item["source_id"],
        "visual_ground_truth": item["visual_ground_truth"],
        "audio_provenance": item["audio_provenance"],
        "audio_type": item["audio_type"],
        "duration_seconds": round(duration, 2),
        "resolution": f"{w}x{h}",
        "audio_stream": audio_info,
        "whisper_transcript": txt.get("combined_text", ""),
        "dmv_visual_prediction": vis.get("prediction", "N/A"),
        "visual_confidence": round(vis.get("confidence", 0.0), 4),
        "face_detected": (vis.get("status") == "SUCCESS"),
        "frames_analyzed": vis.get("frames_analyzed", 0),
        "gradcam_generated": has_gradcam,
        "representative_frame_ts": rep_frame.get("timestamp"),
        "propaganda_prediction": prop.get("prediction", "N/A"),
        "propaganda_confidence": round(prop.get("confidence", 0.0), 4),
        "hate_speech_prediction": hate.get("prediction", "N/A"),
        "hate_speech_confidence": round(hate.get("confidence", 0.0), 4),
        "uncertainty_deepfake_entropy": round(df_uq.get("entropy", 0.0), 4),
        "uncertainty_deepfake_variance": round(df_uq.get("max_variance", 0.0), 6),
        "governance_decision": gov.get("decision_status", "N/A"),
        "review_required": gov.get("review_required", False),
        "review_trigger": gov.get("review_trigger_component"),
        "latency_seconds": round(latency, 2),
        "purpose": item["purpose"]
    }
    manifest_records.append(record)
    
    print(f"  Visual: {record['dmv_visual_prediction'].upper()} (Conf: {record['visual_confidence']:.2%}) | Grad-CAM: {record['gradcam_generated']}")
    print(f"  Whisper Transcript: \"{record['whisper_transcript']}\"")
    print(f"  Propaganda: {record['propaganda_prediction']} ({record['propaganda_confidence']:.2%})")
    print(f"  Hate Speech: {record['hate_speech_prediction']} ({record['hate_speech_confidence']:.2%})")
    print(f"  Governance: {record['governance_decision']} (Review: {record['review_required']}, Trigger: {record['review_trigger']})")
    print("-" * 75)

# Save validation summary
val_out_path = os.path.join(DEMO_DIR, "validation_summary.json")
with open(val_out_path, "w", encoding="utf-8") as f:
    json.dump(manifest_records, f, indent=2)

print(f"\nValidation complete! All records saved to {val_out_path}")
