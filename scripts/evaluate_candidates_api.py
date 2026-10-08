import os
import time
import json
import requests

API_URL = "http://127.0.0.1:8000/verify/video"
candidate_dir = "/mnt/d/DMV_Data/DMV_Demo_Assets/candidate_videos"

candidates = [
    {
        "filename": "A_real_real.mp4",
        "dataset": "FakeAVCeleb",
        "source_id": "FakeAVCeleb_CatA_id01215",
        "ground_truth_visual": "REAL",
        "ground_truth_audio": "REAL",
        "description": "Category A: Real Video + Real Audio (Original authentic YouTube celebrity video)"
    },
    {
        "filename": "B_real_fakeaudio.mp4",
        "dataset": "FakeAVCeleb",
        "source_id": "FakeAVCeleb_CatB_id01215",
        "ground_truth_visual": "REAL",
        "ground_truth_audio": "FAKE",
        "description": "Category B: Real Video + Fake Audio (Authentic face video with SV2TTS cloned voice)"
    },
    {
        "filename": "C_fakevideo_realaudio.mp4",
        "dataset": "FakeAVCeleb",
        "source_id": "FakeAVCeleb_CatC_id01215",
        "ground_truth_visual": "FAKE",
        "ground_truth_audio": "REAL",
        "description": "Category C: Fake Video + Real Audio (FaceSwap manipulated video with authentic audio)"
    },
    {
        "filename": "D_fakevideo_fakeaudio.mp4",
        "dataset": "FakeAVCeleb",
        "source_id": "FakeAVCeleb_CatD_id01215",
        "ground_truth_visual": "FAKE",
        "ground_truth_audio": "FAKE",
        "description": "Category D: Fake Video + Fake Audio (FaceSwap manipulated video + Wav2Lip synthetic speech)"
    },
    {
        "filename": "wav2lip_kennedy_fake.mp4",
        "dataset": "Wav2Lip / In-the-Wild",
        "source_id": "Wav2Lip_JFK_Speech",
        "ground_truth_visual": "FAKE",
        "ground_truth_audio": "REAL",
        "description": "Wav2Lip synthetic lip-synchronization of JFK address"
    },
    {
        "filename": "wav2lip_mona_fake.mp4",
        "dataset": "Wav2Lip / In-the-Wild",
        "source_id": "Wav2Lip_MonaLisa",
        "ground_truth_visual": "FAKE",
        "ground_truth_audio": "REAL",
        "description": "Wav2Lip synthetic portrait animation with audio speech"
    },
    {
        "filename": "real_original_recording5.mp4",
        "dataset": "User / Authentic Camera",
        "source_id": "Camera_Recording_05",
        "ground_truth_visual": "REAL",
        "ground_truth_audio": "REAL",
        "description": "Authentic video recording with clear human speech"
    }
]

results = []

print("=== Running Candidates through POST /verify/video ===")

for c in candidates:
    filepath = os.path.join(candidate_dir, c["filename"])
    if not os.path.exists(filepath):
        print(f"Skipping {c['filename']}: file not found")
        continue

    print(f"\nEvaluating: {c['filename']} ({c['description']})...")
    start_t = time.time()
    
    with open(filepath, "rb") as f:
        files = {"file": (c["filename"], f, "video/mp4")}
        data = {"caption": ""}
        response = requests.post(API_URL, files=files, data=data, timeout=120)
    
    latency = time.time() - start_t
    print(f"  HTTP Status: {response.status_code} | Latency: {latency:.2f}s")
    
    if response.status_code == 200:
        res_json = response.json()
        
        # Correctly map fields from full schema
        vis = res_json.get("visual", {})
        txt = res_json.get("text", {})
        prop = res_json.get("propaganda", {})
        hate = res_json.get("hate_speech", {})
        uq = res_json.get("uncertainty", {})
        gov = res_json.get("governance", {})
        
        df_uq = uq.get("deepfake", {})
        
        face_detected = (vis.get("status") == "SUCCESS" and vis.get("frames_analyzed", 0) > 0)
        gradcam_obj = vis.get("representative_frame", {}).get("explanation", {})
        has_gradcam = bool(gradcam_obj and gradcam_obj.get("heatmap_overlay_path"))
        
        rec = {
            "source_dataset": c["dataset"],
            "source_id": c["source_id"],
            "filename": c["filename"],
            "visual_ground_truth": c["ground_truth_visual"],
            "audio_ground_truth": c["ground_truth_audio"],
            "description": c["description"],
            "latency_s": round(latency, 2),
            "face_detected": face_detected,
            "frames_analyzed": vis.get("frames_analyzed", 0),
            "visual_prediction": vis.get("prediction", "N/A"),
            "visual_confidence": round(vis.get("confidence", 0.0), 4),
            "gradcam_generated": has_gradcam,
            "whisper_transcript": txt.get("combined_text", ""),
            "propaganda_prediction": prop.get("prediction", "N/A"),
            "propaganda_confidence": round(prop.get("confidence", 0.0), 4),
            "hate_speech_prediction": hate.get("prediction", "N/A"),
            "hate_speech_confidence": round(hate.get("confidence", 0.0), 4),
            "uncertainty_entropy": round(df_uq.get("entropy", 0.0), 4),
            "uncertainty_variance": round(df_uq.get("max_variance", 0.0), 6),
            "governance_decision": gov.get("decision_status", "N/A"),
            "review_required": gov.get("review_required", False),
            "review_trigger": gov.get("review_trigger_component")
        }
        results.append(rec)
        print(f"  Face Detected: {rec['face_detected']} | Frames Analyzed: {rec['frames_analyzed']}")
        print(f"  Visual Prediction: {rec['visual_prediction']} (Conf: {rec['visual_confidence']:.2%}) | Grad-CAM: {rec['gradcam_generated']}")
        print(f"  Whisper Transcript: \"{rec['whisper_transcript'][:80]}...\"" if len(rec['whisper_transcript']) > 80 else f"  Whisper Transcript: \"{rec['whisper_transcript']}\"")
        print(f"  Propaganda: {rec['propaganda_prediction']} ({rec['propaganda_confidence']:.2%}) | Hate Speech: {rec['hate_speech_prediction']} ({rec['hate_speech_confidence']:.2%})")
        print(f"  UQ: Entropy={rec['uncertainty_entropy']}, Var={rec['uncertainty_variance']}")
        print(f"  Governance: {rec['governance_decision']} (Review: {rec['review_required']}, Trigger: {rec['review_trigger']})")
    else:
        print(f"  Error response: {response.text[:200]}")

out_json = "/mnt/d/DMV_Data/DMV_Demo_Assets/candidate_evaluation_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nAll candidate evaluations saved to {out_json}")
