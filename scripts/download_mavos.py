import os
import subprocess
import cv2
from huggingface_hub import hf_hub_download

out_dir = "/mnt/d/DMV_Data/DMV_Demo_Assets/candidate_videos"
os.makedirs(out_dir, exist_ok=True)

repo_id = "unibuc-cs/MAVOS-DD"
candidates = [
    ("mavos_roop_01.mp4", "english/roop/0120beb2-0fe8-11f1-857c-ac1f6b8d2e48.mp4", "MAVOS-DD Roop Deepfake"),
    ("mavos_liveportrait_01.mp4", "english/liveportrait/00bbb69c-10ac-11f1-b2c7-ac1f6b417054.mp4", "MAVOS-DD LivePortrait Deepfake"),
    ("mavos_inswapper_01.mp4", "english/inswapper/10000-7GN10u6F9m0_27_1.mp4", "MAVOS-DD InSwapper Deepfake"),
    ("mavos_real_01.mp4", "english/real/-1_SwSYMu2A_12_1.mp4", "MAVOS-DD Authentic Video")
]

for filename, hf_path, desc in candidates:
    target_path = os.path.join(out_dir, filename)
    print(f"\nDownloading {filename} ({desc})...")
    try:
        downloaded = hf_hub_download(repo_id=repo_id, filename=hf_path, repo_type="dataset")
        # Copy to target_path
        with open(downloaded, "rb") as src, open(target_path, "wb") as dst:
            dst.write(src.read())
        print(f"Saved: {target_path} ({os.path.getsize(target_path):,} bytes)")
        
        cap = cv2.VideoCapture(target_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        probe_cmd = f"ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,sample_rate,channels -of default=noprint_wrappers=1 '{target_path}'"
        r = subprocess.run(probe_cmd, shell=True, capture_output=True, text=True)
        print(f"  Resolution: {width}x{height} | FPS: {fps:.2f} | Frames: {frame_count} | Duration: {duration:.2f}s")
        print(f"  Audio: {r.stdout.strip().replace(chr(10), ', ')}")
    except Exception as e:
        print(f"Failed: {e}")
