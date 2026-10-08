import requests
import os
import cv2
import subprocess

url = "https://raw.githubusercontent.com/saifhassan/Wav2Lip-HD/main/input_videos/kennedy.mp4"
target_path = "/mnt/d/DMV_Data/DMV_Demo_Assets/candidate_videos/real_kennedy.mp4"

print("Downloading authentic JFK video...")
r = requests.get(url, timeout=30)
if r.status_code == 200:
    with open(target_path, "wb") as f:
        f.write(r.content)
    print(f"Saved real_kennedy.mp4: {len(r.content):,} bytes")
    
    cap = cv2.VideoCapture(target_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    probe_cmd = f"ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,sample_rate,channels -of default=noprint_wrappers=1 '{target_path}'"
    res = subprocess.run(probe_cmd, shell=True, capture_output=True, text=True)
    print(f"Resolution: {width}x{height} | FPS: {fps:.2f} | Frames: {frame_count} | Duration: {duration:.2f}s")
    print(f"Audio: {res.stdout.strip().replace(chr(10), ', ')}")
else:
    print(f"Failed: {r.status_code}")
