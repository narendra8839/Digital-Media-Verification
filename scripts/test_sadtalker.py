import requests
import os
import subprocess
import cv2

url = "https://user-images.githubusercontent.com/4397546/231495639-5d4bb925-ea64-4a36-a519-6389917dac29.mp4"
out_path = "/mnt/d/DMV_Data/DMV_Demo_Assets/candidate_videos/sadtalker_demo_01.mp4"

print(f"Downloading SadTalker demo from {url}...")
res = requests.get(url, stream=True, timeout=20)
print(f"Status: {res.status_code}")
if res.status_code == 200:
    with open(out_path, "wb") as f:
        for chunk in res.iter_content(32768):
            if chunk:
                f.write(chunk)
    size = os.path.getsize(out_path)
    print(f"Saved {out_path}: {size:,} bytes")
    
    cap = cv2.VideoCapture(out_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    probe_cmd = f"ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,sample_rate,channels -of default=noprint_wrappers=1 '{out_path}'"
    r = subprocess.run(probe_cmd, shell=True, capture_output=True, text=True)
    print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {frame_count}, Duration: {duration:.2f}s")
    print(f"Audio: {r.stdout.strip()}")
