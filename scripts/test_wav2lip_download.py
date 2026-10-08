import requests
import os
import subprocess
import cv2

urls = [
    ("wav2lip_kennedy_fake.mp4", "https://raw.githubusercontent.com/saifhassan/Wav2Lip-HD/main/output_videos_wav2lip/kennedy.mp4", "Wav2Lip Kennedy Deepfake"),
    ("wav2lip_mona_fake.mp4", "https://raw.githubusercontent.com/saifhassan/Wav2Lip-HD/main/output_videos_wav2lip/mona.mp4", "Wav2Lip Mona Deepfake")
]

out_dir = "/mnt/d/DMV_Data/DMV_Demo_Assets/candidate_videos"
os.makedirs(out_dir, exist_ok=True)

for name, url, desc in urls:
    path = os.path.join(out_dir, name)
    print(f"Downloading {name} from {url}...")
    res = requests.get(url, timeout=30)
    if res.status_code == 200:
        with open(path, "wb") as f:
            f.write(res.content)
        print(f"Saved {name}: {len(res.content):,} bytes")
        
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        probe_cmd = f"ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,sample_rate,channels -of default=noprint_wrappers=1 '{path}'"
        r = subprocess.run(probe_cmd, shell=True, capture_output=True, text=True)
        print(f"  {width}x{height} | {fps:.2f} FPS | {duration:.2f}s | Audio: {r.stdout.strip().replace(chr(10), ', ')}")
    else:
        print(f"Failed {name}: {res.status_code}")
