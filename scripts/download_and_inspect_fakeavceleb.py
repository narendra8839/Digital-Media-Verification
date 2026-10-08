import requests
import os
import subprocess
import cv2

doc_ids = {
    "A_real_real.mp4": ("1j2J0RCSTEHkzR7k-i9UT03vc4Xy2yFDG", "FakeAVCeleb Category A: Real Video + Real Audio"),
    "B_real_fakeaudio.mp4": ("18nA84Yb0ZU5E7e3Z-sqe_8ZDGSYdAAUe", "FakeAVCeleb Category B: Real Video + Fake Audio"),
    "C_fakevideo_realaudio.mp4": ("1CG7F1n_B_mEHaXL6-ojwpJFa26HL92dL", "FakeAVCeleb Category C: Fake Video + Real Audio"),
    "D_fakevideo_fakeaudio.mp4": ("1TCKZAKhdgNJdze3njJHuOmOTaMf_XBld", "FakeAVCeleb Category D: Fake Video + Fake Audio")
}

out_dir = "/mnt/d/DMV_Data/DMV_Demo_Assets/candidate_videos"
os.makedirs(out_dir, exist_ok=True)

session = requests.Session()

print(f"Target directory: {out_dir}")
for name, (doc_id, desc) in doc_ids.items():
    path = os.path.join(out_dir, name)
    if not os.path.exists(path) or os.path.getsize(path) < 10000:
        url = f"https://drive.google.com/uc?id={doc_id}&export=download"
        print(f"Downloading {name} ({doc_id})...")
        res = session.get(url, stream=True)
        if res.status_code == 200:
            with open(path, "wb") as f:
                for chunk in res.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            print(f"  Downloaded: {os.path.getsize(path):,} bytes")
        else:
            print(f"  Failed: status {res.status_code}")
    else:
        print(f"{name} already exists: {os.path.getsize(path):,} bytes")
    
    # Analyze video properties
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    # ffprobe for audio
    probe_cmd = f"ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,sample_rate,channels -of default=noprint_wrappers=1 '{path}'"
    res = subprocess.run(probe_cmd, shell=True, capture_output=True, text=True)
    audio_info = res.stdout.strip().replace("\n", ", ")
    
    print(f"[{name}] - {desc}")
    print(f"  Resolution: {width}x{height} | FPS: {fps:.2f} | Frames: {frame_count} | Duration: {duration:.2f}s")
    print(f"  Audio Stream: {audio_info if audio_info else 'NO AUDIO'}\n")
