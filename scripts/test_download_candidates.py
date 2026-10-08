import os
import subprocess
from huggingface_hub import hf_hub_download

repo_id = "belkhir-nacim/deepfake-videos"

# Let's test downloading one real and one fake from DFD and UADFV
test_files = [
    # DFD original (Google DeepFake Detection)
    ("DFD_real_podium", "videos/DFD/DFD_original_sequences/01__podium_speech_happy.mp4"),
    ("DFD_real_wall", "videos/DFD/DFD_original_sequences/01__talking_against_wall.mp4"),
    # DFD manipulated (Google DeepFake Detection)
    ("DFD_fake_wall", "videos/DFD/DFD_manipulated_sequences/DFD_manipulated_sequences/01_02__talking_against_wall__YVGY8LOK.mp4"),
    ("DFD_fake_couch", "videos/DFD/DFD_manipulated_sequences/DFD_manipulated_sequences/01_02__talking_angry_couch__YVGY8LOK.mp4"),
    # UADFV fake
    ("UADFV_fake_0", "videos/UADFV/fake/0000_fake.mp4"),
    ("UADFV_real_0", "videos/UADFV/real/0000.mp4")
]

download_dir = "/tmp/dmv_scout_videos"
os.makedirs(download_dir, exist_ok=True)

for name, rel_path in test_files:
    print(f"\nFetching {name} ({rel_path})...")
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=rel_path, repo_type="dataset", local_dir=download_dir)
        print(f"Downloaded to {local_path}, size: {os.path.getsize(local_path)} bytes")
        
        # Check audio stream via ffprobe
        probe_cmd = f"ffprobe -v error -show_entries stream=codec_type,codec_name -of json '{local_path}'"
        res = subprocess.run(probe_cmd, shell=True, capture_output=True, text=True)
        print("Streams:", res.stdout.strip())
    except Exception as e:
        print(f"Failed for {name}: {e}")
