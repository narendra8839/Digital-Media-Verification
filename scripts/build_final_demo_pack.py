import os
import shutil
import subprocess

OUT_DIR = "/mnt/d/DMV_Data/DMV_Demo_Assets/final_video_demo"
SRC_DIR = "/mnt/d/DMV_Data/DMV_Demo_Assets/candidate_videos"
AUDIO_DIR = "/mnt/d/DMV_Data/DMV_Demo_Assets"

os.makedirs(OUT_DIR, exist_ok=True)

# 1. 01_real_normal.mp4
print("Creating 01_real_normal.mp4...")
src_01 = os.path.join(SRC_DIR, "A_real_real.mp4")
dst_01 = os.path.join(OUT_DIR, "01_real_normal.mp4")
shutil.copy2(src_01, dst_01)
print(f"  01 saved: {os.path.getsize(dst_01):,} bytes")

# 2. 02_external_deepfake_normal.mp4
print("Creating 02_external_deepfake_normal.mp4...")
src_02 = os.path.join(SRC_DIR, "C_fakevideo_realaudio.mp4")
dst_02 = os.path.join(OUT_DIR, "02_external_deepfake_normal.mp4")
shutil.copy2(src_02, dst_02)
print(f"  02 saved: {os.path.getsize(dst_02):,} bytes")

# 3. 03_external_deepfake_propaganda.mp4
print("Creating 03_external_deepfake_propaganda.mp4...")
dst_03 = os.path.join(OUT_DIR, "03_external_deepfake_propaganda.mp4")
audio_prop = os.path.join(AUDIO_DIR, "propaganda_16k.wav")
cmd_03 = [
    "ffmpeg", "-y",
    "-i", src_02,
    "-i", audio_prop,
    "-c:v", "copy",
    "-c:a", "aac",
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-shortest",
    dst_03
]
subprocess.run(cmd_03, check=True, capture_output=True)
print(f"  03 saved: {os.path.getsize(dst_03):,} bytes")

# 4. 04_external_deepfake_hate.mp4
print("Creating 04_external_deepfake_hate.mp4...")
dst_04 = os.path.join(OUT_DIR, "04_external_deepfake_hate.mp4")
audio_hate = os.path.join(AUDIO_DIR, "hate_16k.wav")
cmd_04 = [
    "ffmpeg", "-y",
    "-i", src_02,
    "-i", audio_hate,
    "-c:v", "copy",
    "-c:a", "aac",
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-shortest",
    dst_04
]
subprocess.run(cmd_04, check=True, capture_output=True)
print(f"  04 saved: {os.path.getsize(dst_04):,} bytes")

# 5. 05_real_propaganda.mp4
print("Creating 05_real_propaganda.mp4...")
dst_05 = os.path.join(OUT_DIR, "05_real_propaganda.mp4")
cmd_05 = [
    "ffmpeg", "-y",
    "-i", src_01,
    "-i", audio_prop,
    "-c:v", "copy",
    "-c:a", "aac",
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-shortest",
    dst_05
]
subprocess.run(cmd_05, check=True, capture_output=True)
print(f"  05 saved: {os.path.getsize(dst_05):,} bytes")

print("\n--- Summary of Final Files in", OUT_DIR, "---")
for f in sorted(os.listdir(OUT_DIR)):
    fp = os.path.join(OUT_DIR, f)
    print(f"  {f} ({os.path.getsize(fp):,} bytes)")
