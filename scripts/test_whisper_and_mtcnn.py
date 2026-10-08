import os
import torch
import whisper
import cv2
from facenet_pytorch import MTCNN

video_dir = "/mnt/d/DMV_Data/DMV_Demo_Assets/candidate_videos"
videos = [
    "A_real_real.mp4",
    "B_real_fakeaudio.mp4",
    "C_fakevideo_realaudio.mp4",
    "D_fakevideo_fakeaudio.mp4"
]

print("Loading Whisper model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)

mtcnn = MTCNN(keep_all=False, device=device)

print("\n=== Testing Whisper and MTCNN on FakeAVCeleb Samples ===")
for v in videos:
    vpath = os.path.join(video_dir, v)
    print(f"\n--- Video: {v} ---")
    
    # Check face detection on first few frames
    cap = cv2.VideoCapture(vpath)
    face_count = 0
    total_sampled = 0
    while cap.isOpened() and total_sampled < 10:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)
        if boxes is not None and len(boxes) > 0:
            face_count += 1
        total_sampled += 1
    cap.release()
    print(f"Face detected in {face_count}/{total_sampled} sampled frames")
    
    # Transcribe with Whisper
    res = whisper_model.transcribe(vpath)
    text = res.get("text", "").strip()
    print(f"Whisper Transcript: \"{text}\"")
    print(f"Detected language: {res.get('language')}")
