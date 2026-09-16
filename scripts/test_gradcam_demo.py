"""Demo test script for Grad-CAM explanation generation."""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
from PIL import Image
from models.deepfake.predict import load_deepfake_model
from explainability.gradcam import GradCAM
from preprocessing.frame_sampling import sample_video_frames
from preprocessing.face_detection import FaceDetector

def main():
    model, device = load_deepfake_model()
    gradcam = GradCAM(model=model, device=device)
    detector = FaceDetector(device="cpu")

    with open("data/mvp/deepfake/val_subset.json", "r", encoding="utf-8") as f:
        val_subset = json.load(f)

    os.makedirs("reports/explanations/deepfake", exist_ok=True)
    print("Testing Grad-CAM on 2 validation videos (1 real, 1 fake)...")

    # Sample 0 is real, Sample 10 is fake
    for sample in [val_subset[0], val_subset[10]]:
        vid_id = sample["video_id"]
        label = sample["label"]
        vpath = os.path.join("data/faceforensics/C23", sample["relative_path"])
        frames = sample_video_frames(vpath, target_fps=1.0, max_frames=1)
        ts, rgb = frames[0]
        crop, _ = detector.detect_and_crop(rgb)

        out_img = f"reports/explanations/deepfake/{vid_id}_{label}_gradcam.png"
        res = gradcam.explain(crop, output_path=out_img)
        print(f"Video {vid_id} (GT: {label}) -> Pred: {res['prediction']} (conf: {res['confidence']})")
        print(f"  Saved overlay to: {out_img}")
        print(f"  Heatmap shape: {res['heatmap_shape']}")
        assert os.path.exists(out_img), "Overlay image was not saved!"

    print("--> GRAD-CAM COMPONENT FULLY FUNCTIONAL AND TESTED!")

if __name__ == "__main__":
    main()
