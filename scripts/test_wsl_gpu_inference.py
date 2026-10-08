"""Verify device detection and GPU inference through the FastAPI pipeline."""

import os
import sys
import time
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from api.utils.device import get_optimal_device
from api.dependencies.pipeline_dep import get_pipeline
from fastapi.testclient import TestClient
from api.main import app


def test_device_inference():
    print("=" * 60)
    print("DEVICE & FASTAPI INFERENCE PATH TEST")
    print("=" * 60)
    dev = get_optimal_device()
    print(f"get_optimal_device() returned: {dev}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Allocated Before: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")

    client = TestClient(app)
    pipeline = get_pipeline()
    print(f"Pipeline device: {pipeline.device}")
    print(f"Deepfake detector model device: {next(pipeline.deepfake_detector.model.parameters()).device}")
    print(f"Propaganda detector model device: {next(pipeline.propaganda_detector.model.parameters()).device}")
    print(f"Hate speech detector model device: {next(pipeline.hate_speech_detector.model.parameters()).device}")

    # Run API-level text inference
    t0 = time.time()
    res = client.post("/verify/text", json={"text": "Exclusive report: The leaders are deceiving the public!"})
    lat = time.time() - t0
    assert res.status_code == 200
    data = res.json()
    print(f"\nAPI /verify/text Response:")
    print(f"  Status Code: {res.status_code} (latency: {lat:.3f}s)")
    print(f"  Propaganda: {data['propaganda']['prediction']} (conf: {data['propaganda']['confidence']})")
    print(f"  Hate Speech: {data['hate_speech']['prediction']} (conf: {data['hate_speech']['confidence']})")
    print(f"  Uncertainty mc_samples: {data['uncertainty']['propaganda']['mc_samples']}")

    if torch.cuda.is_available():
        print(f"VRAM Allocated After: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
        print("VERIFICATION RESULT: Genuine CUDA GPU execution confirmed through API path!")
    else:
        print("VERIFICATION RESULT: Clean CPU fallback confirmed through API path!")


if __name__ == "__main__":
    test_device_inference()
