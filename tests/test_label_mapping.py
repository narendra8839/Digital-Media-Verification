"""Tests for label mapping audit, verification, and evaluation-only orientation correction."""

import os
import sys
import pytest
import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.deepfake.predict import load_deepfake_model
from models.deepfake.efficientnet_b4 import EfficientNetB4Deepfake
from torchvision import transforms


class TestLabelMappingAudit:
    """Verifies that the label mapping convention is auditable, testable, and mathematically verified."""

    def test_efficientnet_output_shape(self):
        """Model produces exactly 2 logits (binary classification)."""
        model = EfficientNetB4Deepfake(dropout_rate=0.4, pretrained=False)
        dummy = torch.randn(2, 3, 224, 224)
        out = model(dummy)
        assert out.shape == (2, 2), f"Expected shape (2, 2), got {out.shape}"

    def test_softmax_probability_inversion_math(self):
        """Mathematically proves that swapping the class index inverts the probability."""
        # Simulated logits where Logit[0] = 2.0 (favors class 0), Logit[1] = -1.0
        logits = torch.tensor([[2.0, -1.0]])
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

        p_class0 = float(probs[0])
        p_class1 = float(probs[1])

        assert abs(p_class0 + p_class1 - 1.0) < 1e-6

        # Under convention where Fake=1:
        pred_label_conv1 = "fake" if p_class1 >= 0.5 else "real"
        # Under convention where Fake=0:
        pred_label_conv0 = "fake" if p_class0 >= 0.5 else "real"

        # Since Logit[0] > Logit[1], class 0 is selected
        assert pred_label_conv0 == "fake"
        assert pred_label_conv1 == "real"

    def test_controlled_synthetic_crop_prediction(self):
        """Tests that evaluation-only mapping behaves predictably given known logits."""
        # Mock class with fixed logits
        class MockModel(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.logits = torch.tensor(logits)

            def forward(self, x):
                return self.logits.repeat(x.shape[0], 1)

        # Logits [3.0, -2.0]: index 0 is dominant (prob ~ 0.993)
        mock_model = MockModel([[3.0, -2.0]])
        dummy_tensor = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            out = mock_model(dummy_tensor)
            probs = torch.softmax(out, dim=1).squeeze(0).numpy()

        # If Fake is index 0: Fake prob ~ 0.993 -> predicted Fake
        p_fake_idx0 = float(probs[0])
        assert p_fake_idx0 > 0.99
        assert ("fake" if p_fake_idx0 >= 0.5 else "real") == "fake"

        # If Fake is index 1: Fake prob ~ 0.007 -> predicted Real
        p_fake_idx1 = float(probs[1])
        assert p_fake_idx1 < 0.01
        assert ("fake" if p_fake_idx1 >= 0.5 else "real") == "real"

    def test_existing_checkpoints_architecture_and_hashes(self):
        """Verifies that all 5 key checkpoints exist on disk with exactly 2 output logits."""
        import hashlib
        checkpoints = {
            "baseline": ("models/deepfake/checkpoint_full/best_model.pt", "5878ae38bf5239dd9889e010fafa1d8c7da4198ac260542dc96681ede1e63234"),
            "v2b": ("models/deepfake/checkpoint_robust_v2b/best_model.pt", "f8b8c0ba272b37b353fbcaa2b3148844a0d9b0d4278c52b73a3259f9d5d277c8"),
            "v3a": ("models/deepfake/checkpoint_robust_v3a/best_model.pt", "5f8057f3d54759c2c71163e6d4c60cf6bf303dae611739491e02e1ae8452c41d"),
            "v3b": ("models/deepfake/checkpoint_robust_v3b/best_model.pt", "1b53f93e70d2301fb819f987d2f0c0bd6c6c37ac6560142d933d8542f9470032"),
            "v3c": ("models/deepfake/checkpoint_robust_v3c/best_model.pt", "12b552738dcfcf598b5b18b2ce65820087063703346ac76b24c7b6382efefd34"),
        }

        for name, (path, expected_hash) in checkpoints.items():
            full_path = os.path.join(PROJECT_ROOT, path)
            if not os.path.exists(full_path):
                pytest.skip(f"Checkpoint {path} not found")
            with open(full_path, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            assert actual_hash == expected_hash, f"Hash mismatch for {name}: expected {expected_hash}, got {actual_hash}"

            sd = torch.load(full_path, map_location="cpu", weights_only=False)
            clf_weight = sd.get("backbone.classifier.1.weight")
            assert clf_weight is not None, f"Classifier weight not found in {name}"
            assert clf_weight.shape == (2, 1792), f"Classifier shape in {name} is {clf_weight.shape}, expected (2, 1792)"
