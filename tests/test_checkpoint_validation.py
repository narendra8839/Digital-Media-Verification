"""Tests for model checkpoint validation, missing weight prevention, and mock injection."""

import os
import sys
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.deepfake.predict import load_deepfake_model
from models.hate_speech.predict import load_hate_speech_model
from models.propaganda.predict import load_propaganda_model
from detection.deepfake_detector import DeepfakeDetector
from detection.hate_speech_detector import HateSpeechDetector
from detection.propaganda_detector import PropagandaDetector


class DummyClassifier(nn.Module):
    """Minimal mock model for testing mock injection."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, **kwargs):
        return self.fc(torch.zeros(x.shape[0], 10))


class DummyTokenizer:
    """Minimal mock tokenizer."""
    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]])
        }

    def convert_ids_to_tokens(self, ids):
        return ["[CLS]", "[SEP]"]


class TestMissingCheckpointValidation:
    """Verify that loaders fail clearly with FileNotFoundError when weights are missing."""

    def test_deepfake_loader_missing_checkpoint_raises(self, tmp_path):
        fake_path = str(tmp_path / "nonexistent_model.pt")
        with pytest.raises(FileNotFoundError) as exc_info:
            load_deepfake_model(checkpoint_path=fake_path)
        err_msg = str(exc_info.value)
        assert "Deepfake detector checkpoint not found" in err_msg
        assert "deepfake_efficientnet_b4_checkpoint.tar.gz" in err_msg

    def test_hate_speech_loader_missing_checkpoint_raises(self, tmp_path):
        fake_dir = str(tmp_path / "nonexistent_hs_dir")
        with pytest.raises(FileNotFoundError) as exc_info:
            load_hate_speech_model(checkpoint_dir=fake_dir)
        err_msg = str(exc_info.value)
        assert "Hate speech model weights not found" in err_msg
        assert "hate_speech_bert_checkpoint.tar.gz" in err_msg

    def test_propaganda_loader_missing_checkpoint_raises(self, tmp_path):
        fake_dir = str(tmp_path / "nonexistent_prop_dir")
        with pytest.raises(FileNotFoundError) as exc_info:
            load_propaganda_model(checkpoint_dir=fake_dir)
        err_msg = str(exc_info.value)
        assert "Propaganda model weights not found" in err_msg
        assert "propaganda_roberta_exp1_checkpoint.tar.gz" in err_msg

    def test_deepfake_detector_wrapper_missing_checkpoint_raises(self, tmp_path):
        fake_path = str(tmp_path / "nonexistent_model.pt")
        with pytest.raises(FileNotFoundError):
            DeepfakeDetector(checkpoint_path=fake_path)


class TestMockModelInjection:
    """Verify that unit tests can inject mock models and subcomponents without touching the filesystem."""

    def test_deepfake_detector_mock_injection(self):
        mock_model = DummyClassifier(num_classes=2)
        mock_gradcam = object()
        mock_uq = object()
        detector = DeepfakeDetector(
            model=mock_model,
            gradcam=mock_gradcam,
            uncertainty_engine=mock_uq,
            device="cpu"
        )
        assert detector.model is mock_model
        assert detector.gradcam is mock_gradcam
        assert detector.uncertainty_engine is mock_uq
        assert detector.device == "cpu"

    def test_hate_speech_detector_mock_injection(self):
        mock_model = DummyClassifier(num_classes=3)
        mock_tok = DummyTokenizer()
        mock_explainer = object()
        mock_uq = object()
        detector = HateSpeechDetector(
            model=mock_model,
            tokenizer=mock_tok,
            explainer=mock_explainer,
            uncertainty_engine=mock_uq,
            device="cpu"
        )
        assert detector.model is mock_model
        assert detector.tokenizer is mock_tok
        assert detector.explainer is mock_explainer
        assert detector.uncertainty_engine is mock_uq
        assert detector.device == "cpu"

    def test_propaganda_detector_mock_injection(self):
        mock_model = DummyClassifier(num_classes=14)
        mock_tok = DummyTokenizer()
        mock_explainer = object()
        mock_uq = object()
        detector = PropagandaDetector(
            model=mock_model,
            tokenizer=mock_tok,
            explainer=mock_explainer,
            uncertainty_engine=mock_uq,
            device="cpu"
        )
        assert detector.model is mock_model
        assert detector.tokenizer is mock_tok
        assert detector.explainer is mock_explainer
        assert detector.uncertainty_engine is mock_uq
        assert detector.device == "cpu"
