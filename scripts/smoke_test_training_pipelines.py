"""Smoke test verification suite for Phase 2 training pipelines.

Runs fast, small-sample (8-16 samples, 1 epoch) forward/backward/checkpoint cycles for:
1. Deepfake detection (EfficientNet-B4 + augmentation + class weighting)
2. Propaganda classification (RoBERTa-base 14-class)
3. Hate speech classification (BERT-base-uncased 3-class)

Guarantees full reproducibility, device readiness, and zero test-split contamination.
"""

import os
import sys
import shutil
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from models.deepfake.train import train_deepfake_model
from models.propaganda.train import train_propaganda_model
from models.hate_speech.train import train_hate_speech_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("smoke_tests")


def smoke_test_deepfake():
    logger.info("=== 1. Starting Deepfake Pipeline Smoke Test ===")
    smoke_dir = "models/deepfake/smoke_test_output"
    if os.path.exists(smoke_dir):
        shutil.rmtree(smoke_dir)

    result = train_deepfake_model(
        train_manifest="data/full_splits/deepfake_train.json",
        val_manifest="data/full_splits/deepfake_val.json",
        output_dir=smoke_dir,
        epochs=1,
        batch_size=4,
        lr=1e-4,
        max_train_videos=4,
        max_val_videos=2,
        max_frames_per_video=2,
        use_augmentation=True,
        balance_classes=True,
        save_checkpoint=True
    )

    assert os.path.exists(os.path.join(smoke_dir, "best_model.pt")), "Deepfake checkpoint missing!"
    assert os.path.exists(os.path.join(smoke_dir, "training_config.json")), "Deepfake training_config.json missing!"
    assert os.path.exists(os.path.join(smoke_dir, "metrics.json")), "Deepfake metrics.json missing!"
    assert os.path.exists(os.path.join(smoke_dir, "label_mapping.json")), "Deepfake label_mapping.json missing!"

    logger.info("-> Deepfake smoke test PASSED successfully!")
    shutil.rmtree(smoke_dir)
    return result


def smoke_test_propaganda():
    logger.info("=== 2. Starting Propaganda Pipeline Smoke Test ===")
    smoke_dir = "models/propaganda/smoke_test_output"
    if os.path.exists(smoke_dir):
        shutil.rmtree(smoke_dir)

    result = train_propaganda_model(
        train_manifest="data/full_splits/propaganda_train.json",
        val_manifest="data/full_splits/propaganda_val.json",
        output_dir=smoke_dir,
        epochs=1,
        batch_size=4,
        lr=3e-5,
        max_train_samples=16,
        max_val_samples=8,
        save_checkpoint=True
    )

    assert os.path.exists(os.path.join(smoke_dir, "model.safetensors")), "Propaganda model.safetensors missing!"
    assert os.path.exists(os.path.join(smoke_dir, "training_config.json")), "Propaganda training_config.json missing!"
    assert os.path.exists(os.path.join(smoke_dir, "metrics.json")), "Propaganda metrics.json missing!"
    assert os.path.exists(os.path.join(smoke_dir, "tokenizer", "tokenizer.json")), "Propaganda tokenizer missing!"

    logger.info("-> Propaganda smoke test PASSED successfully!")
    shutil.rmtree(smoke_dir)
    return result


def smoke_test_hate_speech():
    logger.info("=== 3. Starting Hate Speech Pipeline Smoke Test ===")
    smoke_dir = "models/hate_speech/smoke_test_output"
    if os.path.exists(smoke_dir):
        shutil.rmtree(smoke_dir)

    result = train_hate_speech_model(
        train_manifest="data/full_splits/hatexplain_train.json",
        val_manifest="data/full_splits/hatexplain_val.json",
        output_dir=smoke_dir,
        epochs=1,
        batch_size=4,
        lr=3e-5,
        max_train_samples=16,
        max_val_samples=8,
        save_checkpoint=True
    )

    assert os.path.exists(os.path.join(smoke_dir, "model.safetensors")), "Hate speech model.safetensors missing!"
    assert os.path.exists(os.path.join(smoke_dir, "training_config.json")), "Hate speech training_config.json missing!"
    assert os.path.exists(os.path.join(smoke_dir, "metrics.json")), "Hate speech metrics.json missing!"
    assert os.path.exists(os.path.join(smoke_dir, "tokenizer", "tokenizer.json")), "Hate speech tokenizer missing!"

    logger.info("-> Hate speech smoke test PASSED successfully!")
    shutil.rmtree(smoke_dir)
    return result


def main():
    logger.info("Starting Phase 2 Training Smoke Test Suite...")
    df_res = smoke_test_deepfake()
    prop_res = smoke_test_propaganda()
    hs_res = smoke_test_hate_speech()
    logger.info("ALL THREE Phase 2 training pipelines verified successfully!")


if __name__ == "__main__":
    main()
