# Model Checkpoints and Weights Distribution

This directory houses the model configurations, label mappings, training scripts, and evaluation metrics for the Digital Media Verification (DMV) pipeline.

Large binary weights (`.pt`, `.safetensors`) are **not** committed directly to the Git repository to keep the repository lightweight and portable. Instead, production model weights are distributed via project GitHub Releases.

---

## 1. Required Model Checkpoints

The multimodal inference pipeline requires three primary neural models:

| Component | Architecture | Checkpoint Designation | Local Destination Path | Format | Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Visual Deepfake Detector** | EfficientNet-B4 | `best_model.pt` (Trained on FaceForensics++ C23) | `models/deepfake/checkpoint_full/best_model.pt` | PyTorch State Dict | ~63 MB |
| **Hate Speech Detector** | BERT-base-uncased | Fine-tuned on HateXplain (3-class) | `models/hate_speech/checkpoint_full/model.safetensors` | Hugging Face Safetensors | ~388 MB |
| **Propaganda Detector** | RoBERTa-base | Phase 5 Exp 1: Class-Weighted + Scheduled Fine-tuned on SemEval-2020 Task 11 (14-class) | `models/propaganda/optimization/exp1_sqrt_weighted_scheduled/model.safetensors` | Hugging Face Safetensors | ~409 MB |

> **Note on Propaganda Taxonomy & Tokenizers:**
> The propaganda detector is fine-tuned on the SemEval-2020 Task 11 dataset and outputs 14 consolidated rhetorical technique classes (spanning Appeal to Authority, Fear/Prejudice, Bandwagon/Reductio ad Hitlerum, Black-and-White Fallacy, Causal Oversimplification, Doubt, Exaggeration/Minimisation, Flag-Waving, Loaded Language, Name Calling/Labeling, Repetition, Slogans, Thought-terminating Clichés, and Whataboutism/Straw Men/Red Herring).
> The BERT and RoBERTa tokenizers and vocabulary configuration files (`tokenizer.json`, `tokenizer_config.json`) are bundled directly alongside their respective model weights.

---

## 2. Directory Layout for Local Execution

When model weights are installed, the expected directory layout is:

```
models/
├── deepfake/
│   ├── checkpoint_full/
│   │   ├── best_model.pt               <-- PyTorch binary
│   │   ├── config.json
│   │   ├── label_mapping.json
│   │   ├── metrics.json
│   │   └── training_config.json
│   ├── efficientnet_b4.py
│   ├── predict.py
│   └── train.py
│
├── hate_speech/
│   ├── checkpoint_full/
│   │   ├── model.safetensors           <-- Safetensors binary
│   │   ├── config.json
│   │   ├── label_mapping.json
│   │   ├── metrics.json
│   │   ├── training_config.json
│   │   └── tokenizer/
│   │       ├── tokenizer.json
│   │       └── tokenizer_config.json
│   ├── bert.py
│   ├── predict.py
│   └── train.py
│
└── propaganda/
    ├── optimization/
    │   └── exp1_sqrt_weighted_scheduled/
    │       ├── model.safetensors       <-- Safetensors binary
    │       ├── config.json
    │       ├── label_mapping.json
    │       ├── metrics.json
    │       ├── training_config.json
    │       └── tokenizer/
    │           ├── tokenizer.json
    │           └── tokenizer_config.json
    ├── roberta.py
    ├── predict.py
    └── train.py
```

---

## 3. Obtaining Model Weights

### From GitHub Releases

Production model weights are distributed as downloadable `.tar.gz` release assets under release tag [`v1.0.0`](https://github.com/narendra8839/Digital-Media-Verification/releases/tag/v1.0.0) (active once published by the maintainer):

| Archive File | Size (Bytes / MB) | SHA-256 Checksum | Target Extraction Path |
| :--- | :--- | :--- | :--- |
| `deepfake_efficientnet_b4_checkpoint.tar.gz` | 65,963,478 (~62.91 MB) | `7df27eb2dfe1c39759508778a7c37eb056a1ac4c88265a3ce2e861fe225d609b` | `models/deepfake/` |
| `hate_speech_bert_checkpoint.tar.gz` | 407,151,028 (~388.29 MB) | `fd4988f1492d7a6a37fed8fa1891db17d31982fcf2252c92479d4d1b6b6fa822` | `models/hate_speech/` |
| `propaganda_roberta_exp1_checkpoint.tar.gz` | 428,494,811 (~408.64 MB) | `f4cc85cafcfc16a9f6cb3ffb55e06bbc2f1bdc0cba516a4e3567964b630f3cb0` | `models/propaganda/optimization/` |

To install:
1. Navigate to the project's **Releases** tab on GitHub: [`https://github.com/narendra8839/Digital-Media-Verification/releases`](https://github.com/narendra8839/Digital-Media-Verification/releases).
2. Download the three checkpoint archives.
3. Extract each archive into its designated location:
   ```bash
   # Deepfake detector (creates models/deepfake/checkpoint_full/best_model.pt)
   tar -xzf deepfake_efficientnet_b4_checkpoint.tar.gz -C models/deepfake/

   # Hate speech detector (creates models/hate_speech/checkpoint_full/model.safetensors)
   tar -xzf hate_speech_bert_checkpoint.tar.gz -C models/hate_speech/

   # Propaganda detector (creates models/propaganda/optimization/exp1_sqrt_weighted_scheduled/model.safetensors)
   mkdir -p models/propaganda/optimization
   tar -xzf propaganda_roberta_exp1_checkpoint.tar.gz -C models/propaganda/optimization/
   ```

### Training from Scratch (Alternative)

If you prefer to train new checkpoints rather than downloading pre-trained weights:
```bash
# Train EfficientNet-B4 deepfake detector
python -m models.deepfake.train --data_dir data/faceforensics --output_dir models/deepfake/checkpoint_full

# Train BERT hate speech classifier
python -m models.hate_speech.train --data_dir data/hatexplain --output_dir models/hate_speech/checkpoint_full

# Train RoBERTa propaganda detector
python -m models.propaganda.train --data_dir data/semeval2020_task11 --output_dir models/propaganda/optimization/exp1_sqrt_weighted_scheduled
```
