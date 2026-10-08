# Model Checkpoints and Weights Distribution

This directory houses the model configurations, label mappings, training scripts, and evaluation metrics for the Digital Media Verification (DMV) pipeline.

Large binary weights (`.pt`, `.safetensors`) are **not** committed directly to the Git repository to keep the repository lightweight and portable. Instead, production model weights are distributed via project GitHub Releases.

---

## 1. Required Model Checkpoints

The multimodal inference pipeline requires three primary neural models:

| Component | Architecture | Checkpoint Designation | Local Destination Path | Format | Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Visual Deepfake Detector** | EfficientNet-B4 | `best_model.pt` (Trained on FaceForensics++ C23) | `models/deepfake/checkpoint_full/best_model.pt` | PyTorch State Dict | ~68 MB |
| **Hate Speech Detector** | BERT-base-uncased | Fine-tuned on HateXplain (3-class) | `models/hate_speech/checkpoint_full/model.safetensors` | Hugging Face Safetensors | ~418 MB |
| **Propaganda Detector** | RoBERTa-base | Phase 5 Exp 1: Class-Weighted + Scheduled Fine-tuned on SemEval-2020 Task 11 (9-class) | `models/propaganda/optimization/exp1_sqrt_weighted_scheduled/model.safetensors` | Hugging Face Safetensors | ~476 MB |

> **Note on Tokenizers:**  
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

When publishing the project, the release assets are packaged as:
- `deepfake_efficientnet_b4_checkpoint.tar.gz`
- `hate_speech_bert_checkpoint.tar.gz`
- `propaganda_roberta_exp1_checkpoint.tar.gz`

To install:
1. Navigate to the project's **Releases** tab on GitHub (`https://github.com/<owner>/Digital-Media-Verification/releases`).
2. Download the checkpoint archive corresponding to each model.
3. Extract each archive into its respective destination directory:
   ```bash
   # Deepfake detector
   tar -xzf deepfake_efficientnet_b4_checkpoint.tar.gz -C models/deepfake/

   # Hate speech detector
   tar -xzf hate_speech_bert_checkpoint.tar.gz -C models/hate_speech/

   # Propaganda detector
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
