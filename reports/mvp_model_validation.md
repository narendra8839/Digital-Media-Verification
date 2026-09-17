# MVP Model Validation & Training Sanity Check Report

## Executive Summary
This document provides the formal validation and sanity check for the three MVP models fine-tuned for the Ethical and Responsible AI course project (**Digital Media Verification: Explainable and Uncertainty-Aware Deepfake, Propaganda and Hate-Speech Detection**).

All model architectures, weights, gradients, data splits, and checkpoint serialization mechanisms have been systematically inspected and validated on disk.

---

## 1. Deepfake Model Sanity Check (EfficientNet-B4)

### Model & Pretrained Base
- **Architecture**: `EfficientNet-B4` (`torchvision.models.efficientnet_b4`) with custom binary classification head (`Dropout(p=0.4) -> Linear(1792, 2)`).
- **Initialization**: Verified initialized from ImageNet pretrained weights (`EfficientNet_B4_Weights.DEFAULT`). Weight difference norm between random init and pretrained is `58.54`.
- **Classification Head**: All 418 parameter tensors had `requires_grad=True`; the optimizer received all model parameters and updated the classification head.

### Dataset & Provenance
- **Dataset**: FaceForensics++ C23.
- **Provenance Statement**: *"Development dataset sourced from unofficial Kaggle mirror; used for MVP engineering and pipeline validation. Official FaceForensics++ access is pending/being pursued."*
- **Subset Count**:
  - Train: 40 videos (20 Real, 20 Fake across Deepfakes, Face2Face, FaceSwap, NeuralTextures).
  - Validation: 18 videos (10 Real, 8 Fake).
  - Test: 18 videos (10 Real, 8 Fake).
- **Sequence Disjointness**: Verified 0 video ID overlap between Train, Validation, and Test.

### Frame Extraction & Face Detection Rate
- **Sampling**: 4 frames per video sampled at 1.0 FPS.
- **Face Detection Engine**: `facenet-pytorch` MTCNN (`keep_all=False, select_largest=True`).
- **Detection Rate**:
  - Train: **160 / 160 frames (100.0%)** detected actual faces; 0 fallbacks (0.0%).
  - Validation: **72 / 72 frames (100.0%)** detected actual faces; 0 fallbacks (0.0%).
  - Test: **72 / 72 frames (100.0%)** detected actual faces; 0 fallbacks (0.0%).
- **Training Frames**: Exactly **160 frames** (80 Real, 80 Fake; 1:1 balanced).

### Training Setup & Verification
- **Epochs**: 2
- **Batch Size**: 8
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW (weight decay 1e-2)
- **Loss Function**: `nn.CrossEntropyLoss()`
- **Label Mapping**: `{"0": "real", "1": "fake"}` (verified aligned with `label_id`).
- **Video Aggregation**: Mean fake probability across sampled frames; thresholded at `>= 0.5`.

### Actual Verified Metrics
- **Training Set (Best Checkpoint Evaluation)**:
  - Frame-Level: Accuracy: **95.63%**, Macro F1: **0.9562**
  - Video-Level: Accuracy: **97.50%** (39/40 correct), Macro F1: **0.9750**
  - Confusion Matrix: `[[19, 1], [0, 20]]`
- **Validation Set (Unseen Videos)**:
  - Frame-Level: Accuracy: **54.17%**, Macro F1: **0.5394**
  - Video-Level: Accuracy: **55.56%** (10/18 correct), Macro F1: **0.5500**
  - Confusion Matrix: `[[4, 6], [2, 6]]`
- **Test Set (Unseen Videos)**:
  - Frame-Level: Accuracy: **58.33%**, Macro F1: **0.5781**
  - Video-Level: Accuracy: **50.00%** (9/18 correct), Macro F1: **0.4857**
  - Confusion Matrix: `[[6, 4], [5, 3]]`

### Analysis of Weak Generalization
The low generalization score (55.56% val, 50.00% test) is a **genuine, unmanipulated MVP result**:
1. The model fitted the training distribution rapidly (97.5% training accuracy), indicating that the classification head, gradients, and loss function operated correctly.
2. An 19-million parameter network trained on only 160 images (40 identities) without heavy data augmentation naturally memorizes source identity artifacts rather than subtle compression artifacts.
3. This is an authentic limitation of the MVP scale and should be reported honestly without artificial metrics.

---

## 2. SemEval Model Sanity Check (RoBERTa 14-Class)

### Model & Pretrained Base
- **Architecture**: `RoBERTa-base` (`AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=14)`).
- **Classification Head**: Output dimension = 14 (`out_proj.out_features = 14`).
- **Tokenizer**: `AutoTokenizer.from_pretrained("roberta-base")` with max length 128 and truncation.

### Dataset & Subsets
- **Dataset**: SemEval-2020 Task 11 (Propaganda Techniques Corpus).
- **Definition of One Sample**: A **propaganda text span** extracted from a news article, accompanied by its surrounding sentence `context`, character offsets (`start_char`, `end_char`), and `article_id`.
- **Target Label**: One of 14 fine-grained propaganda technique names.
- **Sample Count**:
  - Train: 560 samples (balanced: exactly 40 per technique).
  - Validation: 140 samples (balanced: exactly 10 per technique).
- **Article-Level Split**:
  - Train articles: 195 unique articles.
  - Validation articles: 50 unique articles.
  - Article overlap between Train and Validation: **0 (Strict Article-Level Isolation)**.
- **Duplicate Check**:
  - 548 out of 560 training spans are completely unique. The few recurring phrases correspond to brief colloquial expressions (e.g. repeated slogans).

### Label Mapping (Exact 14 Techniques)
```json
{
  "0": "Appeal_to_Authority",
  "1": "Appeal_to_fear-prejudice",
  "2": "Bandwagon,Reductio_ad_hitlerum",
  "3": "Black-and-White_Fallacy",
  "4": "Causal_Oversimplification",
  "5": "Doubt",
  "6": "Exaggeration,Minimisation",
  "7": "Flag-Waving",
  "8": "Loaded_Language",
  "9": "Name_Calling,Labeling",
  "10": "Repetition",
  "11": "Slogans",
  "12": "Thought-terminating_Cliches",
  "13": "Whataboutism,Straw_Men,Red_Herring"
}
```

### Training Setup & Verification
- **Epochs**: 2
- **Batch Size**: 16
- **Learning Rate**: 3e-5
- **Optimizer**: AdamW (weight decay 0.01)

### Actual Verified Metrics
- **Training Set (Best Checkpoint Evaluation)**:
  - Accuracy: **47.68%**, Macro F1: **0.4619** (substantially learned above random 14-class guessing baseline of 7.14%).
- **Validation Set (Unseen Articles)**:
  - Accuracy: **29.29%**, Macro F1: **0.2481**, Weighted F1: **0.2481**.

---

## 3. HateXplain Model Sanity Check (BERT 3-Class)

### Model & Pretrained Base
- **Architecture**: `BERT-base-uncased` (`AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)`).
- **Output Classes**: Exactly 3 classes (`0: normal`, `1: offensive`, `2: hatespeech`).
- **Tokenizer**: WordPiece tokenizer with max length 128 and truncation.

### Dataset & Subsets
- **Dataset**: HateXplain benchmark.
- **Sample Count**:
  - Train: 600 samples (200 normal, 200 offensive, 200 hatespeech).
  - Validation: 150 samples (50 normal, 50 offensive, 50 hatespeech).
  - Test: 150 samples (50 normal, 50 offensive, 50 hatespeech).
- **Benchmark Split Preservation**: Verified that 100% of train, val, and test posts strictly belong to their respective official benchmark divisions (`post_id_divisions.json`). Zero cross-split leakage.

### Ground-Truth Rationale Preservation
- **Rationale Status**: Retained on 100% of samples across all three subsets:
  - Train: 600 / 600 samples have rationales preserved.
  - Validation: 150 / 150 samples have rationales preserved.
  - Test: 150 / 150 samples have rationales preserved.
- Token-level binary rationale masks match the length of `post_tokens` for all annotated posts.

### Training Setup & Verification
- **Epochs**: 2
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW (weight decay 0.01)

### Actual Verified Metrics
- **Training Set (Best Checkpoint Evaluation)**:
  - Accuracy: **81.33%**, Macro F1: **0.8067** (vs 33.33% random baseline).
- **Validation Set**:
  - Accuracy: **60.67%**, Macro F1: **0.5942**, Weighted F1: **0.5942**.
- **Test Set**:
  - Accuracy: **60.00%**, Macro F1: **0.5773**, Weighted F1: **0.5773**.

---

## 4. Checkpoint Independent Reload & Output Dimension Verification

All three checkpoints were reloaded from disk and evaluated using standalone scripts independent of the training pipeline:

| Modality | Checkpoint Path | Expected Dim | Verified Output Shape | Probability Sum | Decoded Labels | Reload Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Deepfake** | `models/deepfake/checkpoint/best_model.pt` | 2 | `(1, 2)` | 1.0000 | `real`, `fake` | **PASSED** |
| **Propaganda** | `models/propaganda/checkpoint/` | 14 | `(1, 14)` | 1.0000 | 14 Official Techniques | **PASSED** |
| **Hate Speech** | `models/hate_speech/checkpoint/` | 3 | `(1, 3)` | 1.0000 | `normal`, `offensive`, `hatespeech` | **PASSED** |

---

## 5. Reproducibility & Integrity Summary
- **Random Seed**: 42 across all data sampling and model training scripts.
- **Leakage Status**:
  - Deepfake: 0 sequence ID overlap across train, val, and test.
  - SemEval: 0 article ID overlap between train and val.
  - HateXplain: Official benchmark boundaries 100% intact.
- **Readiness**: All models produce valid logits, probability distributions, and classification outputs. The checkpoints are ready for the explainability (Grad-CAM, token rationales) and uncertainty quantification (MC Dropout) phase.
