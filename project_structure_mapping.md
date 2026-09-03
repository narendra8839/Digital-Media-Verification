# How the Project Structure Maps to the Report

## Project Title

**Digital Media Verification: Explainable and Uncertainty-Aware Deepfake, Propaganda and Hate-Speech Detection**

This document maps the proposed software project structure to the architecture and methodology described in the Ethical and Responsible AI project synopsis.

---

## 1. Overall System Architecture

The system follows a multi-modal, multi-branch workflow:

```text
                ┌──────────────────────┐
                │    USER UPLOAD       │
                │ Image / Video / Text │
                └──────────┬───────────┘
                           │
                    ┌──────▼──────┐
                    │    ROUTER   │
                    └──────┬──────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
   ┌──────────────────┐        ┌──────────────────┐
   │ VISUAL FORENSICS │        │ TEXT EXTRACTION  │
   │                  │        │                  │
   │ Frame Sampling   │        │ EasyOCR          │
   │ MTCNN            │        │ Whisper          │
   │ EfficientNet-B4  │        │ Text Aggregator  │
   │ Grad-CAM         │        │                  │
   └────────┬─────────┘        └────────┬─────────┘
            │                           │
            │                           ▼
            │                 ┌────────────────────┐
            │                 │   NLP ANALYTICS    │
            │                 │                    │
            │                 │ RoBERTa            │
            │                 │ Propaganda         │
            │                 │                    │
            │                 │ BERT               │
            │                 │ Hate Speech        │
            │                 └─────────┬──────────┘
            │                           │
            └─────────────┬─────────────┘
                          ▼
              ┌─────────────────────────┐
              │ UNCERTAINTY ENGINE      │
              │                         │
              │ MC Dropout              │
              │ T = 20 passes           │
              │ Variance Calculation    │
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │ HUMAN REVIEW│
                    │  DASHBOARD   │
                    │             │
                    │ Predictions │
                    │ Explanations│
                    │ Uncertainty │
                    │ Audit Log   │
                    └─────────────┘
```

The architecture follows the system workflow specified in the project synopsis: user upload ingestion, visual forensics, text extraction, NLP analytics, uncertainty and governance, and a human-in-the-loop dashboard.

---

# 2. Project Folder Structure

```text
digital-media-verification/
│
├── app.py                         # Main Streamlit application
│
├── config/
│   ├── config.yaml                # Model/configuration settings
│   └── thresholds.yaml            # Uncertainty thresholds
│
├── data/
│   ├── raw/
│   │   ├── images/
│   │   ├── videos/
│   │   └── text/
│   │
│   ├── processed/
│   │   ├── faces/
│   │   ├── frames/
│   │   └── extracted_text/
│   │
│   └── datasets/
│       ├── faceforensics/
│       ├── semeval/
│       └── hatexplain/
│
├── models/
│   ├── deepfake/
│   │   ├── efficientnet_b4.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── checkpoint/
│   │
│   ├── propaganda/
│   │   ├── roberta.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── checkpoint/
│   │
│   └── hate_speech/
│       ├── bert.py
│       ├── train.py
│       ├── predict.py
│       └── checkpoint/
│
├── preprocessing/
│   ├── image_preprocessing.py
│   ├── video_preprocessing.py
│   ├── face_detection.py          # MTCNN
│   ├── frame_sampling.py
│   └── text_cleaning.py
│
├── extraction/
│   ├── ocr.py                     # EasyOCR
│   ├── asr.py                     # Whisper
│   ├── audio_extraction.py        # FFmpeg
│   └── text_aggregator.py
│
├── detection/
│   ├── deepfake_detector.py
│   ├── propaganda_detector.py
│   └── hate_speech_detector.py
│
├── explainability/
│   ├── gradcam.py
│   ├── text_attention.py
│   └── explanation_generator.py
│
├── uncertainty/
│   ├── mc_dropout.py
│   ├── variance.py
│   └── uncertainty_engine.py
│
├── governance/
│   ├── human_review.py
│   ├── audit_logger.py
│   ├── safety_checks.py
│   └── decision_policy.py
│
├── pipeline/
│   ├── router.py
│   ├── image_pipeline.py
│   ├── video_pipeline.py
│   ├── text_pipeline.py
│   └── multimodal_pipeline.py
│
├── dashboard/
│   ├── upload.py
│   ├── results.py
│   ├── visualizations.py
│   └── review_panel.py
│
├── evaluation/
│   ├── evaluate_deepfake.py
│   ├── evaluate_propaganda.py
│   ├── evaluate_hate_speech.py
│   ├── calibration.py
│   └── metrics.py
│
├── utils/
│   ├── logger.py
│   ├── file_utils.py
│   └── common.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 3. Mapping the Structure to the Report

## 3.1 User Upload Ingestion

### Report Requirement

The system accepts:

- Image
- Video
- User Caption

A media stream router separates visual and textual channels.

### Project Implementation

```text
app.py
   ↓
pipeline/router.py
   ↓
Image / Video / Text Pipeline
```

### Relevant Files

```text
app.py
pipeline/router.py
pipeline/image_pipeline.py
pipeline/video_pipeline.py
pipeline/text_pipeline.py
```

---

# 4. Visual Forensics Channel

The visual branch is responsible for detecting facial manipulation.

The report specifies:

- Video frame sampling at 1 frame per second
- MTCNN for face detection and cropping
- EfficientNet-B4 for deepfake classification
- Maximum probability pooling for video-level aggregation
- Grad-CAM for spatial explanation

## Flow

```text
Image / Video
      ↓
Frame Sampling
      ↓
MTCNN Face Detection
      ↓
Face Crop
      ↓
EfficientNet-B4
      ↓
Real / Fake Probability
      ↓
Grad-CAM
      ↓
Facial Heatmap
```

## Project Mapping

```text
preprocessing/
├── video_preprocessing.py
├── frame_sampling.py
└── face_detection.py

models/deepfake/
├── efficientnet_b4.py
├── train.py
└── predict.py

detection/
└── deepfake_detector.py

explainability/
└── gradcam.py
```

---

# 5. Image Analysis Pipeline

For an uploaded image:

```text
IMAGE
  │
  ├──────────────► MTCNN
  │                   │
  │                   ▼
  │              Face Crop
  │                   │
  │                   ▼
  │             EfficientNet-B4
  │                   │
  │                   ▼
  │             Real / Fake
  │                   │
  │                   ▼
  │                Grad-CAM
  │                   │
  │                   ▼
  │              Heatmap Output
  │
  └──────────────► EasyOCR
                       │
                       ▼
                 Extracted Text
                       │
              ┌────────┴────────┐
              ▼                 ▼
           RoBERTa             BERT
              │                 │
              ▼                 ▼
         Propaganda         Hate Speech
```

---

# 6. Video Analysis Pipeline

For videos, the report specifies sampling frames at **1 frame per second** rather than processing every frame.

## Visual Path

```text
Video
  ↓
Sample 1 Frame / Second
  ↓
MTCNN Face Detection
  ↓
Face Cropping
  ↓
EfficientNet-B4
  ↓
Frame-Level Probabilities
  ↓
Maximum Probability Pooling
  ↓
Overall Video Score
```

## Text Extraction Path

```text
Video
 │
 ├──► Video Frames ──► EasyOCR
 │                         │
 │                         ▼
 │                    OCR Text
 │
 └──► Audio ──► FFmpeg ──► Whisper
                              │
                              ▼
                         ASR Transcript
```

## Text Combination

```text
OCR Text
    +
ASR Transcript
    +
User Caption
    ↓
Text Payload Aggregator
    ↓
┌───────────────┐
│               │
▼               ▼
RoBERTa        BERT
│               │
▼               ▼
Propaganda    Hate Speech
```

---

# 7. Text Extraction Layer

The project extracts textual information from multiple sources.

## OCR

**EasyOCR** extracts visible text from:

- Images
- Video keyframes
- Memes
- Headlines
- On-screen overlays

For videos, consecutive OCR results should be deduplicated using string similarity checks.

### Files

```text
extraction/
├── ocr.py
└── text_aggregator.py
```

## ASR

The audio path uses:

```text
Video
 ↓
FFmpeg
 ↓
Audio
 ↓
Whisper-base
 ↓
Transcript
```

### Files

```text
extraction/
├── audio_extraction.py
└── asr.py
```

## Null Text Handling

If no OCR text, speech, or user caption is available:

```text
No textual content available for
propaganda/hate-speech analysis.
```

The NLP modules should be skipped instead of attempting to infer nonexistent text.

---

# 8. NLP Analytics Channel

The NLP branch contains two independent models.

## 8.1 Propaganda Detection

The project uses:

```text
RoBERTa-base
      ↓
SemEval-2020 Task 11
      ↓
14 Propaganda Techniques
      ↓
Technique Label
      +
Highlighted Text Span
```

Examples of techniques include:

- Loaded Language
- Appeal to Fear

### Files

```text
models/propaganda/
├── roberta.py
├── train.py
└── predict.py

detection/
└── propaganda_detector.py
```

---

# 9. Hate-Speech Detection

The project uses:

```text
BERT-base
    ↓
HateXplain
    ↓
Hate Speech
Offensive
Normal
```

The system also uses human rationale annotations to support explainability.

### Files

```text
models/hate_speech/
├── bert.py
├── train.py
└── predict.py

detection/
└── hate_speech_detector.py
```

---

# 10. Explainability Layer

Explainability is a central part of the project.

```text
explainability/
│
├── gradcam.py
├── text_attention.py
└── explanation_generator.py
```

## 10.1 Deepfake Explainability

```text
EfficientNet Prediction
          ↓
       Grad-CAM
          ↓
 Facial Heatmap
          ↓
Regions influencing prediction
```

Example:

```text
Prediction: LIKELY FAKE
Probability: 91%

[Face Image + Grad-CAM Heatmap]

Important region:
Facial region highlighted by the model
```

The heatmap is a **post-hoc feature attribution**, not absolute proof that manipulation occurred.

---

# 11. Text Explainability

For text models:

```text
Text
 ↓
RoBERTa / BERT
 ↓
Influential Tokens
 ↓
Highlighted Words / Spans
```

Example:

```text
"This is a TERRIFYING threat to our country."

          ^^^^^^^^^
       highlighted span
```

The explanation should communicate which words or spans influenced the model's classification.

---

# 12. Uncertainty Engine

The uncertainty module is one of the main Responsible AI components.

```text
uncertainty/
│
├── mc_dropout.py
├── variance.py
└── uncertainty_engine.py
```

The report specifies:

```text
Dropout probability = 0.2
Monte Carlo passes = 20
```

## Process

```text
Input
  │
  ├──► Prediction 1
  ├──► Prediction 2
  ├──► Prediction 3
  │
  │       ...
  │
  └──► Prediction 20
          │
          ▼
    Mean + Variance
          │
     ┌────┴─────┐
     │          │
Low Variance  High Variance
     │          │
     ▼          ▼
 CONFIDENT   INCONCLUSIVE
             HUMAN REVIEW
```

The predictive mean and epistemic variance are calculated from the stochastic forward passes.

---

# 13. Governance Layer

The governance layer implements the Ethical and Responsible AI requirements.

```text
governance/
│
├── human_review.py
├── audit_logger.py
├── safety_checks.py
└── decision_policy.py
```

## Decision Policy

```text
IF uncertainty < threshold
        ↓
CONFIDENT PREDICTION

IF uncertainty > threshold
        ↓
INCONCLUSIVE
        ↓
HUMAN REVIEW RECOMMENDED
```

The system should remain a **decision-support tool**, not an autonomous authority.

It should NOT automatically:

```text
Fake → Delete Content
Hate → Ban User
Propaganda → Censor Content
```

Final decisions remain with human reviewers.

---

# 14. Human-in-the-Loop Dashboard

The Streamlit dashboard should provide:

```text
┌───────────────────────────────────────────┐
│       DIGITAL MEDIA VERIFICATION          │
├───────────────────────────────────────────┤
│                                           │
│ Upload Image / Video / Text               │
│                                           │
│             [ Upload ]                    │
│                                           │
└───────────────────────────────────────────┘
```

After processing:

```text
┌───────────────────────────────────────────┐
│ DEEPFAKE ANALYSIS                         │
│                                           │
│ Prediction: LIKELY FAKE                   │
│ Probability: 91%                         │
│ Uncertainty: LOW                          │
│                                           │
│ [Original Face] [Grad-CAM Heatmap]        │
└───────────────────────────────────────────┘
```

Text results:

```text
┌───────────────────────────────────────────┐
│ TEXT ANALYSIS                             │
│                                           │
│ Extracted Text:                           │
│ "This is a terrifying threat..."          │
│                                           │
│ Propaganda: Appeal to Fear                │
│ Hate Speech: Offensive                    │
│                                           │
│ [Highlighted Text]                        │
└───────────────────────────────────────────┘
```

Uncertainty:

```text
┌───────────────────────────────────────────┐
│ MODEL RELIABILITY                         │
│                                           │
│ Confidence: 91%                           │
│ Uncertainty: LOW                          │
│                                           │
│ Status: CONFIDENT PREDICTION              │
│                                           │
│ [Accept] [Reject] [Mark Inconclusive]     │
└───────────────────────────────────────────┘
```

For high uncertainty:

```text
⚠ INCONCLUSIVE

The model is uncertain about this prediction.

HUMAN REVIEW RECOMMENDED
```

---

# 15. Audit Logging

Every analysis should generate an audit record containing relevant information such as:

```text
{
    "input_metadata": "...",
    "predictions": "...",
    "confidence_scores": "...",
    "uncertainty_variance": "...",
    "xai_information": "...",
    "human_reviewer": "...",
    "review_decision": "..."
}
```

### Implementation

```text
governance/audit_logger.py
```

This supports accountability and traceability.

---

# 16. Evaluation Structure

The project should include a separate evaluation module:

```text
evaluation/
│
├── evaluate_deepfake.py
├── evaluate_propaganda.py
├── evaluate_hate_speech.py
├── calibration.py
└── metrics.py
```

The evaluation should report:

- Precision
- Recall
- F1-score
- Calibration performance
- Performance across compression levels
- Performance on out-of-distribution samples

The benchmark datasets specified in the report include:

```text
FaceForensics++ c23
SemEval-2020 Task 11
HateXplain
```

---

# 17. Complete Data Flow

The complete system can be represented as:

```text
                         USER
                          │
                          ▼
                 ┌─────────────────┐
                 │   MEDIA UPLOAD  │
                 │ Image / Video / │
                 │     Caption     │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  STREAM ROUTER  │
                 └────────┬────────┘
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
     ┌───────────────┐         ┌────────────────┐
     │ VISUAL BRANCH │         │ TEXT EXTRACTION│
     └───────┬───────┘         └───────┬────────┘
             │                         │
             ▼                         ▼
          MTCNN                  EasyOCR / Whisper
             │                         │
             ▼                         ▼
      EfficientNet-B4             Text Payload
             │                         │
             ▼                  ┌──────┴───────┐
       Deepfake Score            │              │
             │                 RoBERTa         BERT
             │                   │              │
             │                   ▼              ▼
             │              Propaganda     Hate Speech
             │                   │              │
             └───────────┬───────┴──────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ EXPLAINABLE  │
                  │    AI       │
                  │             │
                  │ Grad-CAM    │
                  │ Text Span   │
                  │ Highlights  │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ UNCERTAINTY  │
                  │ MC Dropout   │
                  │ 20 Passes    │
                  └──────┬───────┘
                         │
                 ┌───────┴────────┐
                 │                │
                 ▼                ▼
          LOW UNCERTAINTY   HIGH UNCERTAINTY
                 │                │
                 ▼                ▼
          CONFIDENT          INCONCLUSIVE
          PREDICTION         HUMAN REVIEW
                 │                │
                 └───────┬────────┘
                         ▼
                ┌─────────────────┐
                │   STREAMLIT     │
                │   DASHBOARD     │
                │                 │
                │ Results         │
                │ Explanations    │
                │ Uncertainty     │
                │ Human Review    │
                │ Audit Logs      │
                └─────────────────┘
```

---

# 18. Recommended Development Order

Follow this order when implementing the project:

## Phase 1 — Project Setup

Set up:

- Python
- PyTorch
- Torchvision
- HuggingFace Transformers
- OpenCV
- EasyOCR
- Whisper
- FFmpeg
- Streamlit

## Phase 2 — Deepfake Detection

```text
MTCNN
  ↓
Face Crops
  ↓
EfficientNet-B4
  ↓
Real / Fake
```

## Phase 3 — OCR + ASR

```text
EasyOCR
+
Whisper
```

## Phase 4 — NLP

```text
RoBERTa → Propaganda
BERT → Hate Speech
```

## Phase 5 — Explainability

```text
Grad-CAM
+
Text Highlighting
```

## Phase 6 — Uncertainty

```text
MC Dropout
20 Passes
Variance
```

## Phase 7 — Governance + Dashboard

```text
Streamlit
+
Human Review
+
Audit Logging
```

## Phase 8 — Evaluation

```text
Precision
Recall
F1
Calibration
Compression Robustness
OOD Testing
```

---

# 19. Key Design Principle

The project should be implemented as independent but connected modules:

```text
                    DIGITAL MEDIA VERIFICATION
                              │
               ┌──────────────┴──────────────┐
               │                             │
        VISUAL BRANCH                   TEXT BRANCH
               │                             │
        MTCNN + CNN                    OCR + Whisper
               │                             │
        EfficientNet-B4                     │
               │                     ┌───────┴───────┐
          Deepfake                  │               │
               │                RoBERTa           BERT
               │                Propaganda      Hate Speech
               │                     │               │
               └────────────┬────────┴───────────────┘
                            │
                     EXPLAINABILITY
                     Grad-CAM / Text
                            │
                      UNCERTAINTY
                      MC Dropout
                            │
                     GOVERNANCE
                            │
                      HUMAN REVIEW
                            │
                       STREAMLIT
```

This modular structure keeps the implementation aligned with the project synopsis while making the codebase easier to develop, test, debug, and demonstrate.
