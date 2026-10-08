# Digital Media Verification — Project Context

## 1. Project

**Title:** Digital Media Verification: Explainable and Uncertainty-Aware Deepfake, Propaganda and Hate-Speech Detection

This project is a multimodal digital-media analysis system for academic demonstration. It combines visual analysis, text/speech extraction, NLP classification, explainability, uncertainty estimation, and human review signals in one web-based workflow.

The system is a **decision-support tool**. It must not claim to determine objective truth, and it must not automatically delete, suspend, punish, or moderate users/content.

## 2. Problem

Digital content can contain multiple forms of risk at the same time:

- manipulated faces or visual media;
- misleading or propagandistic on-screen text;
- harmful or hateful language;
- spoken content that is not visible in the image.

A single-modality detector can miss useful evidence from another modality. The project therefore combines:

**Visual analysis + OCR/ASR + NLP + Explainability + Uncertainty + Human Review**

## 3. Objectives

The implementation must support:

1. Multimodal ingestion of image, video, and text.
2. Face-based deepfake analysis.
3. OCR extraction from visible text.
4. ASR extraction from video/audio.
5. Propaganda-technique detection.
6. Hate-speech/toxicity classification.
7. Visual explanations using Grad-CAM.
8. Text explanations using token attribution.
9. Uncertainty estimation using Monte Carlo Dropout.
10. Governance outputs that clearly indicate when human review is appropriate.

## 4. Scope

### In scope

- Static images.
- Sampled short videos.
- Face detection.
- Deepfake detection.
- OCR.
- ASR.
- English propaganda analysis.
- English hate-speech analysis.
- Grad-CAM.
- Token attribution.
- Monte Carlo Dropout uncertainty.
- Unified result aggregation.
- React analyst dashboard.
- Human-review recommendation.

### Out of scope

- Real-time stream monitoring.
- Guaranteed deepfake/truth detection.
- Determining whether a claim is factually true.
- Automatic content deletion or account suspension.
- Replacing human reviewers.
- Full-body manipulation analysis.
- Broad multilingual coverage beyond the implemented English pipeline.

## 5. Current MVP Baseline

The current MVP already contains working backend/frontend pipelines, trained checkpoints, explainability, uncertainty logic, API routes, and automated tests.

The MVP is **not the final research/training state**.

Its main limitations are:

- training data is currently represented by small MVP subsets for some training scripts;
- current deepfake generalization is weak on the MVP split;
- uncertainty thresholds are development defaults rather than formally calibrated thresholds;
- FastAPI startup currently initializes inference on CPU;
- Grad-CAM output needs to be reliably browser-renderable in the frontend;
- some older Streamlit/utility scaffolding is legacy and should not be treated as the active architecture.

## 6. Final-State Principle

The final project must move from an MVP subset to a properly evaluated full-data training setup.

“Full data” means:

- all appropriate samples from the intended **training split**;
- full validation/dev data for tuning and model selection;
- untouched test data used only for final evaluation.

**Never train on test data.**

The final system must preserve reproducibility through fixed seeds, recorded configurations, saved checkpoints, and documented dataset splits.

## 7. Responsible AI Principles

The project follows the ART framing:

- **Accountability:** keep model/configuration/audit metadata and clear responsibility boundaries.
- **Responsibility:** provide decision support rather than punitive automated actions.
- **Transparency:** expose predictions, evidence, explanations, uncertainty, and limitations.
- **Human-in-the-loop:** high-uncertainty or incomplete cases should be surfaced for human review.

The UI and reports must avoid language such as “this content is definitely fake” or “this person is a hateful person.”

Prefer model-scoped language such as:

- “Model prediction”
- “Detected signal”
- “Review recommended”
- “Analysis incomplete”
- “Confidence”
- “Uncertainty”

## 8. Technology Direction

### Frontend

- React
- Vite
- Vanilla CSS
- Analyst-style dashboard

### Backend

- FastAPI
- Python
- Existing pipeline/router structure
- BackgroundTasks for current video processing

### Models/tools

- MTCNN for face detection
- EfficientNet-B4 for deepfake classification
- RoBERTa-base for propaganda classification
- BERT-base-uncased for hate-speech classification
- EasyOCR for OCR
- Whisper-base for ASR
- Grad-CAM for visual explanations
- Gradient-based token attribution for NLP explanations
- Monte Carlo Dropout for uncertainty

## 9. Important Architecture Decision

Do **not** add a cross-modal transformer solely because it sounds more advanced.

The project specification is satisfied by a modular multimodal architecture in which visual analysis and text/speech analysis produce evidence that is aggregated into a unified result.

Do not introduce major architecture complexity unless it solves a demonstrated project requirement.

## 10. Working Definition of Success

The final system should let a reviewer upload media and understand:

1. What was analyzed.
2. Which models were used.
3. What the models predicted.
4. What evidence contributed to the result.
5. How uncertain the model is.
6. Whether human review is recommended.
7. What limitations apply.

No output should be fabricated for presentation purposes.
