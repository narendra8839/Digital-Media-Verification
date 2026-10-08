# Digital Media Verification — Architecture Context

## 1. Final Architecture

```text
React Analyst Dashboard
        |
        v
FastAPI Backend
        |
        v
Multimodal Pipeline / Router
        |
        +---------------- IMAGE ----------------+
        |                                       |
        |  MTCNN -> EfficientNet-B4             |
        |          |                            |
        |          +-> Grad-CAM                 |
        |          +-> MC Dropout UQ            |
        |                                       |
        |  EasyOCR -> Text Aggregation           |
        |                  |                    |
        |                  +-> RoBERTa            |
        |                  +-> BERT              |
        |                                       |
        +---------------- VIDEO ----------------+
        |                                       |
        |  Frame Sampling -> MTCNN              |
        |                    -> EfficientNet     |
        |                    -> video aggregation
        |                    -> representative frame
        |                    -> Grad-CAM/UQ      |
        |                                       |
        |  Keyframes -> OCR                     |
        |  FFmpeg -> Whisper -> Text Aggregation|
        |                         |              |
        |                         +-> RoBERTa     |
        |                         +-> BERT       |
        |                                       |
        +---------------- TEXT -----------------+
                                                |
                Text Aggregation -> RoBERTa/BERT
                                -> Token Attribution
                                -> MC Dropout UQ
                                                |
                                                v
                              Uncertainty + Governance
                                                |
                                                v
                              Unified Evidence/Result
                                                |
                                                v
                                      Human Reviewer
```

## 2. Image Flow

1. Receive image.
2. Detect face using MTCNN.
3. If no face is found:
   - return an incomplete-analysis state;
   - do not pretend a deepfake result exists.
4. Crop/prepare the face for EfficientNet-B4.
5. Run deepfake classification.
6. Produce class probabilities and prediction.
7. Generate Grad-CAM from the actual model.
8. Run MC Dropout for uncertainty.
9. Run EasyOCR where applicable.
10. Aggregate extracted text.
11. Run propaganda and hate-speech models when text is available.
12. Build the unified response.
13. Apply governance/review policy.

## 3. Video Flow

1. Receive video.
2. Process it asynchronously through the existing API job mechanism.
3. Sample frames at the configured rate.
4. Detect faces on sampled frames.
5. Run EfficientNet-B4 on valid face frames.
6. Aggregate frame-level fake probabilities into a video-level visual result.
7. Select a representative frame.
8. Generate Grad-CAM and MC Dropout on the representative frame.
9. Extract OCR from selected keyframes.
10. Extract audio with FFmpeg.
11. Convert/process audio as required for Whisper.
12. Run Whisper-base ASR.
13. Aggregate OCR + ASR text.
14. Run RoBERTa and BERT on available text.
15. Apply uncertainty and governance rules.
16. Return the unified result.

## 4. Text Flow

1. Accept user-provided text.
2. Normalize/aggregate text through the existing text processing layer.
3. Run RoBERTa propaganda classification.
4. Run BERT hate-speech/toxicity classification.
5. Generate token attribution.
6. Run MC Dropout uncertainty estimation.
7. Apply governance policy.
8. Return structured results.

## 5. Core Model Configuration

### Deepfake

- Architecture: EfficientNet-B4
- Task: binary real/fake classification
- Input: 224x224 RGB
- ImageNet normalization
- Decision threshold: P(fake) >= 0.5
- Model dropout: 0.4

### Propaganda

- Architecture: RoBERTa-base
- Task: 14 propaganda-technique classes
- Maximum sequence length: 128

### Hate Speech

- Architecture: BERT-base-uncased
- Classes:
  - normal
  - offensive
  - hatespeech
- Maximum sequence length: 128

### ASR

- Model: openai/whisper-base
- Audio target: 16 kHz mono

## 6. Explainability Layer

### Visual

Grad-CAM must be generated from the actual EfficientNet-B4 inference computation.

The final product should expose the saved/browser-accessible visualization rather than a placeholder.

### Text

Token attribution is generated from actual model computation using gradient-based attribution.

Live uploads do not have ground-truth rationales. Benchmark rationale alignment should therefore be evaluated only where the dataset provides rationale annotations.

## 7. Uncertainty Layer

Current approach:

- Monte Carlo Dropout
- T = 20 stochastic passes
- dropout layers active during stochastic sampling
- normalization layers kept in evaluation behavior
- predictive mean
- predictive variance
- standard deviation
- Shannon entropy

Current development thresholds:

- variance: 0.02
- entropy: 0.85 bits

These are not to be presented as formally calibrated probabilities. Calibration should be evaluated as a separate improvement.

## 8. Governance Layer

The governance output should be a decision-support layer, not a moderation engine.

Examples:

### High uncertainty

```text
status: REVIEW_RECOMMENDED
reason: model uncertainty is above configured threshold
```

### No face

```text
status: ANALYSIS_INCOMPLETE
reason: NO_FACE_DETECTED
```

### Normal confident analysis

Return the prediction and evidence while preserving the project's uncertainty and limitation disclosures.

## 9. API Direction

Existing endpoints include:

- GET `/health`
- GET `/ready`
- POST `/verify/text`
- POST `/verify/image`
- POST `/verify/video`
- POST `/verify/multimodal`
- GET `/verify/status/{job_id}`

The implementation should preserve compatibility unless a change is genuinely required.

Do not invent a new API schema before inspecting the current schemas and routes.

## 10. Processing and Storage Principle

For the college demo:

- FastAPI BackgroundTasks + polling is acceptable for video jobs.
- Redis/Celery is not required.
- Avoid unnecessary production infrastructure.
- Keep temporary outputs and model artifacts organized.
- Browser-facing explanation artifacts must use paths/URLs that the frontend can actually access.

## 11. Engineering Rule

Do not rebuild the application from scratch.

Extend the current modular implementation after inspecting the existing code. Preserve working components and tests wherever possible.
