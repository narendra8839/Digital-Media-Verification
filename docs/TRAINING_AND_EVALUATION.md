# Training and Evaluation Context

## 1. Goal

The final reported model performance must come from a proper train/validation/test protocol rather than the small MVP training subsets.

The MVP subset is useful for development and smoke tests, but the final academic experiment should use the full intended eligible data.

## 2. Split Rule

The required protocol is:

```text
Training split
    -> model fitting

Validation / development split
    -> hyperparameter tuning
    -> threshold/model selection
    -> checkpoint selection

Test split
    -> final evaluation only
```

**Never use the test split for training, model selection, threshold tuning, or iterative debugging.**

Once the final test evaluation is run, avoid repeatedly tuning against it.

## 3. Datasets

### FaceForensics++

The project uses the local FaceForensics++ C23 data and existing official split metadata.

The repository audit identified an internal inconsistency between the claimed dataset count and the listed folder contents. Before final training, verify the actual local dataset inventory and split mapping from files on disk.

Do not silently assume a count.

### SemEval-2020 Task 11

Used for propaganda-technique detection.

Use the complete intended training corpus for final training and the provided development/validation data for model selection.

### HateXplain

Used for hate-speech classification and rationale-related evaluation.

Use the complete intended training split for final training and validation data for tuning.

## 4. Deepfake Training

Final training should improve over the MVP script rather than blindly reusing its tiny subset settings.

The training pipeline should explicitly document:

- dataset split files;
- number of videos and frames actually used;
- frame-sampling policy;
- face-detection policy;
- image transforms;
- batch size;
- learning rate;
- optimizer;
- scheduler, if used;
- number of epochs;
- checkpoint selection criterion;
- random seed;
- device;
- class balancing strategy, if required.

For video-level evaluation, clearly distinguish frame-level predictions from aggregated video-level predictions.

## 5. NLP Training

The final training process must record:

- exact dataset split;
- tokenizer/model version;
- maximum sequence length;
- label mapping;
- batch size;
- learning rate;
- optimizer;
- epochs;
- checkpoint selection criterion;
- seed;
- class distribution;
- any weighting/sampling strategy.

Do not alter label semantics between training and inference.

## 6. GPU Requirement

Final local training should use the user's NVIDIA RTX 4050 Laptop GPU when available.

The training scripts must verify GPU use rather than merely assuming CUDA exists.

At minimum, verify:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
```

The actual model and tensors used for training must be moved to the selected CUDA device.

During training, verify GPU activity externally with:

```bash
nvidia-smi
```

If CUDA is unavailable, stop and diagnose the environment before claiming GPU training succeeded.

## 7. CPU vs GPU Inference

The current FastAPI startup path initializes the inference pipeline on CPU.

Do not claim the application is GPU-accelerated until the startup/device path is changed and verified.

Training GPU usage and inference GPU usage are separate requirements and must each be verified.

## 8. Metrics

Use metrics appropriate to the task.

### Binary deepfake

Report at minimum:

- accuracy
- precision
- recall
- F1
- confusion matrix

Also report video-level performance separately when the aggregation operates at video level.

### Multiclass propaganda

Report:

- accuracy
- macro F1
- per-class precision/recall/F1
- confusion matrix where practical

Class imbalance must be considered.

### Hate speech

Report:

- accuracy
- macro F1
- per-class precision/recall/F1
- confusion matrix

## 9. Explainability Evaluation

Explainability should not be treated as visually convincing only.

Where ground-truth rationales or localization annotations exist:

- evaluate alignment quantitatively where feasible;
- separately show qualitative examples.

For live uploads without ground truth:

- present the explanation as model evidence;
- do not call it objectively correct.

## 10. Uncertainty Evaluation

The current MC Dropout thresholds are development defaults.

The final work should ideally include a calibration/uncertainty evaluation such as:

- reliability analysis;
- confidence vs correctness;
- expected calibration error, if implemented;
- uncertainty distribution;
- review-trigger analysis.

Do not describe raw softmax confidence as calibrated certainty without evidence.

## 11. Final Reporting

The final report should clearly separate:

1. MVP baseline results.
2. Final full-data training results.
3. Validation/tuning decisions.
4. Untouched final test results.
5. Explainability examples.
6. Uncertainty/review results.
7. Limitations.

Never hide weak baseline results. Explain why they occurred and how the final experiment addresses them.
