# Explainability & Uncertainty Quantification Report

## Executive Summary
This report documents the design, mathematical formulation, empirical implementation, and Responsible AI governance principles for the Explainable AI (XAI) and Uncertainty Quantification (UQ) components of the **Digital Media Verification** system.

The implementation bridges model predictions with human-interpretable signals across all three project modalities:
1. **Spatial Visual Explainability**: Grad-CAM saliency overlays for `EfficientNet-B4` deepfake detection.
2. **Textual Feature Attribution**: Gradient-based token attribution (Grad $\times$ Input) for `RoBERTa` propaganda classification and `BERT` hate-speech classification.
3. **Empirical Rationale Alignment**: Benchmark comparison against HateXplain human rationale ground truth.
4. **Epistemic Uncertainty Estimation**: Monte Carlo Dropout ($T=20$ stochastic forward passes) across all three models.
5. **Responsible Human-in-the-Loop Safeguards**: Uncertainty-triggered review recommendation flags (`REVIEW_RECOMMENDED`).

---

## 1. Grad-CAM Methodology (Deepfake Detection)

### Target Layer & Architecture
- **Model**: `EfficientNet-B4` binary classifier (`Real` vs `Fake`).
- **Target Convolutional Layer**: Final stage block `model.backbone.features[-1]` (`Conv2dNormActivation`), outputting activation maps $A \in \mathbb{R}^{C \times H \times W}$ where $C = 1792$ and $H = 7, W = 7$ for $224 \times 224$ input face crops.

### Mathematical Formulation
1. **Gradient Computation**: For a predicted or target class $c$, compute the gradient of the pre-softmax logit $y^c$ with respect to the activation feature map $A^k$:
   $$\frac{\partial y^c}{\partial A_{i, j}^k}$$
2. **Importance Weights via Global Average Pooling**:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{i, j}^k}$$
3. **Linear Combination and Rectification**:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^C \alpha_k^c A^k \right)$$
4. **Bilinear Upsampling & Normalization**: The coarse $7 \times 7$ heatmap is resized to $224 \times 224$ and min-max normalized to $[0, 1]$.
5. **Colormap Blending**: The normalized map is colorized with the `cv2.COLORMAP_JET` spectrum and alpha-blended ($\alpha = 0.5$) with the original face crop.

> [!IMPORTANT]
> **Mandatory Responsible AI Disclaimer**:
> *"Grad-CAM and token attribution are post-hoc explanatory signals and should not be interpreted as causal proof."*
> High activation values indicate regions that contributed strongly to the model's forward activations; they do not prove that facial pixels were manually synthesized or tampered with.

---

## 2. Token-Level Attribution Methodology (Propaganda & Hate Speech)

### Mathematical Formulation: Grad $\times$ Input
For transformer-based sequence classifiers (`RoBERTa-base` for propaganda and `BERT-base-uncased` for hate speech), token importance is computed using the gradient of the target class logit with respect to the input word embeddings:

1. Let $E = [e_1, e_2, \dots, e_L] \in \mathbb{R}^{L \times D}$ denote the input embedding vectors for a sequence of length $L$ with hidden dimension $D$.
2. Compute the gradient of the predicted class score $y^c$ with respect to the input embeddings:
   $$G_i = \frac{\partial y^c}{\partial e_i} \in \mathbb{R}^D$$
3. Compute the token attribution saliency score using the dot-product norm:
   $$S_i = \| G_i \odot e_i \|_2$$
4. Normalize attribution scores across the sequence to $[0, 1]$:
   $$\bar{S}_i = \frac{S_i - \min_j S_j}{\max_j S_j - \min_j S_j + \epsilon}$$

### Subword Reconstruction & Cleaning
- Token representations handle tokenizer-specific subword markers:
  - BERT WordPiece subwords (`##`) are mapped back to their root word tokens.
  - RoBERTa byte-level BPE whitespace markers (`Ġ`) are cleaned and sanitized to prevent console and serialization encoding failures.

---

## 3. HateXplain Ground-Truth Rationale Alignment

### Objective
HateXplain includes human-annotated binary rationale masks indicating which tokens annotators identified as offensive or hateful. The system computes a lightweight empirical alignment measure between model-attributed tokens and human annotations.

### Metrics Computed
Let $\mathcal{M}$ denote the set of token indices selected by the model (top-$k$ highest attribution scores), and let $\mathcal{G}$ denote the union of human ground-truth rationale indices:
1. **Token Overlap Count**: $|\mathcal{M} \cap \mathcal{G}|$
2. **Rationale Precision**:
   $$P_{\text{rationale}} = \frac{|\mathcal{M} \cap \mathcal{G}|}{|\mathcal{M}|}$$
3. **Rationale Recall**:
   $$R_{\text{rationale}} = \frac{|\mathcal{M} \cap \mathcal{G}|}{|\mathcal{G}|}$$
4. **Rationale F1**:
   $$F1_{\text{rationale}} = \frac{2 \cdot P_{\text{rationale}} \cdot R_{\text{rationale}}}{P_{\text{rationale}} + R_{\text{rationale}}}$$

### Empirical Demonstration
On test sample `"do not be aggressive says the nigger mandela fuck that"` (Ground Truth: `hatespeech`):
- Top Model-Attributed Tokens: `nigger` (1.0000), `mandela` (0.5319), `fuck` (0.3174).
- Ground-Truth Human Rationale: `nigger` (index 6).
- Alignment Results: Overlap = 1 (`nigger`), Precision = 0.5000, Recall = 1.0000, **Rationale F1 = 0.6667**.

---

## 4. Monte Carlo (MC) Dropout Uncertainty Quantification

### Configuration
- **Stochastic Forward Passes**: $T = 20$ (`MC_DROPOUT_SAMPLES = 20`).
- **Selective Layer Activation**:
  - Dropout layers (`nn.Dropout`, `nn.Dropout1d`, `nn.Dropout2d`) are explicitly activated via `m.train()`.
  - Normalization layers (`BatchNorm2d`, `LayerNorm`) remain in `eval()` mode with fixed running statistics to preserve proper feature scaling.

### Mathematical Uncertainty Metrics
For an input $x$, $T=20$ stochastic forward passes yield probability vectors $p^{(1)}, p^{(2)}, \dots, p^{(T)} \in \mathbb{R}^C$:

1. **Predictive Mean Probability**:
   $$\bar{p}_c = \frac{1}{T} \sum_{t=1}^T p_c^{(t)}$$
2. **Predictive Variance (Epistemic Uncertainty)**:
   $$\sigma_c^2 = \frac{1}{T} \sum_{t=1}^T \left( p_c^{(t)} - \bar{p}_c \right)^2$$
3. **Predictive Standard Deviation**:
   $$\sigma_c = \sqrt{\sigma_c^2}$$
4. **Shannon Predictive Entropy**:
   $$H(\bar{p}) = -\sum_{c=1}^C \bar{p}_c \log_2(\bar{p}_c + \epsilon)$$

### Stochastic Output Verification
Verified across all three models that repeated passes produce non-zero variance ($\sigma^2 > 0$):
- **Deepfake EfficientNet-B4**: Confirmed stochastic variations (class 1 variance: $0.2393$).
- **Hate Speech BERT**: Confirmed stochastic variations (variance: $9.3 \times 10^{-4}$).
- **Propaganda RoBERTa**: Confirmed stochastic variations (variance: $8.4 \times 10^{-4}$).

> [!IMPORTANT]
> **Mandatory Model Uncertainty Disclaimer**:
> *"MC Dropout uncertainty is a model uncertainty signal and has not been formally calibrated for deployment."*

---

## 5. Uncertainty Thresholds & Human-in-the-Loop Review Flag

### Policy Architecture
The decision layer acts as a responsible triage filter:

```
                      Input Media / Text
                              ↓
                Model Inference + MC Dropout (T=20)
                              ↓
              Compute Max Variance and Shannon Entropy
                              ↓
         ┌────────────────────┴────────────────────┐
         ↓                                         ↓
Max Variance < Threshold             Max Variance >= Threshold
   AND Entropy < Threshold            OR Entropy >= Threshold
         ↓                                         ↓
   Status: CONFIDENT                     Status: REVIEW_RECOMMENDED
Review Required: FALSE                    Review Required: TRUE
         ↓                                         ↓
Return Prediction Normally           Flag for Human Verification
```

### Configurable Thresholds
Configured in `config/config.yaml`:
- `mc_dropout_samples`: 20
- `variance_threshold`: 0.02
- `entropy_threshold`: 0.85 bits
- **Calibration Label**: *"Development/MVP threshold — not clinically, legally, or production calibrated."*

### Responsible AI Governance Rules
1. **No Automated Punishment**: The system must NOT automatically delete content, ban user accounts, or generate legal accusations based on the uncertainty signal.
2. **Decision Support**: High uncertainty indicates that the model cannot confidently differentiate between classes under stochastic perturbation. It mandates human oversight rather than an automated decision.
3. **Non-Binary Justifications**: Every flagged prediction provides explicit mathematical reasons (e.g. *"predictive variance (0.0512) exceeds threshold (0.0200)"*).

---

## 6. Common Standardized Output Schema

All pipeline predictions conform to a unified extensible JSON contract:

```json
{
  "modality": "deepfake",
  "prediction": "fake",
  "confidence": 0.9245,
  "uncertainty": {
    "mean_probability": [0.0755, 0.9245],
    "variance": [0.003412, 0.003412],
    "std": [0.0584, 0.0584],
    "entropy": 0.3871,
    "max_variance": 0.003412,
    "mc_samples": 20
  },
  "review_required": false,
  "review_status": "CONFIDENT_PREDICTION",
  "review_justification": "Predictive variance (0.0034) and entropy (0.3871 bits) are within acceptable bounds.",
  "explanation": {
    "heatmap_shape": [7, 7],
    "overlay_path": "reports/explanations/deepfake/939_115_fake_gradcam.png",
    "disclaimer": "Grad-CAM is a post-hoc feature attribution method and should not be interpreted as causal proof."
  },
  "metadata": {}
}
```

---

## 7. Known Limitations

1. **Post-Hoc Attribution**: Grad-CAM and token saliency highlight correlated features rather than causal mechanisms. Saliency does not prove semantic intent.
2. **Resolution Bounds**: Grad-CAM heatmap resolution is bounded by the $7 \times 7$ spatial grid of EfficientNet-B4's final convolution layer; fine facial boundaries are smoothed.
3. **Uncalibrated Thresholds**: The variance (0.02) and entropy (0.85) thresholds are development defaults designed for MVP demonstration and require empirical calibration against human reviewer error rates in production.
4. **Tokenization Artifacts**: WordPiece and BPE subwords split slang or uncommon phrases across tokens, requiring subword-to-word heuristics during rationale alignment.
