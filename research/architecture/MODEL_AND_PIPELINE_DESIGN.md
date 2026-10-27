# Model Architectures and Pipeline Design

## 1. Visual Deepfake Detector: EfficientNet-B4

### 1.1 Architecture & Preprocessing
The primary visual detection model uses an **EfficientNet-B4** backbone (`models/deepfake/efficientnet_b4.py`), pre-trained on ImageNet and fine-tuned on the FaceForensics++ (C23) benchmark:
- **Input Dimensions:** $3 \times 224 \times 224$ pixels, normalized using standard ImageNet mean ($[0.485, 0.456, 0.406]$) and standard deviation ($[0.229, 0.224, 0.225]$).
- **Backbone Feature Extractor:** Mobile Inverted Bottleneck Convolutions (MBConv) with compound scaling factor ($\phi = 4$).
- **Classification Head:** Global Average Pooling (GAP) followed by Dropout ($p = 0.4$) and a fully connected linear layer mapping 1,792 latent features to 2 binary logits:
  $$\text{Logits} \in \mathbb{R}^2 \quad \xrightarrow{\text{Softmax}} \quad [P(\text{Class}_0), P(\text{Class}_1)]$$

### 1.2 Explainability via Grad-CAM
To provide visual localization of forged regions without additional supervision, the model integrates Gradient-weighted Class Activation Mapping (**Grad-CAM**):
1. **Target Feature Layer:** Activations $A^k \in \mathbb{R}^{H \times W}$ are tapped from the final convolutional block (`features[8]`).
2. **Neuron Importance Weights:** Gradients of the score for class $c$ (Fake) with respect to feature activation map $k$ are computed:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
3. **Heatmap Synthesis:** A weighted combination of forward activation maps followed by a Rectified Linear Unit (ReLU) isolates features positively contributing to the fake verdict:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
4. **Bilinear Upsampling:** Heatmaps are scaled to $224 \times 224$ and overlaid with a colormap onto the original face crop.

### 1.3 Bayesian Uncertainty via Monte Carlo Dropout
Standard deep neural networks tend to be poorly calibrated and overconfident under distribution shifts. We approximate Bayesian inference using **Monte Carlo Dropout** (`uncertainty/mc_dropout.py`):
- Dropout layers remain active during inference ($p = 0.4$).
- For each face crop, $T = 10$ stochastic forward passes are executed.
- The predictive mean $\bar{p}$ and sample variance $\sigma^2$ are computed:
  $$\bar{p} = \frac{1}{T} \sum_{t=1}^T p_t, \qquad \sigma^2 = \frac{1}{T-1} \sum_{t=1}^T (p_t - \bar{p})^2$$
- A predictive variance $\sigma^2 > 0.08$ triggers an automated human review advisory.

---

## 2. Textual Propaganda Detector: RoBERTa-base

### 2.1 Architecture & SemEval-2020 Task 11 Formulation
Propaganda detection is formulated as a multi-label classification problem across 14 rhetorical propaganda techniques established by Da San Martino et al. (2020):
- **Base Architecture:** **RoBERTa-base** (12 layers, 768 hidden dimensions, 12 attention heads, 125M parameters).
- **Optimization Strategy:** Standard cross-entropy collapses on rare techniques (e.g., *Whataboutism*, *Bandwagon*). In Phase 5, optimization introduced class-balanced focal loss:
  $$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  with focusing parameter $\gamma = 2.0$ and inverse-frequency class weights $\alpha_t$.
- **Result:** Overall Macro F1 improved from $0.4175$ to **$0.5132$** ($+0.0957$), with substantial gains on underrepresented categories.

---

## 3. Hate Speech Detector: BERT-base-uncased

- **Base Architecture:** **BERT-base-uncased** fine-tuned on the HateXplain dataset.
- **Classification Head:** Dense projection layer mapping sequence summary representations (`[CLS]` token embedding) to 3 target categories: `Hate Speech`, `Offensive Language`, and `Normal`.
- **Interpretability:** Token-level gradient attributions quantify word importance, enabling the user interface to highlight salient tokens that drove the classification decision.

---

## 4. Multimodal Aggregation and Pipeline Flow

The orchestrator (`pipeline/video_pipeline.py`) coordinates multimodal ingestion:

```
[Raw Video File]
       │
       ├─► [PyAV / OpenCV] ──► Uniform Sampling (1.0 FPS, max 4 frames)
       │                            │
       │                            ├─► [MTCNN] ──► Face Crops ──► [EfficientNet-B4 + Grad-CAM]
       │                            │
       │                            └─► [EasyOCR] ──► Detected Banner Text
       │                                                    │
       └─► [FFmpeg] ──► 16kHz PCM Audio                     │
                            │                               │
                            ▼                               ▼
                     [Whisper ASR] ───────────────► [Aggregated Text Corpus]
                                                            │
                                            ┌───────────────┴───────────────┐
                                            ▼                               ▼
                                    [RoBERTa Propaganda]            [BERT Hate Speech]
                                            │                               │
                                            └───────────────┬───────────────┘
                                                            ▼
                                                [Multimodal Fusion Engine]
                                                            │
                                                            ▼
                                                 [VerificationResult]
```

- **Temporal Sampling:** Sparse 1.0 FPS sampling (up to 4 frames) ensures responsive verification times ($<2.0$s on CUDA GPU, $<8.0$s on modern multi-core CPU).
- **Missing Modality Resilience:** If video lacks an audio track, Whisper ASR cleanly returns empty text without failing the pipeline. Similarly, if no human faces are present, the system records `faces_detected = 0`, sets `human_review_required = true`, and proceeds with OCR and audio analysis.
