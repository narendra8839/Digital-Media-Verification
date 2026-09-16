# Digital Media Verification: Viva & Defense Preparation Guide

This document compiles high-frequency technical questions and rigorous, implementation-aligned answers for oral defense and project evaluations.

---

### Architecture & Model Choices

#### 1. Why EfficientNet-B4 for deepfake detection?
EfficientNet-B4 uses compound coefficient scaling across depth, width, and image resolution ($380 \times 380$ input), balancing convolutional representation capacity with computational tractability. In facial manipulation detection, high-resolution feature maps retain subtle artifact patterns (such as blending boundaries, color discrepancies, and frequency warping) that smaller architectures miss, while avoiding the parameter bloat and overfitting risk of larger models like EfficientNet-B7.

#### 2. Why RoBERTa for propaganda detection?
RoBERTa (Robustly Optimized BERT Approach) improves upon BERT by eliminating Next Sentence Prediction (NSP), training on larger text corpora with dynamic masking, and utilizing longer training schedules with larger batch sizes. For rhetorical and propaganda detection (SemEval-2020 Task 11), subtle stylistic and persuasive nuances require rich semantic contextualization, for which RoBERTa consistently outperforms standard BERT.

#### 3. Why BERT for hate speech detection?
BERT-base-uncased is the canonical benchmark backbone for the **HateXplain** dataset. Using BERT preserves comparability with established literature, supports the 3-class target taxonomy (`normal`, `offensive`, `hatespeech`), and directly aligns with the word-piece tokenization scheme used by human annotators in HateXplain ground-truth rationales.

#### 4. Why MTCNN for face detection?
MTCNN (Multi-task Cascaded Convolutional Networks) utilizes a 3-stage cascaded architecture (P-Net, R-Net, O-Net) that jointly predicts face bounding boxes and five facial landmark coordinates (eyes, nose, mouth corners). This provides fast, rotation-resilient cropping to $224 \times 224$, isolating facial regions where deepfake synthesis artifacts concentrate and filtering out irrelevant background noise.

#### 5. Why EasyOCR?
EasyOCR implements a deep-learning-based CRAFT (Character Region Awareness for Text Detection) module combined with a ResNet-BiLSTM-CTC recognition pipeline. Unlike traditional morphological OCR engines (e.g., Tesseract), EasyOCR demonstrates significantly higher accuracy on in-the-wild digital media, complex font overlays, lower resolutions, and noisy video banners.

#### 6. Why Whisper-base for speech transcription?
OpenAI's Whisper-base (74 million parameters) provides an optimal trade-off between transcription accuracy, noise robustness, and real-time CPU/GPU latency. Trained on 680,000 hours of multilingual, weak-supervised audio, it robustly transcribes conversational speech, overlapping background sound, and degraded social media audio streams without requiring custom acoustic training.

---

### Datasets, Benchmarks, and Provenance

#### 7. Why FaceForensics++ (FF++)?
FaceForensics++ is the academic gold standard for facial forgery evaluation, offering four distinct manipulation techniques (Deepfakes, Face2Face, FaceSwap, NeuralTextures) at controlled compression levels (c23 high-quality, c40 low-quality) with official, subject-disjoint train/val/test splits.

#### 8. Why is the Kaggle FF++ dataset strictly development-only?
The Kaggle mirror (`xdxd003/ff-c23`) is used exclusively as a temporary engineering MVP dataset. Official authorized FaceForensics++ access requires university institutional agreement verification, which is ongoing. To maintain academic integrity, the Kaggle mirror is explicitly labeled as development-only, and its samples were mapped strictly against our official dataset split definitions without claiming official distribution status.

#### 9. Why SemEval-2020 Task 11?
SemEval-2020 Task 11 provides a granular, linguistically motivated taxonomy of 14 distinct rhetorical propaganda techniques (e.g., *Loaded Language*, *Name Calling*, *Appeal to Fear*, *Doubt*, *Exaggeration/Minimisation*). Rather than treating misinformation as a naive binary problem, it models specific manipulative techniques.

#### 10. Why HateXplain?
HateXplain is the first benchmark dataset that incorporates word-level human rationale annotations alongside multi-class toxicity labels. This enables quantitative validation of model explainability by measuring whether the tokens driving model classifications align with human expert reasoning.

---

### Explainable AI (XAI)

#### 11. What is Grad-CAM?
Grad-CAM (Gradient-weighted Class Activation Mapping) computes the gradients of the target class score with respect to feature activation maps of the final convolutional layer. Global average pooling of these gradients produces importance weights that linearly combine the feature maps into a coarse spatial 2D heatmap, identifying the pixel regions that positively influenced the prediction.

#### 12. What is Gradient × Input (Grad × Input) token attribution?
$Grad \times Input$ is a gradient-based feature attribution method for neural networks. It computes the element-wise dot product of the input token's embedding vector and the gradient of the predicted class logit with respect to that embedding:
$$S_i = \left| x_i \cdot \frac{\partial y_c}{\partial x_i} \right|$$
This measures how sensitive the output logit is to infinitesimal perturbations of each word embedding in the sequence.

#### 13. Why must Grad-CAM and token attribution include disclaimers?
Grad-CAM and token attribution are **post-hoc correlational signals**, not causal proofs. They reflect what the model attended to during feature extraction, which may include dataset artifacts or spurious correlations. Stating that an attribution "proves" manual image tampering or author intent would be unscientific.

---

### Uncertainty Quantification (UQ) & Epistemic Telemetry

#### 14. What is Monte Carlo (MC) Dropout?
MC Dropout treats standard dropout during inference as an approximation of Bayesian inference over Gaussian processes (Gal & Ghahramani, 2016). By keeping dropout layers active at test time and performing $T$ stochastic forward passes, the model samples from the posterior predictive distribution, enabling estimation of epistemic (model) uncertainty.

#### 15. Why $T = 20$?
$T = 20$ represents an empirical sweet spot between statistical variance stability and computational latency. Prior literature shows that the mean and variance of predictive distributions largely stabilize between 15 and 30 passes. Running 20 passes provides reliable dispersion metrics while keeping API latency on CPU within practical bounds.

#### 16. What is predictive entropy?
Shannon predictive entropy $H$ measures the dispersion or spread of the mean predictive class distribution $p$:
$$H(p) = -\sum_{c=1}^{C} p_c \log_2(p_c)$$
When class probabilities are uniform (e.g., 50%/50% in binary classification), entropy reaches its maximum ($1.0$ bit), signaling that the model cannot confidently separate classes.

#### 17. What is the difference between confidence and uncertainty?
- **Confidence**: The maximum softmax probability output of a single forward pass ($\max_c p_c$). Neural networks are notoriously overconfident, often assigning high probabilities ($> 0.90$) even on completely out-of-distribution or corrupted inputs.
- **Uncertainty**: The variance or dispersion across stochastic samples ($\sigma^2$). If slight parameter dropout causes predictions to oscillate wildly between classes, the model is uncertain, regardless of what an individual softmax logit claims.

---

### Governance, Human Review, and Ethical AI

#### 18. Why human review instead of automated action?
Automated decisions on digital media (e.g., content removal, shadowbanning, account suspension) carry severe risks of censorship, algorithmic bias, and false accusations. In high-stakes misinformation contexts, AI must act strictly as a **decision-support tool** that flags elevated uncertainty and provides interpretable evidence for human verification, preserving human agency and procedural justice.

#### 19. Why not classify content as "definitely real" or "definitely fake"?
Digital manipulation exists along a continuous spectrum—from subtle color grading and compression artifacts to full facial reenactment and voice cloning. A statistical model evaluates likelihood based on training distributions; it cannot establish ground-truth physical reality. Claiming definitive truth leads to false confidence and severe epistemic failure.

#### 20. What was corrected regarding the `NO_FACE_DETECTED` status?
Previously, if an image contained no detectable faces, the visual detector bypassed inference, but governance defaulted to `CONFIDENT_PREDICTION`. This was misleading because it implied the media was verified as authentic. The policy was corrected to `ANALYSIS_INCOMPLETE`: the system explicitly states that visual verification could not be performed, strictly avoiding ungrounded conclusions.

---

### Methodological Rigor & Limitations

#### 21. What is data leakage and how was it prevented?
In video deepfake datasets, consecutive frames from the same video share identical identities, lighting, and backgrounds. **Data leakage** occurs when frames from the same original video appear in both training and validation/test splits, causing models to memorize video backgrounds rather than learning forgery artifacts. We prevented leakage by enforcing **strict video-level, subject-disjoint splits** matching official FaceForensics++ video pair lists.

#### 22. Why is the MVP deepfake accuracy around 55–60%?
The MVP model was trained on a small development subset with early stopping to rapidly validate pipeline architecture and explainability modules. Furthermore, detecting deepfakes across compressed c23 video is challenging under cross-manipulation splits. Scaling training epochs, using full-resolution FF++ data, and incorporating frequency-domain analysis (e.g., DCT/FFT) will increase accuracy.

#### 23. What are the key current limitations?
1. **Facial Boundary Focus**: Deepfake detection relies on MTCNN facial crops and does not evaluate whole-scene background anomalies.
2. **Audio/Video Cross-Modal Fusion**: Text from OCR and ASR is aggregated sequentially rather than cross-attending directly with video feature representations.
3. **In-Memory Job State**: Background video task statuses are maintained in volatile memory and reset on server restarts.
4. **Computational Latency on CPU**: Whisper-base and EasyOCR introduce noticeable latency on long video clips without GPU acceleration.
