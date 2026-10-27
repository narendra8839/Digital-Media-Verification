# Evaluation Protocol and Metrics

This document details the mathematical formulations, evaluation metrics, video sampling methodologies, label mapping conventions, and uncertainty estimation protocols utilized in the Digital Media Verification (DMV) framework.

---

## 1. Evaluation Metrics

To account for class imbalance (such as the $4:1$ ratio in the FaceForensics++ test partition and extreme long-tail distributions in SemEval propaganda techniques), we report four complementary metrics:

### 1.1 Classification Accuracy & Balanced Accuracy
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Balanced Accuracy} = \frac{1}{2}\left(\frac{TP}{TP + FN} + \frac{TN}{TN + FP}\right) = \frac{\text{Recall}_{\text{Fake}} + \text{Recall}_{\text{Real}}}{2}$$

### 1.2 Macro F1 & Weighted F1
For $K$ classes with support $N_k$ and total samples $N = \sum_k N_k$:
$$\text{Macro F1} = \frac{1}{K}\sum_{k=1}^K F_{1,k}, \qquad \text{Weighted F1} = \sum_{k=1}^K \frac{N_k}{N} F_{1,k}$$
where $F_{1,k} = \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$.

### 1.3 Class Marginals (Precision & Recall)
$$\text{Precision}_{\text{Fake}} = \frac{TP}{TP + FP}, \qquad \text{Recall}_{\text{Fake}} = \frac{TP}{TP + FN}$$
$$\text{Precision}_{\text{Real}} = \frac{TN}{TN + FN}, \qquad \text{Recall}_{\text{Real}} = \frac{TN}{TN + FP}$$

---

## 2. Video Frame Sampling and Aggregation

Because processing all frames of a 30 FPS video is computationally prohibitive and introduces severe temporal redundancy:
1. **Sampling Rate:** 1.0 FPS (1 frame sampled per second of video).
2. **Cap:** At most 4 uniformly spaced keyframes are extracted per video.
3. **Face Extraction:** For each sampled keyframe, MTCNN detects facial bounding boxes. If multiple faces exist, the highest-confidence face is selected.
4. **Video-Level Aggregation:**
   - For a video with $M$ extracted faces yielding frame-level probabilities $p_1, \dots, p_M$:
     $$P_{\text{video}}(\text{Fake}) = \max_{m=1\dots M} p_m(\text{Fake})$$
   - This max-pooling aggregation follows the forensic principle that a video is classified as manipulated if any verified facial frame exhibits synthetic manipulation artifacts.

---

## 3. Label Orientation and Softmax Mapping Audit

In binary neural network classifiers with output logits $\mathbf{z} = [z_0, z_1] \in \mathbb{R}^2$:
$$p_0 = \frac{e^{z_0}}{e^{z_0} + e^{z_1}}, \qquad p_1 = \frac{e^{z_1}}{e^{z_0} + e^{z_1}}$$

### 3.1 Mapping Audit Findings
- **Production Baseline Model:** Trained with **Logit 0 = Real, Logit 1 = Fake**. Softmax probability $p_1 \ge 0.5$ triggers a "Fake" verdict.
- **Experimental Candidates (V3):** The training pipeline in `train_robust_v3.py` mapped class index 0 to Fake and class index 1 to Real.
- **Resolution:** Rather than retraining models or modifying production weights, the evaluation harness parameterizes `fake_class_idx`:
  ```python
  p_fake = float(probs[fake_class_idx])
  pred_label = "fake" if p_fake >= 0.5 else "real"
  ```
- **Integrity Proof:** Mathematically verified by unit tests in `tests/test_label_mapping.py::TestLabelMappingAudit`.

---

## 4. Bayesian Uncertainty Estimation Protocol

To quantify epistemic model uncertainty under distribution shifts:
- **Technique:** Monte Carlo Dropout (Gal & Ghahramani, 2016).
- **Inference Configuration:** Dropout layers in the classification head remain enabled with $p = 0.4$.
- **Number of Stochastic Passes:** $T = 10$.
- **Predictive Variance:**
  $$\sigma^2 = \frac{1}{T-1} \sum_{t=1}^{T} \big(p_t(\text{Fake}) - \bar{p}(\text{Fake})\big)^2$$
- **Operational Threshold:** $\sigma^2 \ge 0.02$ (or predictive entropy $H \ge 0.85$ bits) flags the media for mandatory human review (`REVIEW_RECOMMENDED`).
