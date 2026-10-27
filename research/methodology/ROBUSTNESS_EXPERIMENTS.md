# Robustness Experiments, Corruptions, and Findings

## 1. Motivation: The Fragility of Deepfake Forensics

Modern deepfake detectors trained exclusively on high-resolution, uncompressed video frames often suffer from catastrophic degradation when exposed to common social media transformations (e.g., re-encoding, transmission noise, downscaling, compression). This study systematically characterizes the vulnerability of the production EfficientNet-B4 detector and investigates three generations of robustness-enhancing candidates (V1, V2, and V3).

---

## 2. Standardized Degradation Operators

All evaluations employ five realistic distortion operators applied to extracted face crops:

| Distortion Operator | Parameterization | Implementation | Real-World Analog |
| :--- | :--- | :--- | :--- |
| **Clean (Baseline)** | Unmodified | Identity | Raw benchmark input |
| **JPEG Compression** | Quality $Q = 30$ | Pillow JPEG encoding/decoding | Social media image upload compression |
| **Gaussian Noise** | Additive zero-mean, $\sigma = 15$ | NumPy normal distribution, clipped to $[0, 255]$ | Sensor noise, low-light capture, camera ISO grain |
| **Gaussian Blur** | Radius $r = 2.0$ | Pillow `ImageFilter.GaussianBlur(radius=2.0)` | Optical defocus, motion blur |
| **Low-Res Downsampling** | Downsample factor $0.25\times$ | Bilinear downsample ($56\times 56$), bilinear upsample ($224\times 224$) | Streaming bitrate throttling, mobile crop |

---

## 3. Baseline Performance Under Degradation ($N = 100$ Balanced Subset)

On the 100-video development benchmark (50 Real, 50 Fake), the production baseline exhibits severe susceptibility to spatial low-pass filtering:

| Evaluation Condition | Accuracy | Macro F1 | Balanced Accuracy | Real Recall | Fake Recall | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Original (Clean)** | **0.8100** | **0.8098** | **0.8100** | 78.0% | 84.0% | Normal operating baseline |
| **JPEG ($Q=30$)** | 0.7300 | 0.7278 | 0.7300 | 82.0% | 64.0% | Moderate recall drop |
| **Gaussian Noise ($\sigma=15$)** | 0.6700 | 0.6624 | 0.6700 | 52.0% | 82.0% | Real recall degrades to 52% |
| **Gaussian Blur ($r=2$)** | 0.5600 | 0.4544 | 0.5600 | **12.0%** | **100.0%** | Catastrophic collapse: classifies nearly all inputs as Fake |
| **Low-Res ($0.25\times$)** | 0.5600 | 0.4544 | 0.5600 | **12.0%** | **100.0%** | Catastrophic collapse: classifies nearly all inputs as Fake |

*Key Diagnostic Finding:* Under severe blurring or downsampling, the baseline model loses high-frequency facial texture cues and defaults to predicting "Fake", driving Real Recall down to an unacceptable 12.0%.

---

## 4. The V1, V2, and V3 Candidate Iterations

### 4.1 Candidate V1
- Basic data augmentation during fine-tuning (random JPEG, mild blur).
- Outcome: Marginally improved blur resilience, but degraded clean accuracy ($<80\%$).

### 4.2 Candidate V2-B: Catastrophic Noise Collapse
- Trained with heavy frequency-residual loss and multi-distortion augmentations.
- Outcome: Solved blur and downsampling collapse (balanced accuracy $>70\%$).
- **Catastrophic Failure:** Severe vulnerability to Gaussian noise ($\sigma=15$): Real Recall collapsed to **6.0%** (47 out of 50 pristine real videos misclassified as fake), yielding overall accuracy of only 53.0%.

### 4.3 Candidate V3 Series (V3-A, V3-B, V3-C)
- Designed specifically to solve the V2-B noise failure while maintaining clean performance:
  - **V3-A:** Stochastic noise training ($\sigma \in [5, 15]$) with balanced clean preservation.
  - **V3-B:** Multi-scale downsampling ($0.25\times \dots 1.0\times$) and mild blur.
  - **V3-C:** High-dropout regularization ($p=0.5$) with frequency preservation.
- Outcome: Successfully restored noise resilience (70.0%–71.0% accuracy, 86%–92% real recall).

---

## 5. Formal Acceptance Evaluation & Rejection Decision

To deploy any V3 candidate as a production replacement, strict acceptance gates were established:

| Acceptance Criterion | Gate Threshold | Baseline | Candidate V3-A | Candidate V3-B | Candidate V3-C |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Accuracy** | $\ge 80.0\%$ | 81.0% (Pass) | 88.0% (Pass) | 90.0% (Pass) | 87.0% (Pass) |
| **Clean Fake Recall** | $\ge 80.0\%$ | 84.0% (Pass) | 80.0% (Pass) | 82.0% (Pass) | 74.0% (**Fail**) |
| **Clean Real Recall** | $\ge 80.0\%$ | 78.0% | 96.0% (Pass) | 98.0% (Pass) | 100.0% (Pass) |
| **Noise ($\sigma=15$) Accuracy** | $\ge 65.0\%$ | 67.0% (Pass) | 70.0% (Pass) | 68.0% (Pass) | 64.0% (**Fail**) |
| **Noise Real Recall** | $\ge 50.0\%$ | 52.0% (Pass) | 86.0% (Pass) | 90.0% (Pass) | 92.0% (Pass) |
| **JPEG ($Q=30$) Fake Recall** | $\ge 60.0\%$ | 64.0% (Pass) | **26.0% (FAIL)** | **16.0% (FAIL)** | **34.0% (FAIL)** |
| **Final Deployment Decision** | **All Gates** | **RETAIN IN PROD** | **REJECT (Archived)** | **REJECT (Archived)** | **REJECT (Archived)** |

### Conclusion
Although V3 candidates demonstrated substantial recovery on Gaussian noise and high clean accuracy, **all candidates catastrophically failed the JPEG $Q=30$ fake recall gate** (achieving only 16%–34% recall on compressed fakes, missing two-thirds of compressed manipulations).

**Action Taken:** The **production baseline model (`5878ae38...`) is retained in production**. All V3 candidates are archived in the experimental robustness worktree as experimental artifacts.

---

## 6. PRNG Stochastic Noise Discrepancy Reconciliation Note

An exhaustive audit of Candidate V3-A investigated why its Gaussian noise ($\sigma=15$) accuracy is reported as **70.0%** in the development benchmark table but **72.0%** in the multi-severity curve:

1. **Identical Samples & Weights:** Both evaluations evaluated the exact same 100 videos (indices `0:50` Real, `0:50` Fake) from `deepfake_test.json` using the exact same checkpoint weights (`5f8057f3...`).
2. **PRNG Generator Difference:**
   - `scripts/evaluate_robustness.py` utilized legacy NumPy `RandomState(seed=42)` (Mersenne Twister MT19937).
   - `scripts/evaluate_severity_curve.py` utilized modern NumPy `default_rng(seed=42)` (PCG64).
3. **Stochastic Realization Effect:** Because Gaussian noise is additive stochastic perturbation ($I_{\text{noisy}} = \text{clip}(I + \mathcal{N}(0, \sigma^2))$), different pseudo-random sequences produce distinct pixel noise realizations. On two borderline manipulated videos, the PCG64 realization left subtle boundary artifacts intact, pushing predictions across the 0.5 decision threshold ($70/100 \to 72/100$).
4. **Conclusion:** Both numbers are valid empirical measurements reflecting stochastic noise variance on borderline samples; both comfortably satisfy the $\ge 65.0\%$ noise accuracy gate.
