# Responsible AI, Governance, and Ethical Decision Support

The Digital Media Verification (DMV) framework is engineered as a **decision-support platform** for human analysts, fact-checkers, and content moderators. It is explicitly designed **not** to function as an autonomous truth detector, censorship system, or automated suspension engine.

---

## 1. Core Principles

### 1. Decision Support, Not Truth Detection
The system does not claim to detect objective "truth" or "falsehood." Instead, it evaluates specific physical, statistical, and linguistic signals:
- Artifacts characteristic of face-swapping or neural synthesis (via visual feature extractors).
- Rhetorical persuasion patterns matching established psychological techniques (SemEval-2020 Task 11 taxonomy).
- Hate speech and offensive phrasing targeting protected social groups (HateXplain taxonomy).

### 2. Human-in-the-Loop Governance
Automated decisions are never terminal. All outputs serve as evidentiary signals to assist human experts:
- **`CONFIDENT_PREDICTION`**: Model predictive variance and entropy fall below conservative safety thresholds; predictions and explanations are presented to assist rapid triage.
- **`REVIEW_RECOMMENDED`**: Model uncertainty exceeds threshold, or conflicting multimodal evidence is observed; the item is flagged for mandatory human editorial review.
- **`ANALYSIS_INCOMPLETE`**: Required evidence is missing or unextractable (e.g., an image contains no detectable human faces); the system refuses to fabricate a visual deepfake verdict.

### 3. No Automated Censorship or Account Action
The platform does not interface with deletion APIs, user blocking systems, or account throttling endpoints. Content remains untouched; only an analytical verification dossier is compiled.

---

## 2. Uncertainty Quantification and Disclaimers

### Monte Carlo Dropout (UQ)
Uncertainty estimation is achieved via Monte Carlo (MC) Dropout:
- At inference time, dropout layers remain active across $T = 20$ stochastic forward passes.
- We measure:
  - **Predictive Variance ($\sigma^2$)**: Variance of output probabilities across passes.
  - **Predictive Entropy ($H$)**: Shannon entropy computed over the mean predictive distribution.

### Important Epistemic Disclaimers
1. **Uncertainty is Not Error Probability**:
   A high predictive entropy indicates model hesitation among competing candidate classes, but does not provide a Bayesian guarantee of misclassification.
2. **Calibration Not Quantitatively Established**:
   While MC Dropout provides relative dispersion measures, formal temperature scaling, Platt scaling, or conformal prediction sets have not been quantitatively established over unbounded out-of-domain distributions. Analysts must treat uncertainty scores as heuristic triage signals rather than calibrated probabilities.

---

## 3. Explainability and Post-Hoc Interpretability

To prevent black-box opacity, DMV generates multimodal post-hoc explanations:

1. **Visual Grad-CAM**:
   - Visual explanations are computed directly from the final convolutional layer of the EfficientNet-B4 backbone using gradient backpropagation.
   - Heatmaps visualize which facial regions (e.g., eye seams, blend boundaries, teeth, cheek textures) contributed most heavily to the classification.
   - **Limitation**: Grad-CAM is a post-hoc visualization; it reflects model feature attribution, not infallible ground-truth forensic provenance.

2. **Text Token Attribution**:
   - Linguistic attributions are derived from gradient $\times$ input embeddings over RoBERTa and BERT token representations.
   - Highlights highlight specific rhetorical phrases (e.g., fear-inducing terms or slur tokens).
   - **Limitation**: Attribution highlights correlate with model attention, but must be interpreted in full context by human linguistic analysts.

---

## 4. Incomplete Evidence and Fallback Guarantees

In real-world verification, media assets frequently lack the necessary modality inputs:
- When an uploaded image or video contains **no human faces**, MTCNN face detection yields zero detections.
  - DMV explicitly marks the visual component as `ANALYSIS_INCOMPLETE` with status reason `NO_FACE_DETECTED`.
  - The system will **never** assign an arbitrary "real" or "fake" label to landscapes, objects, or text screenshots.
- When a video has **no audio track** or silent audio, ASR reports empty transcript, and NLP analysis is cleanly bypassed without generating speculative text verdicts.

---

## 5. Ethical Guidelines for Evaluators and Developers

1. **Do not weaponize outputs**: The system must not be used to generate automated copyright strikes, defamatory claims, or targeted harassment.
2. **Contextual Awareness**: Satire, parody, artistic CGI, and reenactments frequently exhibit visual or linguistic properties similar to manipulated media. Human judgment is essential to distinguish benign expression from malicious disinformation.
3. **Data Privacy**: Uploaded verification media must be processed in compliance with applicable privacy regulations.
