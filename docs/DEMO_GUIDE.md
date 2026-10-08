# Digital Media Verification System: Demonstration Walkthrough

**Academic Project Title:** Explainable and Uncertainty-Aware Multimodal Digital Media Verification  
**Degree Track:** B.Tech Computer Engineering / AI Capstone  
**Target Audience:** Viva Examiners, Project Reviewers, and Evaluators

---

## Prerequisites & Test Assets

Before beginning the demonstration, ensure that:
1. The virtual environment is active.
2. The working directory is set to the repository root on branch `swaraj`.
3. Pre-prepared evaluation assets in `tests/assets/` are ready:
   - `tests/assets/face_image.jpg` (Image with detectable facial landmarks)
   - `tests/assets/no_face_image.jpg` (Image without human faces / scenery)
   - `tests/assets/audio_video.mp4` (Short video container with speech audio track)
   - `tests/assets/silent_video.mp4` (Silent video container)

---

## Step-by-Step Demonstration Flow

### Step 1: Start FastAPI Backend Service
Launch the Uvicorn server in a terminal:
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Verify initial pipeline pre-warming in console output:
  - MultimodalPipeline singleton loaded.
  - Checkpoint models initialized (EfficientNet-B4, RoBERTa-base, BERT-base, Whisper-base, EasyOCR, MTCNN).
- Open `http://localhost:8000/docs` in a browser to show the interactive OpenAPI/Swagger specification.

### Step 2: Start React + Vite Frontend Interface
In a second terminal window, launch the development frontend server:
```powershell
cd frontend
npm run dev
```
- Open `http://localhost:5173` in a browser.

### Step 3: Open Frontend Dashboard
- Point out the academic branding header: *"Digital Media Verification — Explainable and Uncertainty-Aware AI for Digital Media"*.
- Highlight the academic sub-badge: *Ethical & Responsible AI • B.Tech Capstone*.

### Step 4: Verify Backend Health Status
- Direct attention to the status badge in the upper-right corner:
  - Shows a pulsing green indicator: `Backend Online`.
  - Click the refresh button to trigger an immediate `GET /health` poll.
  - Explain to the examiner that this telemetry prevents blind UI requests if the backend is restarting or unavailable.

### Step 5: Upload Image with Detectable Face
- Select the **Image Verification** tab.
- Drag-and-drop or select `tests/assets/face_image.jpg`.
- Observe the client-side thumbnail preview, filename, and file size metadata.
- Enter an optional caption: *"Breaking news verification test"*.
- Click **Verify Image**.

### Step 6: Inspect Deepfake Prediction & Confidence
- Point out the visual prediction card:
  - EfficientNet-B4 Binary Classification output: Authentic / Real vs Manipulated / Deepfake.
  - Confidence percentage (e.g., stochastic mean softmax probability).
  - Individual class probabilities (Real % vs Fake %).

### Step 7: Examine Grad-CAM Convolutional Saliency Map
- Scroll to the **Spatial Explainability: Grad-CAM** section.
- Explain: Activations are extracted directly from the final convolutional block of EfficientNet-B4 (`features[-1]`).
- Click the **Side-by-Side** toggle button to present the original face crop alongside the saliency overlay.
- Highlight the mandatory responsible AI disclaimer:
  > *"Grad-CAM is a post-hoc explanatory signal and should not be interpreted as causal proof."*

### Step 8: Examine Monte Carlo Dropout ($T = 20$) Uncertainty
- Direct attention to the **Uncertainty Quantification** card:
  - Explain the parameters: Mean Variance ($\sigma^2$), Standard Deviation ($\sigma$), and Shannon Entropy ($H$ in bits).
  - Emphasize that $T = 20$ stochastic forward passes evaluate whether the model's prediction sits in a confident or ambiguous decision region.
  - Show the class probability distribution bar.

### Step 9: Review Governance & Human Oversight Panel
- Inspect the top **Responsible AI / Human Oversight** panel:
  - If entropy or variance exceeds threshold ($H > 0.85$ bits), the badge reads `REVIEW_RECOMMENDED`.
  - Shows the triggering component (e.g., `Visual Deepfake Model`).
  - Read the core ethical principle aloud:
    > *"This system provides decision support and does not establish objective truth or automatically take punitive action."*

---

### Step 10: Demonstrate Text Verification
- Click **Verify Another Media** to return to the workspace.
- Switch to the **Text Verification** tab.
- Click the demo sample chip: **Propaganda Rhetoric**.
- Notice the populated text:
  > *"The mainstream fake news media will never tell you the truth about this dangerous deep-state conspiracy."*
- Click **Verify Text**.

### Step 11: Demonstrate Token-Level Feature Attribution
- Point out the **Propaganda Technique Detection** card (RoBERTa-base, SemEval-2020 14-class benchmark):
  - Displays predicted technique (e.g., `Loaded_Language`).
- Inspect the **Attributed Sequence** visualization:
  - Explain that saliency is computed via $Grad \times Input$ on subword embeddings.
  - Tokens like *"fake"*, *"news"*, *"dangerous"* are highlighted in amber/red proportional to attribution intensity.
- Point out the **Hate Speech & Toxicity** card (BERT-base, HateXplain benchmark):
  - Demonstrates multi-task classification (`normal`, `offensive`, `hatespeech`).
  - Shows the HateXplain human rationale alignment metric (Precision, Recall, Rationale F1).

---

### Step 12: Demonstrate Video Verification (Multimodal Pipeline)
- Click **Verify Another Media**.
- Select the **Video Verification** tab.
- Select `tests/assets/audio_video.mp4`.
- Select frame sampling: **1.0 FPS**.
- Check the box: **Enable Asynchronous Background Job Processing**.
- Click **Verify Video**.
- Show the animated processing spinner with job ID tracking (`QUEUED` $\rightarrow$ `PROCESSING` $\rightarrow$ `COMPLETED`).

### Step 13: Show Optical Character Recognition (OCR)
- Scroll down to the **Extracted Text Provenance** section.
- Identify the **SOURCE: OCR** panel (EasyOCR CRAFT detector).
- Show detected on-screen text segments with frame timestamps and recognition confidence scores.

### Step 14: Show Speech Recognition (ASR)
- In the adjacent panel, identify **SOURCE: ASR** (OpenAI Whisper-base).
- Show the extracted audio transcription with source attribution to the 16 kHz audio waveform.
- Point out the clear visual separation between OCR and ASR, preventing provenance confusion.

### Step 15: Show Downstream Multimodal NLP Outputs
- Show how aggregated OCR and ASR text was forwarded to the RoBERTa and BERT classifiers.
- Show joint predictions for rhetorical techniques and toxicity.

### Step 16: Demonstrate Safe Faceless Handling (Governance Consistency)
- Run a quick test with `tests/assets/no_face_image.jpg`:
  - Show that visual status displays `NO_FACE_DETECTED`.
  - Point out that the governance panel displays `ANALYSIS_INCOMPLETE` instead of falsely reporting `CONFIDENT_PREDICTION`.
  - Explain this key Responsible AI design decision: the system refuses to guess real/fake when facial landmarks are absent.

### Step 17: Inspect Audit & Reproducibility Details
- Click the collapsible **Reproducibility & Audit Trail** accordion at the bottom of the dashboard:
  - Timestamp (UTC)
  - Modality & filename
  - Models executed
  - Checkpoint version paths (`best_model.pt`)
  - Frames sampled & faces detected
  - Total end-to-end execution latency in seconds
- Conclude the demonstration by answering examiner viva questions.
