# Multimodal Digital Media Verification — System Architecture

## 1. Overview and High-Level Design

The **Digital Media Verification (DMV)** framework is an end-to-end multimodal system designed to analyze visual, audio, and textual streams from uploaded digital media (images, videos, and raw text). The architecture integrates forensic computer vision, natural language processing, automated speech recognition (ASR), optical character recognition (OCR), explainable AI (XAI), Bayesian uncertainty quantification, and automated human-in-the-loop triage.

```mermaid
flowchart TD
    subgraph Client ["Frontend User Interface (React 18 / Tailwind / Lucide)"]
        UI_Upload["Upload & Ingestion View<br/>(Drag-and-Drop Video / Image / Text)"]
        UI_Dash["Results Dashboard<br/>(Risk Score, Verdicts, Badges)"]
        UI_XAI["Explainability & Visual View<br/>(Grad-CAM Heatmaps, Token Gradients)"]
        UI_PDF["Download Report Action<br/>(Client-Side Blob PDF Streaming)"]
    end

    subgraph API ["FastAPI Asynchronous Gateway (api/main.py)"]
        EP_Text["POST /api/verify/text"]
        EP_Image["POST /api/verify/image"]
        EP_Video["POST /api/verify/video"]
        EP_Artifacts["GET /api/artifacts/{artifact_id}"]
        EP_PDF["POST /api/verify/report/pdf"]
        ArtMgr["Artifact Lifecycle Manager<br/>(api/utils/artifact_manager.py)<br/>TTL Cleanup, Path Traversal Defense"]
        RepGen["ReportLab PDF Generator<br/>(api/utils/report_generator.py)<br/>Zero-Inference Precomputed Render"]
    end

    subgraph Pipeline ["Multimodal Orchestrator (pipeline/video_pipeline.py)"]
        Orch["Pipeline Controller"]
        FrameExtract["Uniform Frame Sampler<br/>(1.0 FPS, 4 Frames/Video)"]
        AudioExtract["FFmpeg Audio Extractor<br/>(16kHz Mono WAV)"]
        OCR_Engine["EasyOCR Engine<br/>(CRAFT + CRNN)"]
        ASR_Engine["OpenAI Whisper ASR<br/>(Base FP16/FP32 Engine)"]
    end

    subgraph Models ["Forensic Detection Models"]
        subgraph VisualPathway ["Visual Analysis Pathway"]
            MTCNN["MTCNN Face Detector<br/>(P-Net, R-Net, O-Net)"]
            EffNet["EfficientNet-B4 Classifier<br/>(models/deepfake/checkpoint_full)"]
            GradCAM["Grad-CAM Saliency Engine<br/>(Last Conv Feature Map)"]
            MCDropout_V["MC Dropout Engine<br/>(T=10 Stochastic Forward Passes)"]
        end

        subgraph TextPathway ["Text Analysis Pathway"]
            RoBERTa["RoBERTa-base Classifier<br/>(14 SemEval Propaganda Techniques)"]
            BERT["BERT-base-uncased<br/>(3-Class HateXplain Model)"]
            TokenGrad["Integrated Gradients<br/>(Token Saliency Scoring)"]
            MCDropout_T["MC Dropout Engine<br/>(Epistemic Uncertainty)"]
        end
    end

    subgraph Fusion ["Multimodal Fusion & Governance (uncertainty/)"]
        UncertaintyEngine["Uncertainty Quantification Engine"]
        FusionRules["Cross-Modal Heuristic Aggregator"]
        Governance["Human-in-the-Loop Gate<br/>(Threshold > 0.08 / Conflict Flag)"]
    end

    %% Flow connections
    UI_Upload -->|Multipart / JSON| API
    EP_Text --> Orch
    EP_Image --> Orch
    EP_Video --> Orch

    Orch --> FrameExtract
    Orch --> AudioExtract

    FrameExtract --> MTCNN
    MTCNN -->|Cropped Faces (224x224)| EffNet
    EffNet --> GradCAM
    EffNet --> MCDropout_V
    GradCAM --> ArtMgr

    FrameExtract --> OCR_Engine
    AudioExtract --> ASR_Engine

    OCR_Engine -->|Extracted Text| RoBERTa
    ASR_Engine -->|Transcribed Audio| RoBERTa
    OCR_Engine -->|Extracted Text| BERT
    ASR_Engine -->|Transcribed Audio| BERT

    RoBERTa --> TokenGrad
    BERT --> TokenGrad
    RoBERTa --> MCDropout_T
    BERT --> MCDropout_T

    EffNet & MCDropout_V & RoBERTa & BERT & MCDropout_T --> FusionRules
    FusionRules --> UncertaintyEngine
    UncertaintyEngine --> Governance
    Governance -->|Structured VerificationResult| API

    API --> UI_Dash
    API --> UI_XAI
    EP_PDF --> RepGen
    RepGen -->|Binary PDF Stream| UI_PDF
```

---

## 2. Core Processing Pathways

### 2.1 Visual Pathway
1. **Frame Extraction:** Videos are decoded via OpenCV or PyAV with uniform 1.0 FPS temporal sampling (capped at 4 representative frames per video to balance latency and coverage).
2. **Face Detection & Alignment:** MTCNN detects facial bounding boxes with 5 facial landmarks. High-confidence faces ($confidence \ge 0.90$) are cropped, normalized to $224 \times 224 \times 3$, and preprocessed with standard ImageNet statistics.
3. **Deepfake Classification:** Preprocessed crops pass through the production **EfficientNet-B4** backbone (`models/deepfake/checkpoint_full/best_model.pt`).
4. **Explainability (Grad-CAM):** Activation gradients from the final convolutional layer are backpropagated with respect to the "Fake" logit, generating a normalized $224 \times 224$ heatmap overlay highlighting manipulated facial regions (e.g., blending boundaries, eye/mouth artifacts).
5. **Bayesian Uncertainty:** Monte Carlo Dropout ($T=10$ stochastic forward passes with active dropout rate $p=0.4$) computes predictive variance $\sigma^2 = \frac{1}{T}\sum (p_t - \bar{p})^2$.

### 2.2 Audio Pathway
1. **Audio Demuxing:** FFmpeg extracts 16 kHz mono PCM audio streams from container video formats (`mp4`, `mkv`, `avi`).
2. **Automated Speech Recognition (ASR):** OpenAI Whisper (`base` architecture) processes audio segments, generating timestamped text transcriptions.
3. **Downstream Handoff:** Transcribed audio text is forwarded to the text analysis pathway.

### 2.3 Text Pathway (OCR & NLP)
1. **Optical Character Recognition (OCR):** EasyOCR scans keyframes for embedded graphics, banners, and chyrons, extracting on-screen text.
2. **Text Normalization:** Extracted OCR text and Whisper ASR transcripts are sanitized, truncated to maximum token lengths ($512$), and batched.
3. **Propaganda Analysis:** Fine-tuned RoBERTa-base evaluates spans across 14 SemEval-2020 Task 11 rhetorical techniques (e.g., *Loaded Language*, *Name Calling*, *Repetition*).
4. **Hate Speech Analysis:** Fine-tuned BERT-base classifies text into `Hate Speech`, `Offensive`, or `Normal`.
5. **Token Attribution:** Saliency gradients highlight influential words directly in the UI dashboard.

---

## 3. Multimodal Fusion and Human Governance

The system enforces a **Human-in-the-Loop** mandate based on automated epistemic uncertainty and multimodal conflict:
- **Heuristic Risk Score:**
  $$\text{Risk} = \max\Big(\text{Deepfake Score}, \alpha \cdot \text{Propaganda Score} + \beta \cdot \text{Hate Score}\Big)$$
- **Automated Human Review Flag:** Triggered when:
  1. Deepfake predictive variance $\sigma^2 > 0.08$.
  2. Conflict arises between visual and audio modalities (e.g., visual face classified as fake while spoken audio is verified authentic).
  3. No face is detected in an uploaded visual file.
  4. File exhibits extreme degradation (JPEG quality $<30$ or resolution $<100\times 100$).

---

## 4. Zero-Inference Downloadable Reporting Architecture

The report generation module (`api/utils/report_generator.py`) generates formal audit reports without invoking model inference:
- **Client Cache Re-use:** When the user clicks "Download Report", the frontend transmits the already computed `VerificationResult` JSON payload and base64-encoded visual Grad-CAM artifact.
- **Multilingual Unicode PDF Engine:** ReportLab dynamically discovers TrueType system fonts (`Nirmala.ttc` for Hindi, `STSong-Light` for Chinese, `DejaVuSans`/`Arial` for Cyrillic and Arabic) and maps unsupported emojis to text tags (`[ALERT]`, `[SHIELD]`).
- **Memory Streaming:** PDF binaries are synthesized in-memory (`io.BytesIO`) and streamed via FastAPI `Response(content, media_type="application/pdf")`.
