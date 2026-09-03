# Digital Media Verification

An explainable, uncertainty-aware decision-support system for detecting likely deepfakes, propaganda techniques, and hate speech across images, videos, and text.

## Intended workflow

1. Route an uploaded image, video, or caption to the appropriate pipeline.
2. Extract faces, frames, on-screen text, and speech as applicable.
3. Run deepfake, propaganda, and hate-speech classifiers.
4. Produce Grad-CAM and text-attribution explanations.
5. Estimate epistemic uncertainty with Monte Carlo dropout.
6. Escalate inconclusive results for human review and audit logging.

## Quick start

```powershell
cd digital-media-verification
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Model checkpoints and datasets are deliberately excluded from version control. Place them in the matching `checkpoint` and `data/datasets` folders.
