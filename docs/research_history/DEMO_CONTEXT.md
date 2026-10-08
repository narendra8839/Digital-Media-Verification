# Final Demo Context

## 1. Purpose

The final demo should show that the system is not merely a classifier. It should demonstrate:

**analysis + evidence + explanation + uncertainty + responsible review**

## 2. Demo Order

### Case 1 — Image

Upload an image containing a detectable face and, where useful, visible text.

Show:

- face detection status;
- deepfake prediction;
- probabilities/confidence;
- Grad-CAM visualization;
- OCR text;
- propaganda result, if text is available;
- hate-speech result, if text is available;
- uncertainty;
- governance/review status.

The explanation must come from the actual model result.

### Case 2 — Video

Upload a short test video.

Show the asynchronous processing state first.

Then show:

- sampled-frame/video-level visual result;
- representative frame;
- representative Grad-CAM;
- OCR from keyframes;
- Whisper transcription;
- NLP results;
- uncertainty;
- review recommendation;
- relevant audit metadata.

Do not claim the video was analyzed frame-by-frame if the actual configured sampling/limits are different.

### Case 3 — Text

Enter text directly.

Show:

- propaganda prediction and probability distribution;
- hate-speech/toxicity prediction;
- token attribution;
- uncertainty;
- review status.

### Case 4 — High-Uncertainty Example

Use a genuinely ambiguous or uncertain example.

The objective is to show the governance behavior:

```text
REVIEW_RECOMMENDED
```

Explain that uncertainty is a reason to involve a human reviewer, not proof that the content is malicious.

### Case 5 — Incomplete Analysis

A faceless image can demonstrate:

```text
ANALYSIS_INCOMPLETE
NO_FACE_DETECTED
```

This is useful because it shows the system does not fabricate a deepfake prediction when a required visual signal is absent.

## 3. Dashboard Expectations

The frontend should make the following hierarchy obvious:

1. What did the system analyze?
2. What did it predict?
3. What evidence supports the prediction?
4. How uncertain is it?
5. Does a human need to review it?

Use the existing React component structure rather than replacing the UI unnecessarily.

## 4. Design Rule

`DESIGN.md` is the authoritative frontend design/context file.

It is based on a warm cream canvas, warm near-black ink, restrained orange CTA usage, hairline-only depth, editorial typography, and generous spacing. The supplied design tokens/components should be used consistently.

Do not turn the analyst dashboard into a copy of a marketing landing page. Adapt the visual language to an analyst/product surface while preserving the design system's principles.

## 5. Demo Safety

Never:

- fabricate a prediction;
- fabricate a Grad-CAM heatmap;
- fabricate OCR/ASR text;
- claim an uncertain result is certain;
- claim the system proves truth;
- imply that a model output alone should cause punishment.

All visible demo values should come from actual model execution or clearly labeled static test fixtures.
