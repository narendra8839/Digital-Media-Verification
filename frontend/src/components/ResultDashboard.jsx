import React from 'react';
import { RotateCcw, AlertOctagon, Info } from 'lucide-react';
import { GovernancePanel } from './GovernancePanel';
import { VisualResultView } from './VisualResultView';
import { TextExtractionView } from './TextExtractionView';
import { NLPResultView } from './NLPResultView';
import { AuditDetails } from './AuditDetails';
import { formatConfidence } from '../types/schemas';

export function ResultDashboard({ result, inputMeta = {}, onReset }) {
  if (!result) return null;

  const { visual, text, propaganda, hate_speech, governance, audit } = result;

  // Determine primary prediction and confidence for the summary bar
  let primaryLabel = 'Verification Complete';
  let primaryConfidence = 'N/A';
  let isHarmful = false;

  if (visual && visual.status !== 'NO_FACE_DETECTED' && visual.prediction) {
    primaryLabel = visual.prediction === 'fake' ? 'Manipulated Media' : 'Authentic Media';
    primaryConfidence = formatConfidence(visual.confidence);
    isHarmful = visual.prediction === 'fake';
  } else if (hate_speech && hate_speech.status === 'SUCCESS' && hate_speech.prediction) {
    if (hate_speech.prediction === 'hatespeech') {
      primaryLabel = 'Hate Speech Detected';
      primaryConfidence = formatConfidence(hate_speech.confidence);
      isHarmful = true;
    } else if (hate_speech.prediction === 'offensive') {
      primaryLabel = 'Offensive Language';
      primaryConfidence = formatConfidence(hate_speech.confidence);
      isHarmful = true;
    } else {
      primaryLabel = 'Normal / Non-Toxic';
      primaryConfidence = formatConfidence(hate_speech.confidence);
    }
  } else if (propaganda && propaganda.status === 'SUCCESS' && propaganda.prediction) {
    primaryLabel = `Propaganda: ${propaganda.prediction.replace(/_/g, ' ')}`;
    primaryConfidence = formatConfidence(propaganda.confidence);
    isHarmful = true;
  } else if (visual && visual.status === 'NO_FACE_DETECTED') {
    primaryLabel = 'No Face Detected (Bypassed)';
    primaryConfidence = 'N/A';
  }

  const isReviewRequired = governance?.review_required || governance?.decision_status === 'REVIEW_RECOMMENDED';
  const isIncomplete = governance?.decision_status === 'ANALYSIS_INCOMPLETE';

  return (
    <div className="result-dashboard" data-testid="result-dashboard">
      {/* 1. Analyzed Input & Top Navigation Bar */}
      <div className="dashboard-action-bar">
        <div className="dashboard-meta-summary">
          <span className="dashboard-pill">
            Modality: <strong className="font-mono">{audit?.input_type || inputMeta.input_type || 'Media'}</strong>
          </span>
          {inputMeta.filename && (
            <span className="dashboard-pill">
              File: <strong className="font-mono">{inputMeta.filename}</strong>
            </span>
          )}
          {audit?.execution_time_seconds !== undefined && (
            <span className="dashboard-pill">
              Latency: <strong className="font-mono">{audit.execution_time_seconds}s</strong>
            </span>
          )}
        </div>

        {onReset && (
          <button
            type="button"
            className="btn btn-secondary btn-sm reset-btn"
            onClick={onReset}
          >
            <RotateCcw size={14} />
            <span>Verify Another Media</span>
          </button>
        )}
      </div>

      {/* 2. Primary Assessment Banner */}
      <div className="result-summary-card">
        <div className="result-summary-item">
          <span className="summary-label">Primary Assessment</span>
          <span className={`summary-value font-bold ${isHarmful ? 'text-danger' : 'text-main'}`}>
            {primaryLabel}
          </span>
        </div>
        <div className="result-summary-divider" />
        <div className="result-summary-item">
          <span className="summary-label">Aggregate Confidence</span>
          <span className="summary-value font-mono">{primaryConfidence}</span>
        </div>
        <div className="result-summary-divider" />
        <div className="result-summary-item">
          <span className="summary-label">Human Review Status</span>
          <span className={`status-pill ${isReviewRequired ? 'pill-warning' : isIncomplete ? 'pill-info' : 'pill-success'}`}>
            {governance?.decision_status || (isReviewRequired ? 'REVIEW_RECOMMENDED' : 'CONFIDENT_PREDICTION')}
          </span>
        </div>
      </div>

      {/* 3. Responsible AI Oversight & Human Review Recommendation */}
      <GovernancePanel governance={governance} />

      {/* 4. Extracted Supporting Evidence (OCR & Whisper ASR Provenance) */}
      {text && <TextExtractionView textSummary={text} />}

      {/* 5. Modality Forensic Saliency & Uncertainty Analysis */}
      {visual && (
        <VisualResultView
          visual={visual}
          originalImageUrl={inputMeta.originalImageUrl || null}
        />
      )}

      {(propaganda || hate_speech) && (
        <NLPResultView
          propaganda={propaganda}
          hateSpeech={hate_speech}
          textSummary={text}
        />
      )}

      {/* 6. Forensic Limitations & Methodological Constraints Callout */}
      <section className="system-limitations-card">
        <div className="limitations-header">
          <div className="flex-center gap-2">
            <Info size={16} className="text-secondary" />
            <h4 className="limitations-title">Forensic Limitations & Methodological Bounds</h4>
          </div>
          <span className="limitations-scope-tag font-mono">Decision-Support Protocol</span>
        </div>
        <div className="limitations-grid">
          <div className="limitation-item">
            <span className="limitation-label">Uncalibrated Epistemic Uncertainty</span>
            <p className="limitation-text">
              MC Dropout variance reflects stochastic model dispersion (T = {audit?.mc_samples || 20} passes), not calibrated statistical error probability.
            </p>
          </div>
          <div className="limitation-item">
            <span className="limitation-label">Post-Hoc Saliency Explanations</span>
            <p className="limitation-text">
              Grad-CAM and token attributions are post-hoc explanatory signals highlighting internal feature activation, not causal proof of manual tampering.
            </p>
          </div>
          <div className="limitation-item">
            <span className="limitation-label">Sparse Temporal Sampling Protocol</span>
            <p className="limitation-text">
              Videos are uniformly sampled at 1.0 FPS (up to 4 frames). Frame-to-frame temporal inconsistency or intra-second audio-visual desync is not evaluated.
            </p>
          </div>
          <div className="limitation-item">
            <span className="limitation-label">Cascaded Provenance Dependency</span>
            <p className="limitation-text">
              Downstream NLP classification accuracy is conditioned on EasyOCR and Whisper-base transcription fidelity under real-world noise.
            </p>
          </div>
        </div>
      </section>

      {/* 7. Reproducibility & Audit Trail */}
      {audit && <AuditDetails audit={audit} inputMeta={inputMeta} />}
    </div>
  );
}

export default ResultDashboard;
