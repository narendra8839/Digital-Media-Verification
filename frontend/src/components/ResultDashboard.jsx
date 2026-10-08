import React from 'react';
import { RotateCcw } from 'lucide-react';
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
      {/* Top Action Bar */}
      <div className="dashboard-action-bar">
        <div className="dashboard-meta-summary">
          <span className="dashboard-pill">
            Modality: <strong>{audit?.input_type || inputMeta.input_type || 'Media'}</strong>
          </span>
          {inputMeta.filename && (
            <span className="dashboard-pill">
              File: <strong className="font-mono">{inputMeta.filename}</strong>
            </span>
          )}
          {audit?.execution_time_seconds !== undefined && (
            <span className="dashboard-pill">
              Latency: <strong>{audit.execution_time_seconds}s</strong>
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

      {/* 1. Result Summary: Prediction | Confidence | Review Status */}
      <div className="result-summary-card">
        <div className="result-summary-item">
          <span className="summary-label">Primary Prediction</span>
          <span className={`summary-value font-bold ${isHarmful ? 'text-danger' : 'text-main'}`}>
            {primaryLabel}
          </span>
        </div>
        <div className="result-summary-divider" />
        <div className="result-summary-item">
          <span className="summary-label">Confidence</span>
          <span className="summary-value font-mono">{primaryConfidence}</span>
        </div>
        <div className="result-summary-divider" />
        <div className="result-summary-item">
          <span className="summary-label">Review Status</span>
          <span className={`status-pill ${isReviewRequired ? 'pill-warning' : isIncomplete ? 'pill-info' : 'pill-success'}`}>
            {governance?.decision_status || (isReviewRequired ? 'REVIEW_RECOMMENDED' : 'CONFIDENT_PREDICTION')}
          </span>
        </div>
      </div>

      {/* 2. Visual Deepfake & Grad-CAM Analysis */}
      {visual && (
        <VisualResultView
          visual={visual}
          originalImageUrl={inputMeta.originalImageUrl || null}
        />
      )}

      {/* 3. Text Extraction Provenance (OCR & ASR) */}
      {text && <TextExtractionView textSummary={text} />}

      {/* 4. Natural Language Processing (Propaganda & Hate Speech) */}
      {(propaganda || hate_speech) && (
        <NLPResultView
          propaganda={propaganda}
          hateSpeech={hate_speech}
          textSummary={text}
        />
      )}

      {/* 5. Responsible AI / Human Oversight Governance Panel */}
      <GovernancePanel governance={governance} />

      {/* 6. Reproducibility & Audit Trail */}
      {audit && <AuditDetails audit={audit} inputMeta={inputMeta} />}
    </div>
  );
}
