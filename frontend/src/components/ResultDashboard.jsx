import React from 'react';
import { RotateCcw, ArrowLeft, CheckCircle, AlertTriangle } from 'lucide-react';
import { GovernancePanel } from './GovernancePanel';
import { VisualResultView } from './VisualResultView';
import { TextExtractionView } from './TextExtractionView';
import { NLPResultView } from './NLPResultView';
import { AuditDetails } from './AuditDetails';

export function ResultDashboard({ result, inputMeta = {}, onReset }) {
  if (!result) return null;

  const { visual, text, propaganda, hate_speech, governance, audit } = result;

  return (
    <div className="result-dashboard" data-testid="result-dashboard">
      {/* Dashboard Action Header */}
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
            className="btn btn-secondary reset-btn"
            onClick={onReset}
          >
            <RotateCcw size={15} />
            <span>Verify Another Media</span>
          </button>
        )}
      </div>

      {/* 1. Governance / Responsible AI Panel (Top Priority in Decision Support) */}
      <GovernancePanel governance={governance} />

      {/* 2. Visual Deepfake & Grad-CAM Analysis (if applicable) */}
      {visual && (
        <VisualResultView
          visual={visual}
          originalImageUrl={inputMeta.originalImageUrl || null}
        />
      )}

      {/* 3. Text Extraction Provenance (OCR & ASR telemetry) */}
      {text && <TextExtractionView textSummary={text} />}

      {/* 4. Natural Language Processing (Propaganda & Hate Speech with Token Attribution) */}
      {(propaganda || hate_speech) && (
        <NLPResultView
          propaganda={propaganda}
          hateSpeech={hate_speech}
          textSummary={text}
        />
      )}

      {/* 5. Reproducibility & Audit Trail */}
      {audit && <AuditDetails audit={audit} inputMeta={inputMeta} />}
    </div>
  );
}
