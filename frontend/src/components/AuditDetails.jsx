import React, { useState } from 'react';
import { ChevronDown, ChevronUp, FileCode, CheckCircle2, AlertCircle } from 'lucide-react';

export function AuditDetails({ audit, inputMeta }) {
  const [isOpen, setIsOpen] = useState(false);

  if (!audit) return null;

  return (
    <div className="audit-accordion">
      <button
        type="button"
        className="audit-accordion-header"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
      >
        <div className="flex-center gap-2">
          <FileCode size={18} className="text-muted" />
          <span className="audit-accordion-title">Reproducibility & Audit Trail</span>
          <span className="audit-duration-badge">
            {audit.execution_time_seconds !== undefined ? `${audit.execution_time_seconds}s` : 'Logged'}
          </span>
        </div>
        <span className="btn-icon">
          {isOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </span>
      </button>

      {isOpen && (
        <div className="audit-accordion-body">
          <div className="audit-grid">
            <div className="audit-item">
              <span className="audit-label">Timestamp (UTC):</span>
              <span className="audit-value font-mono">{audit.timestamp || 'N/A'}</span>
            </div>

            <div className="audit-item">
              <span className="audit-label">Input Modality & File:</span>
              <span className="audit-value font-mono">
                {audit.input_type || inputMeta?.input_type || 'N/A'} • {audit.filename || inputMeta?.filename || 'N/A'}
              </span>
            </div>

            <div className="audit-item">
              <span className="audit-label">Models Executed:</span>
              <span className="audit-value font-mono">
                {audit.models_executed && audit.models_executed.length > 0
                  ? audit.models_executed.join(', ')
                  : 'None'}
              </span>
            </div>

            <div className="audit-item">
              <span className="audit-label">Monte Carlo Passes:</span>
              <span className="audit-value font-mono">T = {audit.mc_samples || 20} stochastic passes</span>
            </div>

            <div className="audit-item">
              <span className="audit-label">Optical Character Recognition (OCR):</span>
              <span className="audit-value">
                {audit.ocr_executed ? (
                  <span className="text-success flex-center gap-1"><CheckCircle2 size={13} /> Executed (EasyOCR)</span>
                ) : (
                  <span className="text-muted">Not Executed</span>
                )}
              </span>
            </div>

            <div className="audit-item">
              <span className="audit-label">Speech Recognition (ASR):</span>
              <span className="audit-value">
                {audit.asr_executed ? (
                  <span className="text-success flex-center gap-1"><CheckCircle2 size={13} /> Executed (Whisper-base)</span>
                ) : audit.has_audio === false ? (
                  <span className="text-muted">Bypassed (No Audio Stream)</span>
                ) : (
                  <span className="text-muted">Not Executed</span>
                )}
              </span>
            </div>

            {audit.frames_sampled !== undefined && (
              <div className="audit-item">
                <span className="audit-label">Frames Sampled / Faces Analyzed:</span>
                <span className="audit-value font-mono">
                  {audit.frames_sampled} frames sampled, {audit.faces_analyzed || 0} faces detected
                </span>
              </div>
            )}
          </div>

          {audit.checkpoint_versions && (
            <div className="audit-checkpoints-section">
              <span className="audit-section-subtitle">Active Checkpoint Versions:</span>
              <div className="checkpoint-list">
                {Object.entries(audit.checkpoint_versions).map(([mod, path], idx) => (
                  <div key={idx} className="checkpoint-row">
                    <span className="checkpoint-name font-mono">{mod}:</span>
                    <span className="checkpoint-path font-mono text-muted">{path}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {audit.errors && audit.errors.length > 0 && (
            <div className="audit-errors-section">
              <span className="audit-section-subtitle text-danger flex-center gap-1">
                <AlertCircle size={14} /> Logged Non-Fatal Exceptions:
              </span>
              <ul className="audit-errors-list">
                {audit.errors.map((err, i) => (
                  <li key={i} className="audit-error-item font-mono">{err}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
