import React from 'react';
import { FileSearch, Mic, Type, Tag } from 'lucide-react';
import { formatConfidence } from '../types/schemas';

export function TextExtractionView({ textSummary }) {
  if (!textSummary) return null;

  const { sources = [], segments = [], combined_text = '', text_status } = textSummary;

  if (text_status === 'NO_TEXT_AVAILABLE' || segments.length === 0) {
    return (
      <div className="result-card text-extraction-empty">
        <div className="card-header-clean">
          <Type size={18} className="text-muted" />
          <h3 className="section-title">Extracted Text & Provenance</h3>
          <span className="status-badge badge-neutral">NO_TEXT_AVAILABLE</span>
        </div>
        <p className="empty-text-note">
          Neither optical character recognition (OCR) nor speech recognition (ASR) detected readable text in this media.
        </p>
      </div>
    );
  }

  const ocrSegments = segments.filter((s) => s.source === 'OCR');
  const asrSegments = segments.filter((s) => s.source === 'ASR');
  const userSegments = segments.filter((s) => s.source === 'USER_TEXT');

  return (
    <div className="result-card">
      <div className="card-header-clean">
        <FileSearch size={20} className="text-primary" />
        <h3 className="section-title">Extracted Text Provenance & Multi-Source Telemetry</h3>
        <div className="sources-badges-row">
          {sources.map((src, i) => (
            <span key={i} className="source-pill">
              <Tag size={12} />
              <span>{src}</span>
            </span>
          ))}
        </div>
      </div>

      <div className="text-sources-grid">
        {/* OCR Section */}
        {ocrSegments.length > 0 && (
          <div className="source-column">
            <div className="source-column-header">
              <Type size={16} className="text-primary" />
              <span className="source-column-title">Optical Character Recognition (EasyOCR)</span>
              <span className="provenance-tag">SOURCE: OCR</span>
            </div>
            <div className="source-segments-list">
              {ocrSegments.map((seg, idx) => (
                <div key={idx} className="segment-card">
                  <p className="segment-content">"{seg.text}"</p>
                  <div className="segment-meta">
                    {seg.frame_index !== undefined && <span>Frame #{seg.frame_index}</span>}
                    {seg.confidence !== undefined && <span>Confidence: {formatConfidence(seg.confidence)}</span>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ASR Section */}
        {asrSegments.length > 0 && (
          <div className="source-column">
            <div className="source-column-header">
              <Mic size={16} className="text-primary" />
              <span className="source-column-title">Speech Transcription (Whisper-base)</span>
              <span className="provenance-tag">SOURCE: ASR</span>
            </div>
            <div className="source-segments-list">
              {asrSegments.map((seg, idx) => (
                <div key={idx} className="segment-card asr-card">
                  <p className="segment-content">"{seg.text}"</p>
                  <div className="segment-meta">
                    <span>16 kHz Audio Waveform</span>
                    <span>openai/whisper-base</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* User Caption Section */}
        {userSegments.length > 0 && (
          <div className="source-column">
            <div className="source-column-header">
              <FileSearch size={16} className="text-primary" />
              <span className="source-column-title">User Provided Context</span>
              <span className="provenance-tag">SOURCE: USER_TEXT</span>
            </div>
            <div className="source-segments-list">
              {userSegments.map((seg, idx) => (
                <div key={idx} className="segment-card">
                  <p className="segment-content">"{seg.text}"</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {combined_text && (
        <div className="combined-text-box">
          <span className="combined-text-label">Unified Aggregated Text (Forwarded to NLP Classifiers):</span>
          <p className="combined-text-body font-mono">"{combined_text}"</p>
        </div>
      )}
    </div>
  );
}
