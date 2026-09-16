import React, { useState } from 'react';
import { FileText, ArrowRight, Sparkles } from 'lucide-react';

const SAMPLES = [
  {
    label: 'Propaganda Rhetoric',
    text: 'The mainstream fake news media will never tell you the truth about this dangerous deep-state conspiracy.'
  },
  {
    label: 'Election Fraud Claim',
    text: 'They say millions of fraudulent illegal ballots were dumped at midnight, corrupting our democratic integrity.'
  },
  {
    label: 'Neutral News Statement',
    text: 'The meteorological agency announced that seasonal rainfall will peak next week across coastal districts.'
  },
  {
    label: 'Potentially Toxic Rhetoric',
    text: 'These corrupt illegal intruders are destroying our communities and must be driven out immediately.'
  }
];

export function TextInput({ onVerify, isLoading, disabled = false }) {
  const [text, setText] = useState('');
  const maxLength = 10000;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim() && !isLoading) {
      onVerify(text.trim());
    }
  };

  const handleClear = () => {
    setText('');
  };

  const loadSample = (sampleText) => {
    setText(sampleText);
  };

  return (
    <form className="input-card" onSubmit={handleSubmit}>
      <div className="input-card-header">
        <div className="card-title-group">
          <FileText size={20} className="text-primary" />
          <h3 className="card-title">Direct Text Verification</h3>
        </div>
        <span className="card-hint">
          {text.length} / {maxLength} characters
        </span>
      </div>

      <div className="sample-chips-row">
        <span className="sample-chips-label">
          <Sparkles size={14} />
          <span>Demo Samples:</span>
        </span>
        {SAMPLES.map((s, idx) => (
          <button
            key={idx}
            type="button"
            className="chip-btn"
            onClick={() => loadSample(s.text)}
            disabled={disabled || isLoading}
          >
            {s.label}
          </button>
        ))}
      </div>

      <div className="form-group">
        <textarea
          className="form-textarea"
          rows={6}
          placeholder="Paste or enter text content here to analyze for 14 propaganda techniques and hate speech classification..."
          value={text}
          maxLength={maxLength}
          onChange={(e) => setText(e.target.value)}
          disabled={disabled || isLoading}
        />
      </div>

      <div className="form-actions">
        {text && (
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleClear}
            disabled={isLoading}
          >
            Clear
          </button>
        )}
        <button
          type="submit"
          className="btn btn-primary"
          disabled={!text.trim() || isLoading || disabled}
        >
          <span>{isLoading ? 'Analyzing Text...' : 'Verify Text'}</span>
          <ArrowRight size={16} />
        </button>
      </div>
    </form>
  );
}
