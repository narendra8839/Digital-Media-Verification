import React from 'react';
import { Loader2, Clock, Cpu } from 'lucide-react';

export function ProcessingIndicator({ message, jobId, jobStatus }) {
  return (
    <div className="processing-card" role="status" aria-live="polite">
      <div className="processing-spinner-wrap">
        <Loader2 size={36} className="spin text-primary" />
      </div>
      <div className="processing-content">
        <h4 className="processing-title">Verification in Progress</h4>
        <p className="processing-message">
          {message || 'Executing multimodal pipeline models, explainability, and uncertainty estimation...'}
        </p>

        {jobId && (
          <div className="processing-job-meta">
            <div className="job-meta-item">
              <Clock size={14} />
              <span>Job ID: <code>{jobId}</code></span>
            </div>
            <div className="job-meta-item">
              <Cpu size={14} />
              <span>Status: <strong className="status-badge-inline">{jobStatus || 'PROCESSING'}</strong></span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
