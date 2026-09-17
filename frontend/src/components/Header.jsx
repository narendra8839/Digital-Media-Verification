import React from 'react';
import { ShieldCheck, BookOpen } from 'lucide-react';
import { HealthIndicator } from './HealthIndicator';

export function Header({ isOnline, isChecking, onRefreshHealth }) {
  return (
    <header className="app-header">
      <div className="header-container">
        <div className="header-brand">
          <div className="brand-icon">
            <ShieldCheck size={28} className="icon-shield" />
          </div>
          <div>
            <div className="brand-title-row">
              <h1 className="brand-title">Digital Media Verification</h1>
              <span className="academic-badge">
                <BookOpen size={12} />
                <span>Ethical & Responsible AI • B.Tech Capstone</span>
              </span>
            </div>
            <p className="brand-subtitle">
              Explainable and Uncertainty-Aware Multimodal Verification (EfficientNet-B4 • RoBERTa • BERT • Whisper • Grad-CAM)
            </p>
          </div>
        </div>
        <div className="header-actions">
          <HealthIndicator
            isOnline={isOnline}
            isChecking={isChecking}
            onRefresh={onRefreshHealth}
          />
        </div>
      </div>
    </header>
  );
}
