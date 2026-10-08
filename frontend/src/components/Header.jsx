import React from 'react';
import { ShieldCheck, BookOpen } from 'lucide-react';
import { HealthIndicator } from './HealthIndicator';

export function Header({ isOnline, isChecking, onRefresh }) {
  return (
    <header className="app-header">
      <div className="header-container">
        <div className="header-brand">
          <div className="brand-icon">
            <ShieldCheck size={24} className="icon-shield" />
          </div>
          <div>
            <div className="brand-title-row">
              <h1 className="brand-title">Digital Media Verification</h1>
              <span className="academic-badge">
                <BookOpen size={12} />
                <span>B.Tech Project • Decision Support</span>
              </span>
            </div>
            <p className="brand-subtitle">
              Explainable and Uncertainty-Aware Multimodal Verification
            </p>
          </div>
        </div>
        <div className="header-actions">
          <HealthIndicator
            isOnline={isOnline}
            isChecking={isChecking}
            onRefresh={onRefresh}
          />
        </div>
      </div>
    </header>
  );
}
