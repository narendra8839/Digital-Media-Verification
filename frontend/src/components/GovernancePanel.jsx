import React from 'react';
import { AlertTriangle, CheckCircle, ShieldAlert, Info, HelpCircle } from 'lucide-react';
import { GOVERNANCE_STATUS } from '../types/schemas';

export function GovernancePanel({ governance }) {
  if (!governance) return null;

  const isReview = governance.review_required || governance.decision_status === GOVERNANCE_STATUS.REVIEW_RECOMMENDED;
  const isIncomplete = governance.decision_status === GOVERNANCE_STATUS.ANALYSIS_INCOMPLETE;
  const trigger = governance.review_trigger_component;
  const justification = governance.review_justification || governance.disclaimer;

  return (
    <section className={`governance-panel ${isReview ? 'governance-alert' : isIncomplete ? 'governance-incomplete' : 'governance-confident'}`}>
      <div className="governance-header">
        <div className="governance-icon-wrap">
          {isReview ? (
            <AlertTriangle size={24} className="text-warning" />
          ) : isIncomplete ? (
            <HelpCircle size={24} className="text-info" />
          ) : (
            <CheckCircle size={24} className="text-success" />
          )}
        </div>
        <div className="governance-title-group">
          <span className="governance-pretitle">Responsible AI • Decision Support Oversight</span>
          <h3 className="governance-title">
            {isReview
              ? 'Human Review Recommended'
              : isIncomplete
              ? 'Analysis Incomplete (Safe Bypass)'
              : 'Confident Automated Prediction'}
          </h3>
        </div>
        <div className="governance-badge-wrap">
          <span className={`status-pill ${isReview ? 'pill-warning' : isIncomplete ? 'pill-info' : 'pill-success'}`}>
            {governance.decision_status || (isReview ? 'REVIEW_RECOMMENDED' : 'CONFIDENT_PREDICTION')}
          </span>
        </div>
      </div>

      <div className="governance-body">
        <div className="governance-main-text">
          {isReview ? (
            <p className="governance-recommendation">
              <strong>Model uncertainty is elevated.</strong> Human verification is strongly recommended prior to drawing any conclusions or making moderation decisions.
            </p>
          ) : isIncomplete ? (
            <p className="governance-recommendation">
              <strong>Automated verification was bypassed or incomplete.</strong> The system strictly refrains from guessing or making ungrounded conclusions when required input features (such as facial landmarks or audio streams) are not detectable.
            </p>
          ) : (
            <p className="governance-recommendation">
              <strong>Model uncertainty is below the current development threshold.</strong> Model confidence and statistical consistency meet baseline criteria.
            </p>
          )}
        </div>

        {trigger && (
          <div className="governance-trigger-meta">
            <span className="meta-label">Review Triggering Component:</span>
            <span className="meta-value badge-component">{trigger}</span>
          </div>
        )}

        {justification && (
          <div className="governance-justification">
            <Info size={16} className="text-muted flex-shrink-0" />
            <p className="justification-text">{justification}</p>
          </div>
        )}

        <div className="governance-disclaimer-box">
          <ShieldAlert size={16} className="text-muted flex-shrink-0" />
          <p className="disclaimer-text">
            <strong>Responsible AI Principle:</strong> This system provides decision-support telemetry and does not establish objective truth, absolute veracity, or automatically take punitive action (such as content deletion or account banning).
          </p>
        </div>
      </div>
    </section>
  );
}
