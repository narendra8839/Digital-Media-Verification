import React from 'react';
import { Sparkles, Info, Award } from 'lucide-react';

/**
 * Returns an HSL / RGBA style based on token attribution score [0, 1].
 */
function getScoreStyle(score) {
  const s = Math.max(0, Math.min(1, Number(score) || 0));
  const alpha = 0.12 + s * 0.55;
  return {
    backgroundColor: `rgba(239, 68, 68, ${alpha})`,
    borderBottom: s > 0.35 ? `2px solid rgba(220, 38, 38, ${0.4 + s * 0.6})` : '1px solid transparent',
  };
}

export function TokenAttributionView({
  explanation,
  rationaleAlignment = null,
  title = 'Token Saliency & Feature Attribution'
}) {
  if (!explanation) return null;

  const {
    top_tokens = [],
    token_scores = [],
    tokens = [],
    disclaimer
  } = explanation;

  const tokensToRender = token_scores.length > 0
    ? token_scores
    : tokens.length > 0
    ? tokens
    : top_tokens;

  return (
    <div className="token-attribution-section">
      <div className="attribution-header">
        <div className="flex-center gap-2">
          <Sparkles size={16} className="text-primary" />
          <h4 className="attribution-title">{title}</h4>
        </div>
        <span className="source-pill">Grad × Input Gradient Attribution</span>
      </div>

      <div className="token-stream-box">
        <p className="meta-label">Attributed Sequence (Token Saliency Highlights):</p>
        <div className="tokens-canvas-card">
          {tokensToRender.map((t, idx) => {
            const tokenText = t.clean_token || t.token || (typeof t === 'string' ? t : '');
            if (!tokenText) return null;
            const score = t.score !== undefined ? t.score : 0;
            return (
              <span
                key={idx}
                className="token-pill"
                style={getScoreStyle(score)}
                title={`Token: "${tokenText}" | Attribution Score: ${(score * 100).toFixed(1)}%`}
              >
                <span className="token-text">{tokenText}</span>
                {score > 0.2 && (
                  <span className="token-score font-mono">{score.toFixed(2)}</span>
                )}
              </span>
            );
          })}
        </div>
      </div>

      {top_tokens && top_tokens.length > 0 && (
        <div className="top-probabilities-box">
          <span className="meta-label">Top Influential Tokens Driving Prediction:</span>
          <div className="prob-pill-grid">
            {top_tokens.slice(0, 5).map((tok, i) => (
              <div key={i} className="prob-pill">
                <span className="prob-tech-name">#{i + 1} "{tok.clean_token || tok.token}"</span>
                <span className="prob-tech-val font-mono">{((tok.score || 0) * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {rationaleAlignment && (
        <div className="alignment-box">
          <div className="alignment-title">
            <Award size={15} className="text-primary" />
            <span>HateXplain Ground-Truth Rationale Alignment</span>
          </div>
          {rationaleAlignment.interpretation && (
            <p className="alignment-text">{rationaleAlignment.interpretation}</p>
          )}
          <div className="prediction-summary-row">
            <div className="summary-stat-box">
              <span className="stat-label">Precision</span>
              <span className="stat-value font-mono">
                {((rationaleAlignment.precision || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="summary-stat-box">
              <span className="stat-label">Recall</span>
              <span className="stat-value font-mono">
                {((rationaleAlignment.recall || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="summary-stat-box">
              <span className="stat-label">Rationale F1</span>
              <span className="stat-value font-mono">
                {((rationaleAlignment.f1 || 0) * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      )}

      <div className="token-disclaimer-note">
        <Info size={13} className="text-muted flex-shrink-0" />
        <span>
          {disclaimer || 'Token attribution is a post-hoc explanatory signal and is not causal proof.'}
        </span>
      </div>
    </div>
  );
}
