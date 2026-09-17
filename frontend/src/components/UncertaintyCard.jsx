import React from 'react';
import { Activity, Info, BarChart2, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { formatStat, formatConfidence } from '../types/schemas';

export function UncertaintyCard({
  uncertainty,
  modalityName = 'Model',
  classLabels = null,
  isElevated = null
}) {
  if (!uncertainty) {
    return (
      <div className="uncertainty-empty-card">
        <span className="text-muted">Uncertainty estimation not applicable for this component.</span>
      </div>
    );
  }

  const {
    mean_probability,
    variance,
    std,
    entropy,
    mean_variance,
    max_variance,
    mc_samples = 20,
    is_stochastic
  } = uncertainty;

  const meanVar = mean_variance ?? (Array.isArray(variance) ? variance[0] : variance);
  const stdVal = std ? (Array.isArray(std) ? std[0] : std) : (meanVar !== undefined ? Math.sqrt(Number(meanVar)) : null);
  const elevated = isElevated !== null ? isElevated : uncertainty.is_elevated;

  return (
    <div className="uncertainty-section">
      <div className="uncertainty-header">
        <div className="flex-center gap-2">
          <Activity size={18} className="text-primary" />
          <h4 className="uncertainty-title">{modalityName} Uncertainty Quantification</h4>
        </div>
        <span className="mc-tag">MC Dropout: T = {mc_samples} Passes</span>
      </div>

      <div className="uncertainty-metrics-grid">
        <div className="metric-card">
          <span className="metric-name">Mean Variance (σ²)</span>
          <span className="metric-val font-mono">{formatStat(meanVar, 5)}</span>
          <span className="metric-desc">Predictive dispersion</span>
        </div>

        <div className="metric-card">
          <span className="metric-name">Std Deviation (σ)</span>
          <span className="metric-val font-mono">{formatStat(stdVal, 4)}</span>
          <span className="metric-desc">Spread of predictions</span>
        </div>

        <div className="metric-card">
          <span className="metric-name">Shannon Entropy (H)</span>
          <span className="metric-val font-mono">{formatStat(entropy, 4)} <small>bits</small></span>
          <span className="metric-desc">Distributional entropy</span>
        </div>

        <div className="metric-card">
          <span className="metric-name">Sampling Passes</span>
          <span className="metric-val font-mono">{mc_samples}</span>
          <span className="metric-desc">{is_stochastic !== false ? 'Stochastic MC' : 'Deterministic'}</span>
        </div>
      </div>

      {mean_probability && Array.isArray(mean_probability) && mean_probability.length > 0 && (
        <div className="distribution-box">
          <div className="distribution-title">
            <BarChart2 size={14} />
            <span>Mean Probability Distribution Across Classes:</span>
          </div>
          <div className="class-dist-list">
            {mean_probability.map((prob, idx) => {
              const label = classLabels && classLabels[idx] ? classLabels[idx] : `Class ${idx}`;
              const pct = (prob * 100).toFixed(1);
              return (
                <div key={idx} className="class-dist-row">
                  <span className="class-dist-label" title={label}>{label}</span>
                  <div className="dist-bar-track">
                    <div className="dist-bar-fill" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="class-dist-val font-mono">{pct}%</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {elevated !== undefined && (
        <div className={`entropy-callout ${elevated ? 'text-warning' : 'text-success'}`}>
          {elevated ? (
            <>
              <AlertTriangle size={15} className="flex-shrink-0" />
              <span>Model uncertainty is elevated. Human verification is recommended.</span>
            </>
          ) : (
            <>
              <CheckCircle2 size={15} className="flex-shrink-0" />
              <span>Model uncertainty is below the current development threshold.</span>
            </>
          )}
        </div>
      )}

      <div className="gradcam-disclaimer-note">
        <Info size={13} className="text-muted flex-shrink-0" />
        <span>
          Evaluated via Monte Carlo Dropout inference passes (T = {mc_samples}). Avoid over-interpreting uncalibrated probabilities; these metrics represent model epistemic uncertainty.
        </span>
      </div>
    </div>
  );
}
