import React from 'react';
import { MessageSquare, ShieldAlert, Sparkles, BarChart2 } from 'lucide-react';
import { TokenAttributionView } from './TokenAttributionView';
import { UncertaintyCard } from './UncertaintyCard';
import { formatConfidence } from '../types/schemas';

export function NLPResultView({ propaganda, hateSpeech, textSummary }) {
  const hasText = textSummary?.text_status === 'AVAILABLE' || textSummary?.combined_text?.trim();

  if (!hasText || (!propaganda && !hateSpeech)) {
    return (
      <div className="result-card nlp-empty-card">
        <div className="card-header-clean">
          <MessageSquare size={18} className="text-muted" />
          <h3 className="section-title">Natural Language Processing (Propaganda & Hate Speech)</h3>
          <span className="status-badge badge-neutral">SKIPPED</span>
        </div>
        <p className="empty-text-note">
          NLP verification models were bypassed because no usable textual content was provided or extracted from OCR/ASR.
        </p>
      </div>
    );
  }

  return (
    <div className="nlp-results-container">
      {/* 1. Propaganda Detection Section */}
      {propaganda && propaganda.status === 'SUCCESS' && (
        <div className="result-card">
          <div className="card-header-clean">
            <Sparkles size={20} className="text-primary" />
            <div className="header-title-group">
              <h3 className="section-title">Propaganda Technique Detection (RoBERTa-base)</h3>
              <span className="subtitle-sm">SemEval-2020 Task 11 • 14 Fine-Grained Rhetorical Techniques</span>
            </div>
            <span className="status-badge badge-info">
              {propaganda.prediction || 'UNKNOWN'}
            </span>
          </div>

          <div className="prediction-summary-row">
            <div className="summary-stat-box">
              <span className="stat-label">Predicted Technique</span>
              <span className="stat-value text-primary font-bold">{propaganda.prediction}</span>
              <span className="stat-sub">14-Class Span Classifier</span>
            </div>
            <div className="summary-stat-box">
              <span className="stat-label">Prediction Confidence</span>
              <span className="stat-value font-mono">{formatConfidence(propaganda.confidence)}</span>
              <span className="stat-sub">Stochastic Mean Softmax</span>
            </div>
          </div>

          {/* Top Probability Distribution */}
          {propaganda.probabilities && (
            <div className="top-probabilities-box">
              <div className="top-prob-title">
                <BarChart2 size={14} />
                <span>Top Predicted Rhetorical Techniques:</span>
              </div>
              <div className="prob-pill-grid">
                {Object.entries(propaganda.probabilities)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 4)
                  .map(([technique, prob], idx) => (
                    <div key={idx} className="prob-pill">
                      <span className="prob-tech-name" title={technique}>{technique.replace(/_/g, ' ')}</span>
                      <span className="prob-tech-val font-mono">{formatConfidence(prob)}</span>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Token Attribution */}
          <TokenAttributionView
            explanation={propaganda.explanation}
            title="Propaganda Feature Saliency (RoBERTa Word Embeddings)"
          />

          {/* Uncertainty Metrics */}
          <UncertaintyCard
            uncertainty={propaganda.uncertainty}
            modalityName="Propaganda Model (RoBERTa-base)"
          />
        </div>
      )}

      {/* 2. Hate Speech Detection Section */}
      {hateSpeech && hateSpeech.status === 'SUCCESS' && (
        <div className="result-card">
          <div className="card-header-clean">
            <ShieldAlert size={20} className="text-primary" />
            <div className="header-title-group">
              <h3 className="section-title">Hate Speech & Toxicity Analysis (BERT-base)</h3>
              <span className="subtitle-sm">HateXplain Benchmark • 3-Class Classification (normal, offensive, hatespeech)</span>
            </div>
            <span className={`status-badge ${hateSpeech.prediction === 'hatespeech' ? 'badge-danger' : hateSpeech.prediction === 'offensive' ? 'badge-warning' : 'badge-success'}`}>
              {hateSpeech.prediction ? hateSpeech.prediction.toUpperCase() : 'UNKNOWN'}
            </span>
          </div>

          <div className="prediction-summary-row">
            <div className="summary-stat-box">
              <span className="stat-label">Predicted Category</span>
              <span className={`stat-value font-bold ${hateSpeech.prediction === 'hatespeech' ? 'text-danger' : hateSpeech.prediction === 'offensive' ? 'text-warning' : 'text-success'}`}>
                {hateSpeech.prediction ? hateSpeech.prediction.toUpperCase() : 'N/A'}
              </span>
              <span className="stat-sub">BERT 3-Class Classifier</span>
            </div>
            <div className="summary-stat-box">
              <span className="stat-label">Category Confidence</span>
              <span className="stat-value font-mono">{formatConfidence(hateSpeech.confidence)}</span>
              <span className="stat-sub">
                Normal: {formatConfidence(hateSpeech.probabilities?.normal)} | Offensive: {formatConfidence(hateSpeech.probabilities?.offensive)} | Hate: {formatConfidence(hateSpeech.probabilities?.hatespeech)}
              </span>
            </div>
          </div>

          {/* Token Attribution */}
          <TokenAttributionView
            explanation={hateSpeech.explanation}
            rationaleAlignment={hateSpeech.rationale_alignment}
            title="Hate Speech Feature Saliency (BERT Embeddings)"
          />

          {/* Uncertainty Metrics */}
          <UncertaintyCard
            uncertainty={hateSpeech.uncertainty}
            modalityName="Hate Speech Model (BERT-base)"
            classLabels={['Normal', 'Offensive', 'Hate Speech']}
          />
        </div>
      )}
    </div>
  );
}
