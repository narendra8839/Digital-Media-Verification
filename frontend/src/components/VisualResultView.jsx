import React, { useState } from 'react';
import { Eye, UserX, Sliders, Info, Shield } from 'lucide-react';
import { UncertaintyCard } from './UncertaintyCard';
import { VISUAL_STATUS, formatConfidence } from '../types/schemas';

export function VisualResultView({ visual, originalImageUrl = null }) {
  const [viewMode, setViewMode] = useState('overlay'); // 'overlay' | 'side-by-side'

  if (!visual || visual.status === VISUAL_STATUS.NOT_APPLICABLE) {
    return null;
  }

  // Handle NO_FACE_DETECTED
  if (visual.status === VISUAL_STATUS.NO_FACE_DETECTED || visual.face_detected === false) {
    return (
      <div className="result-card">
        <div className="card-header-clean">
          <UserX size={20} className="text-warning" />
          <h3 className="section-title">Facial Deepfake Analysis</h3>
          <span className="status-badge badge-warning">NO_FACE_DETECTED</span>
        </div>
        <div className="no-face-alert">
          <p className="no-face-text">
            <strong>No human face was detected in the input media.</strong> Automated deepfake detection was safely bypassed.
          </p>
          <span className="text-muted small">
            Decision support protocol: The system strictly refrains from guessing whether faceless media is authentic or manipulated.
          </span>
        </div>
      </div>
    );
  }

  const isFake = visual.prediction === 'fake';
  const confidence = visual.confidence;
  const explanation = visual.explanation || visual.representative_frame?.explanation;
  const overlayUrl = explanation?.saved_path || explanation?.overlay_url;
  const disclaimer = explanation?.disclaimer || "Grad-CAM is a post-hoc explanatory signal and should not be interpreted as causal proof.";

  return (
    <div className="result-card">
      <div className="card-header-clean">
        <Eye size={20} className="text-primary" />
        <h3 className="section-title">Visual Deepfake Detection & Saliency (EfficientNet-B4)</h3>
        <span className={`status-badge ${isFake ? 'badge-danger' : 'badge-success'}`}>
          PREDICTED: {visual.prediction ? visual.prediction.toUpperCase() : 'UNKNOWN'}
        </span>
      </div>

      <div className="prediction-summary-row">
        <div className="summary-stat-box">
          <span className="stat-label">Model Classification</span>
          <span className={`stat-value font-bold ${isFake ? 'text-danger' : 'text-success'}`}>
            {visual.prediction === 'fake' ? 'Manipulated / Deepfake' : 'Authentic / Real'}
          </span>
          <span className="stat-sub">EfficientNet-B4 Binary Classifier</span>
        </div>

        <div className="summary-stat-box">
          <span className="stat-label">Prediction Confidence</span>
          <span className="stat-value font-mono">{formatConfidence(confidence)}</span>
          <span className="stat-sub">
            Real: {formatConfidence(visual.probabilities?.real)} | Fake: {formatConfidence(visual.probabilities?.fake)}
          </span>
        </div>

        {visual.frames_analyzed !== undefined && (
          <div className="summary-stat-box">
            <span className="stat-label">Temporal Aggregation</span>
            <span className="stat-value font-mono">{visual.frames_analyzed} Frames</span>
            <span className="stat-sub">From {visual.total_frames_sampled || visual.frames_analyzed} sampled</span>
          </div>
        )}
      </div>

      {/* Grad-CAM Visualization */}
      <div className="gradcam-section">
        <div className="gradcam-header">
          <div className="flex-center gap-2">
            <Sliders size={16} className="text-primary" />
            <h4 className="gradcam-title">Spatial Explainability: Grad-CAM Convolutional Saliency</h4>
          </div>
          <div className="view-toggle-btns">
            <button
              type="button"
              className={`toggle-btn ${viewMode === 'overlay' ? 'active' : ''}`}
              onClick={() => setViewMode('overlay')}
            >
              Overlay
            </button>
            {originalImageUrl && (
              <button
                type="button"
                className={`toggle-btn ${viewMode === 'side-by-side' ? 'active' : ''}`}
                onClick={() => setViewMode('side-by-side')}
              >
                Side-by-Side
              </button>
            )}
          </div>
        </div>

        <div className="gradcam-viewport">
          {viewMode === 'side-by-side' && originalImageUrl ? (
            <div className="side-by-side-grid">
              <div className="image-panel">
                <span className="panel-tag">Original Input</span>
                <img src={originalImageUrl} alt="Original input crop" className="panel-img" />
              </div>
              <div className="image-panel">
                <span className="panel-tag">Grad-CAM Heatmap Overlay</span>
                {overlayUrl ? (
                  <img src={overlayUrl} alt="Grad-CAM Overlay" className="panel-img" />
                ) : (
                  <div className="heatmap-fallback-box">
                    <span className="text-muted">Heatmap dimensions: {explanation?.heatmap_shape?.join(' × ') || '7 × 7'}</span>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="single-overlay-wrap">
              {overlayUrl ? (
                <img src={overlayUrl} alt="Grad-CAM Overlay" className="gradcam-img" />
              ) : originalImageUrl ? (
                <div className="annotated-image-wrap">
                  <img src={originalImageUrl} alt="Input media crop" className="gradcam-img" />
                  <div className="saliency-note-overlay">
                    <span>Saliency feature map: {explanation?.heatmap_shape?.join(' × ') || '7 × 7'} Conv2d Activation</span>
                  </div>
                </div>
              ) : (
                <div className="heatmap-fallback-box">
                  <p className="text-muted">Grad-CAM activations computed on final convolutional layer <code>features[-1]</code></p>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="gradcam-disclaimer-note">
          <Info size={14} className="text-muted flex-shrink-0" />
          <span>
            <strong>Post-Hoc Signal:</strong> {disclaimer} High saliency highlights pixels that influenced model activations, not causal proof of manual pixel manipulation.
          </span>
        </div>
      </div>

      {/* Uncertainty Card */}
      <UncertaintyCard
        uncertainty={visual.uncertainty}
        modalityName="Deepfake Model (EfficientNet-B4)"
        classLabels={['Real', 'Fake']}
      />
    </div>
  );
}
