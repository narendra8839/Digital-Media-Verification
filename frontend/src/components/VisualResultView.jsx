import React, { useState, useEffect } from 'react';
import { Eye, UserX, Sliders, Info, Shield, AlertTriangle, FileDown, Loader2, AlertCircle } from 'lucide-react';
import { UncertaintyCard } from './UncertaintyCard';
import { VISUAL_STATUS, formatConfidence } from '../types/schemas';
import { API_BASE_URL, downloadImageReport, triggerBlobDownload } from '../services/api';

export function VisualResultView({
  visual,
  originalImageUrl = null,
  onDownloadReport = null,
  isDownloading: externalIsDownloading = false,
  downloadError: externalDownloadError = null,
  result = null,
  filename = null
}) {
  const [viewMode, setViewMode] = useState('overlay'); // 'overlay' | 'side-by-side'
  const [overlayLoadFailed, setOverlayLoadFailed] = useState(false);
  const [internalDownloading, setInternalDownloading] = useState(false);
  const [internalError, setInternalError] = useState(null);

  const isDownloading = onDownloadReport ? externalIsDownloading : internalDownloading;
  const downloadError = onDownloadReport ? externalDownloadError : internalError;

  const handleDownloadClick = async () => {
    if (onDownloadReport) {
      onDownloadReport();
      return;
    }
    if (internalDownloading) return;
    setInternalDownloading(true);
    setInternalError(null);
    try {
      let previewData = null;
      if (originalImageUrl) {
        if (originalImageUrl.startsWith('data:')) {
          previewData = originalImageUrl;
        } else if (originalImageUrl.startsWith('blob:')) {
          try {
            const resp = await fetch(originalImageUrl);
            const blob = await resp.blob();
            previewData = await new Promise((resolve) => {
              const reader = new FileReader();
              reader.onloadend = () => resolve(reader.result);
              reader.onerror = () => resolve(null);
              reader.readAsDataURL(blob);
            });
          } catch {
            previewData = null;
          }
        }
      }
      const activeFilename = filename || 'analyzed_image.png';
      const payloadResult = result || { visual };
      const { blob, filename: suggestedFilename } = await downloadImageReport(
        payloadResult,
        activeFilename,
        previewData
      );
      const fallbackDate = new Date().toISOString().slice(0, 10);
      triggerBlobDownload(blob, suggestedFilename || `dmv-image-report-${fallbackDate}.pdf`);
    } catch (err) {
      setInternalError(err.message || 'Failed to download report.');
    } finally {
      setInternalDownloading(false);
    }
  };

  if (!visual || visual.status === VISUAL_STATUS.NOT_APPLICABLE) {
    return null;
  }

  // Handle NO_FACE_DETECTED
  if (visual.status === VISUAL_STATUS.NO_FACE_DETECTED || visual.face_detected === false) {
    return (
      <div className="result-card">
        <div className="card-header-clean">
          <div className="flex-center gap-2">
            <UserX size={20} className="text-warning" />
            <h3 className="section-title">Facial Deepfake Analysis</h3>
          </div>
          <div className="header-actions-group">
            <button
              type="button"
              className="btn btn-secondary btn-sm download-report-btn"
              onClick={handleDownloadClick}
              disabled={isDownloading}
              aria-label="Download Report"
              data-testid="visual-download-report-btn"
            >
              {isDownloading ? (
                <>
                  <Loader2 size={13} className="spin" />
                  <span>Preparing Report...</span>
                </>
              ) : (
                <>
                  <FileDown size={13} />
                  <span>Download Report</span>
                </>
              )}
            </button>
            <span className="status-badge badge-warning">NO_FACE_DETECTED</span>
          </div>
        </div>
        <div className="no-face-alert">
          <p className="no-face-text">
            <strong>No human face was detected in the input media.</strong> Automated deepfake detection was safely bypassed.
          </p>
          <span className="text-muted small">
            Decision support protocol: The system strictly refrains from guessing whether faceless media is authentic or manipulated.
          </span>
        </div>
        {downloadError && !onDownloadReport && (
          <div className="report-error-banner" role="alert">
            <AlertCircle size={15} className="text-danger flex-shrink-0" />
            <span className="report-error-content">{downloadError}</span>
          </div>
        )}
      </div>
    );
  }

  const isFake = visual.prediction === 'fake';
  const confidence = visual.confidence;
  const explanation = visual.explanation || visual.representative_frame?.explanation;
  const rawOverlay = explanation?.saved_path || explanation?.overlay_url;
  const overlayUrl = rawOverlay
    ? (rawOverlay.startsWith('data:') || rawOverlay.startsWith('http') ? rawOverlay : `${API_BASE_URL}${rawOverlay}`)
    : null;
  const disclaimer = explanation?.disclaimer || "Grad-CAM is a post-hoc explanatory signal and should not be interpreted as causal proof.";

  useEffect(() => {
    setOverlayLoadFailed(false);
  }, [overlayUrl]);

  return (
    <div className="result-card">
      <div className="card-header-clean">
        <div className="flex-center gap-2">
          <Eye size={20} className="text-primary" />
          <h3 className="section-title">Visual Deepfake Detection & Saliency (EfficientNet-B4)</h3>
        </div>
        <div className="header-actions-group">
          <button
            type="button"
            className="btn btn-secondary btn-sm download-report-btn"
            onClick={handleDownloadClick}
            disabled={isDownloading}
            aria-label="Download Report"
            data-testid="visual-download-report-btn"
          >
            {isDownloading ? (
              <>
                <Loader2 size={13} className="spin" />
                <span>Preparing Report...</span>
              </>
            ) : (
              <>
                <FileDown size={13} />
                <span>Download Report</span>
              </>
            )}
          </button>
          <span className={`status-badge ${isFake ? 'badge-danger' : 'badge-success'}`}>
            PREDICTED: {visual.prediction ? visual.prediction.toUpperCase() : 'UNKNOWN'}
          </span>
        </div>
      </div>

      {downloadError && !onDownloadReport && (
        <div className="report-error-banner" role="alert" style={{ margin: '10px 0' }}>
          <AlertCircle size={15} className="text-danger flex-shrink-0" />
          <span className="report-error-content">{downloadError}</span>
        </div>
      )}


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
                {overlayUrl && !overlayLoadFailed ? (
                  <img
                    src={overlayUrl}
                    alt="Grad-CAM Overlay"
                    className="panel-img"
                    onError={() => setOverlayLoadFailed(true)}
                  />
                ) : overlayLoadFailed ? (
                  <div className="artifact-expired-notice" role="alert">
                    <div className="expired-notice-header">
                      <AlertTriangle size={15} className="text-warning flex-shrink-0" />
                      <span className="expired-notice-title">Overlay Expired</span>
                    </div>
                    <p className="expired-notice-desc">
                      The spatial Grad-CAM artifact has expired (24h retention) or is no longer available on the server. Rerun verification to regenerate.
                    </p>
                  </div>
                ) : (
                  <div className="heatmap-fallback-box">
                    <span className="text-muted">Heatmap dimensions: {explanation?.heatmap_shape?.join(' × ') || '7 × 7'}</span>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="single-overlay-wrap">
              {overlayUrl && !overlayLoadFailed ? (
                <img
                  src={overlayUrl}
                  alt="Grad-CAM Overlay"
                  className="gradcam-img"
                  onError={() => setOverlayLoadFailed(true)}
                />
              ) : overlayLoadFailed ? (
                <div className="artifact-expired-notice" role="alert">
                  <div className="expired-notice-header">
                    <AlertTriangle size={16} className="text-warning flex-shrink-0" />
                    <span className="expired-notice-title">Explanation Overlay Expired</span>
                  </div>
                  <p className="expired-notice-desc">
                    The spatial Grad-CAM overlay for this analysis has reached its retention limit (24 hours) or is no longer available on the server. Please rerun verification to regenerate spatial explainability overlays.
                  </p>
                </div>
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
