import React, { act } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ResultDashboard } from '../components/ResultDashboard';
import { VisualResultView } from '../components/VisualResultView';
import * as api from '../services/api';

const mockImageResult = {
  status: 'SUCCESS',
  visual: {
    prediction: 'fake',
    confidence: 0.942,
    probabilities: { real: 0.058, fake: 0.942 },
    face_detected: true,
    status: 'SUCCESS',
    explanation: {
      method: 'grad_cam',
      overlay_url: 'data:image/png;base64,mockOverlayData',
      disclaimer: 'Grad-CAM is a post-hoc explanatory signal and should not be interpreted as causal proof.',
      heatmap_shape: [7, 7]
    },
    uncertainty: {
      mean_variance: 0.0125,
      std: 0.1118,
      entropy: 0.2854,
      mc_samples: 20,
      mean_probability: [0.058, 0.942],
      is_elevated: true
    }
  },
  text: {
    text_status: 'AVAILABLE',
    sources: ['OCR'],
    segments: [{ text: 'Suspicious flyer headline', source: 'OCR', confidence: 0.96 }],
    combined_text: 'Suspicious flyer headline'
  },
  propaganda: {
    status: 'SUCCESS',
    prediction: 'Loaded_Language',
    confidence: 0.88,
    probabilities: { Loaded_Language: 0.88, Doubt: 0.12 }
  },
  hate_speech: null,
  governance: {
    decision_status: 'REVIEW_RECOMMENDED',
    review_required: true,
    review_trigger_component: 'Visual Deepfake Model',
    review_justification: 'Predictive variance 0.0125 exceeds threshold 0.0100',
    disclaimer: 'Human review recommended before action.'
  },
  audit: {
    timestamp: '2026-10-27T12:00:00Z',
    input_type: 'IMAGE',
    filename: 'current_suspect_face.png',
    models_executed: ['EfficientNet-B4', 'EasyOCR'],
    mc_samples: 20,
    execution_time_seconds: 0.85
  }
};

describe('Downloadable Image Analysis Report Frontend Feature', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    if (!window.URL.createObjectURL) {
      window.URL.createObjectURL = vi.fn(() => 'blob:mock-pdf-url');
    }
    if (!window.URL.revokeObjectURL) {
      window.URL.revokeObjectURL = vi.fn();
    }
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('renders Download Report button when valid image analysis result is available', () => {
    render(
      <ResultDashboard
        result={mockImageResult}
        inputMeta={{ input_type: 'IMAGE', filename: 'current_suspect_face.png' }}
        onReset={() => {}}
      />
    );

    const downloadBtns = screen.getAllByRole('button', { name: /download report/i });
    expect(downloadBtns.length).toBeGreaterThan(0);
    expect(downloadBtns[0]).toBeInTheDocument();
  });

  it('exports results already produced by application and triggers download without rerunning inference', async () => {
    const mockBlob = new Blob(['%PDF-1.4 mock content'], { type: 'application/pdf' });
    const downloadSpy = vi.spyOn(api, 'downloadImageReport').mockResolvedValue({
      blob: mockBlob,
      filename: 'dmv-image-report-current_suspect_face-2026-10-27.pdf'
    });
    const triggerSpy = vi.spyOn(api, 'triggerBlobDownload').mockImplementation(() => {});

    render(
      <ResultDashboard
        result={mockImageResult}
        inputMeta={{ input_type: 'IMAGE', filename: 'current_suspect_face.png', originalImageUrl: null }}
        onReset={() => {}}
      />
    );

    const downloadBtn = screen.getByTestId('download-report-btn');

    await act(async () => {
      fireEvent.click(downloadBtn);
    });

    // Verify it called downloadImageReport with exact current result and filename
    expect(downloadSpy).toHaveBeenCalledTimes(1);
    expect(downloadSpy).toHaveBeenCalledWith(
      mockImageResult,
      'current_suspect_face.png',
      null
    );

    // Verify triggerBlobDownload received the returned blob and filename
    expect(triggerSpy).toHaveBeenCalledWith(
      mockBlob,
      'dmv-image-report-current_suspect_face-2026-10-27.pdf'
    );
  });

  it('displays loading state while preparing report', async () => {
    let resolvePromise;
    const downloadPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });
    vi.spyOn(api, 'downloadImageReport').mockReturnValue(downloadPromise);
    vi.spyOn(api, 'triggerBlobDownload').mockImplementation(() => {});

    render(
      <ResultDashboard
        result={mockImageResult}
        inputMeta={{ input_type: 'IMAGE', filename: 'current_suspect_face.png' }}
        onReset={() => {}}
      />
    );

    const downloadBtn = screen.getByTestId('download-report-btn');

    act(() => {
      fireEvent.click(downloadBtn);
    });

    // Buttons should now be in loading state
    const loadingIndicators = screen.getAllByText(/Preparing Report\.\.\./i);
    expect(loadingIndicators.length).toBeGreaterThan(0);
    expect(downloadBtn).toBeDisabled();

    // Resolve the promise
    await act(async () => {
      resolvePromise({
        blob: new Blob(['%PDF-1.4 test'], { type: 'application/pdf' }),
        filename: 'report.pdf'
      });
    });

    // Loading state finishes
    expect(screen.queryByText(/Preparing Report\.\.\./i)).not.toBeInTheDocument();
    expect(downloadBtn).not.toBeDisabled();
  });

  it('displays clear error message if report download fails', async () => {
    vi.spyOn(api, 'downloadImageReport').mockRejectedValue(new Error('Backend report generator unavailable (HTTP 500)'));

    render(
      <ResultDashboard
        result={mockImageResult}
        inputMeta={{ input_type: 'IMAGE', filename: 'current_suspect_face.png' }}
        onReset={() => {}}
      />
    );

    const downloadBtn = screen.getByTestId('download-report-btn');

    await act(async () => {
      fireEvent.click(downloadBtn);
    });

    // Error banner should be displayed
    const errorBanner = screen.getByTestId('report-download-error');
    expect(errorBanner).toBeInTheDocument();
    expect(errorBanner).toHaveTextContent(/Backend report generator unavailable/i);
  });

  it('renders Download Report button in VisualResultView for NO_FACE_DETECTED result', () => {
    const noFaceVisual = {
      status: 'NO_FACE_DETECTED',
      face_detected: false
    };

    render(
      <VisualResultView
        visual={noFaceVisual}
        originalImageUrl={null}
      />
    );

    const downloadBtn = screen.getByTestId('visual-download-report-btn');
    expect(downloadBtn).toBeInTheDocument();
  });

  it('handles missing optional fields in result gracefully', async () => {
    const minimalResult = {
      visual: {
        prediction: 'real',
        confidence: 0.85,
        status: 'SUCCESS'
      }
    };

    const mockBlob = new Blob(['%PDF-1.4 minimal'], { type: 'application/pdf' });
    const downloadSpy = vi.spyOn(api, 'downloadImageReport').mockResolvedValue({
      blob: mockBlob,
      filename: 'dmv-image-report-2026-10-27.pdf'
    });
    vi.spyOn(api, 'triggerBlobDownload').mockImplementation(() => {});

    render(
      <ResultDashboard
        result={minimalResult}
        inputMeta={{ input_type: 'IMAGE' }}
        onReset={() => {}}
      />
    );

    const downloadBtn = screen.getByTestId('download-report-btn');
    expect(downloadBtn).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(downloadBtn);
    });

    expect(downloadSpy).toHaveBeenCalledWith(
      minimalResult,
      'analyzed_image.png',
      null
    );
  });
});
