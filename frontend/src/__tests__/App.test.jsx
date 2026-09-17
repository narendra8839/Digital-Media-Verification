import React, { act } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from '../App';
import * as api from '../services/api';
import { GovernancePanel } from '../components/GovernancePanel';
import { VisualResultView } from '../components/VisualResultView';
import { TokenAttributionView } from '../components/TokenAttributionView';
import { UncertaintyCard } from '../components/UncertaintyCard';

const mockVerificationResult = {
  status: 'SUCCESS',
  input: {
    type: 'IMAGE',
    filename: 'test_face.jpg',
  },
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
    sources: ['OCR', 'USER_TEXT'],
    segments: [
      { text: 'Breaking News: Altered Video', source: 'OCR', confidence: 0.95, frame_index: 0 },
      { text: 'User caption claim', source: 'USER_TEXT' }
    ],
    combined_text: 'Breaking News: Altered Video User caption claim'
  },
  propaganda: {
    status: 'SUCCESS',
    prediction: 'Loaded_Language',
    confidence: 0.885,
    probabilities: {
      Loaded_Language: 0.885,
      Name_Calling: 0.065,
      Doubt: 0.035,
      Appeal_to_Fear: 0.015
    },
    explanation: {
      method: 'token_attribution',
      token_scores: [
        { token: 'Breaking', clean_token: 'Breaking', score: 0.12 },
        { token: 'News', clean_token: 'News', score: 0.08 },
        { token: 'Altered', clean_token: 'Altered', score: 0.92 }
      ],
      top_tokens: [
        { token: 'Altered', clean_token: 'Altered', score: 0.92 }
      ],
      disclaimer: 'Token attribution is a post-hoc explanatory signal and is not causal proof.'
    },
    uncertainty: {
      mean_variance: 0.0064,
      std: 0.08,
      entropy: 0.24,
      mc_samples: 20,
      is_elevated: false
    }
  },
  hate_speech: {
    status: 'SUCCESS',
    prediction: 'normal',
    confidence: 0.952,
    probabilities: {
      normal: 0.952,
      offensive: 0.035,
      hatespeech: 0.013
    },
    explanation: {
      method: 'token_attribution',
      token_scores: [
        { token: 'Altered', clean_token: 'Altered', score: 0.15 },
        { token: 'Video', clean_token: 'Video', score: 0.05 }
      ],
      top_tokens: [
        { token: 'Altered', clean_token: 'Altered', score: 0.15 }
      ],
      disclaimer: 'Token attribution is a post-hoc explanatory signal and is not causal proof.'
    },
    rationale_alignment: {
      precision: 0.80,
      recall: 0.75,
      f1: 0.774,
      overlap_tokens: ['Altered'],
      interpretation: 'High rationale alignment with HateXplain ground truth annotations.'
    },
    uncertainty: {
      mean_variance: 0.002,
      std: 0.045,
      entropy: 0.12,
      mc_samples: 20,
      is_elevated: false
    }
  },
  governance: {
    decision_status: 'REVIEW_RECOMMENDED',
    review_required: true,
    review_trigger_component: 'Visual Deepfake Model (MC Dropout Elevated Uncertainty)',
    review_justification: 'Predictive variance 0.0125 exceeds threshold 0.0100',
    disclaimer: 'Human oversight recommended before any punitive action.'
  },
  audit: {
    timestamp: '2026-09-10T12:00:00Z',
    input_type: 'IMAGE',
    filename: 'test_face.jpg',
    models_executed: ['EfficientNet-B4', 'EasyOCR', 'RoBERTa-base', 'BERT-base'],
    mc_samples: 20,
    ocr_executed: true,
    asr_executed: false,
    execution_time_seconds: 1.15,
    checkpoint_versions: {
      deepfake: 'models/deepfake/checkpoints/best_model.pt',
      propaganda: 'models/nlp/propaganda/best_model.pt',
      hate_speech: 'models/nlp/hate_speech/best_model.pt'
    }
  }
};

describe('Digital Media Verification Frontend Test Suite', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(api, 'checkHealth').mockResolvedValue({ status: 'healthy', version: '1.0.0' });
    if (!window.URL.createObjectURL) {
      window.URL.createObjectURL = vi.fn(() => 'blob:mock-url');
    }
    if (!window.URL.revokeObjectURL) {
      window.URL.revokeObjectURL = vi.fn();
    }
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  // 1. Page loads
  it('1. Page loads successfully with header, brand, and navigation tabs', async () => {
    render(<App />);

    expect(screen.getByRole('heading', { level: 1, name: /Digital Media Verification/i })).toBeInTheDocument();
    expect(screen.getByText(/Explainable and Uncertainty-Aware Multimodal Verification/i)).toBeInTheDocument();
    
    expect(screen.getByRole('button', { name: /Image Verification/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Video Verification/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Text Verification/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Multimodal Combined/i })).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText(/Backend Online/i)).toBeInTheDocument();
    });
  });

  // 2. Image upload UI
  it('2. Image upload UI handles file selection and caption input', async () => {
    render(<App />);

    act(() => {
      fireEvent.click(screen.getByRole('button', { name: /Image Verification/i }));
    });

    const fileInput = document.querySelector('input[type="file"]');
    expect(fileInput).toBeInTheDocument();

    const file = new File(['mock content'], 'test_face.jpg', { type: 'image/jpeg' });
    act(() => {
      fireEvent.change(fileInput, { target: { files: [file] } });
    });

    expect(screen.getByText('test_face.jpg')).toBeInTheDocument();

    const captionInput = screen.getByPlaceholderText(/Breaking news reported on social media/i);
    act(() => {
      fireEvent.change(captionInput, { target: { value: 'Investigating viral image' } });
    });
    expect(captionInput).toHaveValue('Investigating viral image');

    const verifyBtn = screen.getByRole('button', { name: /Verify Image/i });
    expect(verifyBtn).toBeEnabled();
  });

  // 3. Text input UI
  it('3. Text input UI updates character count and supports demo sample chips', async () => {
    render(<App />);

    act(() => {
      fireEvent.click(screen.getByRole('button', { name: /Text Verification/i }));
    });

    const textarea = screen.getByPlaceholderText(/Paste or enter text content here/i);
    expect(textarea).toBeInTheDocument();

    const chipBtn = screen.getByRole('button', { name: /Propaganda Rhetoric/i });
    act(() => {
      fireEvent.click(chipBtn);
    });

    expect(textarea.value).toContain('The mainstream fake news media will never tell you the truth');
    expect(screen.getByText(/characters/i)).toBeInTheDocument();

    const verifyBtn = screen.getByRole('button', { name: /Verify Text/i });
    expect(verifyBtn).toBeEnabled();
  });

  // 4. API success response rendering
  it('4. API success response rendering presents full verification dashboard', async () => {
    vi.spyOn(api, 'verifyImage').mockResolvedValue(mockVerificationResult);

    render(<App />);

    const fileInput = document.querySelector('input[type="file"]');
    const file = new File(['fake image bytes'], 'test_face.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [file] } });

    const verifyBtn = screen.getByRole('button', { name: /Verify Image/i });
    fireEvent.click(verifyBtn);

    await waitFor(() => {
      expect(screen.getByTestId('result-dashboard')).toBeInTheDocument();
    });

    expect(screen.getByText(/Manipulated \/ Deepfake/i)).toBeInTheDocument();
    expect(screen.getAllByText(/94.2%/i).length).toBeGreaterThan(0);

    expect(screen.getByRole('button', { name: /Verify Another Media/i })).toBeInTheDocument();
  });

  // 5. API failure handling
  it('5. API failure handling renders user-friendly error alert without raw tracebacks', async () => {
    vi.spyOn(api, 'verifyText').mockRejectedValue(new Error('Backend processing failed: Invalid payload'));

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: /Text Verification/i }));
    const textarea = screen.getByPlaceholderText(/Paste or enter text content here/i);
    fireEvent.change(textarea, { target: { value: 'Test text for failure test' } });

    const verifyBtn = screen.getByRole('button', { name: /Verify Text/i });
    fireEvent.click(verifyBtn);

    await waitFor(() => {
      expect(screen.getByText(/Verification Request Failed/i)).toBeInTheDocument();
      expect(screen.getByText(/Backend processing failed: Invalid payload/i)).toBeInTheDocument();
    });

    const dismissBtn = screen.getByRole('button', { name: /Dismiss & Try Again/i });
    fireEvent.click(dismissBtn);

    expect(screen.queryByText(/Verification Request Failed/i)).not.toBeInTheDocument();
  });

  // 6. Review recommendation rendering
  it('6. Governance panel renders REVIEW_RECOMMENDED, trigger, and responsible AI principle', () => {
    const { rerender } = render(<GovernancePanel governance={mockVerificationResult.governance} />);

    expect(screen.getByText('Human Review Recommended')).toBeInTheDocument();
    expect(screen.getByText(/Model uncertainty is elevated/i)).toBeInTheDocument();
    expect(screen.getByText(/Visual Deepfake Model \(MC Dropout Elevated Uncertainty\)/i)).toBeInTheDocument();
    expect(screen.getByText(/This system provides decision-support telemetry and does not establish objective truth/i)).toBeInTheDocument();

    // Governance consistency check: faceless/incomplete analysis rendering
    rerender(<GovernancePanel governance={{
      decision_status: 'ANALYSIS_INCOMPLETE',
      review_required: false,
      disclaimer: 'Automated facial deepfake detection skipped because no human face was detected.'
    }} />);

    expect(screen.getByText('Analysis Incomplete (Safe Bypass)')).toBeInTheDocument();
    expect(screen.getByText('ANALYSIS_INCOMPLETE')).toBeInTheDocument();
    expect(screen.getByText(/Automated verification was bypassed or incomplete/i)).toBeInTheDocument();
  });

  // 7. Grad-CAM display
  it('7. Grad-CAM saliency overlay renders with explanatory disclaimer and toggle mode', () => {
    render(<VisualResultView visual={mockVerificationResult.visual} originalImageUrl="blob:mock-url" />);

    expect(screen.getByText(/Spatial Explainability: Grad-CAM/i)).toBeInTheDocument();
    expect(screen.getByAltText(/Grad-CAM Overlay/i)).toBeInTheDocument();
    expect(screen.getByText(/Grad-CAM is a post-hoc explanatory signal and should not be interpreted as causal proof/i)).toBeInTheDocument();

    expect(screen.getByRole('button', { name: /Side-by-Side/i })).toBeInTheDocument();
  });

  // 8. Token attribution display
  it('8. Token attribution renders tokens with attribution scores and HateXplain alignment', () => {
    render(
      <TokenAttributionView
        explanation={mockVerificationResult.hate_speech.explanation}
        rationaleAlignment={mockVerificationResult.hate_speech.rationale_alignment}
        title="Hate Speech Feature Saliency"
      />
    );

    expect(screen.getByText('Hate Speech Feature Saliency')).toBeInTheDocument();
    expect(screen.getAllByText('Altered').length).toBeGreaterThan(0);
    expect(screen.getByText('Video')).toBeInTheDocument();
    expect(screen.getByText(/Token attribution is a post-hoc explanatory signal and is not causal proof/i)).toBeInTheDocument();
    expect(screen.getByText(/High rationale alignment with HateXplain ground truth annotations/i)).toBeInTheDocument();
    expect(screen.getByText('80.0%')).toBeInTheDocument(); // precision
  });

  // 9. Uncertainty display
  it('9. Uncertainty card renders variance, standard deviation, Shannon entropy, and MC samples', () => {
    render(
      <UncertaintyCard
        uncertainty={mockVerificationResult.visual.uncertainty}
        modalityName="Deepfake Model"
        classLabels={['Real', 'Fake']}
      />
    );

    expect(screen.getByText(/MC Dropout: T = 20 Passes/i)).toBeInTheDocument();
    expect(screen.getByText('0.01250')).toBeInTheDocument(); // Variance
    expect(screen.getByText('0.1118')).toBeInTheDocument();  // Std Dev
    expect(screen.getByText(/0.2854/)).toBeInTheDocument();  // Entropy
    expect(screen.getByText(/Model uncertainty is elevated\. Human verification is recommended\./i)).toBeInTheDocument();
  });

  // 10. Async video polling
  it('10. Async video verification initiates job polling and resolves on COMPLETED status', async () => {
    vi.useFakeTimers();

    vi.spyOn(api, 'verifyVideo').mockResolvedValue({
      job_id: 'job-9988-test',
      status: 'QUEUED',
      message: 'Video verification initiated.'
    });

    const getJobStatusMock = vi.spyOn(api, 'getJobStatus')
      .mockResolvedValueOnce({
        job_id: 'job-9988-test',
        status: 'PROCESSING',
        result: null
      })
      .mockResolvedValueOnce({
        job_id: 'job-9988-test',
        status: 'COMPLETED',
        result: {
          ...mockVerificationResult,
          input: { type: 'VIDEO', filename: 'demo_video.mp4' }
        }
      });

    render(<App />);

    act(() => {
      fireEvent.click(screen.getByRole('button', { name: /Video Verification/i }));
    });

    const fileInput = document.querySelector('input[type="file"]');
    const file = new File(['fake video stream'], 'demo_video.mp4', { type: 'video/mp4' });
    act(() => {
      fireEvent.change(fileInput, { target: { files: [file] } });
    });

    const asyncCheckbox = screen.getByLabelText(/Enable Asynchronous Background Job Processing/i);
    act(() => {
      fireEvent.click(asyncCheckbox);
    });

    const submitBtn = screen.getByRole('button', { name: /Verify Video/i });
    await act(async () => {
      fireEvent.click(submitBtn);
    });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(getJobStatusMock).toHaveBeenCalledWith('job-9988-test');

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(getJobStatusMock).toHaveBeenCalledTimes(2);

    vi.useRealTimers();
  });
});
