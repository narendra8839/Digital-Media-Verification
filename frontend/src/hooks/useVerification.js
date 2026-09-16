import { useState, useCallback, useRef, useEffect } from 'react';
import {
  verifyText,
  verifyImage,
  verifyVideo,
  verifyMultimodal,
  getJobStatus
} from '../services/api';

/**
 * Hook to manage verification requests, loading states, and async job polling.
 */
export function useVerification() {
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeJobId, setActiveJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);

  const pollingRef = useRef(null);

  const clearPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => clearPolling();
  }, [clearPolling]);

  const reset = useCallback(() => {
    clearPolling();
    setIsLoading(false);
    setLoadingMessage('');
    setResult(null);
    setError(null);
    setActiveJobId(null);
    setJobStatus(null);
  }, [clearPolling]);

  /**
   * Poll an async background video verification job.
   */
  const pollAsyncJob = useCallback((jobId) => {
    setActiveJobId(jobId);
    setJobStatus('QUEUED');
    setLoadingMessage('Video verification queued in background. Waiting for processor...');

    let attempts = 0;
    const maxAttempts = 120; // 4 minutes at 2s interval

    pollingRef.current = setInterval(async () => {
      attempts += 1;
      try {
        const jobData = await getJobStatus(jobId);
        setJobStatus(jobData.status);

        if (jobData.status === 'PROCESSING') {
          setLoadingMessage('Processing video frames, speech ASR, and optical character recognition...');
        } else if (jobData.status === 'COMPLETED') {
          clearPolling();
          setIsLoading(false);
          setResult(jobData.result);
          setLoadingMessage('');
        } else if (jobData.status === 'FAILED') {
          clearPolling();
          setIsLoading(false);
          setError(jobData.error || 'Video analysis failed in background task.');
          setLoadingMessage('');
        }

        if (attempts >= maxAttempts) {
          clearPolling();
          setIsLoading(false);
          setError('Job polling timed out after 4 minutes. Please check server logs.');
        }
      } catch (err) {
        clearPolling();
        setIsLoading(false);
        setError(err.message || 'Error checking video job status.');
      }
    }, 2000);
  }, [clearPolling]);

  /**
   * Verify text
   */
  const handleVerifyText = useCallback(async (text) => {
    reset();
    setIsLoading(true);
    setLoadingMessage('Analyzing text for propaganda techniques and hate speech...');
    try {
      const data = await verifyText(text);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to verify text.');
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  }, [reset]);

  /**
   * Verify image
   */
  const handleVerifyImage = useCallback(async (file, caption) => {
    reset();
    setIsLoading(true);
    setLoadingMessage('Analyzing image: detecting facial landmarks, running EfficientNet-B4, and extracting OCR...');
    try {
      const data = await verifyImage(file, caption);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to verify image.');
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  }, [reset]);

  /**
   * Verify video
   */
  const handleVerifyVideo = useCallback(async (file, options = {}) => {
    reset();
    setIsLoading(true);
    setLoadingMessage(options.asyncMode ? 'Submitting video for background analysis...' : 'Processing video frames at 1 FPS and extracting audio...');
    try {
      const data = await verifyVideo(file, options);
      if (options.asyncMode && data.job_id) {
        pollAsyncJob(data.job_id);
      } else {
        setResult(data);
        setIsLoading(false);
        setLoadingMessage('');
      }
    } catch (err) {
      setError(err.message || 'Failed to verify video.');
      setIsLoading(false);
      setLoadingMessage('');
    }
  }, [reset, pollAsyncJob]);

  /**
   * Verify multimodal
   */
  const handleVerifyMultimodal = useCallback(async (payload) => {
    reset();
    setIsLoading(true);
    setLoadingMessage('Processing multimodal media and text channels...');
    try {
      const data = await verifyMultimodal(payload);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to verify multimodal content.');
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  }, [reset]);

  return {
    isLoading,
    loadingMessage,
    result,
    error,
    activeJobId,
    jobStatus,
    verifyText: handleVerifyText,
    verifyImage: handleVerifyImage,
    verifyVideo: handleVerifyVideo,
    verifyMultimodal: handleVerifyMultimodal,
    reset,
  };
}
