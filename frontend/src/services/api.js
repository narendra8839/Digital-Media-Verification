/**
 * Dedicated API Service Layer for Digital Media Verification Backend.
 * Handles HTTP requests, file uploads, error parsing, and async job polling.
 */

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/+$/, '');

/**
 * Standard helper to process response and extract error details cleanly.
 */
async function handleResponse(response) {
  if (!response.ok) {
    let errorDetail = `Request failed with status ${response.status}`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorDetail = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
      } else if (errorJson.error) {
        errorDetail = errorJson.error;
      }
    } catch {
      // Non-JSON error body fallback
    }
    const error = new Error(errorDetail);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

/**
 * Check backend liveness.
 */
export async function checkHealth() {
  const res = await fetch(`${API_BASE_URL}/health`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  return handleResponse(res);
}

/**
 * Check backend model readiness.
 */
export async function checkReadiness() {
  const res = await fetch(`${API_BASE_URL}/ready`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  return handleResponse(res);
}

/**
 * Verify raw text for propaganda and hate speech.
 */
export async function verifyText(text) {
  const res = await fetch(`${API_BASE_URL}/verify/text`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({ text }),
  });
  return handleResponse(res);
}

/**
 * Verify still image for deepfakes and text OCR.
 */
export async function verifyImage(file, caption = null) {
  const formData = new FormData();
  formData.append('file', file);
  if (caption && caption.trim()) {
    formData.append('caption', caption.trim());
  }

  const res = await fetch(`${API_BASE_URL}/verify/image`, {
    method: 'POST',
    headers: { 'Accept': 'application/json' },
    body: formData,
  });
  return handleResponse(res);
}

/**
 * Verify video for facial deepfakes, keyframe OCR, and speech transcription.
 */
export async function verifyVideo(file, { caption = null, asyncMode = false, sampleFps = 1.0, maxFrames = 16 } = {}) {
  const formData = new FormData();
  formData.append('file', file);
  if (caption && caption.trim()) {
    formData.append('caption', caption.trim());
  }
  formData.append('async_mode', String(asyncMode));
  formData.append('sample_fps', String(sampleFps));
  formData.append('max_frames', String(maxFrames));

  const res = await fetch(`${API_BASE_URL}/verify/video`, {
    method: 'POST',
    headers: { 'Accept': 'application/json' },
    body: formData,
  });
  return handleResponse(res);
}

/**
 * Poll the status of an asynchronous background video verification job.
 */
export async function getJobStatus(jobId) {
  const res = await fetch(`${API_BASE_URL}/verify/status/${jobId}`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  return handleResponse(res);
}

/**
 * Verify multimodal input (media file + accompanying text).
 */
export async function verifyMultimodal({ file = null, text = null } = {}) {
  const formData = new FormData();
  if (file) {
    formData.append('file', file);
  }
  if (text && text.trim()) {
    formData.append('text', text.trim());
  }

  const res = await fetch(`${API_BASE_URL}/verify/multimodal`, {
    method: 'POST',
    headers: { 'Accept': 'application/json' },
    body: formData,
  });
  return handleResponse(res);
}

export { API_BASE_URL };
