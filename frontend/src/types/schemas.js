/**
 * Constants, schemas, and result field extractors for Digital Media Verification.
 */

export const GOVERNANCE_STATUS = {
  CONFIDENT_PREDICTION: 'CONFIDENT_PREDICTION',
  REVIEW_RECOMMENDED: 'REVIEW_RECOMMENDED',
  ANALYSIS_INCOMPLETE: 'ANALYSIS_INCOMPLETE',
  INPUT_ERROR: 'INPUT_ERROR',
  COMPONENT_FAILURE: 'COMPONENT_FAILURE',
};

export const VISUAL_STATUS = {
  SUCCESS: 'SUCCESS',
  NO_FACE_DETECTED: 'NO_FACE_DETECTED',
  NOT_APPLICABLE: 'NOT_APPLICABLE',
};

export const TEXT_STATUS = {
  AVAILABLE: 'AVAILABLE',
  NO_TEXT_AVAILABLE: 'NO_TEXT_AVAILABLE',
};

export const JOB_STATUS = {
  QUEUED: 'QUEUED',
  PROCESSING: 'PROCESSING',
  COMPLETED: 'COMPLETED',
  FAILED: 'FAILED',
};

/**
 * Format confidence percentage safely.
 */
export function formatConfidence(conf) {
  if (conf === null || conf === undefined) return 'N/A';
  return `${(Number(conf) * 100).toFixed(1)}%`;
}

/**
 * Format small statistical decimals.
 */
export function formatStat(num, digits = 4) {
  if (num === null || num === undefined) return 'N/A';
  return Number(num).toFixed(digits);
}

/**
 * Check if human review is recommended.
 */
export function isReviewRecommended(governance) {
  return governance?.review_required === true || governance?.decision_status === GOVERNANCE_STATUS.REVIEW_RECOMMENDED;
}
