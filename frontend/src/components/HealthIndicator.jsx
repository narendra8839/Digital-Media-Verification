import React from 'react';
import { Activity, RefreshCw } from 'lucide-react';

export function HealthIndicator({ isOnline, isChecking, onRefresh }) {
  let statusText = 'Checking...';
  let badgeClass = 'badge-checking';

  if (isOnline === true) {
    statusText = 'Backend Online';
    badgeClass = 'badge-online';
  } else if (isOnline === false) {
    statusText = 'Backend Unavailable';
    badgeClass = 'badge-offline';
  }

  return (
    <div className="health-indicator-container">
      <div className={`health-badge ${badgeClass}`}>
        <span className="pulse-dot" />
        <Activity size={14} />
        <span>{statusText}</span>
      </div>
      <button
        onClick={onRefresh}
        disabled={isChecking}
        className="btn-icon"
        title="Check Backend Health"
        aria-label="Refresh backend status"
      >
        <RefreshCw size={13} className={isChecking ? 'spin' : ''} />
      </button>
    </div>
  );
}
