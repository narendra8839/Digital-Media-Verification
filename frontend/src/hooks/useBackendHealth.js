import { useState, useEffect, useCallback } from 'react';
import { checkHealth } from '../services/api';

/**
 * Hook to monitor backend liveness without blocking UI interactions.
 */
export function useBackendHealth(pollIntervalMs = 20000) {
  const [isOnline, setIsOnline] = useState(null);
  const [isChecking, setIsChecking] = useState(false);
  const [lastChecked, setLastChecked] = useState(null);

  const check = useCallback(async () => {
    setIsChecking(true);
    try {
      const data = await checkHealth();
      setIsOnline(data?.status === 'healthy');
    } catch {
      setIsOnline(false);
    } finally {
      setIsChecking(false);
      setLastChecked(new Date());
    }
  }, []);

  useEffect(() => {
    check();
    if (pollIntervalMs > 0) {
      const timer = setInterval(check, pollIntervalMs);
      return () => clearInterval(timer);
    }
  }, [check, pollIntervalMs]);

  return { isOnline, isChecking, lastChecked, checkNow: check };
}
