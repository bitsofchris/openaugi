import type { ExplorerData } from './types';

const API_BASE = '/api';

/**
 * Load explorer data from the FastAPI backend.
 * Falls back to /fixture.json if the backend is unavailable (dev mode).
 */
export async function fetchData(): Promise<ExplorerData> {
  try {
    const res = await fetch(`${API_BASE}/data`);
    if (!res.ok) throw new Error(`Backend returned ${res.status}`);
    return res.json();
  } catch {
    // Fall back to static fixture for frontend development
    const res = await fetch('/fixture.json');
    if (!res.ok) throw new Error('Could not load fixture.json either');
    return res.json();
  }
}

export async function isBackendAvailable(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(1000) });
    return res.ok;
  } catch {
    return false;
  }
}
