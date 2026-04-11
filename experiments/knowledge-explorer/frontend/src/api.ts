import type { ExplorerData, ExploreParams, ExploreUmapResult, ExploreClusterResult } from './types';

const API_BASE = '/api';

export async function fetchData(): Promise<ExplorerData> {
  try {
    const res = await fetch(`${API_BASE}/data`);
    if (!res.ok) throw new Error(`Backend returned ${res.status}`);
    return res.json();
  } catch {
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

/**
 * Compute (or load cached) UMAP 2D projection for a given dims + block subset.
 * Slow on first call for a new dims value; instant after that.
 * Call this when dims changes. Reuse coords when only k changes.
 */
export async function postExploreUmap(
  dims: number,
  block_ids: string[] | null = null,
): Promise<ExploreUmapResult> {
  const res = await fetch(`${API_BASE}/explore/umap`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dims, block_ids }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`UMAP failed (${res.status}): ${text}`);
  }
  return res.json();
}

/**
 * Run k-means or HDBSCAN on truncated embeddings. Fast — no UMAP.
 * Call this when k or algo changes; combine returned labels with existing UMAP coords.
 */
export async function postExploreCluster(
  params: ExploreParams,
  block_ids: string[] | null = null,
): Promise<ExploreClusterResult> {
  const res = await fetch(`${API_BASE}/explore/cluster`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...params, block_ids }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Cluster failed (${res.status}): ${text}`);
  }
  return res.json();
}
