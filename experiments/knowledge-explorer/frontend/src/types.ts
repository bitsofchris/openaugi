// A data_block from openaugi's SQLite store, projected to 2D.
export interface Block {
  id: string;
  content: string;
  source_path: string;       // from block.metadata.source_path (vault adapter)
  source_content: string;    // truncated raw note text
  x: number;                 // UMAP 2D projection
  y: number;
  date: string | null;       // "YYYY-MM" from block_time, for monthly bucketing
  // Map of pass_id → label_str (e.g. {"life_areas": "3", "life_areas_fine": "3_7"})
  // Absent key = noise for that pass
  cluster_assignments: Record<string, string>;
}

// A context_block:cluster with its optional LLM-generated summary
export interface ClusterInfo {
  id: string;            // cluster block ID in DB
  pass_id: string;       // which pass created this cluster
  label: string;         // the label string (e.g. "3" or "3_7")
  dims: number;          // embedding dims used (lower = coarser)
  summary: string | null;       // null until LLM summaries generated (Step 3)
  member_count: number;
  temporal: {
    first_block: string | null;
    last_block: string | null;
    monthly_counts: Record<string, number>;
    return_events: number;
  } | null;
}

// Info about one clustering pass
export interface PassInfo {
  id: string;           // pass_id (e.g. "life_areas")
  description: string;
  dims: number;         // lower dims = coarser level
  scope: string;        // "all" or "within"
  parent_pass: string | null;
}

export type ColorMode = 'cluster' | 'date' | 'source';

// ── Explore mode ──────────────────────────────────────────────────────────────

export interface ExploreParams {
  algo: 'hdbscan' | 'kmeans';
  dims: number;
  min_cluster_size: number;
  min_samples: number;
  k: number;
}

export interface ExploreBlock {
  id: string;
  x: number;
  y: number;
  label: string;      // cluster label string; "-1" = noise
  content: string;
  source_path: string;
  date: string | null;
}

export interface ExploreClusterStats {
  count: number;
  sample_titles: string[];
}

export interface ExploreResult {
  blocks: ExploreBlock[];
  stats: Record<string, ExploreClusterStats>;  // label → stats
  noise_count: number;
  cluster_count: number;
  cached: boolean;
  params: ExploreParams;
}

// Breadcrumb entry for drill-down navigation
export interface ExploreCrumb {
  label: string;           // display text e.g. "All" or "cluster 3 (847 blocks)"
  block_ids: string[] | null;  // null = all blocks
  params: ExploreParams;
}

// Top-level fixture/API response shape
export interface ExplorerData {
  generated_at: string;
  block_count: number;
  passes: PassInfo[];           // ordered coarse → fine (by dims asc, then scope)
  blocks: Block[];
  clusters: Record<string, ClusterInfo>;  // cluster_block_id → info
  // pass_id → list of cluster_block_ids for that pass (for level navigation)
  clusters_by_pass: Record<string, string[]>;
}
