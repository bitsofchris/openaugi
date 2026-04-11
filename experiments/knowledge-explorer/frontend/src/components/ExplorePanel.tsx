import { useState, useCallback } from 'react';
import type { ExploreParams, ExploreClusterResult } from '../types';

interface Props {
  onRunUmap: (dims: number, blockIds: string[] | null) => Promise<void>;
  onRunCluster: (params: ExploreParams, blockIds: string[] | null) => Promise<void>;
  clusterResult: ExploreClusterResult | null;
  umapLoading: boolean;
  clusterLoading: boolean;
  umapDims: number | null;      // dims the current UMAP coords were computed at
  breadcrumb: Array<{ label: string; block_ids: string[] | null; params: ExploreParams }>;
  onBreadcrumbNav: (index: number) => void;
  onDrillInto: (label: string) => void;
  selectedLabel: string | null;
  colors: Record<string, string>;
  includeFolders: string[];
}

const DIMS_OPTIONS = [32, 64, 96, 128, 256, 512, 1024, 3072];
const DEFAULT_PARAMS: ExploreParams = {
  algo: 'kmeans',
  dims: 64,
  min_cluster_size: 20,
  min_samples: 5,
  k: 10,
};

function Slider({ label, value, min, max, onChange, colors }: {
  label: string; value: number; min: number; max: number;
  onChange: (v: number) => void; colors: Record<string, string>;
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <span style={{ fontSize: 11, color: colors.textMuted }}>{label}</span>
        <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: colors.accent }}>
          {value}
        </span>
      </div>
      <input type="range" min={min} max={max} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: '100%', accentColor: colors.accent, cursor: 'pointer' }}
      />
    </div>
  );
}

export function ExplorePanel({
  onRunUmap, onRunCluster, clusterResult, umapLoading, clusterLoading,
  umapDims, breadcrumb, onBreadcrumbNav, onDrillInto, selectedLabel, colors, includeFolders,
}: Props) {
  const [params, setParams] = useState<ExploreParams>(DEFAULT_PARAMS);

  const set = useCallback(<K extends keyof ExploreParams>(key: K, val: ExploreParams[K]) => {
    setParams(p => ({ ...p, [key]: val }));
  }, []);

  const currentBlockIds = breadcrumb.length > 0
    ? breadcrumb[breadcrumb.length - 1].block_ids
    : null;

  // When dims changes we need both UMAP + cluster. When only k/algo changes, cluster only.
  const dimsChanged = params.dims !== umapDims;

  const handleRun = useCallback(async () => {
    if (dimsChanged || umapDims === null) {
      // New dims → need fresh UMAP projection first, then cluster
      await onRunUmap(params.dims, currentBlockIds);
      await onRunCluster(params, currentBlockIds);
    } else {
      // Same dims → just recluster (instant recolor, same XY)
      await onRunCluster(params, currentBlockIds);
    }
  }, [params, dimsChanged, umapDims, currentBlockIds, onRunUmap, onRunCluster]);

  const handleCopyParams = useCallback(() => {
    navigator.clipboard.writeText(JSON.stringify(params, null, 2));
  }, [params]);

  const btn = (active: boolean): React.CSSProperties => ({
    padding: '4px 10px', fontSize: 11, fontWeight: 500,
    border: `1px solid ${active ? colors.accent : colors.border}`,
    borderRadius: 4,
    background: active ? `${colors.accent}20` : 'transparent',
    color: active ? colors.accent : colors.textMuted,
    cursor: 'pointer',
  });

  const loading = umapLoading || clusterLoading;
  const sortedClusters = clusterResult
    ? Object.entries(clusterResult.stats).sort((a, b) => b[1].count - a[1].count)
    : [];

  const runLabel = () => {
    if (umapLoading) return 'Projecting…';
    if (clusterLoading) return 'Clustering…';
    if (dimsChanged || umapDims === null) return 'Project + Cluster';
    return 'Recluster';
  };

  return (
    <aside style={{
      width: 260, background: colors.surface,
      borderRight: `1px solid ${colors.border}`,
      display: 'flex', flexDirection: 'column', overflow: 'hidden', flexShrink: 0,
    }}>
      {/* Header */}
      <div style={{ padding: '12px 16px', borderBottom: `1px solid ${colors.border}`, flexShrink: 0 }}>
        <div style={{
          fontSize: 11, fontWeight: 700, color: colors.accent,
          textTransform: 'uppercase', letterSpacing: '0.8px',
          fontFamily: "'JetBrains Mono', monospace",
        }}>
          Explore
        </div>
        {umapDims !== null && (
          <div style={{ fontSize: 10, color: colors.textMuted, marginTop: 3, fontFamily: "'JetBrains Mono', monospace" }}>
            projected at dims={umapDims}
          </div>
        )}
      </div>

      {/* Folder filter */}
      {includeFolders.length > 0 && (
        <div style={{ padding: '8px 16px', borderBottom: `1px solid ${colors.border}`, flexShrink: 0 }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: colors.textMuted, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 5 }}>
            Filtering to folders
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {includeFolders.map(f => {
              const label = f.split('/').pop() ?? f;
              return (
                <div key={f} style={{
                  fontSize: 10, fontFamily: "'JetBrains Mono', monospace",
                  color: colors.accent, background: `${colors.accent}10`,
                  border: `1px solid ${colors.accent}30`,
                  borderRadius: 3, padding: '2px 6px',
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }} title={f}>
                  {label}
                </div>
              );
            })}
          </div>
          <div style={{ fontSize: 9, color: colors.textMuted, marginTop: 5 }}>
            edit ~/.openaugi/explorer_config.json to change
          </div>
        </div>
      )}

      <div style={{ flex: 1, overflow: 'auto' }}>
        {/* Breadcrumb */}
        {breadcrumb.length > 1 && (
          <div style={{ padding: '8px 16px', borderBottom: `1px solid ${colors.border}`, display: 'flex', flexWrap: 'wrap', gap: 4, alignItems: 'center' }}>
            {breadcrumb.map((crumb, i) => (
              <span key={i} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                {i > 0 && <span style={{ color: colors.textMuted, fontSize: 10 }}>›</span>}
                <button onClick={() => onBreadcrumbNav(i)} style={{
                  background: 'none', border: 'none', padding: 0, cursor: 'pointer',
                  fontSize: 10, fontFamily: "'JetBrains Mono', monospace",
                  color: i === breadcrumb.length - 1 ? colors.text : colors.accent,
                  textDecoration: i === breadcrumb.length - 1 ? 'none' : 'underline',
                }}>
                  {crumb.label}
                </button>
              </span>
            ))}
          </div>
        )}

        {/* Params */}
        <div style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 14, borderBottom: `1px solid ${colors.border}` }}>

          {/* Algo */}
          <div>
            <div style={{ fontSize: 11, color: colors.textMuted, marginBottom: 6 }}>Algorithm</div>
            <div style={{ display: 'flex', gap: 4 }}>
              <button style={btn(params.algo === 'kmeans')} onClick={() => set('algo', 'kmeans')}>K-Means</button>
              <button style={btn(params.algo === 'hdbscan')} onClick={() => set('algo', 'hdbscan')}>HDBSCAN</button>
            </div>
          </div>

          {/* Dims */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <span style={{ fontSize: 11, color: colors.textMuted }}>Embedding dims</span>
              {dimsChanged && umapDims !== null && (
                <span style={{ fontSize: 10, color: colors.highlight, fontFamily: "'JetBrains Mono', monospace" }}>
                  will re-project
                </span>
              )}
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
              {DIMS_OPTIONS.map(d => (
                <button key={d}
                  style={{ ...btn(params.dims === d), padding: '3px 7px', fontSize: 10 }}
                  onClick={() => set('dims', d)}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>

          {/* Algo params */}
          {params.algo === 'kmeans' ? (
            <Slider label="k (clusters)" value={params.k} min={2} max={150}
              onChange={v => set('k', v)} colors={colors} />
          ) : (
            <>
              <Slider label="min_cluster_size" value={params.min_cluster_size} min={5} max={200}
                onChange={v => set('min_cluster_size', v)} colors={colors} />
              <Slider label="min_samples" value={params.min_samples} min={1} max={50}
                onChange={v => set('min_samples', v)} colors={colors} />
            </>
          )}

          {/* Run + copy */}
          <div style={{ display: 'flex', gap: 6 }}>
            <button onClick={handleRun} disabled={loading} style={{
              flex: 1, padding: '8px 0', fontSize: 12, fontWeight: 600,
              border: `1px solid ${colors.accent}`,
              borderRadius: 6,
              background: loading ? 'transparent' : `${colors.accent}20`,
              color: loading ? colors.textMuted : colors.accent,
              cursor: loading ? 'default' : 'pointer',
            }}>
              {runLabel()}
            </button>
            <button onClick={handleCopyParams} title="Copy params as JSON" style={{
              padding: '8px 10px', fontSize: 11,
              border: `1px solid ${colors.border}`, borderRadius: 6,
              background: 'transparent', color: colors.textMuted, cursor: 'pointer',
            }}>
              copy
            </button>
          </div>

          {/* Loading hint */}
          {umapLoading && (
            <div style={{ fontSize: 10, color: colors.textMuted, fontStyle: 'italic' }}>
              UMAP projection can take 1-5 min for full vault at high dims. Cached after first run.
            </div>
          )}
        </div>

        {/* Result summary */}
        {clusterResult && (
          <div style={{ padding: '10px 16px', borderBottom: `1px solid ${colors.border}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 11, color: colors.textMuted }}>
                {clusterResult.cluster_count} clusters ·{' '}
                {Math.round(clusterResult.noise_count / (Object.keys(clusterResult.labels).length || 1) * 100)}% noise
              </span>
              {clusterResult.cached && (
                <span style={{
                  fontSize: 9, padding: '1px 5px',
                  border: `1px solid ${colors.border}`, borderRadius: 3,
                  color: colors.textMuted, fontFamily: "'JetBrains Mono', monospace",
                }}>cached</span>
              )}
            </div>
          </div>
        )}

        {/* Cluster list */}
        {sortedClusters.length > 0 && (
          <div style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 5 }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: colors.textMuted, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>
              Clusters
            </div>
            {sortedClusters.map(([label, stats]) => {
              const isSelected = selectedLabel === label;
              return (
                <div key={label} onClick={() => onDrillInto(label)} style={{
                  padding: '8px 10px',
                  background: isSelected ? `${colors.accent}15` : colors.bg,
                  border: `1px solid ${isSelected ? colors.accent : colors.border}`,
                  borderRadius: 5, cursor: 'pointer',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 3 }}>
                    <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: isSelected ? colors.accent : colors.text, fontWeight: 600 }}>
                      cluster {label}
                    </span>
                    <span style={{ fontSize: 10, color: colors.textMuted }}>{stats.count.toLocaleString()}</span>
                  </div>
                  <div style={{ fontSize: 10, color: colors.textMuted, lineHeight: 1.5 }}>
                    {stats.sample_titles.slice(0, 3).join(' · ')}
                  </div>
                  {isSelected && (
                    <button onClick={e => { e.stopPropagation(); onDrillInto(label); }} style={{
                      marginTop: 6, padding: '3px 8px', fontSize: 10,
                      border: `1px solid ${colors.accent}`, borderRadius: 3,
                      background: `${colors.accent}20`, color: colors.accent, cursor: 'pointer',
                    }}>
                      ↳ drill in (re-project subset)
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
}
