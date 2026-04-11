import { useState, useCallback } from 'react';
import type { ExploreParams, ExploreResult, ExploreCrumb } from '../types';

interface Props {
  onRun: (params: ExploreParams, blockIds: string[] | null) => Promise<void>;
  result: ExploreResult | null;
  loading: boolean;
  breadcrumb: ExploreCrumb[];
  onBreadcrumbNav: (index: number) => void;
  onDrillInto: (label: string) => void;
  selectedLabel: string | null;
  colors: Record<string, string>;
}

const DIMS_OPTIONS = [32, 64, 96, 128, 256, 512, 1024, 3072];
const DEFAULT_PARAMS: ExploreParams = {
  algo: 'hdbscan',
  dims: 64,
  min_cluster_size: 20,
  min_samples: 5,
  k: 10,
};

function Slider({
  label, value, min, max, step = 1, onChange, colors,
}: {
  label: string; value: number; min: number; max: number; step?: number;
  onChange: (v: number) => void; colors: Record<string, string>;
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 11, color: colors.textMuted }}>{label}</span>
        <span style={{
          fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
          color: colors.accent, minWidth: 28, textAlign: 'right',
        }}>{value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: '100%', accentColor: colors.accent, cursor: 'pointer' }}
      />
    </div>
  );
}

export function ExplorePanel({
  onRun, result, loading, breadcrumb, onBreadcrumbNav, onDrillInto, selectedLabel, colors,
}: Props) {
  const [params, setParams] = useState<ExploreParams>(DEFAULT_PARAMS);

  const set = useCallback(<K extends keyof ExploreParams>(key: K, val: ExploreParams[K]) => {
    setParams(p => ({ ...p, [key]: val }));
  }, []);

  const handleRun = useCallback(() => {
    const currentCrumb = breadcrumb[breadcrumb.length - 1];
    onRun(params, currentCrumb?.block_ids ?? null);
  }, [onRun, params, breadcrumb]);

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

  const noisePercent = result
    ? Math.round(result.noise_count / (result.blocks.length || 1) * 100)
    : null;

  const sortedClusters = result
    ? Object.entries(result.stats).sort((a, b) => b[1].count - a[1].count)
    : [];

  return (
    <aside style={{
      width: 260,
      background: colors.surface,
      borderRight: `1px solid ${colors.border}`,
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
      flexShrink: 0,
    }}>
      {/* Header */}
      <div style={{
        padding: '12px 16px',
        borderBottom: `1px solid ${colors.border}`,
        flexShrink: 0,
      }}>
        <div style={{
          fontSize: 11, fontWeight: 700, color: colors.accent,
          textTransform: 'uppercase', letterSpacing: '0.8px',
          fontFamily: "'JetBrains Mono', monospace",
        }}>
          Explore
        </div>
      </div>

      <div style={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 0 }}>

        {/* Breadcrumb */}
        {breadcrumb.length > 1 && (
          <div style={{
            padding: '8px 16px',
            borderBottom: `1px solid ${colors.border}`,
            display: 'flex', flexWrap: 'wrap', gap: 4, alignItems: 'center',
          }}>
            {breadcrumb.map((crumb, i) => (
              <span key={i} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                {i > 0 && <span style={{ color: colors.textMuted, fontSize: 10 }}>›</span>}
                <button
                  onClick={() => onBreadcrumbNav(i)}
                  style={{
                    background: 'none', border: 'none', padding: 0, cursor: 'pointer',
                    fontSize: 10, fontFamily: "'JetBrains Mono', monospace",
                    color: i === breadcrumb.length - 1 ? colors.text : colors.accent,
                    textDecoration: i === breadcrumb.length - 1 ? 'none' : 'underline',
                  }}
                >
                  {crumb.label}
                </button>
              </span>
            ))}
          </div>
        )}

        {/* Params */}
        <div style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 14, borderBottom: `1px solid ${colors.border}` }}>

          {/* Algo selector */}
          <div>
            <div style={{ fontSize: 11, color: colors.textMuted, marginBottom: 6 }}>Algorithm</div>
            <div style={{ display: 'flex', gap: 4 }}>
              <button style={btn(params.algo === 'hdbscan')} onClick={() => set('algo', 'hdbscan')}>HDBSCAN</button>
              <button style={btn(params.algo === 'kmeans')} onClick={() => set('algo', 'kmeans')}>K-Means</button>
            </div>
          </div>

          {/* Dims */}
          <div>
            <div style={{ fontSize: 11, color: colors.textMuted, marginBottom: 6 }}>Embedding dims</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
              {DIMS_OPTIONS.map(d => (
                <button key={d} style={{ ...btn(params.dims === d), padding: '3px 7px', fontSize: 10 }}
                  onClick={() => set('dims', d)}>
                  {d}
                </button>
              ))}
            </div>
          </div>

          {/* Algo-specific params */}
          {params.algo === 'hdbscan' ? (
            <>
              <Slider label="min_cluster_size" value={params.min_cluster_size}
                min={5} max={200} onChange={v => set('min_cluster_size', v)} colors={colors} />
              <Slider label="min_samples" value={params.min_samples}
                min={1} max={50} onChange={v => set('min_samples', v)} colors={colors} />
            </>
          ) : (
            <Slider label="k (clusters)" value={params.k}
              min={2} max={50} onChange={v => set('k', v)} colors={colors} />
          )}

          {/* Run + copy */}
          <div style={{ display: 'flex', gap: 6 }}>
            <button
              onClick={handleRun}
              disabled={loading}
              style={{
                flex: 1, padding: '8px 0', fontSize: 12, fontWeight: 600,
                border: `1px solid ${colors.accent}`,
                borderRadius: 6,
                background: loading ? 'transparent' : `${colors.accent}20`,
                color: loading ? colors.textMuted : colors.accent,
                cursor: loading ? 'default' : 'pointer',
              }}
            >
              {loading ? 'Running…' : 'Run'}
            </button>
            <button
              onClick={handleCopyParams}
              title="Copy params as JSON"
              style={{
                padding: '8px 10px', fontSize: 11,
                border: `1px solid ${colors.border}`,
                borderRadius: 6, background: 'transparent',
                color: colors.textMuted, cursor: 'pointer',
              }}
            >
              {}
            </button>
          </div>
        </div>

        {/* Result stats */}
        {result && (
          <div style={{ padding: '12px 16px', borderBottom: `1px solid ${colors.border}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <div style={{ fontSize: 11, color: colors.textMuted }}>
                {result.cluster_count} clusters · {noisePercent}% noise
              </div>
              {result.cached && (
                <span style={{
                  fontSize: 9, padding: '1px 5px',
                  border: `1px solid ${colors.border}`,
                  borderRadius: 3, color: colors.textMuted,
                  fontFamily: "'JetBrains Mono', monospace",
                }}>cached</span>
              )}
            </div>
            <div style={{ fontSize: 10, color: colors.textMuted }}>
              {result.blocks.length.toLocaleString()} blocks projected
            </div>
          </div>
        )}

        {/* Cluster list */}
        {sortedClusters.length > 0 && (
          <div style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{
              fontSize: 10, fontWeight: 600, color: colors.textMuted,
              textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4,
            }}>
              Clusters
            </div>
            {sortedClusters.map(([label, stats]) => {
              const isSelected = selectedLabel === label;
              return (
                <div
                  key={label}
                  style={{
                    padding: '8px 10px',
                    background: isSelected ? `${colors.accent}15` : colors.bg,
                    border: `1px solid ${isSelected ? colors.accent : colors.border}`,
                    borderRadius: 5,
                    cursor: 'pointer',
                  }}
                  onClick={() => onDrillInto(label)}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                    <span style={{
                      fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
                      color: isSelected ? colors.accent : colors.text,
                      fontWeight: 600,
                    }}>
                      cluster {label}
                    </span>
                    <span style={{ fontSize: 10, color: colors.textMuted }}>
                      {stats.count.toLocaleString()}
                    </span>
                  </div>
                  <div style={{ fontSize: 10, color: colors.textMuted, lineHeight: 1.5 }}>
                    {stats.sample_titles.slice(0, 3).join(' · ')}
                  </div>
                  {isSelected && (
                    <button
                      onClick={e => { e.stopPropagation(); onDrillInto(label); }}
                      style={{
                        marginTop: 6, padding: '3px 8px', fontSize: 10,
                        border: `1px solid ${colors.accent}`,
                        borderRadius: 3, background: `${colors.accent}20`,
                        color: colors.accent, cursor: 'pointer',
                      }}
                    >
                      ↳ drill in
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
