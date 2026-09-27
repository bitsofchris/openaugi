import { useState, useCallback, useEffect, useRef } from 'react';
import type { Block, ClusterInfo, PassInfo } from '../types';

interface Props {
  block: Block | null;
  clusters: Record<string, ClusterInfo>;        // cluster_block_id → info
  clustersByPass: Record<string, string[]>;     // pass_id → [cluster_block_id]
  passes: PassInfo[];
  level: string;  // current pass_id
  onBlockClose: () => void;
  onHighlightCluster: (clusterBlockId: string) => void;
  onShowOnlySource: (sourcePath: string) => void;
  onlySource: string | null;
  colors: Record<string, string>;
}

const MIN_WIDTH = 300;
const MAX_WIDTH = 800;
const DEFAULT_WIDTH = 440;

export function DetailPane({
  block,
  clusters,
  passes,
  level,
  onBlockClose,
  onHighlightCluster,
  onShowOnlySource,
  onlySource,
  colors,
}: Props) {
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartX = useRef(0);
  const dragStartWidth = useRef(0);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    dragStartX.current = e.clientX;
    dragStartWidth.current = width;
  }, [width]);

  useEffect(() => {
    if (!isDragging) return;
    const onMove = (e: MouseEvent) => {
      const delta = dragStartX.current - e.clientX;
      setWidth(Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, dragStartWidth.current + delta)));
    };
    const onUp = () => setIsDragging(false);
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'col-resize';
    return () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    };
  }, [isDragging]);

  const resizeHandle = (
    <div
      onMouseDown={handleMouseDown}
      style={{
        position: 'absolute', left: 0, top: 0, bottom: 0, width: 6,
        cursor: 'col-resize',
        background: isDragging ? colors.accent : 'transparent',
        transition: isDragging ? 'none' : 'background 0.15s',
        zIndex: 10,
      }}
      onMouseEnter={e => { if (!isDragging) e.currentTarget.style.background = `${colors.accent}40`; }}
      onMouseLeave={e => { if (!isDragging) e.currentTarget.style.background = 'transparent'; }}
    />
  );

  const panelStyle: React.CSSProperties = {
    width, minWidth: MIN_WIDTH, maxWidth: MAX_WIDTH,
    background: colors.surface,
    borderLeft: `1px solid ${colors.border}`,
    display: 'flex', flexDirection: 'column',
    overflow: 'hidden', position: 'relative',
  };

  if (!block) {
    return (
      <aside style={{ ...panelStyle, alignItems: 'center', justifyContent: 'center', padding: 32, textAlign: 'center' }}>
        {resizeHandle}
        <div style={{ fontSize: 40, marginBottom: 16, opacity: 0.4 }}>◉</div>
        <h3 style={{ fontSize: 15, fontWeight: 500, color: colors.text, marginBottom: 8 }}>
          Click a block to explore
        </h3>
        <p style={{ fontSize: 13, color: colors.textMuted, lineHeight: 1.6 }}>
          Click any point in the scatter plot to see its content and cluster membership.
        </p>
      </aside>
    );
  }

  const isNoise = !block.cluster_assignments[level];
  const noteName = block.source_path.split('/').pop() ?? block.source_path;

  // Find the cluster_block_id for a given pass
  const getClusterBlockId = (passId: string): string | null => {
    const label = block.cluster_assignments[passId];
    if (!label) return null;
    // Find the cluster block that matches this pass + label
    return Object.entries(clusters).find(
      ([, info]) => info.pass_id === passId && info.label === label
    )?.[0] ?? null;
  };

  return (
    <aside style={panelStyle}>
      {resizeHandle}

      {/* Header */}
      <div style={{
        padding: '14px 20px',
        borderBottom: `1px solid ${colors.border}`,
        display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
        flexShrink: 0,
      }}>
        <div>
          <div style={{
            fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
            color: isNoise ? colors.textMuted : colors.accent,
            marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.5px',
          }}>
            {isNoise ? `noise · not in ${level}` : `cluster · ${level}`}
          </div>
          <div style={{ fontSize: 12, color: colors.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>
            {noteName}
            {block.date && <span style={{ marginLeft: 10, opacity: 0.7 }}>{block.date}</span>}
          </div>
        </div>
        <button
          onClick={onBlockClose}
          style={{ background: 'none', border: 'none', color: colors.textMuted, cursor: 'pointer', fontSize: 20, padding: 4, lineHeight: 1 }}
        >
          ×
        </button>
      </div>

      <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
        {/* Block content */}
        <div style={{ padding: '16px 20px', borderBottom: `1px solid ${colors.border}` }}>
          <div style={{
            fontSize: 11, fontWeight: 600, color: colors.textMuted,
            textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 10,
          }}>
            Block content
          </div>
          <p style={{ fontSize: 14, lineHeight: 1.7, color: colors.text, margin: 0 }}>
            {block.content}
          </p>
        </div>

        {/* Cluster membership — one row per pass */}
        <div style={{ padding: '16px 20px', borderBottom: `1px solid ${colors.border}` }}>
          <div style={{
            fontSize: 11, fontWeight: 600, color: colors.textMuted,
            textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 12,
          }}>
            Cluster membership
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {passes.map(pass => {
              const label = block.cluster_assignments[pass.id];
              const clusterBlockId = getClusterBlockId(pass.id);
              const info = clusterBlockId ? clusters[clusterBlockId] : null;
              const isCurrentLevel = pass.id === level;

              return (
                <div
                  key={pass.id}
                  style={{
                    padding: '10px 12px',
                    background: isCurrentLevel ? `${colors.accent}12` : colors.bg,
                    border: `1px solid ${isCurrentLevel ? colors.accent : colors.border}`,
                    borderRadius: 6,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: label ? 6 : 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{
                        fontSize: 10, fontFamily: "'JetBrains Mono', monospace",
                        color: isCurrentLevel ? colors.accent : colors.textMuted,
                        textTransform: 'uppercase', letterSpacing: '0.5px',
                      }}>
                        {pass.id}
                      </span>
                      <span style={{ fontSize: 10, color: colors.textMuted }}>
                        dims={pass.dims}
                      </span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      {info && (
                        <span style={{ fontSize: 11, color: colors.textMuted }}>
                          {info.member_count.toLocaleString()} blocks
                        </span>
                      )}
                      {label && clusterBlockId && (
                        <button
                          onClick={() => onHighlightCluster(clusterBlockId)}
                          style={{
                            padding: '2px 8px', fontSize: 10,
                            border: `1px solid ${colors.border}`, borderRadius: 4,
                            background: 'transparent', color: colors.textMuted,
                            cursor: 'pointer',
                          }}
                          title="Highlight all members on scatter plot"
                        >
                          highlight
                        </button>
                      )}
                    </div>
                  </div>

                  {label ? (
                    info?.summary ? (
                      <p style={{ fontSize: 12, lineHeight: 1.6, color: colors.text, margin: 0 }}>
                        {info.summary}
                      </p>
                    ) : (
                      <p style={{ fontSize: 12, color: colors.textMuted, margin: 0, fontStyle: 'italic' }}>
                        {label && `Label: ${label} · `}Summary not yet generated — run Step 3
                      </p>
                    )
                  ) : (
                    <p style={{ fontSize: 12, color: colors.textMuted, margin: 0 }}>
                      noise at this level
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Temporal info from cluster */}
        {(() => {
          const clusterBlockId = getClusterBlockId(level);
          const info = clusterBlockId ? clusters[clusterBlockId] : null;
          if (!info?.temporal) return null;
          const { first_block, last_block, return_events, monthly_counts } = info.temporal;
          const monthCount = Object.keys(monthly_counts).length;
          return (
            <div style={{ padding: '16px 20px', borderBottom: `1px solid ${colors.border}` }}>
              <div style={{
                fontSize: 11, fontWeight: 600, color: colors.textMuted,
                textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 10,
              }}>
                Cluster temporal
              </div>
              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                {[
                  { label: 'first', value: first_block },
                  { label: 'last', value: last_block },
                  { label: 'active months', value: String(monthCount) },
                  { label: 'return events', value: String(return_events) },
                ].map(({ label, value }) => value && (
                  <div key={label}>
                    <div style={{ fontSize: 10, color: colors.textMuted, textTransform: 'uppercase', marginBottom: 2 }}>{label}</div>
                    <div style={{ fontSize: 12, color: colors.text, fontFamily: "'JetBrains Mono', monospace" }}>{value}</div>
                  </div>
                ))}
              </div>
            </div>
          );
        })()}

        {/* Source note */}
        <div style={{ padding: '16px 20px' }}>
          <div style={{
            fontSize: 11, fontWeight: 600, color: colors.textMuted,
            textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 8,
          }}>
            Source note
          </div>
          <div style={{
            fontSize: 12, color: colors.textMuted, marginBottom: 10,
            fontFamily: "'JetBrains Mono', monospace",
            display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8,
          }}>
            <span style={{ wordBreak: 'break-all', opacity: 0.8 }}>{block.source_path}</span>
            <button
              onClick={() => onShowOnlySource(block.source_path)}
              style={{
                flexShrink: 0, padding: '3px 10px', fontSize: 10,
                border: `1px solid ${onlySource === block.source_path ? colors.accent : colors.border}`,
                borderRadius: 4,
                background: onlySource === block.source_path ? `${colors.accent}20` : 'transparent',
                color: onlySource === block.source_path ? colors.accent : colors.textMuted,
                cursor: 'pointer',
              }}
            >
              {onlySource === block.source_path ? 'showing only' : 'show only'}
            </button>
          </div>
          {block.source_content && (
            <div style={{
              fontSize: 12, lineHeight: 1.7, color: colors.text, opacity: 0.85,
              whiteSpace: 'pre-wrap', maxHeight: 280, overflow: 'auto',
              padding: 12, background: colors.bg, borderRadius: 6,
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              {block.source_content}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
