import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { fetchData, fetchConfig, postExploreUmap, postExploreCluster } from './api';
import type {
  Block, ColorMode, ExplorerData, PassInfo,
  ExplorePoint, ExploreClusterResult, ExploreCrumb, ExploreParams,
} from './types';
import { ScatterPlot } from './components/ScatterPlot';
import type { ScatterPlotRef } from './components/ScatterPlot';
import { DetailPane } from './components/DetailPane';
import { FilterBar } from './components/FilterBar';
import { TimelinePlayer } from './components/TimelinePlayer';
import { ZoomControls } from './components/ZoomControls';
import { ExplorePanel } from './components/ExplorePanel';

const COLORS = {
  bg: '#0d0d14',
  surface: '#13131f',
  surfaceHover: '#1a1a2e',
  border: '#1e1e2e',
  text: '#e2e8f0',
  textMuted: '#64748b',
  accent: '#818cf8',
  accentLight: '#a5b4fc',
  highlight: '#f59e0b',
};

const EXPLORE_LEVEL = '_explore';

export default function App() {
  const [data, setData] = useState<ExplorerData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Browse mode
  const [level, setLevel] = useState<string>('');
  const [colorMode, setColorMode] = useState<ColorMode>('cluster');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedBlock, setSelectedBlock] = useState<Block | null>(null);
  const [highlightedIds, setHighlightedIds] = useState<Set<string>>(new Set());
  const [onlySource, setOnlySource] = useState<string | null>(null);

  // Timeline
  const [timelineActive, setTimelineActive] = useState(true);
  const [timelineDate, setTimelineDate] = useState<Date | null>(null);

  // Explore mode
  const [exploreMode, setExploreMode] = useState(false);
  const [explorePoints, setExplorePoints] = useState<ExplorePoint[] | null>(null);   // UMAP coords
  const [exploreLabels, setExploreLabels] = useState<ExploreClusterResult | null>(null); // cluster colors
  const [exploreUmapDims, setExploreUmapDims] = useState<number | null>(null);
  const [umapLoading, setUmapLoading] = useState(false);
  const [clusterLoading, setClusterLoading] = useState(false);
  const [exploreError, setExploreError] = useState<string | null>(null);
  const [exploreBreadcrumb, setExploreBreadcrumb] = useState<ExploreCrumb[]>([]);
  const [exploreSelectedLabel, setExploreSelectedLabel] = useState<string | null>(null);

  const [includeFolders, setIncludeFolders] = useState<string[]>([]);

  const scatterRef = useRef<ScatterPlotRef>(null);

  const productionBlockById = useMemo(() => {
    if (!data) return new Map<string, Block>();
    return new Map(data.blocks.map(b => [b.id, b]));
  }, [data]);

  useEffect(() => {
    fetchData()
      .then(d => {
        setData(d);
        if (d.passes.length > 0) setLevel(d.passes[0].id);
      })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
    fetchConfig().then(cfg => setIncludeFolders(cfg.include_folders));
  }, []);

  // ── Explore: merge UMAP coords + cluster labels → Block[] for scatter ────────
  // XY comes from explorePoints (UMAP), color comes from exploreLabels (k-means).
  // If only labels changed (k changed), points stay in place — just recolor.
  const exploreBlocks = useMemo((): Block[] => {
    if (!explorePoints) return [];
    return explorePoints.map(pt => {
      const prod = productionBlockById.get(pt.id);
      const label = exploreLabels?.labels[pt.id] ?? '-1';
      return {
        id: pt.id,
        content: prod?.content ?? '',
        source_path: prod?.source_path ?? pt.id,
        source_content: prod?.source_content ?? '',
        x: pt.x,
        y: pt.y,
        date: pt.date,
        cluster_assignments: label === '-1'
          ? {} as Record<string, string>
          : { [EXPLORE_LEVEL]: label },
      };
    });
  }, [explorePoints, exploreLabels, productionBlockById]);

  // ── Active blocks passed to scatter ─────────────────────────────────────────
  const activeLevel = exploreMode ? EXPLORE_LEVEL : level;
  const activeBlocks = useMemo(() => {
    let blocks = exploreMode ? exploreBlocks : (data?.blocks ?? []);

    if (timelineActive && timelineDate) {
      blocks = blocks.filter(b => b.date && new Date(b.date + '-01') <= timelineDate);
    }
    if (!exploreMode) {
      if (onlySource) blocks = blocks.filter(b => b.source_path === onlySource);
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        blocks = blocks.filter(b =>
          b.content.toLowerCase().includes(q) || b.source_path.toLowerCase().includes(q)
        );
      }
    }
    return blocks;
  }, [exploreMode, exploreBlocks, data, timelineActive, timelineDate, onlySource, searchQuery]);

  // ── Highlight ────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!selectedBlock) { setHighlightedIds(new Set()); return; }
    if (exploreMode) {
      const lbl = selectedBlock.cluster_assignments[EXPLORE_LEVEL];
      if (!lbl) { setHighlightedIds(new Set()); return; }
      setHighlightedIds(new Set(exploreBlocks.filter(b => b.cluster_assignments[EXPLORE_LEVEL] === lbl).map(b => b.id)));
    } else {
      if (!data) { setHighlightedIds(new Set()); return; }
      const lbl = selectedBlock.cluster_assignments[level];
      if (!lbl) { setHighlightedIds(new Set()); return; }
      setHighlightedIds(new Set(data.blocks.filter(b => b.cluster_assignments[level] === lbl).map(b => b.id)));
    }
  }, [selectedBlock, level, data, exploreMode, exploreBlocks]);

  // ── Explore actions ──────────────────────────────────────────────────────────
  const handleRunUmap = useCallback(async (dims: number, blockIds: string[] | null) => {
    setUmapLoading(true);
    setExploreError(null);
    try {
      const result = await postExploreUmap(dims, blockIds);
      setExplorePoints(result.points);
      setExploreUmapDims(dims);
    } catch (e) {
      setExploreError(String(e));
    } finally {
      setUmapLoading(false);
    }
  }, []);

  const handleRunCluster = useCallback(async (params: ExploreParams, blockIds: string[] | null) => {
    setClusterLoading(true);
    setExploreError(null);
    try {
      const result = await postExploreCluster(params, blockIds);
      setExploreLabels(result);
      // First run — set breadcrumb root
      if (exploreBreadcrumb.length === 0) {
        setExploreBreadcrumb([{
          label: `All (${data?.block_count.toLocaleString() ?? '?'})`,
          block_ids: null,
          params,
        }]);
      }
    } catch (e) {
      setExploreError(String(e));
    } finally {
      setClusterLoading(false);
    }
  }, [exploreBreadcrumb.length, data]);

  const handleDrillInto = useCallback((label: string) => {
    if (!explorePoints || !exploreLabels) return;
    const memberIds = explorePoints
      .filter(pt => exploreLabels.labels[pt.id] === label)
      .map(pt => pt.id);
    const crumb: ExploreCrumb = {
      label: `cluster ${label} (${memberIds.length})`,
      block_ids: memberIds,
      params: { algo: 'kmeans', dims: exploreUmapDims ?? 64, k: 10, min_cluster_size: 20, min_samples: 5 },
    };
    setExploreBreadcrumb(prev => [...prev, crumb]);
    setExploreSelectedLabel(null);
    setExplorePoints(null);
    setExploreLabels(null);
    setExploreUmapDims(null);
    // Trigger UMAP + cluster on subset
    handleRunUmap(crumb.params.dims, memberIds).then(() =>
      handleRunCluster(crumb.params, memberIds)
    );
  }, [explorePoints, exploreLabels, exploreUmapDims, handleRunUmap, handleRunCluster]);

  const handleBreadcrumbNav = useCallback((index: number) => {
    const crumb = exploreBreadcrumb[index];
    if (!crumb) return;
    setExploreBreadcrumb(prev => prev.slice(0, index + 1));
    setExploreSelectedLabel(null);
    setExplorePoints(null);
    setExploreLabels(null);
    setExploreUmapDims(null);
    handleRunUmap(crumb.params.dims, crumb.block_ids).then(() =>
      handleRunCluster(crumb.params, crumb.block_ids)
    );
  }, [exploreBreadcrumb, handleRunUmap, handleRunCluster]);

  const handleBlockClick = useCallback((block: Block | null) => {
    setSelectedBlock(block);
    if (block && exploreMode) {
      const lbl = block.cluster_assignments[EXPLORE_LEVEL];
      if (lbl) setExploreSelectedLabel(lbl);
    }
  }, [exploreMode]);

  const handleToggleExplore = useCallback(() => {
    setExploreMode(prev => {
      if (prev) {
        setExplorePoints(null);
        setExploreLabels(null);
        setExploreUmapDims(null);
        setExploreBreadcrumb([]);
        setExploreSelectedLabel(null);
        setSelectedBlock(null);
        setHighlightedIds(new Set());
      }
      return !prev;
    });
  }, []);

  const handleShowOnlySource = useCallback((sourcePath: string) => {
    setOnlySource(prev => prev === sourcePath ? null : sourcePath);
  }, []);
  const handleClearFilters = useCallback(() => { setOnlySource(null); setSearchQuery(''); }, []);
  const handleHighlightCluster = useCallback((clusterBlockId: string) => {
    if (!data) return;
    const cluster = data.clusters[clusterBlockId];
    if (!cluster) return;
    setHighlightedIds(new Set(
      data.blocks.filter(b => b.cluster_assignments[cluster.pass_id] === cluster.label).map(b => b.id)
    ));
  }, [data]);

  const timelineItems = useMemo(() => {
    const base = exploreMode ? exploreBlocks : (data?.blocks ?? []);
    return base.filter(b => b.date).map(b => ({ timestamp: b.date! + '-01' }));
  }, [exploreMode, exploreBlocks, data]);

  const detailBlock = useMemo(() => {
    if (!selectedBlock) return null;
    if (exploreMode) return productionBlockById.get(selectedBlock.id) ?? selectedBlock;
    return selectedBlock;
  }, [selectedBlock, exploreMode, productionBlockById]);

  if (loading) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100vh', background: COLORS.bg, color: COLORS.textMuted,
        fontFamily: "'JetBrains Mono', monospace", fontSize: '14px',
      }}>
        Loading knowledge graph…
      </div>
    );
  }

  if (error || !data) {
    return (
      <div style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        height: '100vh', background: COLORS.bg, color: COLORS.text, gap: '12px',
      }}>
        <div style={{ fontSize: '20px', color: '#ef4444' }}>Failed to load data</div>
        <div style={{ fontSize: '13px', color: COLORS.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>{error}</div>
      </div>
    );
  }

  const currentPass: PassInfo | undefined = data.passes.find(p => p.id === level);
  const noiseCount = activeBlocks.filter(b => !b.cluster_assignments[activeLevel]).length;
  const clusteredCount = activeBlocks.length - noiseCount;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden', background: COLORS.bg, color: COLORS.text }}>
      {/* Header */}
      <header style={{
        padding: '10px 24px', borderBottom: `1px solid ${COLORS.border}`,
        background: COLORS.surface, display: 'flex', alignItems: 'center', gap: '16px', flexShrink: 0,
      }}>
        <div style={{ fontSize: '15px', fontWeight: 600, color: COLORS.accent, fontFamily: "'JetBrains Mono', monospace", letterSpacing: '-0.5px' }}>
          openaugi / knowledge-explorer
        </div>
        <div style={{ fontSize: '12px', color: COLORS.textMuted }}>
          {exploreMode
            ? (exploreLabels
              ? `${exploreLabels.cluster_count} clusters · ${exploreLabels.noise_count} noise / ${explorePoints?.length.toLocaleString() ?? '?'} projected`
              : 'explore mode — hit Run to project')
            : `${clusteredCount.toLocaleString()} clustered · ${noiseCount.toLocaleString()} noise / ${data.block_count.toLocaleString()} total`
          }
        </div>
        {!exploreMode && currentPass && (
          <div style={{
            fontSize: '11px', color: COLORS.textMuted, fontFamily: "'JetBrains Mono', monospace",
            padding: '3px 8px', border: `1px solid ${COLORS.border}`, borderRadius: '4px',
          }}>
            {currentPass.description || currentPass.id} · dims={currentPass.dims}
          </div>
        )}
        {exploreError && <div style={{ fontSize: '11px', color: '#ef4444', maxWidth: 300 }}>{exploreError}</div>}
        <div style={{ flex: 1 }} />
        <div style={{ fontSize: '11px', color: COLORS.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>{data.generated_at}</div>
      </header>

      {/* Browse filter bar */}
      {!exploreMode && (
        <FilterBar
          level={level} onLevelChange={setLevel} passes={data.passes}
          colorMode={colorMode} onColorModeChange={setColorMode}
          searchQuery={searchQuery} onSearchChange={setSearchQuery}
          onlySource={onlySource} onClearFilters={handleClearFilters}
          colors={COLORS} timelineActive={timelineActive}
          exploreMode={exploreMode} onToggleExplore={handleToggleExplore}
        />
      )}

      {/* Explore toolbar */}
      {exploreMode && (
        <div style={{
          padding: '8px 16px', background: COLORS.surface, borderBottom: `1px solid ${COLORS.border}`,
          display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0,
        }}>
          <button onClick={handleToggleExplore} style={{
            padding: '5px 12px', fontSize: 11, fontWeight: 600,
            border: `1px solid ${COLORS.accent}`, borderRadius: 6,
            background: `${COLORS.accent}20`, color: COLORS.accent, cursor: 'pointer',
            fontFamily: "'JetBrains Mono', monospace",
          }}>
            ← Browse
          </button>
          <span style={{ fontSize: 11, color: COLORS.textMuted }}>
            Ad-hoc clustering · not saved to DB
          </span>
          <div style={{ flex: 1 }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 11, color: COLORS.textMuted }}>Color:</span>
            {(['cluster', 'date', 'source'] as ColorMode[]).map(m => (
              <button key={m} onClick={() => setColorMode(m)} style={{
                padding: '3px 8px', fontSize: 11,
                border: `1px solid ${colorMode === m ? COLORS.accent : COLORS.border}`,
                borderRadius: 4,
                background: colorMode === m ? `${COLORS.accent}20` : 'transparent',
                color: colorMode === m ? COLORS.accent : COLORS.textMuted, cursor: 'pointer',
              }}>{m}</button>
            ))}
          </div>
        </div>
      )}

      {/* Main */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden', position: 'relative' }}>
        {exploreMode && (
          <ExplorePanel
            onRunUmap={handleRunUmap}
            onRunCluster={handleRunCluster}
            clusterResult={exploreLabels}
            umapLoading={umapLoading}
            clusterLoading={clusterLoading}
            umapDims={exploreUmapDims}
            breadcrumb={exploreBreadcrumb}
            onBreadcrumbNav={handleBreadcrumbNav}
            onDrillInto={handleDrillInto}
            selectedLabel={exploreSelectedLabel}
            colors={COLORS}
            includeFolders={includeFolders}
          />
        )}

        <div style={{ flex: 1, position: 'relative' }}>
          <ScatterPlot
            ref={scatterRef}
            blocks={activeBlocks}
            selectedBlock={selectedBlock}
            highlightedIds={highlightedIds}
            colorMode={colorMode}
            level={activeLevel}
            onBlockClick={handleBlockClick}
            colors={COLORS}
            timelineActive={timelineActive}
            timelineDate={timelineDate}
          />
          <ZoomControls
            onZoomIn={() => scatterRef.current?.zoomIn()}
            onZoomOut={() => scatterRef.current?.zoomOut()}
            onZoomReset={() => scatterRef.current?.zoomReset()}
            colors={COLORS}
          />
          <TimelinePlayer
            items={timelineItems}
            onDateChange={d => setTimelineDate(d)}
            isActive={timelineActive}
            onToggle={() => setTimelineActive(p => !p)}
            colors={COLORS}
          />
        </div>

        <DetailPane
          block={detailBlock}
          clusters={data.clusters}
          clustersByPass={data.clusters_by_pass}
          passes={data.passes}
          level={level}
          onBlockClose={() => setSelectedBlock(null)}
          onHighlightCluster={handleHighlightCluster}
          onShowOnlySource={handleShowOnlySource}
          onlySource={onlySource}
          colors={COLORS}
        />
      </div>
    </div>
  );
}
