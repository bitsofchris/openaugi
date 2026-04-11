import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { fetchData, postExplore } from './api';
import type { Block, ColorMode, ExplorerData, PassInfo, ExploreResult, ExploreCrumb, ExploreParams } from './types';
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

// Synthetic pass_id used when explore mode is active
const EXPLORE_LEVEL = '_explore';

export default function App() {
  const [data, setData] = useState<ExplorerData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Browse mode state
  const [level, setLevel] = useState<string>('');
  const [colorMode, setColorMode] = useState<ColorMode>('cluster');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedBlock, setSelectedBlock] = useState<Block | null>(null);
  const [highlightedIds, setHighlightedIds] = useState<Set<string>>(new Set());
  const [onlySource, setOnlySource] = useState<string | null>(null);

  // Timeline — active by default, starts at right (all blocks visible)
  const [timelineActive, setTimelineActive] = useState(true);
  const [timelineDate, setTimelineDate] = useState<Date | null>(null);

  // Explore mode
  const [exploreMode, setExploreMode] = useState(false);
  const [exploreResult, setExploreResult] = useState<ExploreResult | null>(null);
  const [exploreLoading, setExploreLoading] = useState(false);
  const [exploreError, setExploreError] = useState<string | null>(null);
  const [exploreBreadcrumb, setExploreBreadcrumb] = useState<ExploreCrumb[]>([]);
  const [exploreSelectedLabel, setExploreSelectedLabel] = useState<string | null>(null);

  const scatterRef = useRef<ScatterPlotRef>(null);

  // Lookup map for production blocks by id (used to enrich explore blocks with source_content)
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
  }, []);

  // ── Explore mode: map ExploreBlock → Block for ScatterPlot ──────────────────
  // Noise blocks (label="-1") get empty cluster_assignments so isNoise() works.
  // Use explore x/y coords (re-projected UMAP for the subset).
  const exploreBlocks = useMemo((): Block[] => {
    if (!exploreResult) return [];
    return exploreResult.blocks.map(eb => {
      const prod = productionBlockById.get(eb.id);
      return {
        id: eb.id,
        content: eb.content,
        source_path: eb.source_path,
        source_content: prod?.source_content ?? '',
        x: eb.x,
        y: eb.y,
        date: eb.date,
        cluster_assignments: eb.label === '-1' ? {} as Record<string, string> : { [EXPLORE_LEVEL]: eb.label },
      };
    });
  }, [exploreResult, productionBlockById]);

  // ── Active blocks & level passed to scatter ──────────────────────────────────
  const activeLevel = exploreMode ? EXPLORE_LEVEL : level;
  const activeBlocks = useMemo(() => {
    const base = exploreMode ? exploreBlocks : (data?.blocks ?? []);

    let blocks = base;

    if (timelineActive && timelineDate) {
      blocks = blocks.filter(b => {
        if (!b.date) return false;
        return new Date(b.date + '-01') <= timelineDate;
      });
    }

    if (!exploreMode) {
      if (onlySource) blocks = blocks.filter(b => b.source_path === onlySource);
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        blocks = blocks.filter(b =>
          b.content.toLowerCase().includes(q) ||
          b.source_path.toLowerCase().includes(q)
        );
      }
    }

    return blocks;
  }, [exploreMode, exploreBlocks, data, timelineActive, timelineDate, onlySource, searchQuery]);

  // ── Cluster highlight ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!selectedBlock) {
      setHighlightedIds(new Set());
      return;
    }
    if (exploreMode) {
      // Highlight all blocks in the same explore cluster
      const lbl = selectedBlock.cluster_assignments[EXPLORE_LEVEL];
      if (!lbl) { setHighlightedIds(new Set()); return; }
      const ids = new Set(exploreBlocks.filter(b => b.cluster_assignments[EXPLORE_LEVEL] === lbl).map(b => b.id));
      setHighlightedIds(ids);
    } else {
      if (!data) { setHighlightedIds(new Set()); return; }
      const clusterLabel = selectedBlock.cluster_assignments[level];
      if (!clusterLabel) { setHighlightedIds(new Set()); return; }
      const ids = new Set(data.blocks.filter(b => b.cluster_assignments[level] === clusterLabel).map(b => b.id));
      setHighlightedIds(ids);
    }
  }, [selectedBlock, level, data, exploreMode, exploreBlocks]);

  // ── Explore actions ──────────────────────────────────────────────────────────
  const handleRunExplore = useCallback(async (params: ExploreParams, blockIds: string[] | null) => {
    setExploreLoading(true);
    setExploreError(null);
    setExploreSelectedLabel(null);
    try {
      const result = await postExplore(params, blockIds);
      setExploreResult(result);
      // Update breadcrumb: if blockIds is null this is a fresh top-level run
      if (blockIds === null) {
        setExploreBreadcrumb([{
          label: `All (${data?.block_count.toLocaleString() ?? '?'})`,
          block_ids: null,
          params,
        }]);
      }
      // If blockIds provided, the last crumb already exists (added during drill-in)
    } catch (e) {
      setExploreError(String(e));
    } finally {
      setExploreLoading(false);
    }
  }, [data]);

  const handleDrillInto = useCallback((label: string) => {
    if (!exploreResult) return;
    const memberIds = exploreResult.blocks.filter(b => b.label === label).map(b => b.id);
    const crumb: ExploreCrumb = {
      label: `cluster ${label} (${memberIds.length})`,
      block_ids: memberIds,
      params: exploreResult.params,
    };
    setExploreBreadcrumb(prev => [...prev, crumb]);
    setExploreSelectedLabel(null);
    // Run explore on just this subset with same params
    handleRunExplore(exploreResult.params, memberIds);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exploreResult]);

  const handleBreadcrumbNav = useCallback((index: number) => {
    const crumb = exploreBreadcrumb[index];
    if (!crumb) return;
    setExploreBreadcrumb(prev => prev.slice(0, index + 1));
    setExploreSelectedLabel(null);
    handleRunExplore(crumb.params, crumb.block_ids);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exploreBreadcrumb]);

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
        // Exit explore — clear state
        setExploreResult(null);
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

  const handleClearFilters = useCallback(() => {
    setOnlySource(null);
    setSearchQuery('');
  }, []);

  const handleHighlightCluster = useCallback((clusterBlockId: string) => {
    if (!data) return;
    const cluster = data.clusters[clusterBlockId];
    if (!cluster) return;
    const ids = new Set(
      data.blocks.filter(b => b.cluster_assignments[cluster.pass_id] === cluster.label).map(b => b.id)
    );
    setHighlightedIds(ids);
  }, [data]);

  const handleDateChange = useCallback((date: Date | null) => {
    setTimelineDate(date);
  }, []);

  const handleTimelineToggle = useCallback(() => {
    setTimelineActive(prev => !prev);
  }, []);

  // Timeline items — use active blocks (explore or browse)
  const timelineItems = useMemo(() => {
    const base = exploreMode ? exploreBlocks : (data?.blocks ?? []);
    return base.filter(b => b.date).map(b => ({ timestamp: b.date! + '-01' }));
  }, [exploreMode, exploreBlocks, data]);

  // Detail pane: in explore mode, show the production block for rich context
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
        <div style={{ fontSize: '13px', color: COLORS.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>
          {error}
        </div>
        <div style={{ fontSize: '12px', color: COLORS.textMuted, maxWidth: '500px', textAlign: 'center' }}>
          Start the backend (<code>python backend/server.py</code>) or add
          a <code>public/fixture.json</code> for offline dev.
        </div>
      </div>
    );
  }

  const currentPass: PassInfo | undefined = data.passes.find(p => p.id === level);
  const noiseCount = activeBlocks.filter(b => !b.cluster_assignments[activeLevel]).length;
  const clusteredCount = activeBlocks.length - noiseCount;

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      height: '100vh', overflow: 'hidden',
      background: COLORS.bg, color: COLORS.text,
    }}>
      {/* Header */}
      <header style={{
        padding: '10px 24px',
        borderBottom: `1px solid ${COLORS.border}`,
        background: COLORS.surface,
        display: 'flex', alignItems: 'center', gap: '16px',
        flexShrink: 0,
      }}>
        <div style={{
          fontSize: '15px', fontWeight: 600, color: COLORS.accent,
          fontFamily: "'JetBrains Mono', monospace", letterSpacing: '-0.5px',
        }}>
          openaugi / knowledge-explorer
        </div>
        <div style={{ fontSize: '12px', color: COLORS.textMuted }}>
          {exploreMode
            ? (exploreResult
              ? `${exploreResult.cluster_count} clusters · ${exploreResult.noise_count} noise / ${exploreResult.blocks.length.toLocaleString()} projected`
              : 'explore mode — run to project')
            : `${clusteredCount.toLocaleString()} clustered · ${noiseCount.toLocaleString()} noise / ${data.block_count.toLocaleString()} total`
          }
        </div>
        {!exploreMode && currentPass && (
          <div style={{
            fontSize: '11px', color: COLORS.textMuted,
            fontFamily: "'JetBrains Mono', monospace",
            padding: '3px 8px', border: `1px solid ${COLORS.border}`, borderRadius: '4px',
          }}>
            {currentPass.description || currentPass.id} · dims={currentPass.dims}
          </div>
        )}
        {exploreError && (
          <div style={{ fontSize: '11px', color: '#ef4444', maxWidth: 300 }}>{exploreError}</div>
        )}
        <div style={{ flex: 1 }} />
        <div style={{ fontSize: '11px', color: COLORS.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>
          {data.generated_at}
        </div>
      </header>

      {/* Filter bar — hidden in explore mode (search/filter don't apply to explore) */}
      {!exploreMode && (
        <FilterBar
          level={level}
          onLevelChange={setLevel}
          passes={data.passes}
          colorMode={colorMode}
          onColorModeChange={setColorMode}
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          onlySource={onlySource}
          onClearFilters={handleClearFilters}
          colors={COLORS}
          timelineActive={timelineActive}
          exploreMode={exploreMode}
          onToggleExplore={handleToggleExplore}
        />
      )}

      {/* Explore mode toolbar */}
      {exploreMode && (
        <div style={{
          padding: '8px 16px',
          background: COLORS.surface,
          borderBottom: `1px solid ${COLORS.border}`,
          display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0,
        }}>
          <button
            onClick={handleToggleExplore}
            style={{
              padding: '5px 12px', fontSize: 11, fontWeight: 600,
              border: `1px solid ${COLORS.accent}`,
              borderRadius: 6, background: `${COLORS.accent}20`,
              color: COLORS.accent, cursor: 'pointer',
              fontFamily: "'JetBrains Mono', monospace",
            }}
          >
            ← Browse mode
          </button>
          <span style={{ fontSize: 11, color: COLORS.textMuted }}>
            Ad-hoc clustering · results are not saved to DB
          </span>
          <div style={{ flex: 1 }} />
          {/* Color mode still useful in explore */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 11, color: COLORS.textMuted }}>Color:</span>
            {(['cluster', 'date', 'source'] as ColorMode[]).map(m => (
              <button key={m}
                onClick={() => setColorMode(m)}
                style={{
                  padding: '3px 8px', fontSize: 11,
                  border: `1px solid ${colorMode === m ? COLORS.accent : COLORS.border}`,
                  borderRadius: 4, background: colorMode === m ? `${COLORS.accent}20` : 'transparent',
                  color: colorMode === m ? COLORS.accent : COLORS.textMuted, cursor: 'pointer',
                }}
              >
                {m}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Main area */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden', position: 'relative' }}>
        {/* Explore panel — left sidebar when in explore mode */}
        {exploreMode && (
          <ExplorePanel
            onRun={handleRunExplore}
            result={exploreResult}
            loading={exploreLoading}
            breadcrumb={exploreBreadcrumb}
            onBreadcrumbNav={handleBreadcrumbNav}
            onDrillInto={handleDrillInto}
            selectedLabel={exploreSelectedLabel}
            colors={COLORS}
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
            onDateChange={handleDateChange}
            isActive={timelineActive}
            onToggle={handleTimelineToggle}
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
