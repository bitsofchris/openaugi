import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { fetchData } from './api';
import type { Block, ColorMode, ExplorerData, PassInfo } from './types';
import { ScatterPlot } from './components/ScatterPlot';
import type { ScatterPlotRef } from './components/ScatterPlot';
import { DetailPane } from './components/DetailPane';
import { FilterBar } from './components/FilterBar';
import { TimelinePlayer } from './components/TimelinePlayer';
import { ZoomControls } from './components/ZoomControls';

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

export default function App() {
  const [data, setData] = useState<ExplorerData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // View state — level is a pass_id string
  const [level, setLevel] = useState<string>('');
  const [colorMode, setColorMode] = useState<ColorMode>('cluster');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedBlock, setSelectedBlock] = useState<Block | null>(null);
  const [highlightedIds, setHighlightedIds] = useState<Set<string>>(new Set());
  const [onlySource, setOnlySource] = useState<string | null>(null);

  // Timeline
  const [timelineActive, setTimelineActive] = useState(false);
  const [timelineDate, setTimelineDate] = useState<Date | null>(null);

  const scatterRef = useRef<ScatterPlotRef>(null);

  useEffect(() => {
    fetchData()
      .then(d => {
        setData(d);
        // Default to first (coarsest) pass
        if (d.passes.length > 0) setLevel(d.passes[0].id);
      })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  // Current pass info
  const currentPass: PassInfo | undefined = data?.passes.find(p => p.id === level);

  // Filtered blocks for display
  const visibleBlocks = useMemo(() => {
    if (!data) return [];
    let blocks = data.blocks;

    if (timelineActive && timelineDate) {
      blocks = blocks.filter(b => {
        if (!b.date) return false;
        return new Date(b.date + '-01') <= timelineDate;
      });
    }

    if (onlySource) {
      blocks = blocks.filter(b => b.source_path === onlySource);
    }

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      blocks = blocks.filter(b =>
        b.content.toLowerCase().includes(q) ||
        b.source_path.toLowerCase().includes(q)
      );
    }

    return blocks;
  }, [data, timelineActive, timelineDate, onlySource, searchQuery]);

  // Highlight all members of the selected block's cluster at the current level
  useEffect(() => {
    if (!selectedBlock || !data || !level) {
      setHighlightedIds(new Set());
      return;
    }
    const clusterLabel = selectedBlock.cluster_assignments[level];
    if (!clusterLabel) {
      setHighlightedIds(new Set());
      return;
    }
    const ids = new Set(
      data.blocks
        .filter(b => b.cluster_assignments[level] === clusterLabel)
        .map(b => b.id)
    );
    setHighlightedIds(ids);
  }, [selectedBlock, level, data]);

  const handleBlockClick = useCallback((block: Block | null) => {
    setSelectedBlock(block);
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
      data.blocks
        .filter(b => b.cluster_assignments[cluster.pass_id] === cluster.label)
        .map(b => b.id)
    );
    setHighlightedIds(ids);
  }, [data]);

  const handleDateChange = useCallback((date: Date | null) => {
    setTimelineDate(date);
  }, []);

  const handleTimelineToggle = useCallback(() => {
    setTimelineActive(prev => !prev);
  }, []);

  const timelineItems = useMemo(() => {
    if (!data) return [];
    return data.blocks
      .filter(b => b.date)
      .map(b => ({ timestamp: b.date! + '-01' }));
  }, [data]);

  if (loading) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100vh', background: COLORS.bg, color: COLORS.textMuted,
        fontFamily: "'JetBrains Mono', monospace", fontSize: '14px',
      }}>
        Loading knowledge graph...
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

  // Stats for header
  const noiseCount = visibleBlocks.filter(b => !b.cluster_assignments[level]).length;
  const clusteredCount = visibleBlocks.length - noiseCount;

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
      }}>
        <div style={{
          fontSize: '15px', fontWeight: 600, color: COLORS.accent,
          fontFamily: "'JetBrains Mono', monospace", letterSpacing: '-0.5px',
        }}>
          openaugi / knowledge-explorer
        </div>
        <div style={{ fontSize: '12px', color: COLORS.textMuted }}>
          {clusteredCount.toLocaleString()} clustered · {noiseCount.toLocaleString()} noise
          {' / '}
          {data.block_count.toLocaleString()} total
        </div>
        {currentPass && (
          <div style={{
            fontSize: '11px', color: COLORS.textMuted,
            fontFamily: "'JetBrains Mono', monospace",
            padding: '3px 8px',
            border: `1px solid ${COLORS.border}`,
            borderRadius: '4px',
          }}>
            {currentPass.description || currentPass.id} · dims={currentPass.dims}
          </div>
        )}
        <div style={{ flex: 1 }} />
        <div style={{ fontSize: '11px', color: COLORS.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>
          {data.generated_at}
        </div>
      </header>

      {/* Filter bar */}
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
      />

      {/* Main area */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden', position: 'relative' }}>
        <div style={{ flex: 1, position: 'relative' }}>
          <ScatterPlot
            ref={scatterRef}
            blocks={visibleBlocks}
            selectedBlock={selectedBlock}
            highlightedIds={highlightedIds}
            colorMode={colorMode}
            level={level}
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
          block={selectedBlock}
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
