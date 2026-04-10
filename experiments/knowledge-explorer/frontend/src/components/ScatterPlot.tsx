import { useMemo, useCallback, useState, useImperativeHandle, forwardRef } from 'react';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer } from '@deck.gl/layers';
import { OrthographicView } from '@deck.gl/core';
import type { Block, ColorMode } from '../types';

export interface ScatterPlotRef {
  zoomIn: () => void;
  zoomOut: () => void;
  zoomReset: () => void;
}

interface Props {
  blocks: Block[];
  selectedBlock: Block | null;
  highlightedIds: Set<string>;
  colorMode: ColorMode;
  level: string;  // pass_id
  onBlockClick: (block: Block | null) => void;
  colors: Record<string, string>;
  timelineActive?: boolean;
  timelineDate?: Date | null;
}

// Hash a string to a stable HSL-derived RGB — used for cluster labels and source paths
function hashToRgb(str: string): [number, number, number] {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  const h = Math.abs(hash) % 360;
  const s = 60 + (Math.abs(hash >> 8) % 25);
  const l = 50 + (Math.abs(hash >> 16) % 15);

  const c = (1 - Math.abs(2 * l / 100 - 1)) * s / 100;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l / 100 - c / 2;

  let r = 0, g = 0, b = 0;
  if (h < 60)       { r = c; g = x; }
  else if (h < 120) { r = x; g = c; }
  else if (h < 180) { g = c; b = x; }
  else if (h < 240) { g = x; b = c; }
  else if (h < 300) { r = x; b = c; }
  else              { r = c; b = x; }

  return [
    Math.round((r + m) * 255),
    Math.round((g + m) * 255),
    Math.round((b + m) * 255),
  ];
}

// Blue (old) → amber (recent) over 2-year window
function getDateColor(date: string | null): [number, number, number] {
  if (!date) return [100, 100, 120];
  const d = new Date(date + '-01');
  const now = new Date();
  const twoYearsAgo = new Date(now.getFullYear() - 2, now.getMonth(), 1);
  const range = now.getTime() - twoYearsAgo.getTime();
  const t = Math.max(0, Math.min(1, (d.getTime() - twoYearsAgo.getTime()) / range));
  return [
    Math.round(59 + t * (245 - 59)),
    Math.round(130 + t * (158 - 130)),
    Math.round(246 + t * (11 - 246)),
  ];
}

function getTimelineAgeColor(date: string | null, timelineDate: Date): [number, number, number] {
  if (!date) return [60, 60, 80];
  const blockMonth = new Date(date + '-01');
  const diffMonths = Math.floor(
    (timelineDate.getTime() - blockMonth.getTime()) / (1000 * 60 * 60 * 24 * 30)
  );
  const stops: [number, number, number][] = [
    [255, 220, 100], [249, 115, 22], [239, 68, 68],
    [236, 72, 153], [168, 85, 247], [99, 102, 241],
    [59, 130, 246], [71, 85, 105],
  ];
  return stops[Math.min(Math.max(0, diffMonths), stops.length - 1)];
}

export const ScatterPlot = forwardRef<ScatterPlotRef, Props>(function ScatterPlot({
  blocks,
  selectedBlock,
  highlightedIds,
  colorMode,
  level,
  onBlockClick,
  colors,
  timelineActive,
  timelineDate,
}, ref) {
  const [viewState, setViewState] = useState({
    target: [0, 0, 0] as [number, number, number],
    zoom: 0.8,
    minZoom: -2,
    maxZoom: 8,
  });

  useImperativeHandle(ref, () => ({
    zoomIn:    () => setViewState(p => ({ ...p, zoom: Math.min(p.zoom + 0.5, p.maxZoom) })),
    zoomOut:   () => setViewState(p => ({ ...p, zoom: Math.max(p.zoom - 0.5, p.minZoom) })),
    zoomReset: () => setViewState(p => ({ ...p, target: [0, 0, 0], zoom: 0.8 })),
  }), []);

  const isNoise = useCallback((block: Block): boolean => {
    return !block.cluster_assignments[level];
  }, [level]);

  const getColor = useCallback((block: Block): [number, number, number, number] => {
    const isSelected = selectedBlock?.id === block.id;
    const isHighlighted = highlightedIds.has(block.id);
    const noise = isNoise(block);

    if (noise) {
      const alpha = highlightedIds.size > 0 ? 15 : 70;
      return [80, 80, 100, alpha];
    }

    let rgb: [number, number, number];

    if (colorMode === 'cluster') {
      // Color by "{pass_id}:{label}" for stable, unique colors across passes
      const label = block.cluster_assignments[level] ?? '';
      rgb = hashToRgb(`${level}:${label}`);
    } else if (colorMode === 'date') {
      if (timelineActive && timelineDate) {
        rgb = getTimelineAgeColor(block.date, timelineDate);
      } else {
        rgb = getDateColor(block.date);
      }
    } else {
      rgb = hashToRgb(block.source_path);
    }

    let alpha = 200;
    if (highlightedIds.size > 0) {
      alpha = isSelected ? 255 : isHighlighted ? 230 : 25;
    } else if (isSelected) {
      alpha = 255;
    }

    return [...rgb, alpha];
  }, [colorMode, level, selectedBlock, highlightedIds, isNoise, timelineActive, timelineDate]);

  const getRadius = useCallback((block: Block): number => {
    if (isNoise(block)) return 3;
    if (selectedBlock?.id === block.id) return 12;
    if (highlightedIds.has(block.id)) return 8;
    return 5;
  }, [selectedBlock, highlightedIds, isNoise]);

  const layers = useMemo(() => [
    new ScatterplotLayer<Block>({
      id: 'blocks',
      data: blocks,
      pickable: true,
      opacity: 1,
      stroked: true,
      filled: true,
      radiusMinPixels: 2,
      radiusMaxPixels: 20,
      lineWidthMinPixels: 1,
      getPosition: (d: Block) => [d.x * 500, d.y * 500, 0],
      getRadius: getRadius,
      getFillColor: getColor,
      getLineColor: (d: Block) => {
        if (!d.cluster_assignments[level]) return [255, 255, 255, 10];
        if (selectedBlock?.id === d.id) return [255, 255, 255, 255];
        if (highlightedIds.has(d.id)) return [255, 255, 255, 100];
        return [255, 255, 255, 30];
      },
      getLineWidth: (d: Block) => (selectedBlock?.id === d.id ? 2 : 1),
      updateTriggers: {
        getRadius:    [level, selectedBlock?.id, highlightedIds],
        getFillColor: [colorMode, level, selectedBlock?.id, highlightedIds, timelineActive, timelineDate?.getTime()],
        getLineColor: [level, selectedBlock?.id, highlightedIds],
        getLineWidth: [selectedBlock?.id],
      },
    }),
  ], [blocks, level, selectedBlock, highlightedIds, colorMode, getRadius, getColor]);

  const onClick = useCallback((info: { object?: Block }) => {
    onBlockClick(info.object ?? null);
  }, [onBlockClick]);

  const getCursor = useCallback(({ isHovering }: { isHovering: boolean }) =>
    isHovering ? 'pointer' : 'grab'
  , []);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const onViewStateChange = useCallback(({ viewState: vs }: { viewState: any }) => {
    setViewState(vs);
  }, []);

  const legend = useMemo(() => {
    if (colorMode === 'date') {
      return (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            width: 100, height: 12, borderRadius: 4,
            background: 'linear-gradient(90deg, rgb(59,130,246) 0%, rgb(245,158,11) 100%)',
          }} />
          <span style={{ color: colors.textMuted, fontSize: 11 }}>older → newer</span>
        </div>
      );
    }
    if (colorMode === 'source') {
      return <span style={{ color: colors.textMuted, fontSize: 11 }}>colored by source note · gray = noise</span>;
    }
    return <span style={{ color: colors.textMuted, fontSize: 11 }}>colored by cluster · gray = noise</span>;
  }, [colorMode, colors]);

  return (
    <DeckGL
      views={new OrthographicView({ id: 'main' })}
      viewState={viewState}
      onViewStateChange={onViewStateChange}
      controller={{ dragRotate: false }}
      layers={layers}
      onClick={onClick}
      getCursor={getCursor}
      style={{ background: colors.bg }}
    >
      <div style={{
        position: 'absolute', bottom: 20, left: 20,
        padding: '10px 14px',
        background: 'rgba(13,13,20,0.9)',
        borderRadius: 8,
        border: `1px solid ${colors.border}`,
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        {legend}
      </div>
    </DeckGL>
  );
});
