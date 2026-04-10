import type { ColorMode, PassInfo } from '../types';

interface Props {
  level: string;  // current pass_id
  onLevelChange: (level: string) => void;
  passes: PassInfo[];
  colorMode: ColorMode;
  onColorModeChange: (mode: ColorMode) => void;
  searchQuery: string;
  onSearchChange: (query: string) => void;
  onlySource: string | null;
  onClearFilters: () => void;
  colors: Record<string, string>;
  timelineActive?: boolean;
}

const COLOR_MODES: { id: ColorMode; label: string }[] = [
  { id: 'cluster', label: 'cluster' },
  { id: 'date',    label: 'date' },
  { id: 'source',  label: 'source' },
];

export function FilterBar({
  level,
  onLevelChange,
  passes,
  colorMode,
  onColorModeChange,
  searchQuery,
  onSearchChange,
  onlySource,
  onClearFilters,
  colors,
  timelineActive,
}: Props) {
  const btn = (active: boolean): React.CSSProperties => ({
    padding: '5px 12px',
    fontSize: 12,
    fontWeight: 500,
    border: `1px solid ${active ? colors.accent : colors.border}`,
    borderRadius: 6,
    background: active ? `${colors.accent}20` : 'transparent',
    color: active ? colors.accent : colors.textMuted,
    cursor: 'pointer',
    transition: 'all 0.15s',
    fontFamily: "'Outfit', sans-serif",
  });

  const divider = <div style={{ width: 1, height: 24, background: colors.border }} />;

  return (
    <div style={{
      display: 'flex', alignItems: 'center', flexWrap: 'wrap',
      gap: 14, padding: '9px 24px',
      background: colors.surface,
      borderBottom: `1px solid ${colors.border}`,
    }}>
      {/* Pass (level) selector */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 12, color: colors.textMuted, flexShrink: 0 }}>Pass:</span>
        <div style={{ display: 'flex', gap: 4 }}>
          {passes.map(pass => (
            <button
              key={pass.id}
              onClick={() => onLevelChange(pass.id)}
              style={btn(level === pass.id)}
              title={pass.description || `dims=${pass.dims}`}
            >
              {pass.id}
            </button>
          ))}
        </div>
      </div>

      {divider}

      {/* Color mode */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 12, color: colors.textMuted, flexShrink: 0 }}>Color:</span>
        <div style={{ display: 'flex', gap: 4 }}>
          {COLOR_MODES.map(({ id, label }) => (
            <button key={id} onClick={() => onColorModeChange(id)} style={btn(colorMode === id)}>
              {timelineActive && id === 'date' ? '⏱ age' : label}
            </button>
          ))}
        </div>
      </div>

      {divider}

      {/* Search */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 14, color: colors.textMuted }}>🔍</span>
        <input
          type="text"
          placeholder="Search blocks..."
          value={searchQuery}
          onChange={e => onSearchChange(e.target.value)}
          style={{
            padding: '6px 12px', fontSize: 13,
            border: `1px solid ${colors.border}`,
            borderRadius: 6, background: colors.bg, color: colors.text,
            width: 200, outline: 'none',
            fontFamily: "'Outfit', sans-serif",
          }}
        />
      </div>

      {/* Active source filter chip */}
      {onlySource && (
        <>
          {divider}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '4px 10px',
            background: `${colors.accent}20`, border: `1px solid ${colors.accent}`,
            borderRadius: 16, fontSize: 11, color: colors.accent,
          }}>
            <span>Only: {onlySource.split('/').pop()}</span>
            <button
              onClick={onClearFilters}
              style={{ background: 'none', border: 'none', color: colors.accent, cursor: 'pointer', padding: 0, fontSize: 14, lineHeight: 1 }}
            >×</button>
          </div>
        </>
      )}

      {/* Clear search (no chip needed, just a small button) */}
      {searchQuery && !onlySource && (
        <button
          onClick={onClearFilters}
          style={{ padding: '4px 10px', fontSize: 11, border: 'none', borderRadius: 4, background: '#ef4444', color: 'white', cursor: 'pointer' }}
        >
          Clear
        </button>
      )}
    </div>
  );
}
