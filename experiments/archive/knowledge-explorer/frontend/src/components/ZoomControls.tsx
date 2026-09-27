interface Props {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onZoomReset: () => void;
  colors: Record<string, string>;
}

export function ZoomControls({ onZoomIn, onZoomOut, onZoomReset, colors }: Props) {
  const buttonStyle: React.CSSProperties = {
    width: 36,
    height: 36,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: colors.surface,
    border: `1px solid ${colors.border}`,
    borderRadius: '6px',
    color: colors.text,
    fontSize: '18px',
    cursor: 'pointer',
    transition: 'all 0.15s ease',
    fontFamily: "'JetBrains Mono', monospace",
  };

  return (
    <div style={{
      position: 'absolute',
      bottom: 20,
      right: 20,
      display: 'flex',
      flexDirection: 'column',
      gap: '4px',
      zIndex: 100,
    }}>
      <button
        onClick={onZoomIn}
        style={buttonStyle}
        onMouseEnter={e => {
          e.currentTarget.style.borderColor = colors.accent;
          e.currentTarget.style.color = colors.accent;
        }}
        onMouseLeave={e => {
          e.currentTarget.style.borderColor = colors.border;
          e.currentTarget.style.color = colors.text;
        }}
        title="Zoom in"
      >
        +
      </button>
      <button
        onClick={onZoomOut}
        style={buttonStyle}
        onMouseEnter={e => {
          e.currentTarget.style.borderColor = colors.accent;
          e.currentTarget.style.color = colors.accent;
        }}
        onMouseLeave={e => {
          e.currentTarget.style.borderColor = colors.border;
          e.currentTarget.style.color = colors.text;
        }}
        title="Zoom out"
      >
        −
      </button>
      <button
        onClick={onZoomReset}
        style={{
          ...buttonStyle,
          fontSize: '12px',
          marginTop: '4px',
        }}
        onMouseEnter={e => {
          e.currentTarget.style.borderColor = colors.accent;
          e.currentTarget.style.color = colors.accent;
        }}
        onMouseLeave={e => {
          e.currentTarget.style.borderColor = colors.border;
          e.currentTarget.style.color = colors.text;
        }}
        title="Reset zoom"
      >
        ⟲
      </button>
    </div>
  );
}
