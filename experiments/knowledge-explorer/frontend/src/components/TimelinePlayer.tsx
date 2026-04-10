import { useState, useEffect, useCallback, useMemo, useRef } from 'react';

interface Props {
  /** All items with timestamps */
  items: Array<{ timestamp: string | null }>;
  /** Callback when the current date changes */
  onDateChange: (date: Date | null) => void;
  /** Whether timeline is active (null = show all) */
  isActive: boolean;
  /** Toggle timeline on/off */
  onToggle: () => void;
  /** Color theme */
  colors: Record<string, string>;
}

const SPEEDS = [0.5, 1, 2, 5, 10];
const DEFAULT_SPEED = 1;
const MS_PER_DAY = 1000 * 60 * 60 * 24;

export function TimelinePlayer({ items, onDateChange, isActive, onToggle, colors }: Props) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(DEFAULT_SPEED);
  const [progress, setProgress] = useState(0); // 0-1
  const animationRef = useRef<number>();
  const lastTimeRef = useRef<number>(0);

  // Calculate date range from items
  const { minDate, maxDate, sortedDates } = useMemo(() => {
    const dates: Date[] = [];
    const countMap = new Map<string, number>();

    items.forEach(item => {
      if (item.timestamp) {
        const date = new Date(item.timestamp);
        if (!isNaN(date.getTime())) {
          dates.push(date);
          const key = date.toISOString().split('T')[0];
          countMap.set(key, (countMap.get(key) || 0) + 1);
        }
      }
    });

    if (dates.length === 0) {
      return { minDate: null, maxDate: null, sortedDates: [], dateToCount: new Map() };
    }

    dates.sort((a, b) => a.getTime() - b.getTime());
    const min = dates[0];
    const max = dates[dates.length - 1];

    return {
      minDate: min,
      maxDate: max,
      sortedDates: dates,
    };
  }, [items]);

  // Calculate current date from progress
  const currentDate = useMemo(() => {
    if (!minDate || !maxDate) return null;
    const range = maxDate.getTime() - minDate.getTime();
    const current = new Date(minDate.getTime() + range * progress);
    return current;
  }, [minDate, maxDate, progress]);

  // Count items up to current date
  const visibleCount = useMemo(() => {
    if (!currentDate || !isActive) return items.length;
    return sortedDates.filter(d => d.getTime() <= currentDate.getTime()).length;
  }, [sortedDates, currentDate, isActive, items.length]);

  // Notify parent of date changes
  useEffect(() => {
    if (isActive && currentDate) {
      onDateChange(currentDate);
    } else if (!isActive) {
      onDateChange(null);
    }
  }, [currentDate, isActive, onDateChange]);

  // Animation loop
  useEffect(() => {
    if (!isPlaying || !minDate || !maxDate) return;

    const totalDays = (maxDate.getTime() - minDate.getTime()) / MS_PER_DAY;
    // Complete animation in (totalDays / speed) seconds, minimum 5 seconds
    const animationDuration = Math.max(5000, (totalDays / speed) * 100);

    const animate = (time: number) => {
      if (lastTimeRef.current === 0) {
        lastTimeRef.current = time;
      }

      const delta = time - lastTimeRef.current;
      lastTimeRef.current = time;

      setProgress(prev => {
        const next = prev + (delta / animationDuration);
        if (next >= 1) {
          setIsPlaying(false);
          return 1;
        }
        return next;
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      lastTimeRef.current = 0;
    };
  }, [isPlaying, minDate, maxDate, speed]);

  // Generate activity heatmap data
  const heatmapBars = useMemo(() => {
    if (!minDate || !maxDate) return [];

    const range = maxDate.getTime() - minDate.getTime();
    const numBuckets = 100;
    const bucketSize = range / numBuckets;
    const buckets: number[] = new Array(numBuckets).fill(0);

    sortedDates.forEach(date => {
      const bucketIdx = Math.min(
        numBuckets - 1,
        Math.floor((date.getTime() - minDate.getTime()) / bucketSize)
      );
      buckets[bucketIdx]++;
    });

    const maxCount = Math.max(...buckets);
    return buckets.map(count => count / maxCount);
  }, [minDate, maxDate, sortedDates]);

  const handlePlayPause = useCallback(() => {
    if (progress >= 1) {
      setProgress(0);
    }
    setIsPlaying(prev => !prev);
  }, [progress]);

  const handleScrub = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    setProgress(Math.max(0, Math.min(1, x)));
    setIsPlaying(false);
  }, []);

  const handleSpeedChange = useCallback(() => {
    setSpeed(prev => {
      const idx = SPEEDS.indexOf(prev);
      return SPEEDS[(idx + 1) % SPEEDS.length];
    });
  }, []);

  const handleReset = useCallback(() => {
    setProgress(0);
    setIsPlaying(false);
  }, []);

  const handleSkipToEnd = useCallback(() => {
    setProgress(1);
    setIsPlaying(false);
  }, []);

  const formatDate = (date: Date | null) => {
    if (!date) return '—';
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  if (!minDate || !maxDate) {
    return null; // No timeline data available
  }

  return (
    <div style={{
      position: 'absolute',
      bottom: 20,
      left: '50%',
      transform: 'translateX(-50%)',
      padding: '12px 20px',
      background: 'rgba(18, 18, 26, 0.95)',
      borderRadius: '12px',
      border: `1px solid ${colors.border}`,
      backdropFilter: 'blur(8px)',
      display: 'flex',
      flexDirection: 'column',
      gap: '10px',
      minWidth: '500px',
      zIndex: 100,
    }}>
      {/* Header row */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={onToggle}
            style={{
              padding: '4px 10px',
              fontSize: '11px',
              fontWeight: 600,
              fontFamily: "'JetBrains Mono', monospace",
              background: isActive ? `${colors.accent}20` : 'transparent',
              border: `1px solid ${isActive ? colors.accent : colors.border}`,
              borderRadius: '4px',
              color: isActive ? colors.accent : colors.textMuted,
              cursor: 'pointer',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
            }}
          >
            {isActive ? '⏱ Timeline' : '⏱ Timeline'}
          </button>
          {isActive && (
            <span style={{
              fontSize: '11px',
              fontFamily: "'JetBrains Mono', monospace",
              color: colors.textMuted,
            }}>
              {visibleCount} / {items.length} atoms
            </span>
          )}
        </div>

        {isActive && currentDate && (
          <div style={{
            fontSize: '16px',
            fontWeight: 600,
            fontFamily: "'Outfit', sans-serif",
            color: colors.accent,
            letterSpacing: '-0.5px',
          }}>
            {formatDate(currentDate)}
          </div>
        )}
      </div>

      {isActive && (
        <>
          {/* Activity heatmap + scrubber */}
          <div
            onClick={handleScrub}
            style={{
              position: 'relative',
              height: '32px',
              cursor: 'pointer',
              borderRadius: '4px',
              overflow: 'hidden',
              background: `${colors.border}50`,
            }}
          >
            {/* Heatmap bars */}
            <div style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'flex-end',
              gap: '1px',
              padding: '2px',
            }}>
              {heatmapBars.map((height, i) => (
                <div
                  key={i}
                  style={{
                    flex: 1,
                    height: `${Math.max(2, height * 100)}%`,
                    background: (i / heatmapBars.length) <= progress
                      ? colors.accent
                      : `${colors.accent}40`,
                    borderRadius: '1px',
                    transition: 'background 0.1s ease',
                  }}
                />
              ))}
            </div>

            {/* Progress indicator */}
            <div style={{
              position: 'absolute',
              left: `${progress * 100}%`,
              top: 0,
              bottom: 0,
              width: '3px',
              background: colors.text,
              borderRadius: '2px',
              boxShadow: '0 0 8px rgba(255,255,255,0.5)',
              transform: 'translateX(-50%)',
            }} />
          </div>

          {/* Controls row */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}>
            {/* Date range */}
            <div style={{
              fontSize: '11px',
              fontFamily: "'JetBrains Mono', monospace",
              color: colors.textMuted,
            }}>
              {formatDate(minDate)}
            </div>

            {/* Playback controls */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <button
                onClick={handleReset}
                title="Reset to start"
                style={{
                  padding: '6px 10px',
                  fontSize: '14px',
                  background: 'transparent',
                  border: `1px solid ${colors.border}`,
                  borderRadius: '6px',
                  color: colors.textMuted,
                  cursor: 'pointer',
                }}
              >
                ⏮
              </button>

              <button
                onClick={handlePlayPause}
                title={isPlaying ? 'Pause' : 'Play'}
                style={{
                  padding: '8px 16px',
                  fontSize: '16px',
                  background: `${colors.accent}20`,
                  border: `1px solid ${colors.accent}`,
                  borderRadius: '8px',
                  color: colors.accent,
                  cursor: 'pointer',
                  fontWeight: 600,
                }}
              >
                {isPlaying ? '⏸' : '▶'}
              </button>

              <button
                onClick={handleSkipToEnd}
                title="Skip to end"
                style={{
                  padding: '6px 10px',
                  fontSize: '14px',
                  background: 'transparent',
                  border: `1px solid ${colors.border}`,
                  borderRadius: '6px',
                  color: colors.textMuted,
                  cursor: 'pointer',
                }}
              >
                ⏭
              </button>

              <button
                onClick={handleSpeedChange}
                title="Change speed"
                style={{
                  padding: '6px 12px',
                  fontSize: '12px',
                  fontFamily: "'JetBrains Mono', monospace",
                  background: 'transparent',
                  border: `1px solid ${colors.border}`,
                  borderRadius: '6px',
                  color: colors.textMuted,
                  cursor: 'pointer',
                  minWidth: '50px',
                }}
              >
                {speed}x
              </button>
            </div>

            {/* Date range end */}
            <div style={{
              fontSize: '11px',
              fontFamily: "'JetBrains Mono', monospace",
              color: colors.textMuted,
            }}>
              {formatDate(maxDate)}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
