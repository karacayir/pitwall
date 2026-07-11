import type { Forecast } from "@/lib/types";

// Mini compound fan for tower rows: three tiny p50 curves, no bands.
import { COMPOUND_COLORS } from "@/lib/types";

export function Sparkline({ forecast, width = 64, height = 18 }: { forecast: Forecast | null; width?: number; height?: number }) {
  if (!forecast?.ahead) return <span style={{ width, height }} />;
  const series = ["SOFT", "MEDIUM", "HARD"]
    .filter((c) => forecast.ahead[c]?.length)
    .map((c) => ({ c, pts: forecast.ahead[c].slice(0, 10) }));
  if (!series.length) return <span style={{ width, height }} />;
  const all = series.flatMap((s) => s.pts);
  const min = Math.min(...all);
  const max = Math.max(...all) || min + 1;
  const x = (i: number, n: number) => (i / (n - 1)) * (width - 2) + 1;
  const y = (v: number) => 1 + ((max - v) / (max - min || 1)) * (height - 2);
  return (
    <svg width={width} height={height} aria-hidden="true">
      {series.map((s) => (
        <polyline
          key={s.c}
          points={s.pts.map((v, i) => `${x(i, s.pts.length)},${y(v)}`).join(" ")}
          fill="none"
          stroke={COMPOUND_COLORS[s.c]}
          strokeWidth="1"
          opacity="0.85"
        />
      ))}
    </svg>
  );
}
