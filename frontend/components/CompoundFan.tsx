"use client";

// The signature element: three thin forecast curves (S/M/H) fanning forward
// from the live lap dot, each with a soft uncertainty band. Bands take the
// model's k=1 quantile width and grow it with sqrt(k) — a display heuristic,
// the numbers on screen stay the model's own p50s.

import { COMPOUND_COLORS, type DriverUpdate } from "@/lib/types";

const DRY = ["SOFT", "MEDIUM", "HARD"] as const;

export function CompoundFan({
  driver,
  width = 460,
  height = 200,
  showCurrent = true,
}: {
  driver: DriverUpdate;
  width?: number;
  height?: number;
  showCurrent?: boolean;
}) {
  const fc = driver.forecast;
  if (!fc || !fc.current) {
    return (
      <div
        className="flex items-center justify-center rounded border border-(--hairline) bg-(--surface) text-xs text-(--muted)"
        style={{ maxWidth: width, height }}
      >
        forecast warming up…
      </div>
    );
  }

  const pad = { l: 44, r: 10, t: 10, b: 20 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;

  const curves = DRY.filter((c) => fc.ahead[c]?.length).map((c) => ({
    compound: c as string,
    points: fc.ahead[c],
    q1: fc.fresh[c],
  }));
  if (showCurrent && fc.ahead["current"]?.length && driver.compound) {
    curves.unshift({ compound: "current", points: fc.ahead["current"], q1: fc.current });
  }

  const allVals: number[] = [driver.last_lap_s ?? fc.current.p50];
  for (const c of curves) {
    allVals.push(...c.points);
    if (c.q1) allVals.push(c.q1.p10, c.q1.p90);
  }
  const yMin = Math.min(...allVals) - 0.4;
  const yMax = Math.max(...allVals) + 0.4;
  const n = Math.max(...curves.map((c) => c.points.length));

  const x = (k: number) => pad.l + (k / n) * innerW; // k = 0 is "now"
  const y = (v: number) => pad.t + ((yMax - v) / (yMax - yMin)) * innerH;

  const path = (pts: number[], anchor: number) =>
    [`M ${x(0)} ${y(anchor)}`, ...pts.map((v, i) => `L ${x(i + 1)} ${y(v)}`)].join(" ");

  const band = (pts: number[], q: { p10: number; p50: number } & { p90: number }, anchor: number) => {
    const halfLo = Math.max(q.p50 - q.p10, 0.05);
    const halfHi = Math.max(q.p90 - q.p50, 0.05);
    const upper = pts.map((v, i) => `${x(i + 1)} ${y(v + halfHi * Math.sqrt(i + 1))}`);
    const lower = pts.map((v, i) => `${x(i + 1)} ${y(v - halfLo * Math.sqrt(i + 1))}`).reverse();
    return `M ${x(0)} ${y(anchor)} L ${upper.join(" L ")} L ${lower.join(" L ")} Z`;
  };

  const anchorVal = driver.last_lap_s ?? fc.current.p50;
  const ticks = [yMin + 0.4, (yMin + yMax) / 2, yMax - 0.4];

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="block h-auto w-full"
      style={{ maxWidth: width * 1.6 }}
      role="img"
      aria-label="compound forecast fan"
    >
      {ticks.map((t) => (
        <g key={t}>
          <line x1={pad.l} x2={width - pad.r} y1={y(t)} y2={y(t)} stroke="var(--hairline)" strokeWidth="1" />
          <text x={pad.l - 6} y={y(t) + 3} textAnchor="end" fontSize="9" fill="var(--muted)" fontFamily="var(--font-mono)">
            {t.toFixed(1)}
          </text>
        </g>
      ))}
      {curves.map((c) => {
        const color = c.compound === "current" ? "var(--muted)" : COMPOUND_COLORS[c.compound];
        return (
          <g key={c.compound}>
            {c.q1 && <path d={band(c.points, c.q1, anchorVal)} fill={color} opacity="0.07" />}
            <path
              d={path(c.points, anchorVal)}
              fill="none"
              stroke={color}
              strokeWidth={c.compound === "current" ? 1.2 : 1.6}
              strokeDasharray={c.compound === "current" ? "4 3" : undefined}
              opacity={c.compound === "current" ? 0.8 : 0.95}
            />
            <text
              x={x(c.points.length) + 2}
              y={y(c.points[c.points.length - 1]) + 3}
              fontSize="9"
              fill={color}
              fontFamily="var(--font-mono)"
            >
              {c.compound === "current" ? "cur" : c.compound[0]}
            </text>
          </g>
        );
      })}
      <circle cx={x(0)} cy={y(anchorVal)} r="3.5" fill="var(--race-red)" />
      <text x={pad.l} y={height - 6} fontSize="9" fill="var(--muted)" fontFamily="var(--font-mono)">
        now
      </text>
      <text x={width - pad.r} y={height - 6} fontSize="9" fill="var(--muted)" textAnchor="end" fontFamily="var(--font-mono)">
        +{n} laps
      </text>
    </svg>
  );
}
