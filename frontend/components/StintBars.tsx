import type { LapPoint } from "@/lib/store";
import { COMPOUND_COLORS } from "@/lib/types";

export function StintBars({ laps, lapsTotal }: { laps: LapPoint[]; lapsTotal: number }) {
  if (!laps.length) return <p className="text-xs text-(--muted)">no laps yet</p>;
  const stints: { compound: string | null; from: number; to: number }[] = [];
  for (const p of laps) {
    const last = stints[stints.length - 1];
    if (last && last.compound === p.compound) last.to = p.lap;
    else stints.push({ compound: p.compound, from: p.lap, to: p.lap });
  }
  return (
    <div>
      <div className="flex h-6 w-full overflow-hidden rounded border border-(--hairline)">
        {stints.map((s, i) => (
          <div
            key={i}
            title={`${s.compound ?? "?"} L${s.from}–L${s.to}`}
            style={{
              width: `${((s.to - s.from + 1) / lapsTotal) * 100}%`,
              background: COMPOUND_COLORS[s.compound ?? ""] ?? "#8B8B98",
              opacity: 0.75,
            }}
          />
        ))}
      </div>
      <div className="mt-1 flex flex-wrap gap-2 text-[10px] text-(--muted)">
        {stints.map((s, i) => (
          <span key={i} className="timing">
            <span style={{ color: COMPOUND_COLORS[s.compound ?? ""] }}>{s.compound?.[0] ?? "?"}</span>{" "}
            L{s.from}–{s.to}
          </span>
        ))}
      </div>
    </div>
  );
}
