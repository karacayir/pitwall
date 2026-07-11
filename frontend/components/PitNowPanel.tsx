"use client";

// "If they pit now": fresh-tyre P50s with P10-P90 whiskers per dry compound.

import { usePitwall } from "@/lib/store";
import { COMPOUND_COLORS } from "@/lib/types";

export function PitNowPanel({ driverNumber }: { driverNumber: number | null }) {
  const latest = usePitwall((s) => s.latest);
  const d = latest?.drivers.find((x) => x.driver_number === driverNumber);
  const fresh = d?.forecast?.fresh;
  if (!d || !fresh || !Object.keys(fresh).length) {
    return (
      <div className="p-3 text-xs text-(--muted)">select a driver with a live forecast</div>
    );
  }

  const entries = Object.entries(fresh);
  const lo = Math.min(...entries.map(([, q]) => q.p10)) - 0.2;
  const hi = Math.max(...entries.map(([, q]) => q.p90)) + 0.2;
  const pct = (v: number) => ((v - lo) / (hi - lo)) * 100;
  const current = d.forecast?.current;

  return (
    <div className="space-y-2 p-3">
      <div className="flex items-baseline justify-between">
        <h3 className="display text-xs font-semibold tracking-widest text-(--muted) uppercase">
          If {d.driver_code ?? d.driver_number} pits now
        </h3>
        {current && (
          <span className="timing text-xs text-(--muted)">
            staying out: {current.p50.toFixed(2)}s
          </span>
        )}
      </div>
      {entries.map(([compound, q]) => (
        <div key={compound} className="flex items-center gap-2">
          <span
            className="display w-8 text-xs font-bold"
            style={{ color: COMPOUND_COLORS[compound] }}
          >
            {compound[0]}
          </span>
          <div className="relative h-5 grow">
            <div
              className="absolute top-1/2 h-px -translate-y-1/2"
              style={{
                left: `${pct(q.p10)}%`,
                width: `${pct(q.p90) - pct(q.p10)}%`,
                background: COMPOUND_COLORS[compound],
                opacity: 0.5,
              }}
            />
            {[q.p10, q.p90].map((v, i) => (
              <div
                key={i}
                className="absolute top-1/2 h-2.5 w-px -translate-y-1/2"
                style={{ left: `${pct(v)}%`, background: COMPOUND_COLORS[compound], opacity: 0.7 }}
              />
            ))}
            <div
              className="absolute top-1/2 h-3.5 w-1 -translate-x-1/2 -translate-y-1/2 rounded-sm"
              style={{ left: `${pct(q.p50)}%`, background: COMPOUND_COLORS[compound] }}
            />
            {current && (
              <div
                className="absolute top-0 h-full w-px border-l border-dashed border-(--muted)"
                style={{ left: `${pct(current.p50)}%` }}
                title="current tyres"
              />
            )}
          </div>
          <span className="timing w-16 text-right text-sm">{q.p50.toFixed(2)}</span>
        </div>
      ))}
      <p className="text-[10px] text-(--muted)">
        first flying lap on fresh rubber, P10–P90 whiskers · dashed line = staying out
      </p>
    </div>
  );
}
