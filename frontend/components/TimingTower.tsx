"use client";

import { AnimatePresence, motion } from "framer-motion";
import Link from "next/link";
import { usePitwall } from "@/lib/store";
import type { DriverUpdate } from "@/lib/types";
import { CompoundIcon } from "./CompoundIcon";
import { Sparkline } from "./Sparkline";

const TEAM_COLORS: Record<string, string> = {
  red_bull_racing: "#4781D7",
  mclaren: "#F47600",
  ferrari: "#ED1131",
  mercedes: "#00D7B6",
  aston_martin: "#229971",
  alpine: "#00A1E8",
  williams: "#1868DB",
  racing_bulls: "#6C98FF",
  rb: "#6C98FF",
  kick_sauber: "#01C00E",
  audi: "#01C00E",
  haas_f1_team: "#9C9FA2",
  cadillac: "#B6862D",
};

function fmt(t: number | null | undefined, digits = 3): string {
  if (t == null) return "–";
  if (t >= 60) {
    const m = Math.floor(t / 60);
    return `${m}:${(t - m * 60).toFixed(digits).padStart(digits + 3, "0")}`;
  }
  return t.toFixed(digits);
}

function Row({ d, selected, onClick }: { d: DriverUpdate; selected: boolean; onClick: () => void }) {
  const teamColor = TEAM_COLORS[d.team_id ?? ""] ?? "#8B8B98";
  return (
    <motion.li
      layout
      transition={{ duration: 0.18, ease: "easeOut" }}
      className={`flex cursor-pointer items-center gap-2 border-b border-(--hairline) px-2 py-1.5 text-sm ${
        selected ? "bg-(--raised)" : "hover:bg-(--surface)"
      }`}
      onClick={onClick}
      data-driver={d.driver_number}
    >
      <span className="timing w-6 text-right text-(--muted)">{d.position ?? "–"}</span>
      <span className="h-4 w-0.5 rounded" style={{ background: teamColor }} />
      <span className="display w-11 font-semibold tracking-wider">
        {d.driver_code ?? `#${d.driver_number}`}
      </span>
      <span className="timing w-16 text-right text-(--muted)">
        {d.position === 1 ? "int." : d.gap_ahead_s != null ? `+${d.gap_ahead_s.toFixed(1)}` : "–"}
      </span>
      <span className="w-12">
        <CompoundIcon compound={d.compound} age={d.tyre_age} size={16} />
      </span>
      <span className="timing w-20 text-right">{fmt(d.last_lap_s)}</span>
      <span className="ml-auto">
        <Sparkline forecast={d.forecast} />
      </span>
      <Link
        href={`/driver/${d.driver_number}`}
        className="text-xs text-(--muted) hover:text-(--text)"
        onClick={(e) => e.stopPropagation()}
        aria-label={`driver ${d.driver_number} details`}
      >
        →
      </Link>
    </motion.li>
  );
}

export function TimingTower() {
  const latest = usePitwall((s) => s.latest);
  const selected = usePitwall((s) => s.selected);
  const toggle = usePitwall((s) => s.toggleSelected);

  if (!latest) {
    return <div className="p-4 text-sm text-(--muted)">waiting for timing…</div>;
  }
  return (
    <ul className="overflow-y-auto">
      <AnimatePresence initial={false}>
        {latest.drivers.map((d) => (
          <Row
            key={d.driver_number}
            d={d}
            selected={selected.includes(d.driver_number)}
            onClick={() => toggle(d.driver_number)}
          />
        ))}
      </AnimatePresence>
    </ul>
  );
}
