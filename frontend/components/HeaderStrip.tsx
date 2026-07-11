"use client";

import Link from "next/link";
import { usePitwall } from "@/lib/store";

const FLAG_STYLES: Record<string, { label: string; cls: string }> = {
  green: { label: "GREEN", cls: "bg-[#0f2a16] text-[#3DBE5B] border-[#1d4526]" },
  yellow: { label: "YELLOW", cls: "bg-[#2a250f] text-[#F7C325] border-[#453f1d]" },
  sc: { label: "SAFETY CAR", cls: "bg-[#2a250f] text-[#F7C325] border-[#453f1d]" },
  vsc: { label: "VSC", cls: "bg-[#2a250f] text-[#F7C325] border-[#453f1d]" },
  red: { label: "RED FLAG", cls: "bg-[#2a0f0f] text-[#E10600] border-[#451d1d]" },
};

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded border border-(--hairline) bg-(--surface) px-2 py-0.5 text-xs text-(--muted)">
      {children}
    </span>
  );
}

export function HeaderStrip() {
  const latest = usePitwall((s) => s.latest);
  const session = usePitwall((s) => s.session);
  const status = usePitwall((s) => s.status);

  const flag = FLAG_STYLES[latest?.track_status ?? "green"];
  const w = latest?.weather;

  return (
    <header className="flex flex-wrap items-center gap-3 border-b border-(--hairline) px-4 py-2">
      <Link href="/" className="display text-xl font-bold tracking-[0.25em] uppercase">
        Pitwall
      </Link>
      <span
        className={`inline-flex items-center gap-1.5 text-xs ${
          status === "live" ? "text-(--race-red)" : "text-(--muted)"
        }`}
      >
        <span
          className={`h-2 w-2 rounded-full ${
            status === "live" ? "bg-(--race-red)" : "bg-(--muted)"
          }`}
        />
        {status === "live" ? "LIVE" : status.toUpperCase()}
      </span>

      <span className="display text-sm font-semibold tracking-wider text-(--muted) uppercase">
        {session?.event_name ?? session?.track_id ?? "—"}
      </span>

      {latest && (
        <span
          key={latest.track_status}
          className={`flag-pulse rounded border px-2 py-0.5 text-xs font-bold tracking-wider ${flag.cls}`}
        >
          {flag.label}
        </span>
      )}
      {latest && (
        <span className="timing text-sm">
          LAP {latest.lap}
          <span className="text-(--muted)">/{latest.laps_total}</span>
        </span>
      )}

      <span className="grow" />
      <nav className="flex items-center gap-3 text-sm text-(--muted)">
        <Link href="/" className="hover:text-(--text)">
          Live
        </Link>
        <Link href="/strategy" className="hover:text-(--text)">
          Strategy lab
        </Link>
      </nav>

      {w && (
        <span className="flex items-center gap-2">
          <Chip>
            track <span className="timing text-(--text)">{w.track_temp?.toFixed(0) ?? "–"}°</span>
          </Chip>
          <Chip>
            air <span className="timing text-(--text)">{w.air_temp?.toFixed(0) ?? "–"}°</span>
          </Chip>
          <Chip>
            rain{" "}
            <span className="timing text-(--text)">
              {w.rain ? "YES" : w.rain_prob_15m != null ? `${Math.round(w.rain_prob_15m * 100)}%` : "no"}
            </span>
          </Chip>
        </span>
      )}
    </header>
  );
}
