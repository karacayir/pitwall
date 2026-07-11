"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { HeaderStrip } from "@/components/HeaderStrip";
import { ReplayBanner } from "@/components/ReplayBanner";
import { fetchSession, simulate } from "@/lib/api";
import { usePitwall } from "@/lib/store";
import type { SimulateResponse, StintPlan, StrategyResult } from "@/lib/types";
import { COMPOUND_COLORS } from "@/lib/types";
import { startWs } from "@/lib/ws";

const BAR_COLORS = ["#4781D7", "#F47600", "#00D7B6", "#A855F7"];

export default function StrategyLab() {
  const latest = usePitwall((s) => s.latest);
  const setSession = usePitwall((s) => s.setSession);
  const [driver, setDriver] = useState<number | null>(null);
  const [custom, setCustom] = useState<StintPlan[]>([]);
  const [result, setResult] = useState<SimulateResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    startWs();
    fetchSession().then((s) => s && setSession(s));
  }, [setSession]);

  const drivers = latest?.drivers ?? [];
  const chosen = drivers.find((d) => d.driver_number === driver);

  const run = async (useCustom: boolean) => {
    if (driver == null) return;
    setRunning(true);
    setError(null);
    try {
      const strategies = useCustom && custom.length ? [custom] : null;
      setResult(await simulate(driver, strategies));
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col">
      <ReplayBanner />
      <HeaderStrip />
      <main className="grow space-y-4 p-4">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="display text-xl font-bold tracking-widest uppercase">Strategy lab</h1>
          <select
            className="rounded border border-(--hairline) bg-(--surface) px-2 py-1 text-sm"
            value={driver ?? ""}
            onChange={(e) => setDriver(Number(e.target.value) || null)}
          >
            <option value="">pick a driver…</option>
            {drivers.map((d) => (
              <option key={d.driver_number} value={d.driver_number}>
                P{d.position ?? "?"} {d.driver_code ?? d.driver_number}
              </option>
            ))}
          </select>
          <button
            className="rounded bg-(--race-red) px-3 py-1 text-sm font-semibold text-white disabled:opacity-40"
            disabled={driver == null || running}
            onClick={() => run(false)}
          >
            {running ? "simulating…" : "Run 2000 sims"}
          </button>
          {chosen && (
            <span className="timing text-xs text-(--muted)">
              from lap {(chosen.lap_number ?? 0) + 1} · currently P{chosen.position} on{" "}
              {chosen.compound} ({chosen.tyre_age} laps)
            </span>
          )}
        </div>

        <CustomBuilder
          custom={custom}
          setCustom={setCustom}
          lapsTotal={latest?.laps_total ?? 60}
          fromLap={(chosen?.lap_number ?? 0) + 2}
          onRun={() => run(true)}
          disabled={driver == null || running}
        />

        {error && <p className="text-sm text-(--race-red)">{error}</p>}
        {result && <Results result={result} currentPosition={chosen?.position ?? null} />}
      </main>
    </div>
  );
}

function CustomBuilder({
  custom,
  setCustom,
  lapsTotal,
  fromLap,
  onRun,
  disabled,
}: {
  custom: StintPlan[];
  setCustom: (s: StintPlan[]) => void;
  lapsTotal: number;
  fromLap: number;
  onRun: () => void;
  disabled: boolean;
}) {
  const [lap, setLap] = useState(fromLap + 5);
  const [compound, setCompound] = useState("HARD");
  return (
    <section className="rounded border border-(--hairline) bg-(--surface) p-3">
      <h2 className="display mb-2 text-xs font-semibold tracking-widest text-(--muted) uppercase">
        Custom strategy
      </h2>
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <label className="text-(--muted)">pit at lap</label>
        <input
          type="number"
          min={fromLap}
          max={lapsTotal - 1}
          value={lap}
          onChange={(e) => setLap(Number(e.target.value))}
          className="timing w-20 rounded border border-(--hairline) bg-(--bg) px-2 py-1"
        />
        <select
          value={compound}
          onChange={(e) => setCompound(e.target.value)}
          className="rounded border border-(--hairline) bg-(--bg) px-2 py-1"
        >
          {["SOFT", "MEDIUM", "HARD"].map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
        <button
          className="rounded border border-(--hairline) px-2 py-1 hover:bg-(--raised)"
          onClick={() => setCustom([...custom, { lap, compound }].sort((a, b) => a.lap - b.lap))}
        >
          + add stop
        </button>
        {custom.map((s, i) => (
          <span key={i} className="timing rounded bg-(--raised) px-2 py-0.5 text-xs">
            L{s.lap} <span style={{ color: COMPOUND_COLORS[s.compound] }}>{s.compound}</span>
            <button
              className="ml-1 text-(--muted) hover:text-(--race-red)"
              onClick={() => setCustom(custom.filter((_, j) => j !== i))}
            >
              ×
            </button>
          </span>
        ))}
        {custom.length > 0 && (
          <button
            className="rounded bg-(--raised) px-3 py-1 font-semibold disabled:opacity-40"
            onClick={onRun}
            disabled={disabled}
          >
            simulate this plan
          </button>
        )}
      </div>
    </section>
  );
}

function Results({
  result,
  currentPosition,
}: {
  result: SimulateResponse;
  currentPosition: number | null;
}) {
  const top = result.strategies.slice(0, 8);
  const best = top[0];

  const distData = useMemo(() => {
    const shown = [result.baseline, ...top.slice(0, 3)];
    const positions = new Set<number>();
    for (const s of shown)
      for (const p of Object.keys(s.p_position)) positions.add(Number(p));
    return [...positions]
      .sort((a, b) => a - b)
      .map((pos) => {
        const row: Record<string, number | string> = { position: `P${pos}` };
        shown.forEach((s) => {
          row[s.label] = Math.round((s.p_position[String(pos)] ?? 0) * 100);
        });
        return row;
      });
  }, [result, top]);

  const gain =
    best && result.baseline
      ? result.baseline.finish_time_s.p50 - best.finish_time_s.p50
      : 0;

  return (
    <>
      {best && (
        <section className="rounded border border-(--hairline) bg-(--surface) p-3">
          <h2 className="display mb-1 text-xs font-semibold tracking-widest text-(--race-red) uppercase">
            Recommended
          </h2>
          <p className="text-sm">
            <span className="timing font-semibold">{best.label}</span> — expected finish P
            {best.expected_position.toFixed(1)}
            {best.p_better_than_current != null && currentPosition != null && (
              <span className="text-(--muted)">
                {" "}
                · {Math.round(best.p_better_than_current * 100)}% chance of beating P
                {currentPosition}
              </span>
            )}
          </p>
          <p className="mt-1 max-w-2xl text-xs text-(--muted)">
            Each stop costs pit-lane time, but fresher rubber laps faster every remaining lap. On
            these numbers the plan {gain >= 0 ? "recovers" : "loses"}{" "}
            {Math.abs(gain).toFixed(1)}s vs staying out (median finish{" "}
            {best.finish_time_s.p50.toFixed(0)}s vs {result.baseline.finish_time_s.p50.toFixed(0)}
            s) across {result.n_sims} simulated race endings, safety cars and traffic included.
          </p>
        </section>
      )}

      <section className="rounded border border-(--hairline) bg-(--surface) p-3">
        <h2 className="display mb-2 text-xs font-semibold tracking-widest text-(--muted) uppercase">
          Finish position distribution
        </h2>
        <div className="h-64 w-full">
          <ResponsiveContainer>
            <BarChart data={distData}>
              <CartesianGrid stroke="#26262E" vertical={false} />
              <XAxis dataKey="position" stroke="#8B8B98" fontSize={11} />
              <YAxis stroke="#8B8B98" fontSize={11} unit="%" />
              <Tooltip
                contentStyle={{ background: "#1A1A24", border: "1px solid #26262E" }}
                labelStyle={{ color: "#E8E8EC" }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey={result.baseline.label} fill="#8B8B98" />
              {top.slice(0, 3).map((s, i) => (
                <Bar key={s.label} dataKey={s.label} fill={BAR_COLORS[i]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="overflow-x-auto rounded border border-(--hairline) bg-(--surface)">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-(--hairline) text-left text-xs text-(--muted)">
              <th className="p-2">strategy</th>
              <th className="p-2 text-right">E[pos]</th>
              <th className="p-2 text-right">P(better)</th>
              <th className="p-2 text-right">finish p10</th>
              <th className="p-2 text-right">p50</th>
              <th className="p-2 text-right">p90</th>
            </tr>
          </thead>
          <tbody>
            <Row s={result.baseline} muted />
            {top.map((s) => (
              <Row key={s.label} s={s} />
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}

function Row({ s, muted = false }: { s: StrategyResult; muted?: boolean }) {
  return (
    <tr className={`border-b border-(--hairline) ${muted ? "text-(--muted)" : ""}`}>
      <td className="timing p-2">{s.label}</td>
      <td className="timing p-2 text-right">{s.expected_position.toFixed(2)}</td>
      <td className="timing p-2 text-right">
        {s.p_better_than_current != null ? `${Math.round(s.p_better_than_current * 100)}%` : "–"}
      </td>
      <td className="timing p-2 text-right">{s.finish_time_s.p10.toFixed(0)}s</td>
      <td className="timing p-2 text-right">{s.finish_time_s.p50.toFixed(0)}s</td>
      <td className="timing p-2 text-right">{s.finish_time_s.p90.toFixed(0)}s</td>
    </tr>
  );
}
