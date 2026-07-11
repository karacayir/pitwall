"use client";

import { useEffect } from "react";
import { CompoundFan } from "@/components/CompoundFan";
import { GapChart } from "@/components/GapChart";
import { HeaderStrip } from "@/components/HeaderStrip";
import { ReplayBanner } from "@/components/ReplayBanner";
import { StintBars } from "@/components/StintBars";
import { fetchSession } from "@/lib/api";
import { usePitwall } from "@/lib/store";
import { startWs } from "@/lib/ws";

export function DriverClient({ num }: { num: string }) {
  const driverNumber = Number(num);
  const latest = usePitwall((s) => s.latest);
  const history = usePitwall((s) => s.history);
  const setSession = usePitwall((s) => s.setSession);

  useEffect(() => {
    startWs();
    fetchSession().then((s) => s && setSession(s));
  }, [setSession]);

  const d = latest?.drivers.find((x) => x.driver_number === driverNumber);
  const laps = history[driverNumber] ?? [];

  return (
    <div className="flex min-h-screen flex-col">
      <ReplayBanner />
      <HeaderStrip />
      {!d ? (
        <main className="grow p-8 text-sm text-(--muted)">no data for car #{num} yet…</main>
      ) : (
        <main className="grow space-y-4 p-4">
          <div className="flex flex-wrap items-baseline gap-4">
            <h1 className="display text-3xl font-bold tracking-widest uppercase">
              {d.driver_code ?? `#${driverNumber}`}
            </h1>
            <span className="timing text-sm text-(--muted)">
              P{d.position ?? "–"} · lap {d.lap_number ?? "–"} · {d.compound ?? "?"} ·{" "}
              {d.tyre_age ?? "–"} laps old
            </span>
            <span className="timing rounded border border-(--hairline) bg-(--surface) px-2 py-0.5 text-xs">
              model bias {d.bias_s != null ? `${d.bias_s >= 0 ? "+" : ""}${d.bias_s.toFixed(2)}s` : "–"}{" "}
              <span className="text-(--muted)">vs pre-race model</span>
            </span>
          </div>

          <section className="rounded border border-(--hairline) bg-(--surface) p-3">
            <h2 className="display mb-2 text-xs font-semibold tracking-widest text-(--muted) uppercase">
              Compound fan — next 15 laps
            </h2>
            <div className="overflow-x-auto">
              <CompoundFan driver={d} width={760} height={280} />
            </div>
          </section>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <section className="rounded border border-(--hairline) bg-(--surface) p-3">
              <h2 className="display mb-2 text-xs font-semibold tracking-widest text-(--muted) uppercase">
                Stint history
              </h2>
              <StintBars laps={laps} lapsTotal={latest?.laps_total ?? 60} />
            </section>
            <section className="rounded border border-(--hairline) bg-(--surface) p-3">
              <h2 className="display mb-2 text-xs font-semibold tracking-widest text-(--muted) uppercase">
                Gap to car ahead
              </h2>
              <GapChart laps={laps} />
            </section>
          </div>
        </main>
      )}
    </div>
  );
}
