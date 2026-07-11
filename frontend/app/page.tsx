"use client";

import { useEffect } from "react";
import { CompoundFan } from "@/components/CompoundFan";
import { HeaderStrip } from "@/components/HeaderStrip";
import { LapChart } from "@/components/LapChart";
import { PitNowPanel } from "@/components/PitNowPanel";
import { ReplayBanner } from "@/components/ReplayBanner";
import { TimingTower } from "@/components/TimingTower";
import { fetchSession } from "@/lib/api";
import { usePitwall } from "@/lib/store";
import { startWs } from "@/lib/ws";

export default function LiveBoard() {
  const latest = usePitwall((s) => s.latest);
  const selected = usePitwall((s) => s.selected);
  const setSession = usePitwall((s) => s.setSession);

  useEffect(() => {
    startWs();
    fetchSession().then((s) => s && setSession(s));
  }, [setSession]);

  const focusDriver = selected[0] ?? null;
  const focus = latest?.drivers.find((d) => d.driver_number === focusDriver);

  return (
    <div className="flex min-h-screen flex-col">
      <ReplayBanner />
      <HeaderStrip />
      {!latest ? (
        <EmptyState />
      ) : (
        <main className="flex grow flex-col gap-0 md:flex-row">
          <aside className="border-b border-(--hairline) md:w-105 md:border-r md:border-b-0">
            <TimingTower />
          </aside>
          <section className="flex grow flex-col">
            <div className="border-b border-(--hairline) p-3">
              <h2 className="display mb-1 text-xs font-semibold tracking-widest text-(--muted) uppercase">
                Lap times + forecast
              </h2>
              <LapChart />
            </div>
            <div className="grid grow grid-cols-1 lg:grid-cols-2">
              <div className="border-b border-(--hairline) lg:border-r lg:border-b-0">
                <PitNowPanel driverNumber={focusDriver} />
              </div>
              <div className="p-3">
                <h3 className="display mb-1 text-xs font-semibold tracking-widest text-(--muted) uppercase">
                  Compound fan — {focus?.driver_code ?? "select a driver"}
                </h3>
                {focus && <CompoundFan driver={focus} width={430} height={180} />}
              </div>
            </div>
          </section>
        </main>
      )}
    </div>
  );
}

function EmptyState() {
  return (
    <main className="flex grow flex-col items-center justify-center gap-3 p-8 text-center">
      <h1 className="display text-2xl font-bold tracking-widest uppercase">No session</h1>
      <p className="max-w-md text-sm text-(--muted)">
        Waiting for timing data. In replay mode the stream starts automatically; in live mode the
        board wakes up when the next race session begins.
      </p>
    </main>
  );
}
