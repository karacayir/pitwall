"use client";

import { useEffect, useMemo, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { LapPoint } from "@/lib/store";

export function GapChart({ laps, height = 200 }: { laps: LapPoint[]; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  const data = useMemo<uPlot.AlignedData>(() => {
    const xs = laps.map((p) => p.lap);
    const gaps = laps.map((p) => (p.gap != null && p.gap < 60 ? p.gap : null));
    return [xs, gaps];
  }, [laps]);

  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    plotRef.current?.destroy();
    plotRef.current = new uPlot(
      {
        width: el.clientWidth || 500,
        height,
        series: [
          { label: "lap" },
          { label: "gap (s)", stroke: "#2E8BE0", width: 1.5, spanGaps: true, points: { show: false } },
        ],
        scales: { x: { time: false } },
        axes: [
          { stroke: "#8B8B98", grid: { stroke: "#26262E" }, font: "10px var(--font-mono)" },
          { stroke: "#8B8B98", grid: { stroke: "#26262E" }, font: "10px var(--font-mono)" },
        ],
        legend: { show: false },
      },
      data,
      el,
    );
    return () => {
      plotRef.current?.destroy();
      plotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    plotRef.current?.setData(data);
  }, [data]);

  return <div ref={ref} className="w-full overflow-x-auto" style={{ minHeight: height }} />;
}
