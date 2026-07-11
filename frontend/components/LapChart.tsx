"use client";

// Streaming lap-time chart (uPlot) for selected drivers, with each driver's
// current-tyre forecast extension (dashed) drawn past the live lap.

import { useEffect, useMemo, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { usePitwall } from "@/lib/store";

const LINE_COLORS = ["#4781D7", "#F47600", "#ED1131", "#00D7B6", "#A855F7"];

export function LapChart({ height = 320 }: { height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const latest = usePitwall((s) => s.latest);
  const history = usePitwall((s) => s.history);
  const selected = usePitwall((s) => s.selected);

  const { data, series } = useMemo(() => {
    const lapsTotal = latest?.laps_total ?? 60;
    const xs = Array.from({ length: lapsTotal + 15 }, (_, i) => i + 1);
    const seriesDefs: uPlot.Series[] = [{ label: "lap" }];
    const cols: (number | null | undefined)[][] = [xs];

    selected.forEach((dn, i) => {
      const laps = history[dn] ?? [];
      const col: (number | null)[] = xs.map(() => null);
      for (const p of laps) if (p.lap <= xs.length && p.time != null && p.time < 200) col[p.lap - 1] = p.time;
      const d = latest?.drivers.find((x) => x.driver_number === dn);
      const fCol: (number | null)[] = xs.map(() => null);
      const ahead = d?.forecast?.ahead?.["current"];
      if (d?.lap_number && ahead) {
        ahead.forEach((v, k) => {
          const lap = d.lap_number! + 1 + k;
          if (lap <= xs.length) fCol[lap - 1] = v;
        });
        if (d.last_lap_s != null) fCol[d.lap_number - 1] = d.last_lap_s; // join the dots
      }
      const color = LINE_COLORS[i % LINE_COLORS.length];
      cols.push(col);
      seriesDefs.push({
        label: d?.driver_code ?? `#${dn}`,
        stroke: color,
        width: 1.5,
        spanGaps: true,
        points: { show: false },
      });
      cols.push(fCol);
      seriesDefs.push({
        label: `${d?.driver_code ?? dn} fc`,
        stroke: color,
        width: 1,
        dash: [5, 4],
        spanGaps: true,
        points: { show: false },
      });
    });
    return { data: cols as uPlot.AlignedData, series: seriesDefs };
  }, [latest, history, selected]);

  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    const make = () => {
      plotRef.current?.destroy();
      plotRef.current = new uPlot(
        {
          width: el.clientWidth,
          height,
          series,
          scales: { x: { time: false } },
          axes: [
            { stroke: "#8B8B98", grid: { stroke: "#26262E" }, font: "10px var(--font-mono)" },
            {
              stroke: "#8B8B98",
              grid: { stroke: "#26262E" },
              font: "10px var(--font-mono)",
              values: (_u, vals) => vals.map((v) => v.toFixed(1)),
            },
          ],
          legend: { show: true },
          cursor: { points: { size: 5 } },
        },
        data,
        el,
      );
    };
    make();
    const ro = new ResizeObserver(() => {
      if (plotRef.current && el.clientWidth > 0)
        plotRef.current.setSize({ width: el.clientWidth, height });
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      plotRef.current?.destroy();
      plotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series.length]);

  useEffect(() => {
    plotRef.current?.setData(data);
  }, [data]);

  return <div ref={ref} className="w-full overflow-x-auto" style={{ minHeight: height }} />;
}
