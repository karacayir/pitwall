import { create } from "zustand";
import type { LapUpdate, SessionInfo } from "./types";

export interface LapPoint {
  lap: number;
  time: number | null;
  compound: string | null;
  gap: number | null;
  position: number | null;
}

export type ConnStatus = "connecting" | "live" | "stale" | "closed";

interface PitwallState {
  latest: LapUpdate | null;
  session: SessionInfo | null;
  history: Record<number, LapPoint[]>; // per driver, appended as laps complete
  selected: number[];
  status: ConnStatus;
  lastId: number | null;
  applyUpdate: (u: LapUpdate) => void;
  seedHistory: (h: Record<number, LapPoint[]>) => void;
  setSession: (s: SessionInfo) => void;
  setStatus: (s: ConnStatus) => void;
  toggleSelected: (driver: number) => void;
}

export const usePitwall = create<PitwallState>((set, get) => ({
  latest: null,
  session: null,
  history: {},
  selected: [],
  status: "connecting",
  lastId: null,

  applyUpdate: (u) => {
    const history = { ...get().history };
    for (const d of u.drivers) {
      if (d.lap_number == null) continue;
      const laps = history[d.driver_number] ?? [];
      const prev = laps[laps.length - 1];
      if (!prev || prev.lap < d.lap_number) {
        history[d.driver_number] = [
          ...laps,
          {
            lap: d.lap_number,
            time: d.last_lap_s,
            compound: d.compound,
            gap: d.gap_ahead_s,
            position: d.position,
          },
        ];
      }
    }
    const selected = get().selected.length
      ? get().selected
      : u.drivers.slice(0, 3).map((d) => d.driver_number);
    set({ latest: u, history, lastId: u._id ?? get().lastId, selected, status: "live" });
  },

  seedHistory: (h) =>
    set((st) => {
      // server backlog wins for laps we do not have yet
      const merged: Record<number, LapPoint[]> = { ...h };
      for (const [dn, laps] of Object.entries(st.history)) {
        const base = merged[Number(dn)] ?? [];
        const lastServer = base[base.length - 1]?.lap ?? 0;
        merged[Number(dn)] = [...base, ...laps.filter((p) => p.lap > lastServer)];
      }
      return { history: merged };
    }),

  setSession: (s) => set({ session: s }),
  setStatus: (s) => set({ status: s }),
  toggleSelected: (driver) =>
    set((st) => ({
      selected: st.selected.includes(driver)
        ? st.selected.filter((d) => d !== driver)
        : [...st.selected, driver].slice(-5),
    })),
}));
