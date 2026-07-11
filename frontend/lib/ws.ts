// WebSocket client: resume from last _id, auto-reconnect with backoff,
// staleness flag after 15s of silence.

import { fetchHistory } from "./api";
import { usePitwall } from "./store";
import type { LapUpdate } from "./types";

const STALE_MS = 15_000;

export function wsUrl(): string {
  return process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/live";
}

let started = false;

export function startWs(): void {
  if (started || typeof window === "undefined") return;
  started = true;

  let attempt = 0;
  let staleTimer: ReturnType<typeof setTimeout> | null = null;

  const armStaleTimer = () => {
    if (staleTimer) clearTimeout(staleTimer);
    staleTimer = setTimeout(() => usePitwall.getState().setStatus("stale"), STALE_MS);
  };

  const connect = () => {
    const lastId = usePitwall.getState().lastId;
    const url = lastId != null ? `${wsUrl()}?since=${lastId}` : wsUrl();
    const ws = new WebSocket(url);

    ws.onopen = () => {
      attempt = 0;
      armStaleTimer();
      // backfill lap history for charts (late joiners see the whole race)
      fetchHistory().then((h) => h && usePitwall.getState().seedHistory(h));
    };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data) as LapUpdate;
      if (msg.type === "lap_update") usePitwall.getState().applyUpdate(msg);
      armStaleTimer();
    };
    ws.onclose = () => {
      usePitwall.getState().setStatus("closed");
      const delay = Math.min(1000 * 2 ** attempt, 15_000);
      attempt += 1;
      setTimeout(connect, delay);
    };
    ws.onerror = () => ws.close();
  };

  connect();
}
