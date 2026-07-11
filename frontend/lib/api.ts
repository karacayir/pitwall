import type { SessionInfo, SimulateResponse, StintPlan } from "./types";

export function apiBase(): string {
  return process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
}

export async function fetchSession(): Promise<SessionInfo | null> {
  try {
    const r = await fetch(`${apiBase()}/api/session`);
    return r.ok ? ((await r.json()) as SessionInfo) : null;
  } catch {
    return null;
  }
}

export async function fetchHistory(): Promise<Record<
  number,
  { lap: number; time: number | null; compound: string | null; gap: number | null; position: number | null }[]
> | null> {
  try {
    const r = await fetch(`${apiBase()}/api/history`);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

export async function simulate(
  driver_number: number,
  strategies: StintPlan[][] | null = null,
  n_sims = 2000,
): Promise<SimulateResponse> {
  const r = await fetch(`${apiBase()}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ driver_number, strategies, n_sims }),
  });
  if (!r.ok) throw new Error(`simulate failed: ${r.status} ${await r.text()}`);
  return (await r.json()) as SimulateResponse;
}
