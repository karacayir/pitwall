"use client";

import { usePitwall } from "@/lib/store";

export function ReplayBanner() {
  const session = usePitwall((s) => s.session);
  if (!session || session.data_source !== "replay") return null;
  return (
    <div className="flex items-center gap-2 border-b border-(--hairline) bg-(--raised) px-4 py-1 text-xs text-(--muted)">
      <span className="rounded bg-(--surface) px-1.5 py-0.5 font-bold tracking-wider text-(--medium)">
        REPLAY
      </span>
      <span>
        {session.session_id} at {session.replay_speed ?? "?"}× speed — historical data, not a live
        session
      </span>
    </div>
  );
}
