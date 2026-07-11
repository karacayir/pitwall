"""Headless WebSocket smoke test (Phase 5 acceptance).

Connects to a running Pitwall backend during a replay, consumes /ws/live for
--seconds, and validates EVERY message against the pydantic schemas. Exits
non-zero on any invalid message.

    uv run python scripts/ws_smoke.py --url ws://localhost:8000/ws/live --seconds 20
"""

import argparse
import asyncio
import json
import sys
import time

sys.path.insert(0, ".")  # run from backend/

from app import schemas  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/ws/live")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--min-messages", type=int, default=3)
    args = parser.parse_args()

    import websockets  # bundled with uvicorn[standard]

    n, last_id = 0, None
    deadline = time.time() + args.seconds
    async with websockets.connect(args.url) as ws:
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(deadline - time.time(), 0.1))
            except TimeoutError:
                break
            msg = json.loads(raw)
            update = schemas.LapUpdate.model_validate(msg)  # raises on contract breach
            mid = msg.get("_id")
            if mid is not None and last_id is not None and mid <= last_id:
                print(f"FAIL: _id went backwards ({last_id} -> {mid})")
                return 1
            last_id = mid
            n += 1
            print(
                f"ok #{n} _id={mid} lap {update.lap}/{update.laps_total} "
                f"{update.track_status} drivers={len(update.drivers)}"
            )
    if n < args.min_messages:
        print(f"FAIL: only {n} messages received (need >= {args.min_messages})")
        return 1
    print(f"PASS: {n} valid messages")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
