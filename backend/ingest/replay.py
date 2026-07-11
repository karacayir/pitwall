"""Replay harness — THE key dev tool.

Streams a historical race's events (laps + weather) chronologically through
the exact same RaceEngine code path the live client uses, at configurable
speed (1x-60x, or 0 = as fast as possible for backtests).

CLI (headless): DATA_SOURCE=replay REPLAY_RACE=2025_monza REPLAY_SPEED=60 \
    uv run python -m ingest.replay
or: uv run python -m ingest.replay --race 2025_monza --speed 60
"""

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import polars as pl

from app import config
from app.state import RaceEngine, SessionMeta
from app.tracks import load_tracks
from models.predict import PaceModel

log = logging.getLogger("pitwall.replay")

RAW_DIR = config.DATA_DIR / "raw"


def resolve_race(race: str) -> Path:
    """ "2025_monza" or a full session_id ("2025_16_monza") -> raw race dir."""
    exact = RAW_DIR / race
    if (exact / "_complete").exists():
        return exact
    parts = race.split("_", 1)
    if len(parts) == 2:
        year, track = parts
        matches = sorted(RAW_DIR.glob(f"{year}_*_{track}"))
        matches = [m for m in matches if (m / "_complete").exists()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise SystemExit(
                f"ambiguous race {race!r}: {[m.name for m in matches]} — use the full session_id"
            )
    raise SystemExit(f"race {race!r} not found under {RAW_DIR} (run `make data` first)")


def load_session_meta(race_dir: Path, tracks: dict | None = None) -> SessionMeta:
    from ingest.fastf1_pull import canon_track

    meta = json.loads((race_dir / "session.json").read_text())
    tracks = tracks or load_tracks()
    track_id = canon_track(meta["track_id"])
    track = tracks.get(track_id, {})
    return SessionMeta(
        session_id=meta["session_id"],
        track_id=track_id,
        laps_total=meta["laps_total"],
        lap_length_km=track.get("lap_length_km", 5.0),
        abrasiveness=track.get("abrasiveness", 3.0),
        season=meta["season"],
        pole_time_s=meta.get("pole_time_s"),
        event_name=meta.get("event_name"),
        pit_loss_s=track.get("pit_loss_s", 22.0),
        sc_hazard_per_lap=track.get("sc_hazard_per_lap", 0.0035),
        overtake_difficulty=track.get("overtake_difficulty", 0.5),
    )


def event_stream(race_dir: Path) -> Iterator[tuple[float, str, dict]]:
    """(session_time_s, kind, payload) in chronological order.

    Lap events fire at the lap's END (time_session_s); weather at its sample
    time. Laps with no usable clock sort at their estimated position.
    """
    laps = pl.read_parquet(race_dir / "laps.parquet")
    weather = pl.read_parquet(race_dir / "weather.parquet")

    events: list[tuple[float, int, str, dict]] = []
    for row in laps.iter_rows(named=True):
        t = row["time_session_s"]
        if t is None:
            start = row["lap_start_session_s"]
            t = (start + (row["lap_time_s"] or 120.0)) if start is not None else float("inf")
        events.append((float(t), 1, "lap", row))
    for row in weather.iter_rows(named=True):
        events.append((float(row["time_session_s"]), 0, "weather", row))

    events.sort(key=lambda e: (e[0], e[1]))
    for t, _, kind, payload in events:
        if t != float("inf"):
            yield t, kind, payload


def run_replay(
    race_dir: Path,
    speed: float,
    engine: RaceEngine,
    on_update: Callable | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    """Feed the event stream into the engine. speed<=0 means flat out.
    Returns the number of lap updates emitted."""
    n_updates = 0
    prev_t: float | None = None
    started = False  # fast-forward the pre-race dead air (weather-only feed)
    for t, kind, payload in event_stream(race_dir):
        started = started or kind == "lap"
        if started and speed > 0 and prev_t is not None and t > prev_t:
            sleeper(min((t - prev_t) / speed, 60.0))
        if started:
            prev_t = t
        if kind == "weather":
            engine.on_weather(payload)
            continue
        update = engine.on_lap(payload)
        if update is not None:
            n_updates += 1
            if on_update is not None:
                on_update(update)
    return n_updates


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Replay a historical race headlessly")
    parser.add_argument("--race", default=config.REPLAY_RACE)
    parser.add_argument("--speed", type=float, default=config.REPLAY_SPEED)
    parser.add_argument("--no-model", action="store_true", help="skip forecasts")
    args = parser.parse_args()

    race_dir = resolve_race(args.race)
    meta = load_session_meta(race_dir)
    model = None
    if not args.no_model:
        try:
            model = PaceModel.latest()
        except FileNotFoundError:
            log.warning("no trained model found — replaying without forecasts")
    engine = RaceEngine(meta, model)

    last_lap_seen = 0

    def print_lap(update):
        nonlocal last_lap_seen
        if update.lap != last_lap_seen:
            last_lap_seen = update.lap
            leader = update.drivers[0] if update.drivers else None
            fc = leader.forecast.current if leader and leader.forecast else None
            log.info(
                "lap %2d/%d %-6s ref=%.3fs leader=%s last=%s next_p50=%s",
                update.lap,
                update.laps_total,
                update.track_status,
                update.reference_pace_s or float("nan"),
                leader.driver_code if leader else "?",
                f"{leader.last_lap_s:.3f}" if leader and leader.last_lap_s else "-",
                f"{fc.p50:.3f}" if fc else "-",
            )

    n = run_replay(race_dir, args.speed, engine, on_update=print_lap)
    log.info("replay complete: %d lap updates, final lap %d", n, engine.max_lap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
