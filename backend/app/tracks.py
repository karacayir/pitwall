"""tracks.yaml loader. Static per-circuit metadata + fields computed from the
historical parquet (see ingest/tracks_build.py)."""

from functools import lru_cache

import yaml

from app import config

REQUIRED = (
    "lat",
    "lon",
    "laps_total",
    "lap_length_km",
    "pit_loss_s",
    "sc_hazard_per_lap",
    "overtake_difficulty",
    "abrasiveness",
)


@lru_cache(maxsize=1)
def load_tracks() -> dict[str, dict]:
    with open(config.TRACKS_YAML) as fh:
        data = yaml.safe_load(fh)
    tracks = data["tracks"]
    for track_id, meta in tracks.items():
        missing = [k for k in REQUIRED if k not in meta]
        if missing:
            raise ValueError(f"tracks.yaml: {track_id} missing {missing}")
    return tracks
