"""tracks.yaml integrity: loads, has every required field, and (when the
consolidated parquet exists) covers every track in the data."""

import polars as pl
import pytest

from app import config
from app.tracks import REQUIRED, load_tracks


def test_tracks_yaml_loads_and_is_complete():
    tracks = load_tracks()
    assert len(tracks) >= 25
    for track_id, meta in tracks.items():
        for key in REQUIRED:
            assert key in meta, f"{track_id} missing {key}"
        assert 2.5 < meta["lap_length_km"] < 8.0, track_id
        assert 40 <= meta["laps_total"] <= 87, track_id
        assert 5 < meta["pit_loss_s"] < 45, track_id
        assert 0 <= meta["sc_hazard_per_lap"] < 0.45, track_id  # hockenheim 2019 was SC chaos
        assert 0 <= meta["overtake_difficulty"] <= 1, track_id
        assert 1 <= meta["abrasiveness"] <= 5, track_id


def test_every_session_track_has_metadata():
    if not config.SESSIONS_PARQUET.exists():
        pytest.skip("consolidated data not present")
    from ingest.fastf1_pull import canon_track

    sessions = pl.read_parquet(config.SESSIONS_PARQUET)
    tracks = load_tracks()
    missing = {canon_track(t) for t in sessions["track_id"].unique().to_list()} - set(tracks)
    assert not missing, f"tracks.yaml missing: {missing}"


def test_monza_pit_loss_computed_sanely():
    tracks = load_tracks()
    # Monza's pit loss is famously ~20-26s; the computed value must be in that zone
    assert 15 < tracks["monza"]["pit_loss_s"] < 30
    assert tracks["monza"]["laps_total"] == 53
