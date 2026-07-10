"""RaceEngine + replay harness tests: the full live code path driven by the
Monza fixture, with and without a trained model."""

import pytest

from app.state import RaceEngine, SessionMeta, status_flag
from features.build import (
    classify_lap_scalar,
    classify_laps,
    prepare_session_laps,
    reference_series,
)
from tests.conftest import MONZA_META

MONZA_SESSION = SessionMeta(
    session_id="2025_16_monza",
    track_id="monza",
    laps_total=MONZA_META["laps_total"],
    lap_length_km=MONZA_META["lap_length_km"],
    abrasiveness=3.0,
    season=2025,
    pole_time_s=MONZA_META["pole_time_s"],
)


def feed_engine(engine: RaceEngine, race: dict):
    """Feed lap + weather events in chronological order (like the replay)."""
    events = []
    for row in race["laps"].iter_rows(named=True):
        t = row["time_session_s"]
        events.append((t if t is not None else float("inf"), 1, "lap", row))
    for row in race["weather"].iter_rows(named=True):
        events.append((row["time_session_s"], 0, "weather", row))
    events.sort(key=lambda e: (e[0], e[1]))
    updates = []
    for _, _, kind, payload in events:
        if kind == "weather":
            engine.on_weather(payload)
        else:
            u = engine.on_lap(payload)
            if u:
                updates.append(u)
    return updates


def test_scalar_classifier_matches_frame_classifier(both_races):
    for race in both_races:
        frame = classify_laps(race["laps"])
        for row in frame.iter_rows(named=True):
            got = classify_lap_scalar(
                row["lap_number"], row["pit_in"], row["pit_out"], row["track_status"]
            )
            assert got == row["lap_class"], row


def test_status_flag():
    assert status_flag("1") == "green"
    assert status_flag("12") == "yellow"
    assert status_flag("45") == "red"
    assert status_flag("671") == "vsc"
    assert status_flag(None) == "green"


def test_engine_reference_tracks_batch_reference(monza):
    engine = RaceEngine(MONZA_SESSION, model=None)
    feed_engine(engine, monza)
    prepared = prepare_session_laps(
        monza["laps"], MONZA_SESSION.laps_total, MONZA_SESSION.lap_length_km
    )
    batch = reference_series(prepared, MONZA_SESSION.pole_time_s)
    live_refs = engine._refs
    checked = 0
    for lap, batch_ref in zip(batch["lap_number"], batch["reference_s"], strict=True):
        if lap in live_refs and batch_ref is not None and lap >= 6:
            assert live_refs[lap] == pytest.approx(batch_ref, rel=0.02), f"lap {lap}"
            checked += 1
    assert checked > 30, "too few laps compared"


def test_engine_full_replay_with_model(monza, tiny_model):
    model, _ = tiny_model
    engine = RaceEngine(MONZA_SESSION, model=model)
    updates = feed_engine(engine, monza)

    assert len(updates) == len(monza["laps"])  # one snapshot per completed lap
    final = updates[-1]
    assert final.lap == MONZA_SESSION.laps_total
    assert 18 <= len(final.drivers) <= 22

    forecasted = [d for d in final.drivers if d.forecast and d.forecast.current]
    assert len(forecasted) >= 15, "most drivers should carry forecasts at the flag"
    for d in forecasted:
        fc = d.forecast
        assert fc.current.p10 <= fc.current.p50 <= fc.current.p90
        assert 75 < fc.current.p50 < 110, f"implausible Monza forecast {fc.current}"
        assert set(fc.fresh) == {"SOFT", "MEDIUM", "HARD"}
        for q in fc.fresh.values():
            assert q.p10 <= q.p50 <= q.p90
        assert len(fc.ahead["current"]) == 15
        for curve in fc.ahead.values():
            assert len(curve) == 15

    # online bias engaged for most drivers by the flag
    biased = [d for d in final.drivers if d.bias_s is not None and d.bias_s != 0.0]
    assert len(biased) >= 10

    # degradation curves fitted for compounds seen in the race
    assert engine.degradation.curves, "no degradation curves fitted"
    for curve in engine.degradation.curves.values():
        assert curve.ratio(30) >= curve.ratio(0) - 1e-6  # no negative-quadratic silliness

    # snapshot serialises for the WebSocket
    assert '"type":"lap_update"' in final.model_dump_json()


def test_engine_forecast_reacts_to_pace(monza, tiny_model):
    """A driver consistently slower than the model must get a positive bias."""
    model, _ = tiny_model
    engine = RaceEngine(MONZA_SESSION, model=model)
    rows = monza["laps"].sort("time_session_s")
    slow_driver = None
    for row in rows.iter_rows(named=True):
        r = dict(row)
        if slow_driver is None and r["lap_number"] > 5:
            slow_driver = r["driver_number"]
        if r["driver_number"] == slow_driver and r["lap_time_s"] is not None:
            r["lap_time_s"] = r["lap_time_s"] + 2.5  # sandbag by 2.5s every lap
        engine.on_lap(r)
    assert slow_driver is not None
    assert engine.bias.get(slow_driver) > 0.005, "bias failed to learn a 2.5s deficit"
