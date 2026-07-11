"""OpenF1 mapper tests against real API fixtures (2025 Monza, session 9912)."""

import json
from pathlib import Path

import pytest

from ingest.openf1 import MessageLog, OpenF1Mapper, _epoch

FIXTURES = Path(__file__).parent / "fixtures" / "openf1"


def load(name: str) -> list[dict]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.fixture()
def mapper() -> OpenF1Mapper:
    m = OpenF1Mapper("live_9912")
    for msg in load("stints"):
        m.on_stint(msg)
    for msg in load("position"):
        m.on_position(msg)
    for msg in load("pit"):
        m.on_pit(msg)
    for msg in load("race_control"):
        m.on_race_control(msg)
    return m


def test_lap_row_fields(mapper):
    laps = load("laps")
    lap1 = next(lap for lap in laps if lap["lap_number"] == 1)
    row = mapper.on_lap(lap1)
    assert row is not None
    assert row["driver_number"] == 1
    assert row["lap_time_s"] == pytest.approx(87.498)
    assert row["compound"] == "MEDIUM"  # from the stints endpoint
    assert row["tyre_life"] == 1  # tyre_age_at_start 0 + first lap
    assert row["stint"] == 1
    assert row["pit_out"] is False
    assert row["time_session_s"] == pytest.approx(_epoch(lap1["date_start"]) + lap1["lap_duration"])
    assert row["sector1_s"] == pytest.approx(lap1["duration_sector_1"])


def test_tyre_life_advances_with_stint(mapper):
    laps = load("laps")
    lap5 = next(lap for lap in laps if lap["lap_number"] == 5)
    row = mapper.on_lap(lap5)
    assert row["tyre_life"] == 5  # stint 1 started lap 1 at age 0


def test_incomplete_lap_not_emitted(mapper):
    partial = dict(load("laps")[0], lap_number=99, lap_duration=None)
    assert mapper.on_lap(partial) is None


def test_lap_emitted_once(mapper):
    lap = load("laps")[2]
    assert mapper.on_lap(lap) is not None
    assert mapper.on_lap(lap) is None  # _key update replay: no double emit


def test_pit_lap_flagged(mapper):
    # fixture: driver 30 pitted on lap 9
    fake_lap = dict(load("laps")[0], driver_number=30, lap_number=9)
    row = mapper.on_lap(fake_lap)
    assert row["pit_in"] is True


def test_weather_row(mapper):
    w = load("weather")[0]
    row = mapper.on_weather(w)
    assert row["air_temp"] == pytest.approx(26.0)
    assert row["track_temp"] == pytest.approx(43.5)
    assert row["rainfall"] is False
    assert row["time_session_s"] == pytest.approx(_epoch(w["date"]))


def test_race_control_status_codes():
    m = OpenF1Mapper("s")
    t0 = "2025-09-07T13:00:00+00:00"
    m.on_race_control({"date": t0, "category": "Flag", "flag": "YELLOW", "message": "YELLOW"})
    assert m._active_yellow
    m.on_race_control(
        {"date": t0, "category": "SafetyCar", "message": "SAFETY CAR DEPLOYED", "flag": None}
    )
    assert m._active_sc == "4"
    status = m._status_during(_epoch(t0) - 1, _epoch(t0) + 1)
    assert "2" in status and "4" in status
    m.on_race_control({"date": t0, "category": "Flag", "flag": "CLEAR", "message": "TRACK CLEAR"})
    assert m._active_sc is None and not m._active_yellow
    m.on_race_control(
        {"date": t0, "category": "SafetyCar", "message": "VIRTUAL SAFETY CAR DEPLOYED"}
    )
    assert m._active_sc == "6"
    m.on_race_control({"date": t0, "category": "SafetyCar", "message": "VIRTUAL SAFETY CAR ENDING"})
    assert m._active_sc is None


def test_message_log_dedupe_and_order():
    ml = MessageLog()
    assert ml.accept({"_id": 1, "_key": "a", "x": 1})
    assert not ml.accept({"_id": 1, "_key": "a", "x": 1})  # replay
    assert ml.accept({"_id": 2, "_key": "a", "x": 2})  # genuine update
    assert not ml.accept({"_id": 1, "_key": "b", "x": 9})  # stale out-of-order
    assert ml.accept({"x": "no envelope"})  # REST rows pass through


def test_driver_meta(mapper):
    meta = mapper.enrich_driver_meta(load("drivers"))
    assert meta[1]["driver_code"] == "VER"
    assert "red_bull" in meta[1]["team_id"]
