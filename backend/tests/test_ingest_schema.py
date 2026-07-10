"""Schema + internal-consistency tests for the FastF1 -> parquet converters,
running against raw fixture races (2025 Monza, 2024 São Paulo)."""

import polars as pl
import pytest

from app import config
from ingest.fastf1_pull import (
    LAPS_SCHEMA,
    RESULTS_SCHEMA,
    WEATHER_SCHEMA,
    make_session_id,
    slugify,
)


def test_slugify():
    assert slugify("São Paulo") == "sao_paulo"
    assert slugify("Monza") == "monza"
    assert slugify("Mexico City") == "mexico_city"
    assert slugify("Red Bull Racing") == "red_bull_racing"


def test_session_id():
    assert make_session_id(2025, 16, "Monza") == "2025_16_monza"
    assert make_session_id(2024, 3, "Melbourne") == "2024_03_melbourne"


@pytest.mark.parametrize("race", ["monza", "sao_paulo"])
def test_laps_schema(race, request):
    laps: pl.DataFrame = request.getfixturevalue(race)["laps"]
    assert dict(laps.schema) == LAPS_SCHEMA
    key_cols = [
        "session_id",
        "season",
        "round",
        "track_id",
        "driver_number",
        "driver_code",
        "team_id",
        "lap_number",
        "pit_in",
        "pit_out",
    ]
    for col in key_cols:
        assert laps[col].null_count() == 0, f"{col} has nulls"


@pytest.mark.parametrize("race", ["monza", "sao_paulo"])
def test_one_row_per_driver_lap(race, request):
    laps: pl.DataFrame = request.getfixturevalue(race)["laps"]
    assert laps.select(["driver_number", "lap_number"]).is_duplicated().sum() == 0


def test_lap_times_sane(monza):
    timed = monza["laps"].filter(pl.col("lap_time_s").is_not_null())
    assert len(timed) > 800
    assert timed["lap_time_s"].min() > 60  # Monza flying lap ~80s
    assert timed["lap_time_s"].max() < 600


def test_compounds_recognised(both_races):
    for race in both_races:
        seen = set(race["laps"]["compound"].drop_nulls().unique())
        assert seen <= set(config.ALL_COMPOUNDS) | {"UNKNOWN", "TEST_UNKNOWN"}, seen


def test_sao_paulo_is_wet(sao_paulo):
    seen = set(sao_paulo["laps"]["compound"].drop_nulls().unique())
    assert seen <= {"INTERMEDIATE", "WET"}


def test_tyre_life_increments_within_stint(both_races):
    """Internal consistency: tyre_life increases by 1 lap-over-lap within a stint."""
    for race in both_races:
        laps = race["laps"].filter(
            pl.col("stint").is_not_null() & pl.col("tyre_life").is_not_null()
        )
        deltas = (
            laps.sort(["driver_number", "lap_number"])
            .group_by(["driver_number", "stint"], maintain_order=True)
            .agg(pl.col("tyre_life").diff().drop_nulls().alias("d"))
            .explode("d")
            .drop_nulls()
        )
        ok_share = (deltas["d"] == 1).mean()
        assert ok_share is not None and ok_share > 0.98, f"{race['session_id']}: {ok_share}"


def test_pit_laps_flagged(both_races):
    """Every stint change (except the first stint) must come right after a pit-in
    or start with a pit-out — allowing red-flag tyre changes in São Paulo."""
    for race in both_races:
        laps = race["laps"]
        n_stints = laps.group_by("driver_number").agg(pl.col("stint").n_unique().alias("n"))
        n_pit_in = laps.group_by("driver_number").agg(pl.col("pit_in").sum().alias("p"))
        joined = n_stints.join(n_pit_in, on="driver_number")
        # stints - 1 <= pit stops + slack (red flag swaps don't go through the pit lane)
        slack = 3 if race["session_id"].startswith("2024") else 0
        bad = joined.filter(pl.col("n") - 1 > pl.col("p") + slack)
        assert len(bad) == 0, f"{race['session_id']}: {bad}"


@pytest.mark.parametrize("race", ["monza", "sao_paulo"])
def test_weather_schema(race, request):
    weather: pl.DataFrame = request.getfixturevalue(race)["weather"]
    assert dict(weather.schema) == WEATHER_SCHEMA
    assert weather["time_session_s"].null_count() == 0
    assert weather["air_temp"].is_between(-10, 60).all()
    assert weather["track_temp"].is_between(-10, 80).all()


def test_sao_paulo_rainfall_flagged(sao_paulo):
    assert sao_paulo["weather"]["rainfall"].any()


@pytest.mark.parametrize("race", ["monza", "sao_paulo"])
def test_results_schema(race, request):
    results: pl.DataFrame = request.getfixturevalue(race)["results"]
    assert dict(results.schema) == RESULTS_SCHEMA
    assert results["driver_number"].null_count() == 0
    assert 18 <= len(results) <= 22
