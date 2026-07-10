import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from ingest.fastf1_pull import convert_laps, convert_results, convert_weather, make_session_id

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(slug: str) -> dict:
    d = FIXTURES / slug
    meta = json.loads((d / "meta.json").read_text())
    return {
        "meta": meta,
        "raw_laps": pd.read_parquet(d / "raw_laps.parquet"),
        "raw_weather": pd.read_parquet(d / "raw_weather.parquet"),
        "raw_results": pd.read_parquet(d / "raw_results.parquet"),
    }


def _converted(fx: dict) -> dict:
    meta = fx["meta"]
    sid = make_session_id(meta["season"], meta["round"], meta["location"])
    track_id = sid.split("_", 2)[2]
    return {
        **fx,
        "session_id": sid,
        "laps": convert_laps(fx["raw_laps"], sid, meta["season"], meta["round"], track_id),
        "weather": convert_weather(fx["raw_weather"], sid),
        "results": convert_results(fx["raw_results"], sid),
    }


@pytest.fixture(scope="session")
def monza() -> dict:
    """2025 Monza: clean dry race."""
    return _converted(_load_fixture("monza_2025"))


@pytest.fixture(scope="session")
def sao_paulo() -> dict:
    """2024 São Paulo: wet, SC + VSC + red flag."""
    return _converted(_load_fixture("sao_paulo_2024"))


@pytest.fixture(scope="session")
def both_races(monza: dict, sao_paulo: dict) -> list[dict]:
    return [monza, sao_paulo]


def combined_laps(races: list[dict]) -> pl.DataFrame:
    return pl.concat([r["laps"] for r in races])
