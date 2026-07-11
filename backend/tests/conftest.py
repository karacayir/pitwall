import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from ingest.fastf1_pull import convert_laps, convert_results, convert_weather, make_session_id

FIXTURES = Path(__file__).parent / "fixtures"

MONZA_META = {"track_id": "monza", "laps_total": 53, "lap_length_km": 5.793,
              "pole_time_s": 78.79, "season": 2025, "round": 16}  # fmt: skip
SAO_PAULO_META = {"track_id": "sao_paulo", "laps_total": 69, "lap_length_km": 4.309,
                  "pole_time_s": 84.0, "season": 2024, "round": 21}  # fmt: skip


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


@pytest.fixture(scope="session")
def tiny_model(monza, sao_paulo, tmp_path_factory):
    """A small but real quantile model trained on the two fixture races,
    round-tripped through the artifact format. Shared by model + engine tests."""
    from app import config
    from features.build import FEATURE_COLUMNS, training_mask
    from models.predict import PaceModel
    from models.train import build_training_frame, train_quantile_models

    sessions = pl.DataFrame(
        [
            {"session_id": monza["session_id"], **MONZA_META},
            {"session_id": sao_paulo["session_id"], **SAO_PAULO_META},
        ]
    ).drop("lap_length_km")
    tracks = {
        "monza": {"lap_length_km": 5.793, "abrasiveness": 3.0},
        "sao_paulo": {"lap_length_km": 4.309, "abrasiveness": 3.0},
    }
    laps = pl.concat([monza["laps"], sao_paulo["laps"]])
    weather = pl.concat([monza["weather"], sao_paulo["weather"]])
    df = build_training_frame(laps, weather, sessions, tracks)
    rows = df.filter(training_mask(df))
    assert len(rows) > 800, "fixture training frame unexpectedly small"

    valid = rows.filter(pl.col("session_id") == monza["session_id"])
    boosters, info = train_quantile_models(rows, valid, current_season=2025, n_estimators=80)

    out = tmp_path_factory.mktemp("artifact") / "20990101"
    out.mkdir()
    for alpha, booster in boosters.items():
        booster.save_model(str(out / f"q{int(alpha * 100)}.txt"))
    (out / "meta.json").write_text(
        json.dumps(
            {
                "features": FEATURE_COLUMNS,
                "categorical_features": ["compound", "track_id", "driver_id", "team_id"],
                "quantiles": list(config.QUANTILES),
                "vocab": info["vocab"],
                "calibration": info["calibration"],
                "metrics": info["metrics"],
            }
        )
    )
    return PaceModel.latest(out.parent), rows
