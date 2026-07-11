"""The autoregressive features must be lagged: at lap L they equal the
driver's green-lap ratio strictly before L — never the lap's own."""

import polars as pl

from features.build import build_feature_frame, prepare_session_laps, reference_series
from tests.conftest import MONZA_META


def test_last_green_ratio_is_previous_green_lap(monza):
    prepared = prepare_session_laps(
        monza["laps"], MONZA_META["laps_total"], MONZA_META["lap_length_km"]
    )
    refs = reference_series(prepared, MONZA_META["pole_time_s"])
    df = build_feature_frame(prepared, monza["weather"], refs, MONZA_META["laps_total"], 3.0)

    checked = 0
    for driver in df["driver_number"].unique().to_list()[:6]:
        sub = df.filter(pl.col("driver_number") == driver).sort("lap_number")
        rows = sub.to_dicts()
        greens: list[tuple[int, float]] = []
        for r in rows:
            expected = greens[-1][1] if greens else None
            got = r["last_green_ratio"]
            if expected is None:
                assert got is None, f"driver {driver} lap {r['lap_number']}: expected null"
            else:
                assert got is not None and abs(got - expected) < 1e-9, (
                    f"driver {driver} lap {r['lap_number']}"
                )
                checked += 1
            if r["lap_class"] == "green" and r["y"] is not None:
                greens.append((r["lap_number"], r["y"]))
    assert checked > 200


def test_rolling_ratio_never_uses_own_lap(monza):
    prepared = prepare_session_laps(
        monza["laps"], MONZA_META["laps_total"], MONZA_META["lap_length_km"]
    )
    refs = reference_series(prepared, MONZA_META["pole_time_s"])
    df = build_feature_frame(prepared, monza["weather"], refs, MONZA_META["laps_total"], 3.0)
    greens = df.filter((pl.col("lap_class") == "green") & pl.col("y").is_not_null())
    # a green lap's own y can never equal its rolling feature unless it also
    # equals the median of the *previous* laps — check they are not identical
    # across the board (which would betray an off-by-one)
    same = (greens["rolling_ratio_3"] - greens["y"]).abs() < 1e-12
    assert same.mean() < 0.5, "rolling_ratio_3 looks like it includes the current lap"
