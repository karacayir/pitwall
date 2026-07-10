"""Golden tests for reference pace: synthetic lap sets with hand-computed
expected values, incl. cold start, window widening, hold, and rain paths."""

import polars as pl
import pytest

from app import config
from features.build import extrapolate_reference, reference_series


def synth_session(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal prepared-laps frame for reference computation."""
    base = {
        "session_id": "t",
        "lap_class": "green",
        "clean_air": True,
        "is_outlier": False,
    }
    return pl.DataFrame([{**base, **r} for r in rows]).with_columns(
        pl.col("lap_number").cast(pl.Int32),
        pl.col("driver_number").cast(pl.Int32),
        pl.col("fuel_corrected_s").cast(pl.Float64),
    )


def grid(drivers: int, laps: range, time_fn) -> list[dict]:
    return [
        {"driver_number": d, "lap_number": lap, "fuel_corrected_s": time_fn(d, lap)}
        for d in range(1, drivers + 1)
        for lap in laps
    ]


def test_reference_median_of_top5_bests():
    # 6 drivers, laps 2..11, driver d always laps at 90 + d/10.
    # At L=12 the window [7,11] has 30 laps (>=8, no widening).
    # Bests: 90.1..90.6 -> five fastest 90.1..90.5 -> median 90.3.
    laps = synth_session(grid(6, range(2, 12), lambda d, lap: 90.0 + d / 10))
    refs = reference_series(laps, pole_time_s=88.0, max_lap=12)
    assert refs.filter(pl.col("lap_number") == 12)["reference_s"][0] == pytest.approx(90.3)


def test_cold_start_uses_pole():
    laps = synth_session(grid(6, range(2, 12), lambda d, lap: 90.0 + d / 10))
    refs = reference_series(laps, pole_time_s=88.0, max_lap=12)
    for lap in (1, 2, 3):
        got = refs.filter(pl.col("lap_number") == lap)["reference_s"][0]
        assert got == pytest.approx(88.0 * config.COLD_START_POLE_FACTOR)


def test_cold_start_backfill_without_pole():
    laps = synth_session(grid(6, range(2, 12), lambda d, lap: 90.0 + d / 10))
    refs = reference_series(laps, pole_time_s=None, max_lap=12)
    lap4 = refs.filter(pl.col("lap_number") == 4)["reference_s"][0]
    assert refs.filter(pl.col("lap_number") == 1)["reference_s"][0] == pytest.approx(lap4)


def test_window_widens_when_sparse():
    # Only laps 4..6 have clean green laps (3 drivers); at L=12 the narrow
    # window [7,11] is empty and [4,11] has 9 laps -> widened window used.
    laps = synth_session(grid(3, range(4, 7), lambda d, lap: 91.0 + d / 10))
    refs = reference_series(laps, pole_time_s=None, max_lap=12)
    # bests 91.1, 91.2, 91.3 -> median 91.2
    assert refs.filter(pl.col("lap_number") == 12)["reference_s"][0] == pytest.approx(91.2)


def test_reference_holds_when_no_laps_qualify():
    # Clean laps only up to lap 6; far past the widened window the last
    # computed reference must hold rather than vanish.
    laps = synth_session(grid(6, range(2, 7), lambda d, lap: 90.0 + d / 10))
    refs = reference_series(laps, pole_time_s=88.0, max_lap=30)
    ref14 = refs.filter(pl.col("lap_number") == 14)["reference_s"][0]  # widened window hit
    ref30 = refs.filter(pl.col("lap_number") == 30)["reference_s"][0]
    assert ref30 == pytest.approx(ref14)
    assert ref30 is not None


def test_extrapolation_trend_and_clip():
    # steady -0.02 s/lap trend on a 90s reference: |trend| < 0.045 clip -> kept
    refs = [90.0 - 0.02 * i for i in range(10)]
    got = extrapolate_reference(refs, k=5)
    assert got == pytest.approx(refs[-1] - 0.02 * 5, abs=1e-6)
    # a crash-dieting reference (-1 s/lap) must clip to 0.05% of level per lap
    steep = [90.0 - 1.0 * i for i in range(10)]
    clip = config.REF_TREND_CLIP_PER_LAP * steep[-1]
    assert extrapolate_reference(steep, k=4) == pytest.approx(steep[-1] - clip * 4)


def test_extrapolation_rain_holds_flat():
    refs = [90.0 - 0.05 * i for i in range(10)]
    assert extrapolate_reference(refs, k=8, rain=True) == pytest.approx(refs[-1])
