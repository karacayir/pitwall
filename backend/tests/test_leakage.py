"""THE leakage tests: features and reference at lap L may only use information
available before lap L runs.

Two corruption experiments on a real race (Monza fixture):

1. Lap data: garble lap/sector times, track status, positions and clocks for
   every lap >= L (and tyre/stint structure for laps > L). Every feature and
   the reference for laps <= L must be bit-identical — the lap being predicted
   only contributes pre-lap knowledge (its tyre age / stint / start clock).

2. Weather: garble every weather sample taken at/after the moment the first
   driver starts lap L. That driver's features for laps <= L must be identical
   (for later-starting drivers, samples after this boundary are legitimately
   pre-lap information, so only the earliest starter gives an exact boundary).
"""

import polars as pl

from features.build import (
    FEATURE_COLUMNS,
    build_feature_frame,
    prepare_session_laps,
    reference_series,
)

L = 30  # corruption point (mid-race)
LAPS_TOTAL = 53
LAP_LENGTH_KM = 5.793
WEATHER_FEATURES = ["track_temp", "air_temp", "humidity", "rainfall_flag", "wind_speed"]
LAP_FEATURES = [c for c in FEATURE_COLUMNS if c not in WEATHER_FEATURES]
KEY = ["driver_number", "lap_number"]


def _features(laps_raw: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    prepared = prepare_session_laps(laps_raw, LAPS_TOTAL, LAP_LENGTH_KM)
    refs = reference_series(prepared, pole_time_s=78.8)
    return build_feature_frame(prepared, weather, refs, LAPS_TOTAL, abrasiveness=3.0)


def _corrupt_laps(laps: pl.DataFrame, lap: int) -> pl.DataFrame:
    """Garble everything measured during/after `lap` (+1000s offsets keep
    causal ordering so the corruption stays physically representable)."""
    tampered = pl.when(pl.col("lap_number") >= lap)
    return laps.with_columns(
        tampered.then(pl.col("lap_time_s") * 3 + 500)
        .otherwise(pl.col("lap_time_s"))
        .alias("lap_time_s"),
        tampered.then(pl.lit("45")).otherwise(pl.col("track_status")).alias("track_status"),
        tampered.then(pl.col("time_session_s") + 1000)
        .otherwise(pl.col("time_session_s"))
        .alias("time_session_s"),
        tampered.then(pl.lit(None, dtype=pl.Int32)).otherwise(pl.col("position")).alias("position"),
        tampered.then(pl.col("sector1_s") * 2).otherwise(pl.col("sector1_s")).alias("sector1_s"),
        # structure of *later* laps (lap L's own tyre/stint state is pre-lap knowledge)
        pl.when(pl.col("lap_number") > lap)
        .then(pl.col("tyre_life") + 40)
        .otherwise(pl.col("tyre_life"))
        .alias("tyre_life"),
        pl.when(pl.col("lap_number") > lap)
        .then(pl.col("lap_start_session_s") + 1000)
        .otherwise(pl.col("lap_start_session_s"))
        .alias("lap_start_session_s"),
    )


def _corrupt_weather(weather: pl.DataFrame, t_boundary: float) -> pl.DataFrame:
    after = pl.when(pl.col("time_session_s") >= t_boundary)
    return weather.with_columns(
        after.then(pl.col("air_temp") + 25).otherwise(pl.col("air_temp")).alias("air_temp"),
        after.then(pl.col("track_temp") + 25).otherwise(pl.col("track_temp")).alias("track_temp"),
        after.then(pl.lit(True)).otherwise(pl.col("rainfall")).alias("rainfall"),
    )


def test_lap_features_use_only_past_laps(monza):
    laps, weather = monza["laps"], monza["weather"]
    clean = _features(laps, weather)
    dirty = _features(_corrupt_laps(laps, L), weather)
    cols = list(dict.fromkeys([*KEY, "reference_pace_s", *LAP_FEATURES]))
    a = clean.filter(pl.col("lap_number") <= L).select(cols).sort(KEY)
    b = dirty.filter(pl.col("lap_number") <= L).select(cols).sort(KEY)
    assert a.equals(b), "lap-derived features before/at the corruption lap changed — leakage!"


def test_weather_features_use_only_pre_lap_samples(monza):
    laps, weather = monza["laps"], monza["weather"]
    starts = laps.filter(pl.col("lap_number") == L).sort("lap_start_session_s")
    first_driver = starts["driver_number"][0]
    t_boundary = float(starts["lap_start_session_s"][0])

    clean = _features(laps, weather)
    dirty = _features(laps, _corrupt_weather(weather, t_boundary))
    cols = [*KEY, *WEATHER_FEATURES]
    a = (
        clean.filter((pl.col("driver_number") == first_driver) & (pl.col("lap_number") <= L))
        .select(cols)
        .sort(KEY)
    )
    b = (
        dirty.filter((pl.col("driver_number") == first_driver) & (pl.col("lap_number") <= L))
        .select(cols)
        .sort(KEY)
    )
    assert a.equals(b), "weather features peeked at samples taken after lap start — leakage!"


def test_target_before_corruption_unchanged(monza):
    """y (the target) for laps < L must be untouched: the reference is causal."""
    laps, weather = monza["laps"], monza["weather"]
    clean = _features(laps, weather)
    dirty = _features(_corrupt_laps(laps, L), weather)
    a = clean.filter(pl.col("lap_number") < L).select([*KEY, "y"]).sort(KEY)
    b = dirty.filter(pl.col("lap_number") < L).select([*KEY, "y"]).sort(KEY)
    assert a.equals(b)
