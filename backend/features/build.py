"""Feature pipeline part 1: lap classification, fuel correction, outlier marking.

All functions are pure (frame in -> frame out) so the replay/live pipeline can
reuse them lap-by-lap. Reference pace + the feature matrix live here too (Phase 2).
"""

import polars as pl

from app import config

LAP_CLASSES = ("green", "sc", "vsc", "yellow", "red", "in", "out", "lap1")

# fastf1 TrackStatus digit codes (concatenated when several occur in one lap):
# 1 AllClear, 2 Yellow, 3 unused/flag, 4 SCDeployed, 5 Red, 6 VSCDeployed, 7 VSCEnding
_RED, _SC, _VSC_CODES, _YELLOW_CODES = "5", "4", ("6", "7"), ("2", "3")


def classify_laps(laps: pl.DataFrame) -> pl.DataFrame:
    """Add a ``lap_class`` column.

    Precedence (first match wins): lap1 > in > out > red > sc > vsc > yellow > green.
    Pit in/out outranks track status because a pit lap is useless for pace no
    matter the flag; a null track_status counts as green.
    """
    status = pl.col("track_status").fill_null("1")
    lap_class = (
        pl.when(pl.col("lap_number") == 1)
        .then(pl.lit("lap1"))
        .when(pl.col("pit_in"))
        .then(pl.lit("in"))
        .when(pl.col("pit_out"))
        .then(pl.lit("out"))
        .when(status.str.contains(_RED, literal=True))
        .then(pl.lit("red"))
        .when(status.str.contains(_SC, literal=True))
        .then(pl.lit("sc"))
        .when(
            status.str.contains(_VSC_CODES[0], literal=True)
            | status.str.contains(_VSC_CODES[1], literal=True)
        )
        .then(pl.lit("vsc"))
        .when(
            status.str.contains(_YELLOW_CODES[0], literal=True)
            | status.str.contains(_YELLOW_CODES[1], literal=True)
        )
        .then(pl.lit("yellow"))
        .otherwise(pl.lit("green"))
        .alias("lap_class")
    )
    return laps.with_columns(lap_class)


def fuel_correction_s(lap_number: pl.Expr, laps_total: pl.Expr, lap_length_km: pl.Expr) -> pl.Expr:
    """Seconds of lap time attributable to remaining fuel (approximation).

    fuel_kg ~= 110 * (1 - lap/laps_total); effect ~= 0.033 s/kg * (lap_len/5.0).
    Subtract from lap_time to compare laps at equal (empty-tank) fuel load.
    """
    fuel_kg = config.FUEL_START_KG * (1 - lap_number / laps_total)
    per_kg = config.FUEL_EFFECT_S_PER_KG_5KM * (lap_length_km / config.FUEL_EFFECT_REF_LAP_KM)
    return (fuel_kg * per_kg).alias("fuel_correction_s")


def with_fuel_corrected(laps: pl.DataFrame, laps_total: int, lap_length_km: float) -> pl.DataFrame:
    """Add ``fuel_correction_s`` and ``fuel_corrected_s`` columns."""
    return laps.with_columns(
        fuel_correction_s(pl.col("lap_number"), pl.lit(laps_total), pl.lit(lap_length_km))
    ).with_columns((pl.col("lap_time_s") - pl.col("fuel_correction_s")).alias("fuel_corrected_s"))


def mark_outliers(laps: pl.DataFrame) -> pl.DataFrame:
    """Add ``is_outlier``: green laps slower than median + 3*MAD within
    (session, driver, stint). Non-green laps are never marked (they are
    excluded from training by class already)."""
    green_time = pl.when(pl.col("lap_class") == "green").then(pl.col("lap_time_s")).otherwise(None)
    grp = ["session_id", "driver_number", "stint"]
    med = green_time.median().over(grp)
    mad = (green_time - med).abs().median().over(grp)
    threshold = med + config.OUTLIER_MAD_MULTIPLIER * mad
    is_outlier = (
        (pl.col("lap_class") == "green")
        & pl.col("lap_time_s").is_not_null()
        & (pl.col("lap_time_s") > threshold)
    ).fill_null(False)
    return laps.with_columns(is_outlier.alias("is_outlier"))
