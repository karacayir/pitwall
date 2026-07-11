"""Feature pipeline: lap classification, fuel correction, outlier marking,
reference pace (the normalisation at the heart of the system), and the
training feature matrix.

All functions are pure (frame in -> frame out) so the replay/live pipeline can
reuse them lap-by-lap. Leakage rule: anything computed "at lap L" may only use
laps with lap_number < L — tested explicitly in tests/test_leakage.py.
"""

from collections.abc import Sequence

import numpy as np
import polars as pl

from app import config

LAP_CLASSES = ("green", "sc", "vsc", "yellow", "red", "in", "out", "lap1")

# fastf1 TrackStatus digit codes (concatenated when several occur in one lap):
# 1 AllClear, 2 Yellow, 3 unused/flag, 4 SCDeployed, 5 Red, 6 VSCDeployed, 7 VSCEnding
_RED, _SC, _VSC_CODES, _YELLOW_CODES = "5", "4", ("6", "7"), ("2", "3")


def classify_lap_scalar(
    lap_number: int, pit_in: bool, pit_out: bool, track_status: str | None
) -> str:
    """Scalar twin of classify_laps for the live/replay event path.
    tests/test_classifier.py asserts the two stay in lockstep."""
    if lap_number == 1:
        return "lap1"
    if pit_in:
        return "in"
    if pit_out:
        return "out"
    status = track_status or "1"
    if _RED in status:
        return "red"
    if _SC in status:
        return "sc"
    if any(c in status for c in _VSC_CODES):
        return "vsc"
    if any(c in status for c in _YELLOW_CODES):
        return "yellow"
    return "green"


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


def compute_gaps(laps: pl.DataFrame) -> pl.DataFrame:
    """Add ``gap_ahead_s`` (seconds to the previous line crossing by ANY car,
    i.e. the car physically ahead on track) and ``clean_air`` (> 2.0s or leader).

    Uses time_session_s only, so it works identically on historical and live data.
    """
    crossings = (
        laps.filter(pl.col("time_session_s").is_not_null())
        .sort(["session_id", "time_session_s"])
        .with_columns(pl.col("time_session_s").diff().over("session_id").alias("gap_ahead_s"))
        .select(["session_id", "driver_number", "lap_number", "gap_ahead_s"])
    )
    out = laps.join(crossings, on=["session_id", "driver_number", "lap_number"], how="left")
    return out.with_columns(
        (pl.col("gap_ahead_s").is_null() | (pl.col("gap_ahead_s") > config.CLEAN_AIR_GAP_S)).alias(
            "clean_air"
        )
    )


def _windowed_reference(
    prior_laps: pl.DataFrame, lap: int, window: int
) -> tuple[float | None, int]:
    """Median of the 5 fastest drivers' best clean green fuel-corrected laps in
    [lap-window, lap-1]. Returns (reference, n_qualifying_laps).

    Deliberately does NOT use the outlier flag: outliers are computed over the
    whole stint (future information), and only slow laps are ever flagged, so
    they cannot influence per-driver bests anyway (leakage rule)."""
    win = prior_laps.filter(
        (pl.col("lap_number") >= lap - window)
        & (pl.col("lap_number") <= lap - 1)
        & (pl.col("lap_class") == "green")
        & pl.col("clean_air")
        & pl.col("fuel_corrected_s").is_not_null()
    )
    n = len(win)
    if n == 0:
        return None, 0
    bests = (
        win.group_by("driver_number")
        .agg(pl.col("fuel_corrected_s").min().alias("best"))
        .sort("best")
        .head(config.REF_TOP_DRIVERS)
    )
    return float(bests["best"].median()), n


def reference_series(
    session_laps: pl.DataFrame, pole_time_s: float | None, max_lap: int | None = None
) -> pl.DataFrame:
    """Reference pace for every lap of one session.

    Per the spec: window [L-5, L-1] (widened to 8 back if < 8 laps qualify);
    cold start (L < 4) uses pole * 1.03; if a window yields nothing, the last
    known reference holds. Only laps < L are ever used (leakage rule).

    Returns a frame (lap_number i32, reference_s f64?).
    """
    if max_lap is None:
        max_lap = int(session_laps["lap_number"].max() or 0)
    cold = pole_time_s * config.COLD_START_POLE_FACTOR if pole_time_s else None

    refs: list[float | None] = []
    last: float | None = cold
    for lap in range(1, max_lap + 1):
        if lap < config.COLD_START_MAX_LAP:
            refs.append(cold)
            continue
        ref, n = _windowed_reference(session_laps, lap, config.REF_WINDOW_LAPS)
        if n < config.REF_MIN_QUALIFYING_LAPS:
            wide_ref, wide_n = _windowed_reference(session_laps, lap, config.REF_WINDOW_WIDE_LAPS)
            if wide_n > n:
                ref = wide_ref
        if ref is None:
            ref = last  # e.g. long SC train: hold the last known level
        refs.append(ref)
        if ref is not None:
            last = ref

    # cold-start backfill when no pole time is known: use the first computed value
    if cold is None:
        first = next((r for r in refs if r is not None), None)
        for i in range(min(config.COLD_START_MAX_LAP - 1, len(refs))):
            refs[i] = first

    return pl.DataFrame(
        {"lap_number": list(range(1, max_lap + 1)), "reference_s": refs},
        schema={"lap_number": pl.Int32, "reference_s": pl.Float64},
    )


def extrapolate_reference(refs: Sequence[float], k: int, rain: bool = False) -> float:
    """Reference k laps ahead: linear trend over the last 10 values, clipped to
    +/-0.05%/lap; under rain hold flat (uncertainty is widened elsewhere)."""
    if not refs:
        raise ValueError("need at least one reference value")
    last = float(refs[-1])
    if rain or len(refs) < 2:
        return last
    tail = np.asarray(refs[-config.REF_TREND_WINDOW :], dtype=float)
    slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0])
    clip = config.REF_TREND_CLIP_PER_LAP * last
    return last + float(np.clip(slope, -clip, clip)) * k


def sc_restart_flags(laps: pl.DataFrame) -> pl.DataFrame:
    """Track-wide ``sc_restart`` per (session, lap): SC/VSC ran in the previous
    3 laps (restart laps behave differently: cold tyres, bunched field)."""
    lap_status = laps.group_by(["session_id", "lap_number"]).agg(
        pl.col("lap_class").is_in(["sc", "vsc"]).any().alias("_sc_now")
    )
    out = (
        lap_status.sort(["session_id", "lap_number"])
        .with_columns(
            pl.col("_sc_now")
            .shift(1)
            .rolling_max(config.SC_RESTART_LOOKBACK_LAPS, min_samples=1)
            .over("session_id")
            .fill_null(False)
            .alias("sc_restart")
        )
        .select(["session_id", "lap_number", "sc_restart"])
    )
    return out


FEATURE_COLUMNS = [
    "compound",
    "tyre_age",
    "tyre_age_sq",
    "stint_no",
    "race_progress",
    "laps_total",
    "track_id",
    "abrasiveness",
    "track_temp",
    "air_temp",
    "humidity",
    "rainfall_flag",
    "wind_speed",
    "driver_id",
    "team_id",
    "season",
    "gap_ahead_s",
    "clean_air",
    "sc_restart",
    "position",
    "reference_pace_s",
    # autoregressive form (added after the first backtest: persistence beat the
    # curve-only model — the GBM needs the driver's live pace, laps < L only)
    "last_green_ratio",
    "rolling_ratio_3",
]
CATEGORICAL_FEATURES = ["compound", "track_id", "driver_id", "team_id"]
TARGET_COLUMN = "y"


def prepare_session_laps(laps: pl.DataFrame, laps_total: int, lap_length_km: float) -> pl.DataFrame:
    """Classification + fuel correction + gaps + outliers for one session's laps.
    This is the shared entry point for batch training and the live/replay path."""
    out = classify_laps(laps)
    out = with_fuel_corrected(out, laps_total, lap_length_km)
    out = compute_gaps(out)
    out = mark_outliers(out)
    return out


def join_weather(laps: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    """Weather at lap START (latest sample at or before lap_start_session_s):
    strictly information available before the lap runs (leakage rule)."""
    w = weather.sort("time_session_s").select(
        "session_id",
        pl.col("time_session_s").alias("_w_time"),
        pl.col("air_temp"),
        pl.col("track_temp"),
        pl.col("humidity"),
        pl.col("rainfall").alias("rainfall_flag"),
        pl.col("wind_speed"),
    )
    return (
        laps.sort("lap_start_session_s")
        .join_asof(
            w,
            left_on="lap_start_session_s",
            right_on="_w_time",
            by="session_id",
            strategy="backward",
        )
        .drop("_w_time")
    )


def build_feature_frame(
    prepared: pl.DataFrame,
    weather: pl.DataFrame,
    reference: pl.DataFrame,
    laps_total: int,
    abrasiveness: float,
) -> pl.DataFrame:
    """Assemble the model feature frame for one session from prepared laps.

    Every feature of lap L uses only pre-lap information: gap_ahead_s and
    position come from the END of lap L-1, weather from the last sample before
    lap start, the reference from laps < L. Emits one row per lap with
    FEATURE_COLUMNS (+ y where lap_time and reference exist). Rows are NOT yet
    filtered to training rows — the trainer filters on lap_class/outliers so
    the live path can reuse this for any lap.
    """
    df = prepared.join(reference, on="lap_number", how="left")
    df = join_weather(df, weather)
    df = df.join(sc_restart_flags(prepared), on=["session_id", "lap_number"], how="left")
    df = df.sort(["driver_number", "lap_number"]).with_columns(
        pl.col("gap_ahead_s").shift(1).over(["session_id", "driver_number"]).alias("_gap_prev"),
        pl.col("position").shift(1).over(["session_id", "driver_number"]).alias("_pos_prev"),
    )
    df = df.with_columns(
        (pl.col("tyre_life") - 1).cast(pl.Float64).alias("tyre_age"),
        pl.col("stint").cast(pl.Float64).alias("stint_no"),
        (pl.col("lap_number") / laps_total).alias("race_progress"),
        pl.lit(float(laps_total)).alias("laps_total"),
        pl.lit(float(abrasiveness)).alias("abrasiveness"),
        pl.col("driver_code").alias("driver_id"),
        pl.col("_gap_prev").clip(0.0, config.GAP_AHEAD_CAP_S).alias("gap_ahead_s"),
        (pl.col("_gap_prev").is_null() | (pl.col("_gap_prev") > config.CLEAN_AIR_GAP_S)).alias(
            "clean_air"
        ),
        pl.col("clean_air").alias("clean_air_actual"),  # lap-local; for eval metrics only
        pl.col("_pos_prev").cast(pl.Float64).alias("position"),
        pl.col("reference_s").alias("reference_pace_s"),
        pl.col("sc_restart").fill_null(False),
    )
    df = df.with_columns(
        (pl.col("tyre_age") ** 2).alias("tyre_age_sq"),
        (pl.col("lap_time_s") / pl.col("reference_s")).alias(TARGET_COLUMN),
    )
    df = _with_recent_pace(df)
    return df.drop(["_gap_prev", "_pos_prev"])


def _with_recent_pace(df: pl.DataFrame) -> pl.DataFrame:
    """Autoregressive features: the driver's most recent green-lap ratio and
    the median of their last 3, taken strictly from laps < L (asof on L-1)."""
    greens = (
        df.filter((pl.col("lap_class") == "green") & pl.col(TARGET_COLUMN).is_not_null())
        .sort(["driver_number", "lap_number"])
        .select(
            "session_id",
            "driver_number",
            pl.col("lap_number").alias("_g_lap"),
            pl.col(TARGET_COLUMN).alias("last_green_ratio"),
            pl.col(TARGET_COLUMN)
            .rolling_median(3, min_samples=1)
            .over(["session_id", "driver_number"])
            .alias("rolling_ratio_3"),
        )
    )
    out = (
        df.with_columns((pl.col("lap_number") - 1).alias("_prev_lap"))
        .sort("_prev_lap")
        .join_asof(
            greens.sort("_g_lap"),
            left_on="_prev_lap",
            right_on="_g_lap",
            by=["session_id", "driver_number"],
            strategy="backward",
        )
        .drop(["_prev_lap", "_g_lap"])
    )
    return out


def training_mask(df: pl.DataFrame) -> pl.Series:
    """Rows eligible to train the pace model: green, timed, accurate, not
    deleted, not outlier, with a reference."""
    return df.select(
        (
            (pl.col("lap_class") == "green")
            & ~pl.col("is_outlier")
            & ~pl.col("deleted")
            & pl.col("is_accurate")
            & pl.col("lap_time_s").is_not_null()
            & pl.col("reference_pace_s").is_not_null()
            & pl.col("tyre_age").is_not_null()
            & (pl.col("y") > 0.85)
            & (pl.col("y") < 1.35)
        ).alias("m")
    )["m"]
