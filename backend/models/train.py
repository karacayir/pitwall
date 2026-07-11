"""Train the three LightGBM quantile models on the ratio target and export a
versioned artifact directory models/YYYYMMDD/ containing:

    q10.txt / q50.txt / q90.txt   LightGBM boosters
    meta.json                     feature list, categorical vocabularies,
                                  training config, validation metrics
    degradation_priors.json       quadratic priors per (track_id, compound)

Run: uv run python -m models.train   (from backend/, after `make data`)
"""

import argparse
import datetime as dt
import json
import logging
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from app import config
from app.tracks import load_tracks
from features.build import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_frame,
    prepare_session_laps,
    reference_series,
    training_mask,
)
from models.degradation import fit_prior_curves

log = logging.getLogger("pitwall.train")

# NOTE: the brief asked for LightGBM monotone constraints on tyre_age, but the
# quantile objective rejects monotone_constraints outright (LightGBMError).
# Monotonicity in tyre age is therefore enforced at inference time via an
# isotonic pass along age sweeps — see PaceModel.predict_quantiles(age_groups=).


def build_training_frame(
    laps: pl.DataFrame,
    weather: pl.DataFrame,
    sessions: pl.DataFrame,
    tracks: dict,
) -> pl.DataFrame:
    """Feature frames for every session, concatenated (rows NOT yet filtered)."""
    frames = []
    for sess in sessions.iter_rows(named=True):
        sid = sess["session_id"]
        track = tracks.get(sess["track_id"])
        if track is None:
            log.warning("no track metadata for %s, skipping", sid)
            continue
        sl = laps.filter(pl.col("session_id") == sid)
        sw = weather.filter(pl.col("session_id") == sid)
        if sl.is_empty():
            continue
        prepared = prepare_session_laps(sl, sess["laps_total"], track["lap_length_km"])
        refs = reference_series(prepared, sess["pole_time_s"])
        frames.append(
            build_feature_frame(prepared, sw, refs, sess["laps_total"], track["abrasiveness"])
        )
    return pl.concat(frames, how="vertical_relaxed")


def sample_weights(seasons: pl.Series, current_season: int) -> np.ndarray:
    """Recency decay exp(-age/2), x2 on the current season (2026 reg reset)."""
    age = (current_season - seasons).cast(pl.Float64).to_numpy()
    w = np.exp(-age / config.RECENCY_HALFLIFE_SEASONS)
    w[age == 0] *= config.CURRENT_SEASON_WEIGHT_MULT
    return w


def to_lgbm_frame(df: pl.DataFrame, vocab: dict[str, list] | None = None):
    """Polars -> pandas with proper categorical dtypes; returns (X, vocab)."""
    x = df.select(FEATURE_COLUMNS).to_pandas()
    if vocab is None:
        vocab = {c: sorted(x[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES}
    for c in CATEGORICAL_FEATURES:
        x[c] = pd.Categorical(x[c], categories=vocab[c])
    for c in FEATURE_COLUMNS:
        if c not in CATEGORICAL_FEATURES:
            x[c] = x[c].astype(float)
    return x, vocab


def train_quantile_models(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    current_season: int,
    n_estimators: int | None = None,
) -> tuple[dict[float, lgb.Booster], dict]:
    x_train, vocab = to_lgbm_frame(train_df)
    y_train = train_df[TARGET_COLUMN].to_numpy()
    w_train = sample_weights(train_df["season"], current_season)
    x_valid, _ = to_lgbm_frame(valid_df, vocab)
    y_valid = valid_df[TARGET_COLUMN].to_numpy()

    params = dict(config.LGBM_PARAMS)
    num_boost_round = int(n_estimators or params.pop("n_estimators"))
    params.pop("n_estimators", None)
    params.update(objective="quantile", verbosity=-1)

    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        weight=w_train,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    )

    boosters: dict[float, lgb.Booster] = {}
    metrics: dict[str, float] = {}
    for alpha in config.QUANTILES:
        valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set)
        booster = lgb.train(
            {**params, "alpha": alpha},
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        boosters[alpha] = booster
        pred = booster.predict(x_valid, num_iteration=booster.best_iteration)
        pinball = np.mean(np.maximum(alpha * (y_valid - pred), (alpha - 1) * (y_valid - pred)))
        metrics[f"pinball_q{int(alpha * 100)}"] = float(pinball)
        log.info("alpha=%.1f best_iter=%s pinball=%.5f", alpha, booster.best_iteration, pinball)

    p50 = boosters[0.5].predict(x_valid)
    metrics["valid_mae_ratio"] = float(np.mean(np.abs(y_valid - p50)))
    metrics["valid_mae_s"] = float(
        np.mean(np.abs((y_valid - p50) * valid_df["reference_pace_s"].to_numpy()))
    )
    lo = boosters[0.1].predict(x_valid)
    hi = boosters[0.9].predict(x_valid)
    metrics["valid_coverage_raw"] = float(np.mean((y_valid >= lo) & (y_valid <= hi)))

    # split-conformal calibration: LightGBM's quantile heads come out too
    # narrow, so scale each half-width until the validation fold hits its
    # nominal one-sided 90% coverage. Factors never narrow (>=1), capped at 5.
    eps = 1e-4
    tri = np.sort(np.column_stack([lo, p50, hi]), axis=1)
    lo_s, p50_s, hi_s = tri[:, 0], tri[:, 1], tri[:, 2]
    r_hi = (y_valid - p50_s) / np.maximum(hi_s - p50_s, eps)
    r_lo = (p50_s - y_valid) / np.maximum(p50_s - lo_s, eps)
    s_hi = float(np.clip(np.quantile(r_hi, 0.9), 1.0, 5.0))
    s_lo = float(np.clip(np.quantile(r_lo, 0.9), 1.0, 5.0))
    cal_lo = p50_s - s_lo * (p50_s - lo_s)
    cal_hi = p50_s + s_hi * (hi_s - p50_s)
    metrics["valid_coverage_p10_p90"] = float(np.mean((y_valid >= cal_lo) & (y_valid <= cal_hi)))
    metrics["calibration_s_lo"] = s_lo
    metrics["calibration_s_hi"] = s_hi
    return boosters, {
        "vocab": vocab,
        "metrics": metrics,
        "calibration": {"s_lo": s_lo, "s_hi": s_hi},
    }


def degradation_priors(train_df: pl.DataFrame) -> dict:
    """Quadratic priors per (track_id, compound) + per-compound global fallback,
    fit on fuel-corrected ratios (fuel_corrected_s / reference)."""
    df = train_df.with_columns(
        (pl.col("fuel_corrected_s") / pl.col("reference_pace_s")).alias("_fc_ratio")
    ).filter(pl.col("_fc_ratio").is_between(0.85, 1.35) & pl.col("tyre_age").is_not_null())

    priors: dict[str, dict] = {}

    def _key_fit(key: str, sub: pl.DataFrame, min_laps: int):
        rows = list(zip(sub["tyre_age"].to_list(), sub["_fc_ratio"].to_list(), strict=True))
        curve = fit_prior_curves(
            rows, min_laps=min_laps, ridge_alpha=config.DEGRADATION_RIDGE_ALPHA
        )
        if curve:
            priors[key] = {"c0": curve.c0, "c1": curve.c1, "c2": curve.c2, "n": curve.n_laps}

    for (compound,), sub in df.group_by(["compound"]):
        _key_fit(f"__global__/{compound}", sub, min_laps=200)
    for (track_id, compound), sub in df.group_by(["track_id", "compound"]):
        _key_fit(f"{track_id}/{compound}", sub, min_laps=60)
    return priors


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="artifact dir (default models/YYYYMMDD)")
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument(
        "--valid-sessions",
        type=int,
        default=6,
        help="most recent N races form the early-stopping fold",
    )
    args = parser.parse_args()

    laps = pl.read_parquet(config.LAPS_PARQUET)
    weather = pl.read_parquet(config.WEATHER_PARQUET)
    sessions = pl.read_parquet(config.SESSIONS_PARQUET).sort(["season", "round"])
    tracks = load_tracks()

    df = build_training_frame(laps, weather, sessions, tracks)
    rows = df.filter(training_mask(df))
    log.info("training rows: %d of %d laps", len(rows), len(df))

    valid_ids = sessions["session_id"].to_list()[-args.valid_sessions :]
    train_df = rows.filter(~pl.col("session_id").is_in(valid_ids))
    valid_df = rows.filter(pl.col("session_id").is_in(valid_ids))
    current_season = int(sessions["season"].max())

    boosters, info = train_quantile_models(
        train_df, valid_df, current_season, n_estimators=args.n_estimators
    )
    priors = degradation_priors(rows)

    out_dir = config.MODELS_DIR / (args.out or dt.date.today().strftime("%Y%m%d"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for alpha, booster in boosters.items():
        booster.save_model(str(out_dir / f"q{int(alpha * 100)}.txt"))
    (out_dir / "degradation_priors.json").write_text(json.dumps(priors, indent=2))
    meta = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "features": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "quantiles": list(config.QUANTILES),
        "vocab": info["vocab"],
        "calibration": info["calibration"],
        "metrics": info["metrics"],
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "valid_sessions": valid_ids,
        "current_season": current_season,
        "lgbm_params": config.LGBM_PARAMS,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("artifacts written to %s | metrics: %s", out_dir, info["metrics"])
    print(json.dumps({"out": str(out_dir), **info["metrics"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
