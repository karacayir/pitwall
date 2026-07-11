"""Inference: load versioned artifacts, predict ratio quantiles, build
counterfactual compound scenarios, and track per-driver online bias."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from app import config
from features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS


class PaceModel:
    """The three quantile boosters + categorical vocabulary from one artifact dir."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        meta = json.loads((self.model_dir / "meta.json").read_text())
        if meta["features"] != FEATURE_COLUMNS:
            raise ValueError(
                f"artifact {model_dir} was trained on different features: {meta['features']}"
            )
        self.meta = meta
        self.vocab: dict[str, list] = meta["vocab"]
        cal = meta.get("calibration") or {}
        self.cal_lo: float = float(cal.get("s_lo", 1.0))
        self.cal_hi: float = float(cal.get("s_hi", 1.0))
        self.boosters = {
            q: lgb.Booster(model_file=str(self.model_dir / f"q{int(q * 100)}.txt"))
            for q in config.QUANTILES
        }
        priors_file = self.model_dir / "degradation_priors.json"
        self.degradation_priors: dict = (
            json.loads(priors_file.read_text()) if priors_file.exists() else {}
        )

    @classmethod
    def latest(cls, models_dir: Path | None = None) -> "PaceModel":
        root = Path(models_dir or config.MODELS_DIR)
        candidates = sorted(d for d in root.iterdir() if (d / "meta.json").exists())
        if not candidates:
            raise FileNotFoundError(f"no model artifacts under {root}")
        return cls(candidates[-1])

    def _to_pandas(self, features: pl.DataFrame) -> pd.DataFrame:
        x = features.select(FEATURE_COLUMNS).to_pandas()
        for c in CATEGORICAL_FEATURES:
            x[c] = pd.Categorical(x[c], categories=self.vocab[c])
        for c in FEATURE_COLUMNS:
            if c not in CATEGORICAL_FEATURES:
                x[c] = x[c].astype(float)
        return x

    def predict_quantiles(
        self, features: pl.DataFrame, age_groups: np.ndarray | list | None = None
    ) -> np.ndarray:
        """(n, 3) array of ratio quantiles [p10, p50, p90], sorted row-wise so
        quantile crossing can never leak out.

        ``age_groups``: optional group id per row for rows that form tyre-age
        sweeps in ascending age order (e.g. forecast curves). Within each group
        a cumulative-max pass enforces "lap time never improves with tyre age"
        — LightGBM's quantile objective rejects monotone_constraints, so
        monotonicity is applied here instead.
        """
        x = self._to_pandas(features)
        preds = np.column_stack([self.boosters[q].predict(x) for q in config.QUANTILES])
        preds = np.sort(preds, axis=1)
        # split-conformal widening fitted on the validation fold (meta.json)
        preds[:, 0] = preds[:, 1] - self.cal_lo * (preds[:, 1] - preds[:, 0])
        preds[:, 2] = preds[:, 1] + self.cal_hi * (preds[:, 2] - preds[:, 1])
        if age_groups is not None:
            groups = np.asarray(age_groups)
            for g in np.unique(groups):
                idx = np.where(groups == g)[0]
                preds[idx] = np.maximum.accumulate(preds[idx], axis=0)
        return preds


def counterfactual_frame(
    base: dict, laps_ahead: int = config.FORECAST_AHEAD_LAPS
) -> tuple[pl.DataFrame, list[dict]]:
    """Scenario rows for one driver from their current feature state.

    Scenarios (one row each, tagged in the returned index):
      {kind: "current", k}          next k-th lap on current tyres (age+k)
      {kind: "fresh", compound, k}  k-th lap after pitting now (age k-1)

    race_progress advances with k; the caller converts predicted ratios to
    seconds with the k-extrapolated reference and adds the driver bias.
    """
    rows, index = [], []
    laps_total = base["laps_total"]
    lap_now = base["race_progress"] * laps_total

    def scenario(compound: str, age: float, k: int) -> dict:
        row = dict(base)
        row["compound"] = compound
        row["tyre_age"] = age
        row["tyre_age_sq"] = age * age
        row["race_progress"] = min((lap_now + k) / laps_total, 1.0)
        if compound != base["compound"] or age < base["tyre_age"]:
            row["stint_no"] = (base.get("stint_no") or 1) + 1
        return row

    for k in range(1, laps_ahead + 1):
        rows.append(scenario(base["compound"], base["tyre_age"] + k, k))
        index.append({"kind": "current", "k": k})
    for compound in config.DRY_COMPOUNDS:
        for k in range(1, laps_ahead + 1):
            rows.append(scenario(compound, float(k - 1), k))
            index.append({"kind": "fresh", "compound": compound, "k": k})
    return pl.DataFrame(rows), index


class OnlineBias:
    """Per-driver exponentially-weighted bias between actual and predicted
    ratio. Adapts the pre-trained model to this weekend's form in ~5 laps.
    b <- 0.85*b + 0.15*(actual - predicted); reset at race start."""

    def __init__(self, decay: float = config.BIAS_DECAY):
        self.decay = decay
        self._bias: dict[int, float] = {}

    def update(self, driver_number: int, actual_ratio: float, predicted_ratio: float) -> float:
        err = actual_ratio - predicted_ratio
        b = self.decay * self._bias.get(driver_number, 0.0) + (1 - self.decay) * err
        self._bias[driver_number] = b
        return b

    def get(self, driver_number: int) -> float:
        return self._bias.get(driver_number, 0.0)

    def reset(self) -> None:
        self._bias.clear()
