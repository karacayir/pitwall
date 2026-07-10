"""Model-layer tests: online bias, degradation curves, and the Phase 3
acceptance gate — monotonicity in tyre_age on a real (fixture-trained) model,
via the full save/load artifact round-trip."""

import json

import numpy as np
import polars as pl
import pytest

from app import config
from features.build import FEATURE_COLUMNS, TARGET_COLUMN, training_mask
from models.degradation import DegradationCurve, fit_curve
from models.predict import OnlineBias, PaceModel, counterfactual_frame
from models.train import build_training_frame, train_quantile_models

# --- online bias -------------------------------------------------------------


def test_bias_update_math():
    b = OnlineBias(decay=0.85)
    assert b.update(1, actual_ratio=1.02, predicted_ratio=1.00) == pytest.approx(0.15 * 0.02)
    assert b.update(1, 1.02, 1.00) == pytest.approx(0.85 * 0.003 + 0.15 * 0.02)
    assert b.get(99) == 0.0


def test_bias_converges_to_constant_offset():
    b = OnlineBias()
    for _ in range(60):
        b.update(44, actual_ratio=1.05, predicted_ratio=1.00)
    assert b.get(44) == pytest.approx(0.05, abs=1e-4)


def test_bias_reset():
    b = OnlineBias()
    b.update(1, 1.1, 1.0)
    b.reset()
    assert b.get(1) == 0.0


# --- degradation curves --------------------------------------------------------


def test_fit_recovers_known_quadratic():
    ages = np.arange(0, 25, dtype=float)
    ratios = 1.0 + 0.002 * ages + 0.0001 * ages**2
    curve = fit_curve(ages, ratios, ridge_alpha=1e-6)
    assert curve.c0 == pytest.approx(1.0, abs=2e-3)
    assert curve.c1 == pytest.approx(0.002, abs=1e-3)
    assert curve.c2 == pytest.approx(0.0001, abs=1e-4)


def test_fit_shrinks_to_prior_with_few_points():
    prior = DegradationCurve(1.0, 0.003, 0.0, n_laps=500)
    # two wildly noisy live laps must not overthrow the prior anywhere on the curve
    curve = fit_curve(np.array([3.0, 4.0]), np.array([1.30, 0.80]), prior=prior, prior_weight=8.0)
    ages = np.arange(0, 26, dtype=float)
    max_dev = np.max(np.abs(np.asarray(curve.ratio(ages)) - np.asarray(prior.ratio(ages))))
    assert max_dev < 0.05, f"prediction drifted {max_dev:.3f} from prior on 2 noisy laps"


def test_fit_empty_returns_prior_or_flat():
    prior = DegradationCurve(1.01, 0.004, 0.0001, 300)
    got = fit_curve(np.array([]), np.array([]), prior=prior)
    assert (got.c0, got.c1, got.c2) == (1.01, 0.004, 0.0001)
    flat = fit_curve(np.array([]), np.array([]), prior=None)
    assert flat.ratio(10) == 1.0


def test_c2_never_negative():
    ages = np.arange(0, 10, dtype=float)
    ratios = 1.1 - 0.001 * ages**2  # tyre "improving" quadratically: unphysical
    assert fit_curve(ages, ratios).c2 >= 0.0


# --- trained model: monotonicity + artifact round-trip ---------------------------


@pytest.fixture(scope="module")
def tiny_model(monza, sao_paulo, tmp_path_factory):
    """Train a small but real quantile model on the two fixture races and
    round-trip it through the artifact format."""
    sessions = pl.DataFrame(
        [
            {"session_id": monza["session_id"], "track_id": "monza",
             "laps_total": 53, "pole_time_s": 78.79, "season": 2025, "round": 16},
            {"session_id": sao_paulo["session_id"], "track_id": "sao_paulo",
             "laps_total": 69, "pole_time_s": 84.0, "season": 2024, "round": 21},
        ]
    )  # fmt: skip
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
                "metrics": info["metrics"],
            }
        )
    )
    model = PaceModel.latest(out.parent)
    return model, rows


def test_monotone_in_tyre_age(tiny_model):
    """Acceptance: predictions non-decreasing in tyre_age, all else fixed.

    LightGBM's quantile objective rejects monotone_constraints, so the model
    wrapper enforces monotonicity on age sweeps (the API every product path —
    forecast curves, fresh-tyre points, simulator stints — goes through)."""
    model, rows = tiny_model
    base = rows.filter(pl.col("session_id").str.contains("monza")).row(50, named=True)
    sweep = pl.DataFrame(
        [
            {**{c: base[c] for c in FEATURE_COLUMNS}, "tyre_age": age, "tyre_age_sq": age * age}
            for age in range(0, 35)
        ]
    )
    preds = model.predict_quantiles(sweep, age_groups=[0] * len(sweep))
    for col in range(3):
        deltas = np.diff(preds[:, col])
        assert (deltas >= -1e-9).all(), f"quantile {col} decreased with tyre age"


def test_quantiles_ordered_and_plausible(tiny_model):
    model, rows = tiny_model
    valid = rows.filter(pl.col("session_id").str.contains("monza")).head(200)
    preds = model.predict_quantiles(valid)
    assert (preds[:, 0] <= preds[:, 1]).all() and (preds[:, 1] <= preds[:, 2]).all()
    assert preds[:, 1].mean() == pytest.approx(valid[TARGET_COLUMN].mean(), rel=0.05), (
        "median ratio prediction far from actual ratios"
    )


def test_unseen_categories_dont_crash(tiny_model):
    """A 2026 rookie / new team must fall through as missing, not explode."""
    model, rows = tiny_model
    row = {c: rows.row(10, named=True)[c] for c in FEATURE_COLUMNS}
    row["driver_id"] = "ROOKIE_99"
    row["team_id"] = "brand_new_team"
    preds = model.predict_quantiles(pl.DataFrame([row]))
    assert np.isfinite(preds).all()


# --- counterfactual frame ---------------------------------------------------------


def test_counterfactual_frame_shapes():
    base = {c: 0.0 for c in FEATURE_COLUMNS}
    base.update(
        compound="MEDIUM", tyre_age=12.0, tyre_age_sq=144.0, stint_no=2.0,
        race_progress=0.5, laps_total=53.0, track_id="monza", driver_id="VER",
        team_id="red_bull_racing", season=2025.0, reference_pace_s=82.0,
    )  # fmt: skip
    frame, index = counterfactual_frame(base, laps_ahead=15)
    assert len(frame) == len(index) == 15 * (1 + len(config.DRY_COMPOUNDS))
    cur1 = frame.row(0, named=True)
    assert index[0] == {"kind": "current", "k": 1} and cur1["tyre_age"] == 13.0
    fresh = [i for i, ix in enumerate(index) if ix["kind"] == "fresh" and ix["k"] == 1]
    for i in fresh:
        row = frame.row(i, named=True)
        assert row["tyre_age"] == 0.0 and row["stint_no"] == 3.0
    # race_progress advances with k and caps at 1.0
    assert frame["race_progress"].max() <= 1.0
