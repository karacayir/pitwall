"""Walk-forward replay backtest.

For each held-out race (strictly temporal): train on everything before it,
stream the race through the live RaceEngine, record the engine's forecasts as
they were made, then score them against what actually happened.

Metrics: next-lap MAE (clean-air green primary, all green secondary),
5-lap-ahead MAE, pinball loss per quantile, [P10,P90] coverage; baselines:
persistence (last green lap), rolling-median(5), linear stint extrapolation.

Gates (from the validation spec): P50 beats persistence by >=20% and
rolling-median by >=8% on clean-air next-lap MAE; coverage in 80 +/- 7.

Run: uv run python -m backtest.replay_eval --seasons 2024 2025 2026
"""

import argparse
import datetime as dt
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from app import config
from app.state import RaceEngine
from app.tracks import load_tracks
from features.build import training_mask
from ingest.replay import RAW_DIR, event_stream, load_session_meta
from models.predict import PaceModel
from models.train import build_training_frame, degradation_priors, train_quantile_models

log = logging.getLogger("pitwall.backtest")


@dataclass
class Prediction:
    driver: int
    target_lap: int
    p10: float
    p50: float
    p90: float
    ahead5_p50: float | None
    stint_at_pred: int
    persistence: float | None
    persistence_green: float | None
    rolling_median: float | None
    stint_linear: float | None


@dataclass
class RaceEval:
    session_id: str
    track_id: str
    rows: pl.DataFrame  # one row per scored (driver, lap)

    def metric(self, expr: pl.Expr, mask: pl.Expr | None = None) -> float | None:
        df = self.rows.filter(mask) if mask is not None else self.rows
        if df.is_empty():
            return None
        val = df.select(expr.alias("m"))["m"][0]
        return float(val) if val is not None else None


@dataclass
class _History:
    """Per-driver lap history for the baselines (uses laps <= L only)."""

    green_times: list[float] = field(default_factory=list)
    all_times: list[float] = field(default_factory=list)
    stint_laps: list[tuple[int, float]] = field(default_factory=list)
    stint_no: int = 1

    def update(self, lap_number: int, lap_time_s: float | None, lap_class: str, stint: int):
        if stint != self.stint_no:
            self.stint_no = stint
            self.stint_laps = []
        if lap_time_s is not None:
            self.all_times.append(lap_time_s)
        if lap_class == "green" and lap_time_s is not None:
            self.green_times.append(lap_time_s)
            self.stint_laps.append((lap_number, lap_time_s))

    def persistence(self) -> float | None:
        """Spec baseline: the driver's last completed lap, whatever it was."""
        return self.all_times[-1] if self.all_times else None

    def persistence_green(self) -> float | None:
        """Harder diagnostic baseline: last GREEN lap."""
        return self.green_times[-1] if self.green_times else None

    def rolling_median(self, n: int = 5) -> float | None:
        if not self.green_times:
            return None
        return float(np.median(self.green_times[-n:]))

    def stint_linear(self, target_lap: int) -> float | None:
        if len(self.stint_laps) < 3:
            return self.persistence()
        laps = np.array([p[0] for p in self.stint_laps[-8:]])
        times = np.array([p[1] for p in self.stint_laps[-8:]])
        slope, intercept = np.polyfit(laps, times, 1)
        return float(slope * target_lap + intercept)


def evaluate_race(race_dir: Path, model: PaceModel, tracks: dict) -> RaceEval | None:
    meta = load_session_meta(race_dir, tracks)
    engine = RaceEngine(meta, model)
    preds: dict[tuple[int, int], Prediction] = {}
    history: dict[int, _History] = defaultdict(_History)
    scored: list[dict] = []

    for _, kind, payload in event_stream(race_dir):
        if kind == "weather":
            engine.on_weather(payload)
            continue

        dn = int(payload["driver_number"])
        lap = int(payload["lap_number"])
        engine.on_lap(payload)
        enriched = engine.enriched[-1]
        d = engine.drivers[dn]

        # score any prediction that targeted this lap
        p = preds.pop((dn, lap), None)
        if (
            p is not None
            and enriched.lap_class == "green"
            and payload.get("lap_time_s") is not None
            and payload.get("is_accurate", True)
            and not payload.get("deleted", False)
        ):
            actual = float(payload["lap_time_s"])
            scored.append(
                {
                    "driver": dn,
                    "lap": lap,
                    "actual": actual,
                    "p10": p.p10,
                    "p50": p.p50,
                    "p90": p.p90,
                    "clean_air": enriched.clean_air,
                    "compound": d.compound,
                    "persistence": p.persistence,
                    "persistence_green": p.persistence_green,
                    "rolling_median": p.rolling_median,
                    "stint_linear": p.stint_linear,
                }
            )
        p5 = preds.pop((dn, -lap), None)  # 5-ahead predictions keyed negative
        if (
            p5 is not None
            and enriched.lap_class == "green"
            and payload.get("lap_time_s") is not None
            and int(payload.get("stint") or 0) == p5.stint_at_pred  # no pit in between
            and p5.ahead5_p50 is not None
        ):
            scored.append(
                {
                    "driver": dn,
                    "lap": lap,
                    "actual": float(payload["lap_time_s"]),
                    "p10": None,
                    "p50": None,
                    "p90": None,
                    "ahead5": p5.ahead5_p50,
                    "clean_air": enriched.clean_air,
                    "compound": d.compound,
                }
            )

        # record the forecast the engine just made for this driver's next laps
        h = history[dn]
        h.update(lap, payload.get("lap_time_s"), enriched.lap_class, int(payload.get("stint") or 1))
        fc = d.forecast
        if fc and fc.current:
            preds[(dn, lap + 1)] = Prediction(
                driver=dn,
                target_lap=lap + 1,
                p10=fc.current.p10,
                p50=fc.current.p50,
                p90=fc.current.p90,
                ahead5_p50=None,
                stint_at_pred=int(payload.get("stint") or 1),
                persistence=h.persistence(),
                persistence_green=h.persistence_green(),
                rolling_median=h.rolling_median(),
                stint_linear=h.stint_linear(lap + 1),
            )
            ahead = fc.ahead.get("current") or []
            if len(ahead) >= 5:
                preds[(dn, -(lap + 5))] = Prediction(
                    driver=dn, target_lap=lap + 5, p10=0, p50=0, p90=0,
                    ahead5_p50=ahead[4], stint_at_pred=int(payload.get("stint") or 1),
                    persistence=None, persistence_green=None,
                    rolling_median=None, stint_linear=None,
                )  # fmt: skip

    if not scored:
        return None
    rows = pl.DataFrame(scored, infer_schema_length=None)
    if "ahead5" not in rows.columns:
        rows = rows.with_columns(pl.lit(None, dtype=pl.Float64).alias("ahead5"))
    return RaceEval(meta.session_id, meta.track_id, rows)


def _pinball(actual: pl.Expr, pred: pl.Expr, q: float) -> pl.Expr:
    diff = actual - pred
    return pl.when(diff >= 0).then(q * diff).otherwise((q - 1) * diff)


def aggregate(evals: list[RaceEval]) -> dict:
    rows = pl.concat([e.rows for e in evals], how="diagonal")
    next_rows = rows.filter(pl.col("p50").is_not_null())
    clean = next_rows.filter(pl.col("clean_air"))
    ahead_rows = rows.filter(pl.col("ahead5").is_not_null())

    def mae(df: pl.DataFrame, col: str) -> float | None:
        df = df.filter(pl.col(col).is_not_null())
        if df.is_empty():
            return None
        return float(df.select((pl.col("actual") - pl.col(col)).abs().mean())[0, 0])

    out = {
        "n_next": len(next_rows),
        "n_clean": len(clean),
        "n_ahead5": len(ahead_rows),
        "mae_clean_next": mae(clean, "p50"),
        "mae_green_next": mae(next_rows, "p50"),
        "mae_ahead5": mae(ahead_rows, "ahead5"),
        "mae_persistence": mae(clean, "persistence"),
        "mae_persistence_green": mae(clean, "persistence_green"),
        "mae_rolling_median": mae(clean, "rolling_median"),
        "mae_stint_linear": mae(clean, "stint_linear"),
        "coverage": float(
            next_rows.select(
                ((pl.col("actual") >= pl.col("p10")) & (pl.col("actual") <= pl.col("p90"))).mean()
            )[0, 0]
        ),
    }
    for q, col in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
        out[f"pinball_q{int(q * 100)}"] = float(
            next_rows.select(_pinball(pl.col("actual"), pl.col(col), q).mean())[0, 0]
        )
    out["gate_persistence"] = (
        out["mae_clean_next"] is not None
        and out["mae_persistence"] is not None
        and out["mae_clean_next"] <= 0.8 * out["mae_persistence"]
    )
    out["gate_rolling_median"] = (
        out["mae_clean_next"] is not None
        and out["mae_rolling_median"] is not None
        and out["mae_clean_next"] <= 0.92 * out["mae_rolling_median"]
    )
    out["gate_coverage"] = 0.73 <= out["coverage"] <= 0.87
    out["gates_pass"] = all(
        out[k] for k in ("gate_persistence", "gate_rolling_median", "gate_coverage")
    )
    return out


def per_compound_table(evals: list[RaceEval]) -> pl.DataFrame:
    rows = pl.concat([e.rows for e in evals], how="diagonal")
    return (
        rows.filter(pl.col("p50").is_not_null() & pl.col("clean_air"))
        .group_by("compound")
        .agg(
            (pl.col("actual") - pl.col("p50")).abs().mean().round(3).alias("mae_s"),
            ((pl.col("actual") >= pl.col("p10")) & (pl.col("actual") <= pl.col("p90")))
            .mean()
            .round(3)
            .alias("coverage"),
            pl.len().alias("n"),
        )
        .sort("compound")
    )


def md_table(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)


def render_report(
    agg: dict, race_table: pl.DataFrame, compound_table: pl.DataFrame, out_path: Path
) -> None:
    def pct(a: float | None, b: float | None) -> str:
        if not a or not b:
            return "n/a"
        return f"{(1 - a / b) * 100:+.1f}%"

    def tick(flag: bool) -> str:
        return "PASS" if flag else "FAIL"

    pin = f"{agg['pinball_q10']:.4f} / {agg['pinball_q50']:.4f} / {agg['pinball_q90']:.4f}"
    g1 = pct(agg["mae_clean_next"], agg["mae_persistence"])
    g2 = pct(agg["mae_clean_next"], agg["mae_rolling_median"])
    lines = [
        f"# Pitwall backtest — {dt.date.today().isoformat()}",
        "",
        "## Aggregate (walk-forward, held-out races)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| clean-air next-lap MAE | **{agg['mae_clean_next']:.3f} s** ({agg['n_clean']} laps) |",
        f"| all-green next-lap MAE | {agg['mae_green_next']:.3f} s ({agg['n_next']} laps) |",
        f"| 5-lap-ahead MAE | {agg['mae_ahead5']:.3f} s ({agg['n_ahead5']} laps) |"
        if agg["mae_ahead5"]
        else "| 5-lap-ahead MAE | n/a |",
        f"| persistence baseline MAE (spec: last lap) | {agg['mae_persistence']:.3f} s |",
        f"| persistence, last GREEN lap (harder) | {agg['mae_persistence_green']:.3f} s |",
        f"| rolling-median(5) baseline MAE | {agg['mae_rolling_median']:.3f} s |",
        f"| stint-linear baseline MAE | {agg['mae_stint_linear']:.3f} s |",
        f"| [P10,P90] coverage | {agg['coverage'] * 100:.1f}% (target 80±7) |",
        f"| pinball q10 / q50 / q90 | {pin} |",
        "",
        "## Acceptance gates",
        "",
        "| gate | required | actual | pass |",
        "|---|---|---|---|",
        f"| beats persistence | ≥20% | {g1} | {tick(agg['gate_persistence'])} |",
        f"| beats rolling-median(5) | ≥8% | {g2} | {tick(agg['gate_rolling_median'])} |",
        f"| coverage in [73%, 87%] | 80±7 | {agg['coverage'] * 100:.1f}% "
        f"| {tick(agg['gate_coverage'])} |",
        "",
        "## Per compound (clean-air next-lap)",
        "",
        md_table(compound_table),
        "",
        "## Per race",
        "",
        md_table(race_table),
        "",
    ]
    out_path.write_text("\n".join(lines))
    log.info("report written to %s", out_path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="*", default=[2024, 2025, 2026])
    parser.add_argument(
        "--retrain-every",
        type=int,
        default=3,
        help="retrain the model every N held-out races (walk-forward blocks)",
    )
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--max-races", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    laps = pl.read_parquet(config.LAPS_PARQUET)
    weather = pl.read_parquet(config.WEATHER_PARQUET)
    sessions = pl.read_parquet(config.SESSIONS_PARQUET).sort(["season", "round"])
    tracks = load_tracks()

    log.info("building feature frames for %d sessions…", len(sessions))
    all_features = build_training_frame(laps, weather, sessions, tracks)
    all_rows = all_features.filter(training_mask(all_features))

    session_list = sessions.to_dicts()
    targets = [s for s in session_list if s["season"] in args.seasons]
    if args.max_races:
        targets = targets[-args.max_races :]
    log.info("held-out races: %d", len(targets))

    evals: list[RaceEval] = []
    race_metrics: list[dict] = []
    model: PaceModel | None = None
    model_dir = config.MODELS_DIR / "_backtest"

    for i, sess in enumerate(targets):
        sid = sess["session_id"]
        if model is None or i % args.retrain_every == 0:
            block_start = sid
            train_rows = all_rows.filter(
                pl.col("session_id").is_in(
                    [s["session_id"] for s in session_list
                     if (s["season"], s["round"]) < (sess["season"], sess["round"])]
                )
            )  # fmt: skip
            if len(train_rows) < 5000:
                log.warning("only %d training rows before %s — skipping race", len(train_rows), sid)
                continue
            valid_ids = train_rows["session_id"].unique().sort().to_list()[-4:]
            t_df = train_rows.filter(~pl.col("session_id").is_in(valid_ids))
            v_df = train_rows.filter(pl.col("session_id").is_in(valid_ids))
            log.info("[%d/%d] retraining before %s (%d rows)…", i + 1, len(targets), sid, len(t_df))
            boosters, info = train_quantile_models(
                t_df, v_df, current_season=sess["season"], n_estimators=args.n_estimators
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            for alpha, booster in boosters.items():
                booster.save_model(str(model_dir / f"q{int(alpha * 100)}.txt"))
            priors = degradation_priors(t_df)
            (model_dir / "degradation_priors.json").write_text(json.dumps(priors))
            from features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS

            (model_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "features": FEATURE_COLUMNS,
                        "categorical_features": CATEGORICAL_FEATURES,
                        "quantiles": list(config.QUANTILES),
                        "vocab": info["vocab"],
                        "calibration": info["calibration"],
                        "metrics": info["metrics"],
                        "trained_before": block_start,
                    }
                )
            )
            model = PaceModel(model_dir)

        race_dir = RAW_DIR / sid
        if not (race_dir / "_complete").exists():
            log.warning("raw data missing for %s, skipping", sid)
            continue
        ev = evaluate_race(race_dir, model, tracks)
        if ev is None:
            log.warning("no scored laps for %s", sid)
            continue
        evals.append(ev)
        m = aggregate([ev])
        race_metrics.append(
            {
                "session_id": sid,
                "mae_clean": round(m["mae_clean_next"], 3) if m["mae_clean_next"] else None,
                "mae_persist": round(m["mae_persistence"], 3) if m["mae_persistence"] else None,
                "coverage": round(m["coverage"], 3),
                "n": m["n_clean"],
            }
        )
        log.info(
            "[%d/%d] %s: clean MAE %.3fs vs persistence %.3fs, coverage %.1f%%",
            i + 1, len(targets), sid,
            m["mae_clean_next"] or float("nan"),
            m["mae_persistence"] or float("nan"),
            m["coverage"] * 100,
        )  # fmt: skip

    if not evals:
        log.error("nothing evaluated")
        return 1

    agg = aggregate(evals)
    race_table = pl.DataFrame(race_metrics)
    compound_table = per_compound_table(evals)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    out = (
        Path(args.out)
        if args.out
        else config.REPORTS_DIR / f"backtest_{dt.date.today().isoformat()}.md"
    )
    render_report(agg, race_table, compound_table, out)
    print(
        json.dumps({k: v for k, v in agg.items() if not isinstance(v, pl.DataFrame)}, default=str)
    )
    return 0 if agg["gates_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
