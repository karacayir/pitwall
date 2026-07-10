"""Backtest evaluator plumbing test: fixture race through evaluate_race ->
aggregate -> report. (The tiny model has seen Monza, so metric VALUES here
prove nothing — the real gates run on held-out races via `make backtest`.)"""

import json

import polars as pl

from backtest.replay_eval import aggregate, evaluate_race, per_compound_table, render_report
from tests.conftest import MONZA_META

TRACKS = {"monza": {"lap_length_km": 5.793, "abrasiveness": 3.0, "pit_loss_s": 21.0}}


def make_race_dir(tmp_path, race: dict):
    d = tmp_path / race["session_id"]
    d.mkdir()
    race["laps"].write_parquet(d / "laps.parquet")
    race["weather"].write_parquet(d / "weather.parquet")
    meta = {
        "session_id": race["session_id"],
        "track_id": "monza",
        "laps_total": MONZA_META["laps_total"],
        "season": MONZA_META["season"],
        "pole_time_s": MONZA_META["pole_time_s"],
        "event_name": "Italian Grand Prix",
    }
    (d / "session.json").write_text(json.dumps(meta))
    (d / "_complete").touch()
    return d


def test_evaluate_race_and_report(monza, tiny_model, tmp_path):
    model, _ = tiny_model
    race_dir = make_race_dir(tmp_path, monza)

    ev = evaluate_race(race_dir, model, TRACKS)
    assert ev is not None
    next_rows = ev.rows.filter(pl.col("p50").is_not_null())
    ahead_rows = ev.rows.filter(pl.col("ahead5").is_not_null())
    assert len(next_rows) > 400, "expected most green laps to be scored"
    assert len(ahead_rows) > 200, "expected 5-ahead predictions to be scored"
    # baselines recorded alongside
    assert next_rows["persistence"].null_count() < len(next_rows) * 0.2

    agg = aggregate([ev])
    assert 0.0 < agg["mae_clean_next"] < 5.0
    assert 0.2 <= agg["coverage"] <= 1.0
    assert agg["mae_persistence"] is not None
    for key in ("gate_persistence", "gate_rolling_median", "gate_coverage", "gates_pass"):
        assert isinstance(agg[key], bool)

    race_table = pl.DataFrame(
        [{"session_id": ev.session_id, "mae_clean": agg["mae_clean_next"],
          "mae_persist": agg["mae_persistence"], "coverage": agg["coverage"], "n": agg["n_clean"]}]
    )  # fmt: skip
    out = tmp_path / "report.md"
    render_report(agg, race_table, per_compound_table([ev]), out)
    text = out.read_text()
    assert "Acceptance gates" in text and "clean-air next-lap MAE" in text
