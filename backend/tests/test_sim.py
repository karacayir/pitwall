"""Strategy simulator tests: seeded determinism, the <2s/2000-sims acceptance
gate, and the fixture sanity check (sim from 10 laps before the flag lands
within ±20s of the real top-5 finish times)."""

import time

import numpy as np
import polars as pl
import pytest

from app.schemas import StintPlan
from app.state import RaceEngine
from models.degradation import DegradationCurve
from sim.montecarlo import (
    CarSim,
    SimSetup,
    TrackSim,
    label_for,
    legal_strategies,
    run_strategy,
    simulate_strategies,
)
from tests.test_engine import MONZA_SESSION


def toy_setup(n_cars: int = 10, from_lap: int = 30, laps_total: int = 53) -> SimSetup:
    cars = [
        CarSim(
            driver_number=i + 1,
            t0=2500.0 + 2.5 * i,
            compound="MEDIUM" if i % 2 == 0 else "HARD",
            tyre_age=8.0 + i % 5,
            bias_ratio=0.0005 * (i - n_cars / 2),
            compounds_used={"SOFT"},
        )
        for i in range(n_cars)
    ]
    return SimSetup(
        cars=cars,
        chosen=3,
        track=TrackSim(
            laps_total=laps_total,
            pit_loss_s=21.0,
            sc_hazard_per_lap=0.004,
            overtake_difficulty=0.3,
        ),  # fmt: skip
        from_lap=from_lap,
        ref0=82.0,
        ref_slope=-0.02,
        curves={
            "SOFT": DegradationCurve(0.995, 0.0035, 0.00012, 100),
            "MEDIUM": DegradationCurve(1.005, 0.0022, 0.00006, 100),
            "HARD": DegradationCurve(1.012, 0.0012, 0.00003, 100),
        },
        current_position=4,
    )


def test_seeded_determinism():
    setup = toy_setup()
    strat = [StintPlan(lap=38, compound="HARD")]
    f1, p1 = run_strategy(setup, strat, n_sims=500, rng_seed=7)
    f2, p2 = run_strategy(setup, strat, n_sims=500, rng_seed=7)
    assert np.array_equal(f1, f2) and np.array_equal(p1, p2)
    f3, _ = run_strategy(setup, strat, n_sims=500, rng_seed=8)
    assert not np.array_equal(f1, f3)


def test_2000_sims_under_2_seconds():
    """Acceptance: a full simulate call at n_sims=2000 in <2s on one core."""
    setup = toy_setup()
    strategies = legal_strategies(setup)
    assert 10 <= len(strategies) <= 60
    start = time.perf_counter()
    baseline, results = simulate_strategies(setup, strategies=None, n_sims=2000, seed=1)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"simulation took {elapsed:.2f}s"
    assert results == sorted(results, key=lambda r: r.expected_position)
    assert baseline.finish_time_s.p10 <= baseline.finish_time_s.p50 <= baseline.finish_time_s.p90
    assert sum(baseline.p_position.values()) == pytest.approx(1.0, abs=1e-6)


def test_fresh_rubber_beats_dying_tyres():
    """A car on 25-lap-old SOFTs must gain from pitting (physics smoke test)."""
    setup = toy_setup()
    chosen_car = next(c for c in setup.cars if c.driver_number == setup.chosen)
    chosen_car.compound = "SOFT"
    chosen_car.tyre_age = 25.0
    stay, _ = run_strategy(setup, [], n_sims=2000, rng_seed=3)
    pit, _ = run_strategy(setup, [StintPlan(lap=32, compound="MEDIUM")], n_sims=2000, rng_seed=3)
    # 23 laps to go on dying softs vs one 21s stop: pitting must win on median
    assert np.median(pit) < np.median(stay)


def test_two_compound_rule_respected():
    setup = toy_setup()
    chosen_car = next(c for c in setup.cars if c.driver_number == setup.chosen)
    chosen_car.compounds_used = {chosen_car.compound}  # only one dry compound used
    strategies = legal_strategies(setup)
    assert [] not in strategies, "stay-out is illegal with only one dry compound used"
    for s in strategies:
        if len(s) == 1:
            assert (
                len({chosen_car.compound, s[0].compound}) >= 2
                or s[0].compound != chosen_car.compound
            )


def test_sanity_vs_fixture_race(monza, tiny_model):
    """Sim from 10 laps before the end: median finish within ±20s of reality
    for the top 5 classified drivers."""
    model, _ = tiny_model
    sim_from_completed = MONZA_SESSION.laps_total - 10  # laps completed: 43

    engine = RaceEngine(MONZA_SESSION, model=model)
    actual_t0, actual_final = {}, {}
    events = monza["laps"].sort("time_session_s")
    for row in events.iter_rows(named=True):
        dn, lap = row["driver_number"], row["lap_number"]
        if lap <= sim_from_completed:
            engine.on_lap(row)
            if lap == sim_from_completed and row["time_session_s"] is not None:
                actual_t0[dn] = row["time_session_s"]
        if row["time_session_s"] is not None:
            actual_final[dn] = (lap, row["time_session_s"])

    final_frame = monza["laps"].filter(pl.col("lap_number") == MONZA_SESSION.laps_total)
    top5 = final_frame.filter(pl.col("position") <= 5).sort("position")["driver_number"].to_list()
    checked = 0
    for dn in top5:
        if dn not in actual_t0:
            continue
        last_lap, t_final = actual_final[dn]
        if last_lap != MONZA_SESSION.laps_total:
            continue  # lapped car: its "finish" isn't comparable
        actual_duration = t_final - actual_t0[dn]
        setup = engine.to_sim_setup(dn)
        finish, _ = run_strategy(setup, [], n_sims=1000, rng_seed=11)
        assert abs(float(np.median(finish)) - actual_duration) < 20.0, (
            f"driver {dn}: sim {np.median(finish):.1f}s vs actual {actual_duration:.1f}s"
        )
        checked += 1
    assert checked >= 4, f"only {checked} of the top 5 were comparable"


def test_labels():
    assert label_for([]) == "stay out"
    assert "L38 HARD" in label_for([StintPlan(lap=38, compound="HARD")])
