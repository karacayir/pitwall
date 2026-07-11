"""Monte Carlo strategy simulator.

Vectorised across sims (numpy), Python loop over remaining laps. Common random
numbers: every strategy sees the same SC timeline and pace-noise draws, so
strategy deltas aren't drowned in sampling noise.

Per sim & lap (see brief):
  1. SC state machine from the track's sc_hazard_per_lap; under SC all cars lap
     at SC_LAP_FACTOR x reference and pit loss shrinks by SC_PIT_LOSS_FACTOR.
  2. A car's green lap ratio = its compound degradation curve at current age
     + driver bias + two-piece-normal noise scaled from the model's quantiles.
  3. Pitting adds pit_loss_s, resets age, switches compound.
  4. Traffic: within TRAFFIC_FOLLOW_WINDOW_S behind a slower car -> overtake
     with p = logistic((pace_delta - overtake_threshold)/0.15), else follow at
     leader pace + TRAFFIC_FOLLOW_PENALTY_S.
  5. Only cars within SIM_RIVAL_WINDOW_S of the chosen driver are simulated;
     cars outside the window are assumed to keep their relative order.
"""

from dataclasses import dataclass, field

import numpy as np

from app import config
from app.schemas import FinishTime, StintPlan, StrategyResult
from models.degradation import DegradationCurve

# tunable approximations (rival behaviour + noise defaults)
RIVAL_STINT_MAX = {"SOFT": 20, "MEDIUM": 30, "HARD": 42, "INTERMEDIATE": 32, "WET": 30}
DEFAULT_SIGMA_LO = 0.006  # ratio units below the median
DEFAULT_SIGMA_HI = 0.012  # ratio units above (laps are skewed slow)
SC_MEAN_DURATION_LAPS = 4.0
OVERTAKE_THRESHOLD_BASE_S = 0.25  # pace delta for 50% overtake odds at difficulty 0
OVERTAKE_THRESHOLD_SCALE_S = 1.5  # + scale * overtake_difficulty


@dataclass
class CarSim:
    driver_number: int
    t0: float  # cumulative race clock at sim start (s)
    compound: str
    tyre_age: float
    bias_ratio: float = 0.0
    plan: list[StintPlan] = field(default_factory=list)  # rivals: heuristic if empty
    compounds_used: set[str] = field(default_factory=set)


@dataclass
class TrackSim:
    laps_total: int
    pit_loss_s: float
    sc_hazard_per_lap: float
    overtake_difficulty: float


@dataclass
class SimSetup:
    cars: list[CarSim]
    chosen: int  # driver_number
    track: TrackSim
    from_lap: int  # first simulated lap
    ref0: float  # current reference pace (s)
    ref_slope: float = 0.0  # s/lap, already clipped
    curves: dict[str, DegradationCurve] = field(default_factory=dict)
    sigma: dict[str, tuple[float, float]] = field(default_factory=dict)
    current_position: int | None = None


def _rival_plan(car: CarSim, from_lap: int, laps_total: int) -> list[StintPlan]:
    """Heuristic rival strategy: run each stint to the compound's typical life,
    then fit the hardest compound that gets to the flag."""
    plan = []
    lap = from_lap
    compound, age = car.compound, car.tyre_age
    while lap < laps_total:
        remaining_life = RIVAL_STINT_MAX.get(compound, 30) - age
        pit_lap = int(lap + max(remaining_life, 2))
        if pit_lap >= laps_total - 2:
            break
        nxt = "HARD" if compound != "HARD" else "MEDIUM"
        plan.append(StintPlan(lap=pit_lap, compound=nxt))
        lap, compound, age = pit_lap, nxt, 0.0
    return plan


class SimEnvironment:
    """Shared random draws: one environment reused by every strategy (common
    random numbers), so strategy deltas reflect strategy, not sampling luck.

    sc_hazard_per_lap in tracks.yaml is the share of laps run under SC/VSC
    (per the brief); an SC lasts SC_MEAN_DURATION_LAPS on average, so the
    per-lap START probability is share / duration (capped for sanity)."""

    def __init__(self, setup: SimSetup, n_sims: int, seed: int):
        rng = np.random.default_rng(seed)
        n_cars = len(setup.cars)
        n_laps = setup.track.laps_total - setup.from_lap + 1
        p_start = min(setup.track.sc_hazard_per_lap / SC_MEAN_DURATION_LAPS, 0.08)
        self.sc = np.zeros((n_sims, n_laps), dtype=bool)
        state = np.zeros(n_sims, dtype=bool)
        for li in range(n_laps):
            start = rng.random(n_sims) < p_start
            end = rng.random(n_sims) < 1.0 / SC_MEAN_DURATION_LAPS
            state = np.where(state, ~end, start)
            self.sc[:, li] = state
        self.z_all = rng.standard_normal((n_sims, n_laps, n_cars))
        self.u_overtake = rng.random((n_sims, n_laps, n_cars))
        self.n_sims = n_sims


def run_strategy(
    setup: SimSetup,
    strategy: list[StintPlan],
    n_sims: int,
    rng_seed: int,
    env: SimEnvironment | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one strategy for the chosen driver.

    Returns (finish_times (n_sims,), positions (n_sims,)) for the chosen driver;
    positions are global (window offset applied).
    """
    if env is None or env.n_sims != n_sims:
        env = SimEnvironment(setup, n_sims, rng_seed)
    track, cars = setup.track, setup.cars
    n_cars = len(cars)
    laps = range(setup.from_lap, track.laps_total + 1)

    # per-car plans (chosen driver gets the strategy under test)
    plans: list[dict[int, str]] = []
    for car in cars:
        if car.driver_number == setup.chosen:
            plan = strategy
        elif car.plan:
            plan = car.plan
        else:
            plan = _rival_plan(car, setup.from_lap, track.laps_total)
        plans.append({p.lap: p.compound for p in plan})

    # state arrays
    t = np.tile(np.array([c.t0 for c in cars]), (n_sims, 1))  # (N, C)
    age = np.tile(np.array([c.tyre_age for c in cars], dtype=float), (n_sims, 1))
    compound_idx = {c: i for i, c in enumerate(config.ALL_COMPOUNDS)}
    comp = np.tile(np.array([compound_idx.get(c.compound, 1) for c in cars]), (n_sims, 1))
    bias = np.array([c.bias_ratio for c in cars])

    # curve/noise lookup tables per compound
    curve_c = np.zeros((len(config.ALL_COMPOUNDS), 3))
    sig = np.zeros((len(config.ALL_COMPOUNDS), 2))
    for name, i in compound_idx.items():
        curve = setup.curves.get(name) or DegradationCurve(1.02, 0.002, 0.0, 0)
        curve_c[i] = [curve.c0, curve.c1, curve.c2]
        sig[i] = setup.sigma.get(name, (DEFAULT_SIGMA_LO, DEFAULT_SIGMA_HI))

    sc = env.sc
    z_all = env.z_all
    u_overtake = env.u_overtake
    threshold = OVERTAKE_THRESHOLD_BASE_S + OVERTAKE_THRESHOLD_SCALE_S * track.overtake_difficulty

    for li, lap in enumerate(laps):
        ref = setup.ref0 + setup.ref_slope * (lap - setup.from_lap)
        c0 = curve_c[comp, 0]
        c1 = curve_c[comp, 1]
        c2 = curve_c[comp, 2]
        ratio = c0 + c1 * age + c2 * age * age + bias[None, :]
        z = z_all[:, li, :]
        noise = np.where(z >= 0, z * sig[comp, 1], z * sig[comp, 0])
        lap_time = (ratio + noise) * ref

        sc_now = sc[:, li]
        lap_time = np.where(sc_now[:, None], config.SC_LAP_FACTOR * ref, lap_time)

        # pit stops scheduled at the END of this lap
        for ci, plan in enumerate(plans):
            new_comp = plan.get(lap)
            if new_comp is not None:
                loss = track.pit_loss_s * np.where(sc_now, config.SC_PIT_LOSS_FACTOR, 1.0)
                lap_time[:, ci] += loss
                age[:, ci] = 0.0
                comp[:, ci] = compound_idx.get(new_comp, 2)

        # traffic: process cars in running order (argsort of clock before the lap)
        order = np.argsort(t, axis=1)
        t_sorted = np.take_along_axis(t, order, axis=1)
        lt_sorted = np.take_along_axis(lap_time, order, axis=1)
        u_sorted = np.take_along_axis(u_overtake[:, li, :], order, axis=1)
        new_t = t_sorted + lt_sorted
        # vectorised overtake odds for every adjacent pair (exp is costly)
        pace_delta = lt_sorted[:, :-1] - lt_sorted[:, 1:]  # >0: follower faster
        p_pass = 1.0 / (1.0 + np.exp(-(pace_delta - threshold) / config.OVERTAKE_LOGISTIC_SCALE))
        may_pass = (u_sorted[:, 1:] < p_pass) & ~sc_now[:, None]
        in_window = (t_sorted[:, 1:] - t_sorted[:, :-1]) < config.SIM_RIVAL_WINDOW_S
        follow = config.TRAFFIC_FOLLOW_PENALTY_S
        for pos in range(1, n_cars):
            leader_t = new_t[:, pos - 1]
            catching = new_t[:, pos] < leader_t + follow
            blocked = catching & ~(may_pass[:, pos - 1] & in_window[:, pos - 1])
            new_t[:, pos] = np.where(blocked, leader_t + follow, new_t[:, pos])
        # under SC the field compresses toward the delta train
        if sc_now.any():
            compressed = np.maximum.accumulate(new_t, axis=1)  # keep order, close gaps
            new_t = np.where(sc_now[:, None], compressed + np.arange(n_cars) * 1.0, new_t)
        t = np.empty_like(new_t)
        np.put_along_axis(t, order, new_t, axis=1)
        age += 1.0

    chosen_ci = next(i for i, c in enumerate(cars) if c.driver_number == setup.chosen)
    finish = t[:, chosen_ci] - cars[chosen_ci].t0
    local_rank = (t < t[:, chosen_ci : chosen_ci + 1]).sum(axis=1) + 1
    start_rank = sorted(range(n_cars), key=lambda i: cars[i].t0).index(chosen_ci) + 1
    offset = (setup.current_position or start_rank) - start_rank
    positions = local_rank + offset
    return finish, positions


def legal_strategies(
    setup: SimSetup, max_stops: int = 2, lap_step: int = 3
) -> list[list[StintPlan]]:
    """Auto-enumerate stop plans: 0-2 further stops, candidate laps every
    `lap_step`, compound sequences satisfying the two-dry-compound rule."""
    chosen = next(c for c in setup.cars if c.driver_number == setup.chosen)
    used = set(chosen.compounds_used) | {chosen.compound}
    dry_used = used & set(config.DRY_COMPOUNDS)
    wet_race = chosen.compound in ("INTERMEDIATE", "WET")

    first_pit = setup.from_lap + 2
    last_pit = setup.track.laps_total - 3
    candidates = list(range(first_pit, last_pit + 1, lap_step))
    out: list[list[StintPlan]] = []

    if wet_race or len(dry_used) >= 2:
        out.append([])  # staying out is legal
    for lap in candidates:
        for compound in config.DRY_COMPOUNDS:
            if wet_race or len(dry_used | {compound}) >= 2:
                out.append([StintPlan(lap=lap, compound=compound)])
    if max_stops >= 2:
        two_step = lap_step * 2
        for l1 in candidates[::2]:
            for l2 in range(l1 + two_step, last_pit + 1, two_step):
                for c1 in config.DRY_COMPOUNDS:
                    for c2 in config.DRY_COMPOUNDS:
                        if wet_race or len(dry_used | {c1, c2}) >= 2:
                            out.append(
                                [StintPlan(lap=l1, compound=c1), StintPlan(lap=l2, compound=c2)]
                            )
    # keep the search tractable (and the <2s budget honest on a small vCPU)
    max_strategies = 36
    if len(out) > max_strategies:
        keep = out[:1] + out[1 : len(out) : max(1, len(out) // (max_strategies - 1))]
        out = keep[:max_strategies]
    return out


def label_for(strategy: list[StintPlan]) -> str:
    if not strategy:
        return "stay out"
    return " → ".join(f"L{p.lap} {p.compound}" for p in strategy)


def simulate_strategies(
    setup: SimSetup,
    strategies: list[list[StintPlan]] | None,
    n_sims: int = config.SIM_DEFAULT_N,
    seed: int = 42,
) -> tuple[StrategyResult, list[StrategyResult]]:
    """Returns (baseline stay-out/current-plan result, ranked strategy results)."""
    if strategies is None:
        strategies = legal_strategies(setup)
    current_pos = float(setup.current_position or 1)
    env = SimEnvironment(setup, n_sims, seed)

    def result_for(strategy: list[StintPlan]) -> StrategyResult:
        finish, positions = run_strategy(setup, strategy, n_sims, rng_seed=seed, env=env)
        pos_counts = {str(p): float((positions == p).mean()) for p in np.unique(positions)}
        return StrategyResult(
            stops=strategy,
            label=label_for(strategy),
            p_position=pos_counts,
            finish_time_s=FinishTime(
                p10=float(np.quantile(finish, 0.1)),
                p50=float(np.quantile(finish, 0.5)),
                p90=float(np.quantile(finish, 0.9)),
            ),
            expected_position=float(positions.mean()),
            p_better_than_current=float((positions < current_pos).mean()),
        )

    baseline = result_for([])
    results = [result_for(s) for s in strategies if s]
    results.sort(key=lambda r: r.expected_position)
    return baseline, results
