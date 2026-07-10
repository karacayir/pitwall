"""RaceEngine: the live race state store + forecast pipeline.

One engine instance per session. Both the replay harness (ingest/replay.py)
and the live OpenF1 client (ingest/openf1.py) feed the SAME entry points:

    on_lap(row)      a completed driver lap (laps-parquet field names)
    on_weather(row)  a weather sample

so replay exercises exactly the live code path. Each completed lap returns a
schemas.LapUpdate snapshot for the WebSocket/backtest to consume.
"""

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from app import config, schemas
from features.build import classify_lap_scalar, extrapolate_reference, reference_series
from models.degradation import DegradationCurve, fit_curve
from models.predict import OnlineBias, PaceModel, counterfactual_frame

_FLAG_ORDER = [("5", "red"), ("4", "sc"), ("6", "vsc"), ("7", "vsc"), ("2", "yellow")]


def status_flag(track_status: str | None) -> str:
    status = track_status or "1"
    for code, flag in _FLAG_ORDER:
        if code in status:
            return flag
    return "green"


@dataclass
class SessionMeta:
    session_id: str
    track_id: str
    laps_total: int
    lap_length_km: float
    abrasiveness: float
    season: int
    pole_time_s: float | None = None
    event_name: str | None = None
    pit_loss_s: float = 22.0
    sc_hazard_per_lap: float = 0.0035
    overtake_difficulty: float = 0.5


@dataclass
class DriverState:
    driver_number: int
    driver_code: str | None = None
    team_id: str | None = None
    position: int | None = None
    compound: str | None = None
    tyre_life: int | None = None
    stint: int = 1
    last_lap_s: float | None = None
    lap_number: int = 0
    gap_ahead_s: float | None = None
    pitted: bool = False  # pit_in seen on the most recent lap
    forecast: schemas.Forecast | None = None
    # (predicted_raw_p50_ratio, reference_used) for the next lap, for bias updates
    pending: tuple[float, float] | None = None


class LiveDegradation:
    """Per-compound quadratic fits on this race's clean laps, shrunk to the
    artifact's (track, compound) priors; refit every 3 race laps."""

    def __init__(self, track_id: str, priors: dict):
        self.track_id = track_id
        self.priors = priors
        self.points: dict[str, list[tuple[float, float]]] = {}
        self.curves: dict[str, DegradationCurve] = {}

    def prior_for(self, compound: str) -> DegradationCurve | None:
        raw = self.priors.get(f"{self.track_id}/{compound}") or self.priors.get(
            f"__global__/{compound}"
        )
        if raw is None:
            return None
        return DegradationCurve(raw["c0"], raw["c1"], raw["c2"], raw.get("n", 0))

    def add_point(self, compound: str, tyre_age: float, fc_ratio: float) -> None:
        self.points.setdefault(compound, []).append((tyre_age, fc_ratio))

    def refit(self) -> None:
        for compound in set(self.points) | {c for c in config.DRY_COMPOUNDS}:
            pts = self.points.get(compound, [])
            self.curves[compound] = fit_curve(
                np.array([p[0] for p in pts]),
                np.array([p[1] for p in pts]),
                prior=self.prior_for(compound),
                prior_weight=config.DEGRADATION_PRIOR_SHRINKAGE,
                ridge_alpha=config.DEGRADATION_RIDGE_ALPHA,
            )

    def curve(self, compound: str) -> DegradationCurve:
        got = self.curves.get(compound)
        if got is None:
            got = self.prior_for(compound) or DegradationCurve(1.02, 0.002, 0.0, 0)
            self.curves[compound] = got
        return got


@dataclass
class _EnrichedLap:
    driver_number: int
    lap_number: int
    lap_class: str
    clean_air: bool
    fuel_corrected_s: float | None


class RaceEngine:
    def __init__(self, meta: SessionMeta, model: PaceModel | None = None):
        self.meta = meta
        self.model = model
        self.bias = OnlineBias()
        self.degradation = LiveDegradation(meta.track_id, model.degradation_priors if model else {})
        self.drivers: dict[int, DriverState] = {}
        self.enriched: list[_EnrichedLap] = []
        self.weather: dict | None = None
        self.rain_prob_15m: float | None = None
        self.track_status: str = "1"
        self.max_lap: int = 0
        self._last_crossing_t: float | None = None
        self._sc_laps: set[int] = set()  # race laps that saw SC/VSC
        self._refs: dict[int, float] = {}
        self._ref_history: list[float] = []
        self._last_refit_lap = 0

    # --- event entry points -------------------------------------------------

    def on_weather(self, row: dict) -> None:
        self.weather = row

    def on_rain_prob(self, prob: float) -> None:
        self.rain_prob_15m = prob

    def on_lap(self, row: dict) -> schemas.LapUpdate | None:
        """Process one completed driver lap; returns a broadcast snapshot."""
        d = self.drivers.setdefault(
            int(row["driver_number"]), DriverState(int(row["driver_number"]))
        )
        lap_number = int(row["lap_number"])
        lap_class = classify_lap_scalar(
            lap_number, bool(row.get("pit_in")), bool(row.get("pit_out")), row.get("track_status")
        )
        if row.get("track_status"):
            self.track_status = str(row["track_status"])
        if lap_class in ("sc", "vsc"):
            self._sc_laps.add(lap_number)

        # gap to the car physically ahead = time since the previous line crossing
        t = row.get("time_session_s")
        gap = None
        if t is not None and self._last_crossing_t is not None:
            gap = float(t) - self._last_crossing_t
        if t is not None:
            self._last_crossing_t = max(self._last_crossing_t or 0.0, float(t))

        fuel_corrected = None
        if row.get("lap_time_s") is not None:
            fuel_kg = config.FUEL_START_KG * (1 - lap_number / self.meta.laps_total)
            per_kg = config.FUEL_EFFECT_S_PER_KG_5KM * (
                self.meta.lap_length_km / config.FUEL_EFFECT_REF_LAP_KM
            )
            fuel_corrected = float(row["lap_time_s"]) - fuel_kg * per_kg

        clean_air = gap is None or gap > config.CLEAN_AIR_GAP_S
        self.enriched.append(
            _EnrichedLap(d.driver_number, lap_number, lap_class, clean_air, fuel_corrected)
        )
        self.max_lap = max(self.max_lap, lap_number)
        # the reference is core state, model or not (also feeds degradation + UI)
        self.ref_for(min(lap_number + 1, self.meta.laps_total))

        # driver state
        d.driver_code = row.get("driver_code") or d.driver_code
        d.team_id = row.get("team_id") or d.team_id
        d.position = row.get("position") or d.position
        d.compound = row.get("compound") or d.compound
        d.tyre_life = row.get("tyre_life")
        d.stint = int(row.get("stint") or d.stint)
        d.last_lap_s = row.get("lap_time_s")
        d.lap_number = lap_number
        d.gap_ahead_s = gap
        d.pitted = bool(row.get("pit_in"))

        # online bias: compare the arrived green lap against what we predicted
        if d.pending is not None and lap_class == "green" and row.get("lap_time_s"):
            pred_ratio, ref_used = d.pending
            self.bias.update(d.driver_number, float(row["lap_time_s"]) / ref_used, pred_ratio)
        d.pending = None

        # degradation points on clean green laps
        ref_here = self._refs.get(lap_number)
        if (
            lap_class == "green"
            and clean_air
            and fuel_corrected is not None
            and ref_here
            and d.compound
            and d.tyre_life is not None
        ):
            self.degradation.add_point(
                d.compound, float(d.tyre_life - 1), fuel_corrected / ref_here
            )
        if self.max_lap - self._last_refit_lap >= config.DEGRADATION_REFIT_EVERY_LAPS:
            self.degradation.refit()
            self._last_refit_lap = self.max_lap

        self._forecast_driver(d)
        return self.snapshot()

    # --- reference ------------------------------------------------------------

    def _enriched_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "driver_number": [e.driver_number for e in self.enriched],
                "lap_number": [e.lap_number for e in self.enriched],
                "lap_class": [e.lap_class for e in self.enriched],
                "clean_air": [e.clean_air for e in self.enriched],
                "fuel_corrected_s": [e.fuel_corrected_s for e in self.enriched],
            }
        )

    def ref_for(self, lap: int) -> float | None:
        """Reference pace for lap L using only completed laps < L (cached)."""
        if lap in self._refs:
            return self._refs[lap]
        series = reference_series(self._enriched_frame(), self.meta.pole_time_s, max_lap=lap)
        ref = series["reference_s"][lap - 1]
        if ref is not None and math.isfinite(ref):
            self._refs[lap] = float(ref)
            self._ref_history.append(float(ref))
            return float(ref)
        return None

    @property
    def raining(self) -> bool:
        wet_flag = bool(self.weather and self.weather.get("rainfall"))
        prob_high = (self.rain_prob_15m or 0.0) > config.RAIN_PROB_WIDEN_THRESHOLD
        return wet_flag or prob_high

    # --- forecasting ------------------------------------------------------------

    def _base_features(self, d: DriverState, ref_next: float) -> dict | None:
        if d.compound is None or d.tyre_life is None:
            return None
        next_lap = d.lap_number + 1
        w = self.weather or {}
        recent = range(max(1, next_lap - config.SC_RESTART_LOOKBACK_LAPS), next_lap)
        return {
            "compound": d.compound,
            "tyre_age": float(d.tyre_life),  # age at the START of the next lap
            "tyre_age_sq": float(d.tyre_life) ** 2,
            "stint_no": float(d.stint),
            "race_progress": next_lap / self.meta.laps_total,
            "laps_total": float(self.meta.laps_total),
            "track_id": self.meta.track_id,
            "abrasiveness": self.meta.abrasiveness,
            "track_temp": w.get("track_temp"),
            "air_temp": w.get("air_temp"),
            "humidity": w.get("humidity"),
            "rainfall_flag": bool(w.get("rainfall", False)),
            "wind_speed": w.get("wind_speed"),
            "driver_id": d.driver_code,
            "team_id": d.team_id,
            "season": float(self.meta.season),
            "gap_ahead_s": min(d.gap_ahead_s, config.GAP_AHEAD_CAP_S)
            if d.gap_ahead_s is not None
            else None,
            "clean_air": d.gap_ahead_s is None or d.gap_ahead_s > config.CLEAN_AIR_GAP_S,
            "sc_restart": any(lap in self._sc_laps for lap in recent),
            "position": float(d.position) if d.position else None,
            "reference_pace_s": ref_next,
        }

    def _forecast_driver(self, d: DriverState) -> None:
        if self.model is None or d.lap_number >= self.meta.laps_total:
            return
        ref_next = self.ref_for(d.lap_number + 1)
        if ref_next is None:
            return
        base = self._base_features(d, ref_next)
        if base is None:
            return

        frame, index = counterfactual_frame(base, laps_ahead=config.FORECAST_AHEAD_LAPS)
        groups = [
            f"{ix['kind']}/{ix.get('compound', '')}" for ix in index
        ]  # each scenario is an ascending-age sweep
        preds = self.model.predict_quantiles(frame, age_groups=groups)
        preds = preds + self.bias.get(d.driver_number)
        if self.raining:
            mid = preds[:, 1:2]
            preds = np.column_stack(
                [
                    mid[:, 0] - (mid[:, 0] - preds[:, 0]) * config.RAIN_WIDEN_FACTOR,
                    mid[:, 0],
                    mid[:, 0] + (preds[:, 2] - mid[:, 0]) * config.RAIN_WIDEN_FACTOR,
                ]
            )

        refs_k = [
            extrapolate_reference(self._ref_history, k, rain=self.raining)
            for k in range(1, config.FORECAST_AHEAD_LAPS + 1)
        ]

        forecast = schemas.Forecast()
        ahead: dict[str, list[float]] = {}
        for i, ix in enumerate(index):
            k = ix["k"]
            ref_k = refs_k[k - 1]
            q = schemas.Quantiles(
                p10=round(preds[i, 0] * ref_k, 3),
                p50=round(preds[i, 1] * ref_k, 3),
                p90=round(preds[i, 2] * ref_k, 3),
            )
            if ix["kind"] == "current":
                ahead.setdefault("current", []).append(q.p50)
                if k == 1:
                    forecast.current = q
                    # raw model ratio (bias excluded) for the next bias update
                    raw_p50 = float(preds[i, 1]) - self.bias.get(d.driver_number)
                    d.pending = (raw_p50, ref_k)
            else:
                compound = ix["compound"]
                ahead.setdefault(compound, []).append(q.p50)
                if k == 1:
                    forecast.fresh[compound] = q
        forecast.ahead = ahead
        d.forecast = forecast

    # --- snapshot ------------------------------------------------------------------

    def snapshot(self) -> schemas.LapUpdate:
        current_lap = min(self.max_lap + 1, self.meta.laps_total)
        w = self.weather or {}
        drivers = sorted(
            self.drivers.values(), key=lambda d: (d.position is None, d.position or 99)
        )
        return schemas.LapUpdate(
            session_key=self.meta.session_id,
            lap=current_lap,
            laps_total=self.meta.laps_total,
            track_status=status_flag(self.track_status),
            reference_pace_s=self._refs.get(current_lap)
            or (self._ref_history[-1] if self._ref_history else None),
            weather=schemas.WeatherInfo(
                track_temp=w.get("track_temp"),
                air_temp=w.get("air_temp"),
                humidity=w.get("humidity"),
                rain=bool(w.get("rainfall", False)),
                rain_prob_15m=self.rain_prob_15m,
            ),
            drivers=[
                schemas.DriverUpdate(
                    driver_number=d.driver_number,
                    driver_code=d.driver_code,
                    position=d.position,
                    gap_ahead_s=round(d.gap_ahead_s, 3) if d.gap_ahead_s is not None else None,
                    compound=d.compound,
                    tyre_age=int(d.tyre_life - 1) if d.tyre_life is not None else None,
                    stint_no=d.stint,
                    last_lap_s=d.last_lap_s,
                    lap_number=d.lap_number,
                    pitted=d.pitted,
                    bias_s=round(self.bias.get(d.driver_number) * (self._ref_history[-1] or 0), 3)
                    if self._ref_history
                    else None,
                    forecast=d.forecast,
                )
                for d in drivers
            ],
        )
