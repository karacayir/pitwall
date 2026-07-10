"""Pydantic v2 schemas for the WebSocket/REST contracts (see brief)."""

from pydantic import BaseModel, Field


class Quantiles(BaseModel):
    p10: float
    p50: float
    p90: float


class Forecast(BaseModel):
    current: Quantiles | None = None
    fresh: dict[str, Quantiles] = Field(default_factory=dict)  # SOFT/MEDIUM/HARD
    ahead: dict[str, list[float]] = Field(default_factory=dict)  # p50 curves per scenario


class WeatherInfo(BaseModel):
    track_temp: float | None = None
    air_temp: float | None = None
    humidity: float | None = None
    rain: bool = False
    rain_prob_15m: float | None = None


class DriverUpdate(BaseModel):
    driver_number: int
    driver_code: str | None = None
    position: int | None = None
    gap_ahead_s: float | None = None
    compound: str | None = None
    tyre_age: int | None = None
    stint_no: int | None = None
    last_lap_s: float | None = None
    lap_number: int | None = None
    pitted: bool = False
    bias_s: float | None = None
    forecast: Forecast | None = None


class LapUpdate(BaseModel):
    type: str = "lap_update"
    session_key: str
    lap: int
    laps_total: int
    track_status: str  # green | sc | vsc | yellow | red
    reference_pace_s: float | None = None
    weather: WeatherInfo = Field(default_factory=WeatherInfo)
    drivers: list[DriverUpdate] = Field(default_factory=list)


class SessionInfo(BaseModel):
    session_id: str
    event_name: str | None = None
    track_id: str
    laps_total: int
    data_source: str  # replay | live
    replay_speed: float | None = None


class StintPlan(BaseModel):
    lap: int  # pit at the end of this lap
    compound: str


class SimulateRequest(BaseModel):
    driver_number: int
    strategies: list[list[StintPlan]] | None = None
    n_sims: int = Field(default=2000, ge=100, le=20000)


class FinishTime(BaseModel):
    p10: float
    p50: float
    p90: float


class StrategyResult(BaseModel):
    stops: list[StintPlan]
    label: str
    p_position: dict[str, float]
    finish_time_s: FinishTime
    expected_position: float
    p_better_than_current: float | None = None


class SimulateResponse(BaseModel):
    driver_number: int
    from_lap: int
    n_sims: int
    baseline: StrategyResult
    strategies: list[StrategyResult]
