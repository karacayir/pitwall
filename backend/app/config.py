"""Central configuration: every magic number in Pitwall lives here.

All physical constants, thresholds and priors below are tunable approximations
unless stated otherwise. Change them here, never inline.
"""

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
DATA_DIR = Path(os.getenv("PITWALL_DATA_DIR", REPO_ROOT / "data"))
FASTF1_CACHE_DIR = DATA_DIR / "fastf1_cache"
MODELS_DIR = Path(os.getenv("MODEL_DIR", REPO_ROOT / "models"))
REPORTS_DIR = REPO_ROOT / "reports"
TRACKS_YAML = BACKEND_DIR / "tracks.yaml"

LAPS_PARQUET = DATA_DIR / "laps.parquet"
WEATHER_PARQUET = DATA_DIR / "weather.parquet"
SESSIONS_PARQUET = DATA_DIR / "sessions.parquet"

# --- Data sources ------------------------------------------------------------
OPENF1_BASE_URL = "https://api.openf1.org/v1"
OPENF1_MQTT_HOST = "mqtt.openf1.org"
OPENF1_MQTT_PORT = 8883
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

FIRST_SEASON = 2019  # no data before 2019 (non-goal)


# --- Env ---------------------------------------------------------------------
def env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


DATA_SOURCE = os.getenv("DATA_SOURCE", "replay")  # replay | live
REPLAY_RACE = os.getenv("REPLAY_RACE", "2025_monza")
REPLAY_SPEED = float(os.getenv("REPLAY_SPEED", "60"))

# --- Fuel correction (approximation; used for reference pace + degradation only)
FUEL_START_KG = 110.0  # ~race fuel load at lights out
FUEL_EFFECT_S_PER_KG_5KM = 0.033  # s per kg per 5.0 km lap; scaled by lap_length_km / 5.0
FUEL_EFFECT_REF_LAP_KM = 5.0

# --- Lap classification / outlier filtering -----------------------------------
OUTLIER_MAD_MULTIPLIER = 3.0  # drop laps above median + 3*MAD within (session, driver, stint)

# --- Reference pace ------------------------------------------------------------
CLEAN_AIR_GAP_S = 2.0  # gap_ahead above this counts as clean air
REF_WINDOW_LAPS = 5  # look back [L-5, L-1]
REF_WINDOW_WIDE_LAPS = 8  # widen to this if < REF_MIN_QUALIFYING_LAPS qualify
REF_MIN_QUALIFYING_LAPS = 8
REF_TOP_DRIVERS = 5  # median of the 5 fastest drivers' best laps
COLD_START_MAX_LAP = 4  # laps < 4 use the cold-start reference
COLD_START_POLE_FACTOR = 1.03  # reference = pole_time * this, absent long-run data
REF_TREND_WINDOW = 10  # linear extrapolation window for k-lap-ahead forecasts
REF_TREND_CLIP_PER_LAP = 0.0005  # clip trend to +/-0.05% per lap

# --- Features ------------------------------------------------------------------
GAP_AHEAD_CAP_S = 5.0
SC_RESTART_LOOKBACK_LAPS = 3  # sc_restart = SC/VSC within last 3 laps

# --- Model training --------------------------------------------------------------
QUANTILES = (0.1, 0.5, 0.9)
LGBM_PARAMS = {
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 3000,
}
EARLY_STOPPING_ROUNDS = 100
RECENCY_HALFLIFE_SEASONS = 2.0  # weight = exp(-age_in_seasons / 2)
CURRENT_SEASON_WEIGHT_MULT = 2.0  # extra weight on current-season laps

# --- Online bias correction -------------------------------------------------------
BIAS_DECAY = 0.85  # b <- 0.85*b + 0.15*(actual_ratio - predicted_ratio)

# --- Degradation curves ------------------------------------------------------------
DEGRADATION_REFIT_EVERY_LAPS = 3
DEGRADATION_RIDGE_ALPHA = 1.0  # ridge strength for quadratic fit
DEGRADATION_PRIOR_SHRINKAGE = 8.0  # pseudo-laps of prior weight vs live laps

# --- Simulator ---------------------------------------------------------------------
SIM_DEFAULT_N = 2000
SC_LAP_FACTOR = 1.4  # under SC all cars lap at ~1.4x reference
SC_PIT_LOSS_FACTOR = 0.55  # pit loss multiplier under SC
TRAFFIC_FOLLOW_WINDOW_S = 1.0  # within 1.0s behind a slower car -> overtake attempt
TRAFFIC_FOLLOW_PENALTY_S = 0.4  # otherwise follows at leader pace + 0.4s
OVERTAKE_LOGISTIC_SCALE = 0.15
SIM_RIVAL_WINDOW_S = 40.0  # simulate rivals within +/-40s of the chosen driver
FORECAST_AHEAD_LAPS = 15  # counterfactual curves 1..15 laps ahead

# --- Forecast / live behaviour --------------------------------------------------------
RAIN_PROB_WIDEN_THRESHOLD = 0.3  # widen quantiles when rain_prob_15m exceeds this
RAIN_WIDEN_FACTOR = 1.5  # multiply P10-P90 half-widths under rain risk
WS_STALENESS_S = 15.0

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
ALL_COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
