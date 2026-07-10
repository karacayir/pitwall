"""Historical data pull: FastF1 -> parquet.

Downloads every completed race 2019..current season, converts to a fixed schema,
and persists per-race parquet files under ``data/raw/<session_id>/`` (resumable),
then consolidates into ``data/laps.parquet``, ``data/weather.parquet``,
``data/sessions.parquet``, ``data/results.parquet``.

Schemas (all times in seconds as float64 unless noted):

laps.parquet — one row per driver-lap:
    session_id        str   "2025_16_monza" (season_round_trackslug)
    season            i32
    round             i32
    track_id          str   slugified event location ("monza", "sao_paulo")
    driver_number     i32
    driver_code       str   "VER"
    team_id           str   slugified team name ("red_bull_racing")
    lap_number        i32
    stint             i32?  fastf1 Stint (nullable)
    lap_time_s        f64?  null for laps without a timed lap
    lap_start_session_s f64?  session clock at lap start
    time_session_s    f64?  session clock at lap end (line crossing)
    sector1_s/2_s/3_s f64?
    pit_in            bool  lap ends in the pit lane (PitInTime set)
    pit_out           bool  lap starts from the pit lane (PitOutTime set)
    compound          str?  SOFT/MEDIUM/HARD/INTERMEDIATE/WET (fastf1 values kept verbatim)
    tyre_life         i32?  laps on this tyre set incl. this one
    fresh_tyre        bool?
    track_status      str?  concatenated digit codes seen during the lap
                            (1 green, 2 yellow, 4 SC, 5 red, 6 VSC, 7 VSC-ending)
    position          i32?  position at lap end
    deleted           bool
    is_accurate       bool  fastf1 lap-accuracy flag
    fastf1_generated  bool

weather.parquet — one row per weather sample:
    session_id str, time_session_s f64, air_temp f64, track_temp f64,
    humidity f64, pressure f64, rainfall bool, wind_speed f64, wind_direction i32

sessions.parquet — one row per race:
    session_id str, season i32, round i32, track_id str, event_name str,
    country str, location str, laps_total i32, start_utc str (ISO),
    pole_time_s f64? (fastest quali lap; cold-start reference scale)

results.parquet — one row per driver per race:
    session_id str, driver_number i32, driver_code str, full_name str,
    team_id str, team_name str, team_color str, grid_position i32?,
    finish_position i32?, classified_position str?, status str?, points f64?,
    laps_completed i32?
"""

import argparse
import datetime as dt
import json
import logging
import sys
import time
import unicodedata

import fastf1
import pandas as pd
import polars as pl

from app import config

log = logging.getLogger("pitwall.ingest.fastf1")

RAW_DIR = config.DATA_DIR / "raw"

LAPS_SCHEMA: dict[str, pl.DataType] = {
    "session_id": pl.String,
    "season": pl.Int32,
    "round": pl.Int32,
    "track_id": pl.String,
    "driver_number": pl.Int32,
    "driver_code": pl.String,
    "team_id": pl.String,
    "lap_number": pl.Int32,
    "stint": pl.Int32,
    "lap_time_s": pl.Float64,
    "lap_start_session_s": pl.Float64,
    "time_session_s": pl.Float64,
    "sector1_s": pl.Float64,
    "sector2_s": pl.Float64,
    "sector3_s": pl.Float64,
    "pit_in": pl.Boolean,
    "pit_out": pl.Boolean,
    "compound": pl.String,
    "tyre_life": pl.Int32,
    "fresh_tyre": pl.Boolean,
    "track_status": pl.String,
    "position": pl.Int32,
    "deleted": pl.Boolean,
    "is_accurate": pl.Boolean,
    "fastf1_generated": pl.Boolean,
}

WEATHER_SCHEMA: dict[str, pl.DataType] = {
    "session_id": pl.String,
    "time_session_s": pl.Float64,
    "air_temp": pl.Float64,
    "track_temp": pl.Float64,
    "humidity": pl.Float64,
    "pressure": pl.Float64,
    "rainfall": pl.Boolean,
    "wind_speed": pl.Float64,
    "wind_direction": pl.Int32,
}

SESSIONS_SCHEMA: dict[str, pl.DataType] = {
    "session_id": pl.String,
    "season": pl.Int32,
    "round": pl.Int32,
    "track_id": pl.String,
    "event_name": pl.String,
    "country": pl.String,
    "location": pl.String,
    "laps_total": pl.Int32,
    "start_utc": pl.String,
    "pole_time_s": pl.Float64,
}

RESULTS_SCHEMA: dict[str, pl.DataType] = {
    "session_id": pl.String,
    "driver_number": pl.Int32,
    "driver_code": pl.String,
    "full_name": pl.String,
    "team_id": pl.String,
    "team_name": pl.String,
    "team_color": pl.String,
    "grid_position": pl.Int32,
    "finish_position": pl.Int32,
    "classified_position": pl.String,
    "status": pl.String,
    "points": pl.Float64,
    "laps_completed": pl.Int32,
}


def slugify(text: str) -> str:
    """ "São Paulo" -> "sao_paulo"; stable across seasons."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return "_".join("".join(c if c.isalnum() else " " for c in ascii_text).lower().split())


def make_session_id(season: int, rnd: int, location: str) -> str:
    return f"{season}_{rnd:02d}_{slugify(location)}"


def _td_seconds(series: pd.Series) -> pd.Series:
    return series.dt.total_seconds()


def convert_laps(
    laps: pd.DataFrame, session_id: str, season: int, rnd: int, track_id: str
) -> pl.DataFrame:
    """fastf1 Laps dataframe -> LAPS_SCHEMA polars frame."""
    out = pd.DataFrame(
        {
            "session_id": session_id,
            "season": season,
            "round": rnd,
            "track_id": track_id,
            "driver_number": pd.to_numeric(laps["DriverNumber"], errors="coerce"),
            "driver_code": laps["Driver"].astype(str),
            "team_id": laps["Team"].astype(str).map(slugify),
            "lap_number": laps["LapNumber"],
            "stint": laps["Stint"],
            "lap_time_s": _td_seconds(laps["LapTime"]),
            "lap_start_session_s": _td_seconds(laps["LapStartTime"]),
            "time_session_s": _td_seconds(laps["Time"]),
            "sector1_s": _td_seconds(laps["Sector1Time"]),
            "sector2_s": _td_seconds(laps["Sector2Time"]),
            "sector3_s": _td_seconds(laps["Sector3Time"]),
            "pit_in": laps["PitInTime"].notna(),
            "pit_out": laps["PitOutTime"].notna(),
            "compound": laps["Compound"].astype("string").str.upper(),
            "tyre_life": laps["TyreLife"],
            "fresh_tyre": laps["FreshTyre"],
            "track_status": laps["TrackStatus"].astype("string"),
            "position": laps["Position"],
            "deleted": laps["Deleted"].fillna(False).astype(bool),
            "is_accurate": laps["IsAccurate"].fillna(False).astype(bool),
            "fastf1_generated": laps["FastF1Generated"].fillna(False).astype(bool),
        }
    )
    return pl.from_pandas(out).cast(LAPS_SCHEMA).sort(["driver_number", "lap_number"])  # type: ignore[arg-type]


def convert_weather(weather: pd.DataFrame, session_id: str) -> pl.DataFrame:
    out = pd.DataFrame(
        {
            "session_id": session_id,
            "time_session_s": _td_seconds(weather["Time"]),
            "air_temp": weather["AirTemp"],
            "track_temp": weather["TrackTemp"],
            "humidity": weather["Humidity"],
            "pressure": weather["Pressure"],
            "rainfall": weather["Rainfall"].astype(bool),
            "wind_speed": weather["WindSpeed"],
            "wind_direction": weather["WindDirection"],
        }
    )
    return pl.from_pandas(out).cast(WEATHER_SCHEMA).sort("time_session_s")  # type: ignore[arg-type]


def convert_results(results: pd.DataFrame, session_id: str) -> pl.DataFrame:
    out = pd.DataFrame(
        {
            "session_id": session_id,
            "driver_number": pd.to_numeric(results["DriverNumber"], errors="coerce"),
            "driver_code": results["Abbreviation"].astype(str),
            "full_name": results["FullName"].astype(str),
            "team_id": results["TeamName"].astype(str).map(slugify),
            "team_name": results["TeamName"].astype(str),
            "team_color": results["TeamColor"].astype(str),
            "grid_position": results["GridPosition"],
            "finish_position": results["Position"],
            "classified_position": results["ClassifiedPosition"].astype("string"),
            "status": results["Status"].astype("string"),
            "points": results["Points"],
            "laps_completed": results["Laps"],
        }
    )
    return pl.from_pandas(out).cast(RESULTS_SCHEMA).sort("driver_number")  # type: ignore[arg-type]


def pole_time_s(year: int, rnd: int) -> float | None:
    """Fastest lap set in the GP qualifying session (any of Q1/Q2/Q3)."""
    try:
        q = fastf1.get_session(year, rnd, "Q")
        q.load(laps=False, telemetry=False, weather=False, messages=False)
        res = q.results
        best = pd.concat([res["Q1"], res["Q2"], res["Q3"]]).dropna()
        if best.empty:
            return None
        return float(best.min().total_seconds())
    except Exception as exc:  # noqa: BLE001 — pole is optional, never fail the pull
        log.warning("pole time unavailable for %s round %s: %s", year, rnd, exc)
        return None


def pull_race(year: int, rnd: int, location: str, with_pole: bool = True) -> str | None:
    """Download one race, write per-race parquets. Returns session_id or None."""
    session_id = make_session_id(year, rnd, location)
    race_dir = RAW_DIR / session_id
    if (race_dir / "_complete").exists():
        log.info("%s already pulled, skipping", session_id)
        return session_id

    session = None
    for attempt in range(1, 4):
        try:
            session = fastf1.get_session(year, rnd, "R")
            session.load(laps=True, telemetry=False, weather=True, messages=True)
            break
        except Exception as exc:  # noqa: BLE001 — retry then give up on this race
            log.warning("load failed (%s round %s attempt %d): %s", year, rnd, attempt, exc)
            if attempt == 3:
                return None
            time.sleep(5 * attempt)
    assert session is not None

    if session.laps is None or len(session.laps) == 0:
        log.warning("%s has no laps, skipping", session_id)
        return None

    track_id = slugify(location)
    race_dir.mkdir(parents=True, exist_ok=True)

    convert_laps(session.laps, session_id, year, rnd, track_id).write_parquet(
        race_dir / "laps.parquet"
    )
    convert_weather(session.weather_data, session_id).write_parquet(race_dir / "weather.parquet")
    convert_results(session.results, session_id).write_parquet(race_dir / "results.parquet")

    meta = {
        "session_id": session_id,
        "season": year,
        "round": rnd,
        "track_id": track_id,
        "event_name": str(session.event["EventName"]),
        "country": str(session.event["Country"]),
        "location": str(location),
        "laps_total": int(session.total_laps),
        "start_utc": str(session.date),
        "pole_time_s": pole_time_s(year, rnd) if with_pole else None,
    }
    (race_dir / "session.json").write_text(json.dumps(meta, indent=2))
    (race_dir / "_complete").touch()
    log.info("pulled %s (%d laps)", session_id, len(session.laps))
    return session_id


def completed_races(year: int) -> list[tuple[int, str]]:
    """(round, location) for races that have already happened."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    out = []
    for _, ev in schedule.iterrows():
        race_date = ev["Session5DateUtc"]
        if pd.notna(race_date) and race_date + pd.Timedelta(hours=4) < now:
            out.append((int(ev["RoundNumber"]), str(ev["Location"])))
    return out


def consolidate() -> dict[str, int]:
    """Concatenate per-race parquets into the top-level training files."""
    race_dirs = sorted(d for d in RAW_DIR.iterdir() if (d / "_complete").exists())
    laps, weather, results, sessions = [], [], [], []
    for d in race_dirs:
        laps.append(pl.read_parquet(d / "laps.parquet"))
        weather.append(pl.read_parquet(d / "weather.parquet"))
        results.append(pl.read_parquet(d / "results.parquet"))
        sessions.append(json.loads((d / "session.json").read_text()))

    pl.concat(laps).write_parquet(config.LAPS_PARQUET)
    pl.concat(weather).write_parquet(config.WEATHER_PARQUET)
    pl.concat(results).write_parquet(config.DATA_DIR / "results.parquet")
    sessions_df = pl.from_dicts(sessions, schema=SESSIONS_SCHEMA).sort(["season", "round"])
    sessions_df.write_parquet(config.SESSIONS_PARQUET)
    counts = {"races": len(race_dirs), "laps": sum(len(x) for x in laps)}
    log.info("consolidated %(races)d races, %(laps)d laps", counts)
    return counts


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Pull historical F1 race data to parquet")
    current_year = dt.date.today().year
    parser.add_argument(
        "--seasons", type=int, nargs="*", default=list(range(config.FIRST_SEASON, current_year + 1))
    )
    parser.add_argument("--consolidate-only", action="store_true")
    args = parser.parse_args()

    config.FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(config.FASTF1_CACHE_DIR))

    if not args.consolidate_only:
        for year in args.seasons:
            try:
                races = completed_races(year)
            except Exception as exc:  # noqa: BLE001 — a missing season must not kill the pull
                log.error("schedule unavailable for %s: %s", year, exc)
                continue
            log.info("season %s: %d completed races", year, len(races))
            for rnd, location in races:
                pull_race(year, rnd, location)
                time.sleep(1.0)  # be polite to the API

    counts = consolidate()
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
