"""Regenerate test fixtures from the FastF1 cache.

Saves the RAW fastf1 dataframes (laps/weather/results as parquet, metadata as
json) for two races:
  - 2025 Monza      — clean, dry race (mostly green)
  - 2024 São Paulo  — wet chaos: SC, VSC, red flag, only INTERMEDIATE/WET

Run: uv run python tests/fixtures/make_fixtures.py   (from backend/)
"""

import json
from pathlib import Path

import fastf1

from app import config

FIXTURES = Path(__file__).parent
RACES = [(2025, "Monza", "monza_2025"), (2024, "São Paulo", "sao_paulo_2024")]


def main() -> None:
    config.FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(config.FASTF1_CACHE_DIR))
    for year, name, slug in RACES:
        session = fastf1.get_session(year, name, "R")
        session.load(laps=True, telemetry=False, weather=True, messages=True)
        out = FIXTURES / slug
        out.mkdir(exist_ok=True)
        # drop object-typed columns parquet can't take verbatim? none — all serialise
        session.laps.to_parquet(out / "raw_laps.parquet")
        session.weather_data.to_parquet(out / "raw_weather.parquet")
        session.results.to_parquet(out / "raw_results.parquet")
        meta = {
            "season": year,
            "round": int(session.event["RoundNumber"]),
            "location": str(session.event["Location"]),
            "event_name": str(session.event["EventName"]),
            "country": str(session.event["Country"]),
            "laps_total": int(session.total_laps),
            "start_utc": str(session.date),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"wrote {out} ({len(session.laps)} laps)")


if __name__ == "__main__":
    main()
