"""Generate backend/tracks.yaml.

Static fields (lat/lon/lengths/subjective ratings) come from the table below
(hand-maintained, sensible defaults, `# TODO refine` where guessed). Computed
fields come from the consolidated parquet:

  pit_loss_s        median over pit stops of (in-lap + out-lap) - 2x that
                    driver's median green lap in the same race
  sc_hazard_per_lap share of historical laps run under SC or VSC at the track

Run after `make data`: uv run python -m ingest.tracks_build
"""

import logging
import sys

import polars as pl
import yaml

from app import config
from features.build import classify_laps

log = logging.getLogger("pitwall.tracks")

# track_id: (lat, lon, laps_total, lap_length_km, overtake_difficulty 0-1, abrasiveness 1-5)
# subjective ratings are documented approximations  # TODO refine
STATIC: dict[str, tuple[float, float, int, float, float, float]] = {
    "melbourne": (-37.8497, 144.9680, 58, 5.278, 0.60, 2.5),
    "sakhir": (26.0325, 50.5106, 57, 5.412, 0.30, 4.0),
    "shanghai": (31.3389, 121.2200, 56, 5.451, 0.40, 3.0),
    "baku": (40.3725, 49.8533, 51, 6.003, 0.35, 2.0),
    "barcelona": (41.5700, 2.2611, 66, 4.657, 0.65, 4.0),
    "monaco": (43.7347, 7.4206, 78, 3.337, 0.95, 1.0),
    "montreal": (45.5000, -73.5228, 70, 4.361, 0.40, 2.0),
    "le_castellet": (43.2506, 5.7917, 53, 5.842, 0.55, 3.0),
    "spielberg": (47.2197, 14.7647, 71, 4.318, 0.40, 3.0),
    "silverstone": (52.0786, -1.0169, 52, 5.891, 0.45, 4.0),
    "hockenheim": (49.3278, 8.5658, 67, 4.574, 0.50, 3.0),
    "budapest": (47.5789, 19.2486, 70, 4.381, 0.80, 2.5),
    "spa_francorchamps": (50.4372, 5.9714, 44, 7.004, 0.35, 4.0),
    "monza": (45.6156, 9.2811, 53, 5.793, 0.30, 3.0),
    "singapore": (1.2914, 103.8640, 62, 4.940, 0.85, 2.0),
    "sochi": (43.4057, 39.9578, 53, 5.848, 0.50, 1.5),
    "suzuka": (34.8431, 136.5410, 53, 5.807, 0.55, 5.0),
    "mexico_city": (19.4042, -99.0907, 71, 4.304, 0.50, 2.0),
    "austin": (30.1328, -97.6411, 56, 5.513, 0.45, 3.5),
    "sao_paulo": (-23.7036, -46.6997, 71, 4.309, 0.30, 3.0),
    "yas_island": (24.4672, 54.6031, 58, 5.281, 0.50, 2.5),
    "mugello": (43.9975, 11.3719, 59, 5.245, 0.70, 4.5),
    "nurburgring": (50.3356, 6.9475, 60, 5.148, 0.50, 3.0),
    "portimao": (37.2270, -8.6267, 66, 4.653, 0.50, 2.5),
    "imola": (44.3439, 11.7167, 63, 4.909, 0.75, 3.0),
    "istanbul": (40.9517, 29.4050, 58, 5.338, 0.45, 4.5),
    "jeddah": (21.6319, 39.1044, 50, 6.174, 0.40, 2.0),
    "lusail": (25.4900, 51.4542, 57, 5.419, 0.50, 4.5),
    "miami": (25.9581, -80.2389, 57, 5.412, 0.45, 3.0),
    "zandvoort": (52.3888, 4.5409, 72, 4.259, 0.75, 3.5),
    "las_vegas": (36.1147, -115.1730, 50, 6.201, 0.30, 1.5),
    "madrid": (40.4700, -3.5800, 57, 5.470, 0.60, 3.0),
}
DEFAULT_PIT_LOSS_S = 22.0  # typical modern F1 pit loss  # TODO refine
DEFAULT_SC_HAZARD = 0.0035  # ~20% chance over a 60-lap race  # TODO refine


def compute_pit_loss(laps: pl.DataFrame) -> pl.DataFrame:
    """Median over (in-lap + out-lap) - 2x driver's median green lap, per track."""
    df = classify_laps(laps)
    green_med = (
        df.filter((pl.col("lap_class") == "green") & pl.col("lap_time_s").is_not_null())
        .group_by(["session_id", "driver_number"])
        .agg(pl.col("lap_time_s").median().alias("green_med"))
    )
    stops = (
        df.sort(["session_id", "driver_number", "lap_number"])
        .with_columns(
            pl.col("lap_time_s")
            .shift(-1)
            .over(["session_id", "driver_number"])
            .alias("next_lap_s"),
            pl.col("pit_out").shift(-1).over(["session_id", "driver_number"]).alias("next_is_out"),
        )
        .filter(
            pl.col("pit_in")
            & pl.col("next_is_out").fill_null(False)
            & pl.col("lap_time_s").is_not_null()
            & pl.col("next_lap_s").is_not_null()
        )
        .join(green_med, on=["session_id", "driver_number"])
        .with_columns(
            (pl.col("lap_time_s") + pl.col("next_lap_s") - 2 * pl.col("green_med")).alias("loss")
        )
        # SC-window stops and red-flag anomalies distort the loss; keep a sane band
        .filter(pl.col("loss").is_between(5, 60))
    )
    return stops.group_by("track_id").agg(
        pl.col("loss").median().round(2).alias("pit_loss_s"),
        pl.len().alias("n_stops"),
    )


def compute_sc_hazard(laps: pl.DataFrame) -> pl.DataFrame:
    """Share of (session, lap_number) track-laps run under SC or VSC."""
    df = classify_laps(laps)
    lap_status = df.group_by(["track_id", "session_id", "lap_number"]).agg(
        pl.col("lap_class").is_in(["sc", "vsc"]).any().alias("sc")
    )
    return lap_status.group_by("track_id").agg(
        pl.col("sc").mean().round(5).alias("sc_hazard_per_lap"),
        pl.len().alias("n_track_laps"),
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    laps = pl.read_parquet(config.LAPS_PARQUET)
    sessions = pl.read_parquet(config.SESSIONS_PARQUET)

    pit = {r["track_id"]: r for r in compute_pit_loss(laps).iter_rows(named=True)}
    sc = {r["track_id"]: r for r in compute_sc_hazard(laps).iter_rows(named=True)}
    # typical race distance: laps_total by most recent season at that track
    latest = (
        sessions.sort(["season", "round"])
        .group_by("track_id", maintain_order=True)
        .agg(pl.col("laps_total").last())
    )
    laps_total = {r["track_id"]: r["laps_total"] for r in latest.iter_rows(named=True)}

    tracks: dict[str, dict] = {}
    for track_id in sorted(set(sessions["track_id"].to_list())):
        static = STATIC.get(track_id)
        if static is None:
            log.warning("UNKNOWN track %s — stubbed, refine by hand", track_id)
            static = (0.0, 0.0, laps_total.get(track_id, 60), 5.0, 0.5, 3.0)
        lat, lon, laps_default, length, overtake, abrasive = static
        tracks[track_id] = {
            "lat": lat,
            "lon": lon,
            "laps_total": int(laps_total.get(track_id, laps_default)),
            "lap_length_km": length,
            "pit_loss_s": float(pit.get(track_id, {}).get("pit_loss_s", DEFAULT_PIT_LOSS_S)),
            "sc_hazard_per_lap": float(
                sc.get(track_id, {}).get("sc_hazard_per_lap", DEFAULT_SC_HAZARD)
            ),
            "overtake_difficulty": overtake,
            "abrasiveness": abrasive,
        }

    header = (
        "# Generated by ingest/tracks_build.py — static fields hand-maintained there,\n"
        "# pit_loss_s + sc_hazard_per_lap computed from data/laps.parquet.\n"
        "# overtake_difficulty and abrasiveness are documented approximations. TODO refine.\n"
    )
    config.TRACKS_YAML.write_text(header + yaml.safe_dump({"tracks": tracks}, sort_keys=True))
    log.info("wrote %s (%d tracks)", config.TRACKS_YAML, len(tracks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
