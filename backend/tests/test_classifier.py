"""Lap classifier + fuel correction + outlier-filter tests.

Fixture counts were hand-verified against the raw fastf1 dataframes by
independent recomputation (see git history / CLAUDE.md): pit flags from
PitInTime/PitOutTime, statuses from TrackStatus digit codes.
"""

import polars as pl
import pytest

from features.build import classify_laps, mark_outliers, with_fuel_corrected

# lap_class -> expected count, verified against raw fixture data
MONZA_EXPECTED = {"green": 897, "in": 20, "lap1": 19, "out": 19, "yellow": 19}
SAO_PAULO_EXPECTED = {
    "green": 800, "in": 36, "lap1": 18, "out": 31, "sc": 93, "vsc": 13, "yellow": 143,
}  # fmt: skip


@pytest.mark.parametrize(
    ("race", "expected"),
    [("monza", MONZA_EXPECTED), ("sao_paulo", SAO_PAULO_EXPECTED)],
)
def test_classifier_golden_counts(race, expected, request):
    laps = classify_laps(request.getfixturevalue(race)["laps"])
    counts = dict(laps["lap_class"].value_counts().iter_rows())
    assert counts == expected
    assert sum(counts.values()) == len(laps)  # every lap classified


def test_classifier_precedence():
    """A pit-in lap under SC is 'in'; lap 1 always 'lap1'; red beats sc."""
    frame = pl.DataFrame(
        {
            "lap_number": [1, 10, 11, 12, 13, 14],
            "pit_in": [False, True, False, False, False, False],
            "pit_out": [False, False, True, False, False, False],
            "track_status": ["1", "4", "4", "45", "67", None],
        }
    )
    got = classify_laps(frame)["lap_class"].to_list()
    assert got == ["lap1", "in", "out", "red", "vsc", "green"]


def test_fuel_correction_exact():
    laps = pl.DataFrame({"lap_number": [1, 53], "lap_time_s": [90.0, 85.0]})
    out = with_fuel_corrected(laps, laps_total=53, lap_length_km=5.0)
    # lap 53: tank empty -> zero correction
    assert out["fuel_correction_s"][1] == pytest.approx(0.0)
    # lap 1: 110*(52/53) kg * 0.033 s/kg = 3.6034 s
    assert out["fuel_correction_s"][0] == pytest.approx(110 * 52 / 53 * 0.033, abs=1e-6)
    assert out["fuel_corrected_s"][0] == pytest.approx(90.0 - 110 * 52 / 53 * 0.033, abs=1e-6)


def test_outlier_filter_golden():
    frame = pl.DataFrame(
        {
            "session_id": ["s"] * 5,
            "driver_number": [1] * 5,
            "stint": [1] * 5,
            "lap_number": [2, 3, 4, 5, 6],
            "lap_time_s": [90.0, 90.1, 90.2, 95.0, 99.0],
            "lap_class": ["green", "green", "green", "green", "sc"],
        }
    )
    out = mark_outliers(frame)
    # median 90.15, MAD 0.125 -> threshold 90.525: the 95.0 green lap is out;
    # the 99.0 lap is slower still but non-green so never marked.
    assert out["is_outlier"].to_list() == [False, False, False, True, False]


def test_outlier_share_sane(monza):
    laps = mark_outliers(classify_laps(monza["laps"]))
    green = laps.filter(pl.col("lap_class") == "green")
    share = green["is_outlier"].mean()
    assert 0.0 < share < 0.15  # some traffic laps cut, but not a purge
