"""Phase 0 placeholder tests: config imports and constants are sane."""

from app import config


def test_quantiles_ordered():
    assert config.QUANTILES == (0.1, 0.5, 0.9)


def test_fuel_constants_positive():
    assert config.FUEL_START_KG > 0
    assert config.FUEL_EFFECT_S_PER_KG_5KM > 0


def test_compounds():
    assert set(config.DRY_COMPOUNDS) <= set(config.ALL_COMPOUNDS)
    assert "INTERMEDIATE" in config.ALL_COMPOUNDS


def test_data_source_default():
    assert config.DATA_SOURCE in ("replay", "live")
