"""Integration tests for read_ecostress_granule on a synthetic granule.

The synthetic granule has hand-countable geometry (see synthetic_ecostress.py):
AOI = rows 2-6 x cols 2-6 (25 px). Within AOI: water = cols 2-4 (15 px); all rows
2-6 are clear and QC-good. LST is 285.0 K except (4,3)=NaN and (3,2)=272.0 K.
So keep = 15 water px - 1 NaN water px = 14; one of those (272.0 K) is sub-zero degC.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import synthetic_ecostress as syn  # noqa: E402

from eo_waterbody_temp.ecostress import (  # noqa: E402
    EcostressSource,
    find_granule_layers,
    read_ecostress_granule,
)
from eo_waterbody_temp.temperature_source import TemperatureField  # noqa: E402


@pytest.fixture
def granule(tmp_path):
    paths = syn.write_granule(tmp_path / "granule")
    return paths, syn.aoi_4326()


def test_find_granule_layers(granule, tmp_path):
    paths = find_granule_layers(tmp_path / "granule")
    assert set(paths) == {"LST", "SST", "QC", "cloud", "water"}


def test_find_granule_layers_missing_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="LST"):
        find_granule_layers(tmp_path / "empty")


def test_reader_returns_field(granule):
    paths, aoi = granule
    field = read_ecostress_granule(paths, aoi)
    assert isinstance(field, TemperatureField)
    assert field.method == "ecostress_l2t_lste_v003:LST"
    assert field.crs == "EPSG:32610"


def test_reader_known_answer_counts(granule):
    paths, aoi = granule
    field = read_ecostress_granule(paths, aoi)
    d = field.diagnostics
    assert d["aoi_pixel_count"] == 25
    assert d["water_pixel_count"] == 15
    assert d["valid_pixel_count"] == 14  # 15 water - 1 NaN
    assert d["aoi_valid_fraction"] == pytest.approx(14 / 25)
    assert d["qc_pass_fraction"] == pytest.approx(1.0)
    assert d["cloud_fraction_of_observed"] == pytest.approx(0.0)
    assert d["n_subzero_water_px"] == 1


def test_reader_temperatures_in_celsius(granule):
    paths, aoi = granule
    field = read_ecostress_granule(paths, aoi)
    valid = field.data[np.isfinite(field.data)]
    assert valid.size == 14
    # 13 pixels at 285.0 K -> 11.85 degC, 1 pixel at 272.0 K -> -1.15 degC
    assert field.diagnostics["temp_max_c"] == pytest.approx(11.85, abs=1e-4)
    assert field.diagnostics["temp_min_c"] == pytest.approx(-1.15, abs=1e-4)
    assert np.sum(np.isclose(valid, 11.85, atol=1e-4)) == 13


def test_masked_pixels_are_nan_not_zero(granule):
    paths, aoi = granule
    field = read_ecostress_granule(paths, aoi)
    # land/cloud/out-of-AOI pixels must be NaN, never 0.0
    assert not np.any(field.data == 0.0)
    assert np.isnan(field.data).sum() > 0


def test_sst_layer_selects_different_values(granule):
    paths, aoi = granule
    field = read_ecostress_granule(paths, aoi, thermal_layer="SST")
    assert field.method == "ecostress_l2t_lste_v003:SST"
    # SST is 286.0 K everywhere -> 12.85 degC; no NaN/sub-zero seeded in SST
    assert field.diagnostics["temp_max_c"] == pytest.approx(12.85, abs=1e-4)
    assert field.diagnostics["valid_pixel_count"] == 15  # no NaN in SST


def test_accept_degraded_is_recorded(granule):
    paths, aoi = granule
    field = read_ecostress_granule(paths, aoi, accept_degraded=True)
    assert field.diagnostics["accept_degraded_qc"] is True


def test_bad_thermal_layer_raises(granule):
    paths, aoi = granule
    with pytest.raises(ValueError, match="LST.*SST"):
        read_ecostress_granule(paths, aoi, thermal_layer="height")


def test_source_wrapper_from_granule_dir(granule, tmp_path):
    _, aoi = granule
    src = EcostressSource.from_granule_dir(tmp_path / "granule")
    field = src.read(aoi)
    assert field.diagnostics["valid_pixel_count"] == 14


def test_granule_id_and_time_parsed(granule):
    paths, aoi = granule
    field = read_ecostress_granule(paths, aoi)
    assert field.diagnostics["granule_id"] == syn.GRANULE_STEM
    assert field.diagnostics["acquisition_time_utc"] == "2026-02-01T01:42:32+00:00"
