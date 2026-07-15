"""Tests for coupling a WST field to a DO-saturation field."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import synthetic_ecostress as syn  # noqa: E402

from eo_waterbody_temp.dissolved_oxygen import (  # noqa: E402
    DissolvedOxygenField,
    couple_do_saturation,
)
from eo_waterbody_temp.ecostress import read_ecostress_granule  # noqa: E402
from eo_waterbody_temp.temperature_source import TemperatureField  # noqa: E402


def _wst_field(data):
    return TemperatureField(
        data=np.asarray(data, dtype=float),
        crs="EPSG:32610",
        transform=(70.0, 0.0, 535000.0, 0.0, -70.0, 4265000.0),
        method="ecostress_l2t_lste_v003:LST",
        diagnostics={
            "aoi_pixel_count": 4,
            "granule_id": "ECOv003_L2T_LSTE_TEST",
            "acquisition_time_utc": "2026-02-01T01:42:32+00:00",
        },
    )


def test_drops_subzero_and_over_max_counts_them():
    # 5.0 in-range; -2.0 sub-zero; 45.0 over-max; NaN already masked
    field = _wst_field([[5.0, -2.0], [45.0, np.nan]])
    do = couple_do_saturation(field)
    assert isinstance(do, DissolvedOxygenField)
    d = do.diagnostics
    assert d["n_dropped_subzero"] == 1
    assert d["n_dropped_over_max"] == 1
    assert d["valid_pixel_count"] == 1
    # only the 5.0 degC pixel survives -> DO_sat(5 C) ~ 12.77 mg/L
    assert do.data[0, 0] == pytest.approx(12.77, abs=0.01)
    assert np.isnan(do.data[0, 1])  # sub-zero dropped
    assert np.isnan(do.data[1, 0])  # over-max dropped
    assert np.isnan(do.data[1, 1])  # already masked


def test_dropped_and_masked_are_nan_never_zero():
    field = _wst_field([[5.0, -2.0], [45.0, np.nan]])
    do = couple_do_saturation(field)
    assert not np.any(do.data == 0.0)


def test_method_carries_source_provenance():
    field = _wst_field([[10.0, 10.0], [10.0, 10.0]])
    do = couple_do_saturation(field)
    assert do.method == "do_sat_bk1984_freshwater_1atm_v1<-ecostress_l2t_lste_v003:LST"
    assert do.diagnostics["source_granule_id"] == "ECOv003_L2T_LSTE_TEST"
    assert do.diagnostics["source_acquisition_time_utc"] == "2026-02-01T01:42:32+00:00"


def test_valid_mask_property():
    field = _wst_field([[5.0, -2.0], [10.0, np.nan]])
    do = couple_do_saturation(field)
    assert do.valid_mask.tolist() == [[True, False], [True, False]]


def test_do_stats_present():
    field = _wst_field([[0.0, 20.0], [40.0, 10.0]])
    do = couple_do_saturation(field)
    d = do.diagnostics
    assert d["valid_pixel_count"] == 4
    # DO_sat rises as temp falls: min at 40 C (~6.41), max at 0 C (~14.62)
    assert d["do_min_mgl"] == pytest.approx(6.41, abs=0.01)
    assert d["do_max_mgl"] == pytest.approx(14.62, abs=0.01)
    assert d["do_min_mgl"] < d["do_mean_mgl"] < d["do_max_mgl"]


def test_all_masked_input_gives_empty_do():
    field = _wst_field([[np.nan, np.nan], [np.nan, np.nan]])
    do = couple_do_saturation(field)
    assert do.diagnostics["valid_pixel_count"] == 0
    assert np.isnan(do.diagnostics["do_mean_mgl"])
    assert np.all(np.isnan(do.data))


def test_end_to_end_from_reader_lst(tmp_path):
    """Reader -> couple, on the synthetic granule (known-answer)."""
    paths = syn.write_granule(tmp_path / "granule")
    aoi = syn.aoi_4326()
    wst = read_ecostress_granule(paths, aoi)  # 13 px @ 11.85 C, 1 px @ -1.15 C
    do = couple_do_saturation(wst)
    d = do.diagnostics
    assert d["n_dropped_subzero"] == 1  # the -1.15 C water pixel
    assert d["n_dropped_over_max"] == 0
    assert d["valid_pixel_count"] == 13
    # DO_sat(11.85 C) ~ 10.81 mg/L (between table 11.03@11C and 10.78@12C)
    assert d["do_mean_mgl"] == pytest.approx(10.81, abs=0.02)
    assert d["do_min_mgl"] == pytest.approx(d["do_max_mgl"])  # all same temp


def test_end_to_end_from_reader_sst(tmp_path):
    paths = syn.write_granule(tmp_path / "granule")
    aoi = syn.aoi_4326()
    wst = read_ecostress_granule(paths, aoi, thermal_layer="SST")  # 15 px @ 12.85 C
    do = couple_do_saturation(wst)
    assert do.diagnostics["n_dropped_subzero"] == 0
    assert do.diagnostics["valid_pixel_count"] == 15
