"""Pure masking / unit tests for the ECOSTRESS reader (no file I/O)."""

import numpy as np
import pytest

from eo_waterbody_temp.ecostress import (
    build_keep_mask,
    kelvin_to_celsius,
    qc_good_mask,
)


def test_kelvin_to_celsius_value_and_nan():
    out = kelvin_to_celsius(np.array([273.15, 285.0, np.nan]))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(11.85)
    assert np.isnan(out[2])


def test_qc_good_mask_strict():
    qc = np.array([0, 1, 2, 3, 49856, 49857], dtype=np.uint16)
    # bits 1-0: 0->00 good, 1->01, 2->10, 3->11, 49856->00 good, 49857->01
    strict = qc_good_mask(qc)
    assert strict.tolist() == [True, False, False, False, True, False]


def test_qc_good_mask_accept_degraded():
    qc = np.array([0, 1, 2, 3, 49856, 49857], dtype=np.uint16)
    lenient = qc_good_mask(qc, accept_degraded=True)
    assert lenient.tolist() == [True, True, False, False, True, True]


def test_build_keep_mask_conjunction():
    water = np.array([[1, 1], [1, 0]])
    cloud = np.array([[0, 1], [0, 0]])  # (0,1) cloudy -> dropped
    qc = np.array([[0, 0], [3, 0]])  # (1,0) not-produced -> dropped
    thermal_finite = np.array([[True, True], [True, True]])
    inside = np.array([[True, True], [True, True]])
    keep = build_keep_mask(
        water=water,
        cloud=cloud,
        qc=qc,
        thermal_finite=thermal_finite,
        inside_aoi=inside,
    )
    # only (0,0) passes all gates: water=1, clear, qc-good, finite, inside
    assert keep.tolist() == [[True, False], [False, False]]


def test_build_keep_mask_respects_aoi_and_finite():
    ones = np.ones((2, 2), dtype=int)
    keep = build_keep_mask(
        water=ones,
        cloud=np.zeros((2, 2), int),
        qc=np.zeros((2, 2), int),
        thermal_finite=np.array([[True, False], [True, True]]),
        inside_aoi=np.array([[True, True], [False, True]]),
    )
    # (0,1) dropped by finite; (1,0) dropped by AOI
    assert keep.tolist() == [[True, False], [False, True]]
