"""DO_sat core validated against a verbatim USGS DOTABLES reference table.

The fixture in tests/fixtures/ is the literal output of the USGS DOTABLES program
(Benson & Krause equations), pasted unmodified. We assert our numpy implementation
reproduces it to within the table's own 0.01 mg/L rounding, so the test is grounded
on real USGS output rather than numbers typed in from memory.
"""

from pathlib import Path

import numpy as np
import pytest

from eo_waterbody_temp.do_saturation import (
    MODEL_ID,
    VALID_TEMP_C,
    do_saturation_freshwater,
)

FIXTURE = (
    Path(__file__).parent / "fixtures" / "dotables_do_solubility_bk_freshwater.csv"
)

# DOTABLES rounds to 2 decimals, so the true value is within +/-0.005 of what it
# prints. A correct implementation must therefore land within 0.005 mg/L.
ROUNDING_TOL = 0.005


def _load_dotables_760():
    """Parse the DOTABLES fixture -> (temps_C, do_at_760mmHg) as float arrays."""
    temps, do760 = [], []
    for line in FIXTURE.read_text().splitlines():
        parts = line.split(",")
        # data rows look like: "<int temp>,<do760>,<do761>,"
        if len(parts) >= 2 and parts[0].strip().lstrip("-").isdigit():
            temps.append(float(parts[0]))
            do760.append(float(parts[1]))
    return np.array(temps), np.array(do760)


def test_fixture_present_and_shaped():
    temps, do760 = _load_dotables_760()
    assert temps.size == 41  # 0..40 inclusive
    assert temps[0] == 0.0 and temps[-1] == 40.0
    assert do760[0] == pytest.approx(14.62)  # sanity: sea-level 0 C


def test_matches_dotables_within_rounding():
    temps, do760 = _load_dotables_760()
    computed = do_saturation_freshwater(temps)
    max_diff = np.max(np.abs(computed - do760))
    assert (
        max_diff <= ROUNDING_TOL
    ), f"max |diff| = {max_diff:.4f} mg/L exceeds rounding tol {ROUNDING_TOL}"


def test_monotonic_decreasing_in_temperature():
    computed = do_saturation_freshwater(np.arange(0, 41))
    assert np.all(np.diff(computed) < 0)


def test_scalar_input_returns_0d_array():
    out = do_saturation_freshwater(20.0)
    assert out.shape == ()
    assert out == pytest.approx(9.09, abs=ROUNDING_TOL)


def test_nan_passes_through():
    """Masked / no-data pixels stay unknown; they are never turned into a value."""
    out = do_saturation_freshwater(np.array([10.0, np.nan, 25.0]))
    assert np.isnan(out[1])
    assert np.isfinite(out[0]) and np.isfinite(out[2])


def test_preserves_raster_shape():
    grid = np.full((4, 3), 15.0)
    out = do_saturation_freshwater(grid)
    assert out.shape == (4, 3)


@pytest.mark.parametrize("bad", [-0.1, 40.1, np.array([10.0, 45.0])])
def test_out_of_range_raises(bad):
    with pytest.raises(ValueError, match="validity range"):
        do_saturation_freshwater(bad)


def test_out_of_range_ignores_nan():
    # NaN is not "out of range"; a finite in-range value alongside NaN is fine.
    out = do_saturation_freshwater(np.array([np.nan, 20.0]))
    assert np.isnan(out[0]) and np.isfinite(out[1])


def test_model_id_and_range_exposed():
    assert MODEL_ID == "do_sat_bk1984_freshwater_1atm_v1"
    assert VALID_TEMP_C == (0.0, 40.0)
