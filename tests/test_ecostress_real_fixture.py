"""End-to-end test on a REAL (clipped) ECOSTRESS V003 granule.

The fixture in tests/fixtures/ecostress_real/ is a 193x250 window carved from a
real LP DAAC granule (ECOv003_L2T_LSTE_..._10SFH_20260201T014232), over the
water-rich southern band of the Yolo Bypass AOI. It grounds the reader + DO
coupling on genuine ECOSTRESS bytes -- real CRS, dtypes, NaN fill, packed QC, and
water/cloud conventions -- not just synthetic ones.

Portability note: pixel COUNTS depend on reprojecting the AOI (EPSG:4326 ->
32610) and rasterizing it, which routes through PROJ; a different PROJ/GDAL build
can shift a boundary pixel or two. So counts are asserted with a small tolerance
and via structural invariants (valid <= water <= aoi), while CRS, provenance
strings, and the zero-drop facts are asserted exactly. Temperature / DO values
carry 0.01-0.02 tolerances for the same reason.

Skips (does not fail) if the fixture or AOI polygon is absent, so a shallow
checkout stays green; CI has the fixture committed, so it runs there.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from eo_waterbody_temp.dissolved_oxygen import couple_do_saturation  # noqa: E402
from eo_waterbody_temp.ecostress import (  # noqa: E402
    find_granule_layers,
    read_ecostress_granule,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ecostress_real"
GRANULE_STEM = "ECOv003_L2T_LSTE_42945_013_10SFH_20260201T014232_03"
LST_FIXTURE = FIXTURE_DIR / f"{GRANULE_STEM}_LST.tif"
AOI_PATH = Path(__file__).parents[1] / "data" / "polys" / "yolo-bypass-boundary.geojson"

pytestmark = pytest.mark.skipif(
    not LST_FIXTURE.exists() or not AOI_PATH.exists(),
    reason="real ECOSTRESS fixture and/or AOI polygon not present",
)

# reference counts on the committed fixture (approx: PROJ/GDAL may jitter a pixel)
AOI_PX = 12304
VALID_PX = 3405
WATER_PX = 4046


def _aoi():
    from shapely.geometry import mapping, shape

    geom = shape(json.load(open(AOI_PATH))["features"][0]["geometry"])
    return mapping(geom)


@pytest.fixture
def paths():
    return find_granule_layers(FIXTURE_DIR)


def test_fixture_has_all_layers(paths):
    assert set(paths) == {"LST", "SST", "QC", "cloud", "water"}


def test_real_conventions_crs_and_provenance(paths):
    d = read_ecostress_granule(paths, _aoi()).diagnostics
    assert d["granule_crs"] == "EPSG:32610"  # real granule UTM 10N, exact
    assert d["granule_id"] == GRANULE_STEM
    assert d["acquisition_time_utc"] == "2026-02-01T01:42:32+00:00"


def test_real_lst_structural_invariants(paths):
    d = read_ecostress_granule(paths, _aoi()).diagnostics
    # monotone containment holds regardless of PROJ jitter
    assert 0 < d["valid_pixel_count"] <= d["water_pixel_count"] <= d["aoi_pixel_count"]
    # counts near reference (tolerate a little reprojection jitter)
    assert d["aoi_pixel_count"] == pytest.approx(AOI_PX, rel=0.01)
    assert d["water_pixel_count"] == pytest.approx(WATER_PX, rel=0.02)
    assert d["valid_pixel_count"] == pytest.approx(VALID_PX, rel=0.02)
    assert d["n_subzero_water_px"] == 0  # night bypass water, but not freezing
    assert 0.79 < d["qc_pass_fraction"] < 0.82


def test_real_lst_temperature_range(paths):
    d = read_ecostress_granule(paths, _aoi()).diagnostics
    assert d["temp_min_c"] == pytest.approx(6.44, abs=0.05)
    assert d["temp_max_c"] == pytest.approx(12.29, abs=0.05)


def test_real_lst_to_do(paths):
    do = couple_do_saturation(read_ecostress_granule(paths, _aoi()))
    d = do.diagnostics
    assert d["n_dropped_subzero"] == 0
    assert d["n_dropped_over_max"] == 0
    assert d["do_min_mgl"] == pytest.approx(10.71, abs=0.05)
    assert d["do_max_mgl"] == pytest.approx(12.31, abs=0.05)
    assert d["do_mean_mgl"] == pytest.approx(11.03, abs=0.05)
    assert do.method == "do_sat_bk1984_freshwater_1atm_v1<-ecostress_l2t_lste_v003:LST"


def test_real_sst_vs_lst_divergence(paths):
    """Grades the LST-vs-SST prediction on real bypass water: SST's colder floor
    (water emissivity) yields a higher DO ceiling than LST."""
    lst_do = couple_do_saturation(read_ecostress_granule(paths, _aoi()))
    sst_do = couple_do_saturation(
        read_ecostress_granule(paths, _aoi(), thermal_layer="SST")
    )
    assert sst_do.diagnostics["do_max_mgl"] > lst_do.diagnostics["do_max_mgl"]
    assert sst_do.diagnostics["do_max_mgl"] == pytest.approx(13.18, abs=0.05)


def test_masked_pixels_are_nan_not_zero(paths):
    field = read_ecostress_granule(paths, _aoi())
    assert not np.any(field.data == 0.0)
    assert np.isnan(field.data).sum() > 0
