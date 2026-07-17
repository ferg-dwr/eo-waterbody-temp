"""Round-trip, filename, sidecar, and cross-repo schema tests for the COG writer."""

import json
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")  # [io] extra

from eo_waterbody_temp.dissolved_oxygen import couple_do_saturation  # noqa: E402
from eo_waterbody_temp.outputs import (  # noqa: E402
    NODATA,
    validate_sidecar_schema,
    write_dissolved_oxygen,
    write_geotiff,
    write_temperature,
)
from eo_waterbody_temp.temperature_source import TemperatureField  # noqa: E402

TRANSFORM = (70.0, 0.0, 535000.0, 0.0, -70.0, 4265000.0)
GRANULE = "ECOv003_L2T_LSTE_42945_013_10SFH_20260201T014232_03"
CROSS_REPO = (
    Path(__file__).parent / "fixtures" / "cross_repo" / "volume_sample_sidecar.json"
)


def _wst(data):
    return TemperatureField(
        data=np.asarray(data, dtype=np.float64),
        crs="EPSG:32610",
        transform=TRANSFORM,
        method="ecostress_l2t_lste_v003:LST",
        diagnostics={
            "granule_id": GRANULE,
            "acquisition_time_utc": "2026-02-01T01:42:32+00:00",
            "aoi_pixel_count": 4,
            "valid_pixel_count": 3,
            "temp_min_c": 6.44,
        },
    )


def test_filename_scheme_wst(tmp_path):
    cog, side = write_temperature(_wst([[10.0, 11.0], [12.0, np.nan]]), tmp_path)
    assert (
        cog.name
        == "yolo_WST_ecostress_l2t_lste_v003_T10SFH_20260201T014232Z_LST.tif"
    )
    assert side.name.endswith(".json")


def test_filename_scheme_do(tmp_path):
    do = couple_do_saturation(_wst([[10.0, 11.0], [12.0, np.nan]]))
    cog, _ = write_dissolved_oxygen(do, tmp_path)
    assert cog.name == (
        "yolo_DO_do_sat_bk1984_freshwater_1atm_v1_T10SFH_20260201T014232Z_LST.tif"
    )


def test_nan_masked_pixels_become_sentinel(tmp_path):
    data = np.array([[11.85, np.nan], [6.44, 12.29]], dtype=np.float64)
    cog, _ = write_temperature(_wst(data), tmp_path)
    with rasterio.open(cog) as ds:
        assert ds.nodata == NODATA
        rt = ds.read(1)
    assert rt[0, 1] == pytest.approx(NODATA)  # NaN -> -9999
    finite = np.array([[11.85, NODATA], [6.44, 12.29]], dtype=np.float32)
    np.testing.assert_allclose(rt, finite, rtol=0, atol=1e-4)


def test_is_cog_float32_with_band_and_units(tmp_path):
    cog, _ = write_temperature(_wst([[10.0, 11.0], [12.0, 13.0]]), tmp_path)
    with rasterio.open(cog) as ds:
        assert ds.profile["tiled"] is True
        assert ds.dtypes[0] == "float32"
        assert ds.descriptions == ("water_surface_temperature",)
        assert ds.units == ("degC",)


def test_crs_transform_survive(tmp_path):
    cog, _ = write_temperature(_wst([[10.0, 11.0], [12.0, 13.0]]), tmp_path)
    with rasterio.open(cog) as ds:
        assert ds.crs.to_string() == "EPSG:32610"
        assert tuple(ds.transform)[:6] == TRANSFORM


def test_tags_carry_model_and_source(tmp_path):
    cog, _ = write_temperature(_wst([[10.0, 11.0], [12.0, 13.0]]), tmp_path)
    with rasterio.open(cog) as ds:
        t = ds.tags()
    assert t["model_id"] == "ecostress_l2t_lste_v003"
    assert t["variant"] == "LST"
    assert t["nodata_value"] == "-9999.0"
    assert t["source_id"] == GRANULE
    assert t["software_repo"] == "eo-waterbody-temp"
    assert t["diag_valid_pixel_count"] == "3"


def test_sidecar_has_structured_nodata_and_software(tmp_path):
    _, side = write_temperature(_wst([[10.0, 11.0], [12.0, 13.0]]), tmp_path)
    sc = json.loads(side.read_text())
    assert sc["nodata"]["value"] == NODATA
    assert "meaning" in sc["nodata"] and sc["nodata"]["meaning"]
    assert sc["software"]["repo"] == "eo-waterbody-temp"
    assert sc["software"]["version"]
    assert sc["source_id"] == GRANULE
    assert sc["model_id"] == "ecostress_l2t_lste_v003"
    assert sc["diagnostics"]["valid_pixel_count"] == 3


def test_our_sidecar_passes_shared_schema(tmp_path):
    _, side = write_temperature(_wst([[10.0, 11.0], [12.0, 13.0]]), tmp_path)
    sc = json.loads(side.read_text())
    assert validate_sidecar_schema(sc) == []


def test_cross_repo_volume_sidecar_passes_shared_schema():
    """The sibling repo's sidecar must satisfy the shared contract too.

    Swap this fixture for the volume repo's real emitted sidecar once it ships;
    schema drift on either side then breaks CI here.
    """
    sc = json.loads(CROSS_REPO.read_text())
    assert validate_sidecar_schema(sc) == []


def test_write_geotiff_both_ways_signature(tmp_path):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    p1 = write_geotiff(
        data, path=tmp_path / "a.tif", crs="EPSG:32610", transform=TRANSFORM
    )
    with rasterio.open(p1) as ds:
        assert ds.crs.to_string() == "EPSG:32610"

    class _Like:
        pass

    like = _Like()
    like.data = data
    like.crs = "EPSG:32610"
    like.transform = TRANSFORM
    p2 = write_geotiff(data, path=tmp_path / "b.tif", like=like)
    with rasterio.open(p2) as ds:
        assert ds.crs.to_string() == "EPSG:32610"


def test_write_geotiff_requires_georef(tmp_path):
    with pytest.raises(ValueError, match="like= OR both crs"):
        write_geotiff(np.zeros((2, 2)), path=tmp_path / "c.tif")


def test_shape_mismatch_with_like_rejected(tmp_path):
    class _Like:
        pass

    like = _Like()
    like.data = np.zeros((4, 5))
    like.crs = "EPSG:32610"
    like.transform = TRANSFORM
    with pytest.raises(ValueError, match="does not match"):
        write_geotiff(np.zeros((3, 3)), path=tmp_path / "d.tif", like=like)
