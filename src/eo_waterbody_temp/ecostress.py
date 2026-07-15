"""Read water-surface temperature from an ECOSTRESS L2T LSTE V003 granule.

Grounded on the real granule conventions (verified against
ECOv003_L2T_LSTE_..._10SFH_20260201T014232, LP DAAC, Feb 2026):

- Each layer is a separate Cloud-Optimized GeoTIFF: ``<granule>_<LAYER>.tif``.
- ``LST`` and ``SST`` are float32 **Kelvin**, scale 1.0 / offset 0.0 (no decode),
  no-data = ``NaN``. (So we convert K -> degC by subtraction, nothing else.)
- ``water``: uint8, tag ``"0=land 1=water"`` -> keep where ``water == 1``.
- ``cloud``: uint8, tag ``"1 = cloudy"``, no-data ``255`` -> keep where ``cloud == 0``.
- ``QC``: uint16 bit-packed. Mandatory QA is bits 1-0: ``00`` = produced by TES
  (best), ``01`` = produced but accuracy may be degraded, ``11`` = not produced.
  The QC layer does **not** account for clouds (LP DAAC action-required notice) --
  that is exactly why cloud masking is a separate, independent gate below.
- CRS is a UTM zone (EPSG:326xx). The AOI polygon (EPSG:4326) is reprojected into
  the granule CRS before clipping; we never assume they match.

A pixel survives to a temperature only if it passes all three independent gates
AND lies inside the AOI polygon AND has a finite thermal value. Everything else
is NaN.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .temperature_source import TemperatureField, TemperatureSource

PRODUCT = "ecostress_l2t_lste_v003"

#: Layer name -> filename suffix (before ".tif"). Single-token suffixes only.
LAYER_SUFFIXES = {
    "LST": "LST",
    "SST": "SST",
    "QC": "QC",
    "cloud": "cloud",
    "water": "water",
}

_KELVIN = 273.15
_CLOUD_CLOUDY = 1
_CLOUD_NODATA = 255
_WATER_TRUE = 1
_QC_MANDATORY_MASK = 0b11  # bits 1-0
_QC_GOOD = 0b00  # produced by TES
_QC_DEGRADED = 0b01  # produced by TES, accuracy may be degraded


def kelvin_to_celsius(kelvin: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert Kelvin to degrees Celsius (NaN passes through as NaN).

    This is the one hand-off where a unit slip would silently poison every
    downstream dissolved-oxygen number, so it is its own tested function.
    """
    return np.asarray(kelvin, dtype=np.float64) - _KELVIN


def qc_good_mask(
    qc: NDArray[np.integer], *, accept_degraded: bool = False
) -> NDArray[np.bool_]:
    """Boolean mask of usable pixels from the packed QC layer (bits 1-0).

    With ``accept_degraded=False`` (default) only ``00`` (produced by TES) passes.
    With ``accept_degraded=True`` both ``00`` and ``01`` pass.
    """
    mandatory = np.asarray(qc, dtype=np.uint16) & _QC_MANDATORY_MASK
    if accept_degraded:
        return (mandatory == _QC_GOOD) | (mandatory == _QC_DEGRADED)
    return mandatory == _QC_GOOD


def build_keep_mask(
    *,
    water: NDArray[np.integer],
    cloud: NDArray[np.integer],
    qc: NDArray[np.integer],
    thermal_finite: NDArray[np.bool_],
    inside_aoi: NDArray[np.bool_],
    accept_degraded: bool = False,
) -> NDArray[np.bool_]:
    """Conjunction of the independent gates that keep a pixel's temperature.

    keep = inside AOI AND water AND clear AND QC-good AND finite thermal value.
    """
    is_water = np.asarray(water) == _WATER_TRUE
    is_clear = np.asarray(cloud) == 0  # 0 clear; 1 cloudy; 255 no-observation
    is_good = qc_good_mask(qc, accept_degraded=accept_degraded)
    return inside_aoi & is_water & is_clear & is_good & thermal_finite


def find_granule_layers(granule_dir: str | Path) -> dict[str, Path]:
    """Map a downloaded granule directory to ``{layer_name: path}`` for our layers.

    Raises
    ------
    FileNotFoundError
        If any required layer (LST, SST, QC, cloud, water) is missing.
    """
    granule_dir = Path(granule_dir)
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for name, suffix in LAYER_SUFFIXES.items():
        hits = sorted(granule_dir.glob(f"*_{suffix}.tif"))
        if len(hits) == 1:
            paths[name] = hits[0]
        else:
            missing.append(f"{name} ({len(hits)} matches for *_{suffix}.tif)")
    if missing:
        raise FileNotFoundError(
            f"could not resolve required layers in {granule_dir}: {', '.join(missing)}"
        )
    return paths


def _parse_granule_datetime(lst_path: Path) -> str | None:
    """Best-effort acquisition time from the granule filename (UTC ISO), else None."""
    # ...LSTE_<orbit>_<scene>_<tile>_YYYYMMDDThhmmss_<build>_LST.tif
    for token in lst_path.stem.split("_"):
        if len(token) == 15 and token[8] == "T" and token[:8].isdigit():
            try:
                dt = datetime.strptime(token, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
                return dt.isoformat()
            except ValueError:
                return None
    return None


def read_ecostress_granule(
    layer_paths: dict[str, str | Path],
    aoi: Any,
    *,
    thermal_layer: str = "LST",
    accept_degraded: bool = False,
    aoi_crs: str = "EPSG:4326",
) -> TemperatureField:
    """Read a masked water-surface-temperature field (degrees C) over ``aoi``.

    Parameters
    ----------
    layer_paths:
        ``{"LST":..., "SST":..., "QC":..., "cloud":..., "water":...}`` (local COGs).
    aoi:
        A GeoJSON-like geometry mapping (an ``__geo_interface__`` object works too),
        in ``aoi_crs`` (default EPSG:4326).
    thermal_layer:
        Which thermal layer to use as the temperature -- ``"LST"`` or ``"SST"``.
    accept_degraded:
        If True, keep QC-degraded (``01``) pixels as well as best (``00``).
    """
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.warp import transform_geom
    from rasterio.windows import Window, from_bounds
    from shapely.geometry import mapping, shape

    if thermal_layer not in ("LST", "SST"):
        raise ValueError(f"thermal_layer must be 'LST' or 'SST', got {thermal_layer!r}")

    geom = shape(aoi)  # normalize to a shapely geometry
    thermal_path = Path(layer_paths[thermal_layer])

    with rasterio.open(thermal_path) as ref:
        dst_crs = ref.crs
        # reproject AOI (aoi_crs -> granule CRS); never assume they match
        geom_dst = shape(transform_geom(aoi_crs, dst_crs.to_string(), mapping(geom)))
        minx, miny, maxx, maxy = geom_dst.bounds
        win = from_bounds(minx, miny, maxx, maxy, transform=ref.transform)
        win = win.round_offsets().round_lengths()
        full = Window(0, 0, ref.width, ref.height)
        win = win.intersection(full)
        if win.width <= 0 or win.height <= 0:
            raise ValueError("AOI does not overlap the granule tile")
        win_transform = ref.window_transform(win)
        shape_hw = (int(win.height), int(win.width))

        def read_layer(name: str, dtype: str) -> np.ndarray:
            with rasterio.open(Path(layer_paths[name])) as ds:
                return ds.read(1, window=win).astype(dtype)

        thermal_k = read_layer(thermal_layer, "float64")

    water = read_layer("water", "uint16")
    cloud = read_layer("cloud", "uint16")
    qc = read_layer("QC", "uint16")

    inside_aoi = geometry_mask(
        [mapping(geom_dst)], out_shape=shape_hw, transform=win_transform, invert=True
    )
    thermal_finite = np.isfinite(thermal_k)

    keep = build_keep_mask(
        water=water,
        cloud=cloud,
        qc=qc,
        thermal_finite=thermal_finite,
        inside_aoi=inside_aoi,
        accept_degraded=accept_degraded,
    )

    wst_c = np.where(keep, kelvin_to_celsius(thermal_k), np.nan)

    # diagnostics (all computed within the AOI footprint)
    n_aoi = int(inside_aoi.sum())
    n_valid = int(keep.sum())
    aoi_water = inside_aoi & (water == _WATER_TRUE)
    n_water = int(aoi_water.sum())
    n_cloudy = int((inside_aoi & (cloud == _CLOUD_CLOUDY)).sum())
    n_observed = int((inside_aoi & (cloud != _CLOUD_NODATA)).sum())
    n_qc_good = int(
        (inside_aoi & qc_good_mask(qc, accept_degraded=accept_degraded)).sum()
    )
    valid_temps = wst_c[keep]
    n_subzero_water = int((valid_temps < 0).sum()) if valid_temps.size else 0

    diagnostics: dict[str, Any] = {
        "product": PRODUCT,
        "thermal_layer": thermal_layer,
        "granule_id": thermal_path.stem.rsplit(f"_{thermal_layer}", 1)[0],
        "acquisition_time_utc": _parse_granule_datetime(thermal_path),
        "granule_crs": dst_crs.to_string(),
        "aoi_pixel_count": n_aoi,
        "valid_pixel_count": n_valid,
        "aoi_valid_fraction": (n_valid / n_aoi) if n_aoi else 0.0,
        "water_pixel_count": n_water,
        "water_fraction": (n_water / n_aoi) if n_aoi else 0.0,
        "cloud_fraction_of_observed": (
            (n_cloudy / n_observed) if n_observed else float("nan")
        ),
        "qc_pass_fraction": (n_qc_good / n_aoi) if n_aoi else 0.0,
        "temp_min_c": float(np.nanmin(valid_temps)) if n_valid else float("nan"),
        "temp_max_c": float(np.nanmax(valid_temps)) if n_valid else float("nan"),
        "n_subzero_water_px": n_subzero_water,
        "accept_degraded_qc": accept_degraded,
    }

    return TemperatureField(
        data=wst_c,
        crs=dst_crs.to_string(),
        transform=tuple(win_transform)[:6],
        method=f"{PRODUCT}:{thermal_layer}",
        diagnostics=diagnostics,
    )


class EcostressSource(TemperatureSource):
    """A :class:`TemperatureSource` backed by a local ECOSTRESS V003 granule."""

    def __init__(self, layer_paths: dict[str, str | Path]):
        self.layer_paths = layer_paths

    @classmethod
    def from_granule_dir(cls, granule_dir: str | Path) -> EcostressSource:
        return cls(find_granule_layers(granule_dir))

    def read(self, aoi: Any, **kwargs: Any) -> TemperatureField:
        return read_ecostress_granule(self.layer_paths, aoi, **kwargs)
