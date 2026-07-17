"""Write temperature / DO fields to self-describing COGs (+ JSON sidecars).

Implements the cross-repo output contract shared with eo-water-volume-estimator:

- Cloud-Optimized GeoTIFF, ``overview_resampling`` per product (``average`` for
  our continuous WST/DO; the parameter exists so a future categorical product can
  pick ``nearest``).
- float32 on disk by default (float64 override); ``deflate`` compressed.
- nodata sentinel ``-9999.0`` declared in the header (masked pixels are converted
  from NaN to the sentinel on write). The sidecar states the sentinel's *meaning*,
  not just its number.
- Filename: ``yolo_<PRODUCT>_<model_id>_T<tile>_<sensingUTC>Z[_<variant>].tif``
  (UTC sensing time; latest-processing-wins overwrite; full source_id incl.
  processing build preserved in the sidecar).
- JSON sidecar next to each raster (consumers often don't read GeoTIFF tags) with
  the shared minimum keys, structured ``nodata`` object, and a ``software`` block
  so every file is self-citing. GeoTIFF tags mirror the scalar keys.

rasterio is imported lazily, so importing this module needs no ``[io]`` extra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from . import __version__
from .models import (
    model_id_from_method,
    registry_entry,
    variant_from_method,
)

REPO_NAME = "eo-waterbody-temp"
NODATA = -9999.0

#: Minimum sidecar keys guaranteed by the shared cross-repo output contract.
#: Both sibling repos emit these; a fixture of the *other* repo's sidecar is
#: validated in CI so schema drift breaks the build on both sides.
SHARED_SIDECAR_KEYS = (
    "file",
    "aoi",
    "product",
    "description",
    "units",
    "model_id",
    "method",
    "crs",
    "nodata",
    "source_id",
    "acquisition_time_utc",
    "software",
    "diagnostics",
)


def validate_sidecar_schema(sidecar: dict) -> list[str]:
    """Return the shared-contract keys missing from ``sidecar`` ([] if compliant).

    Also checks the two structured fields the contract requires: ``nodata`` must
    carry ``value`` + ``meaning``, and ``software`` must carry ``repo`` + ``version``.
    """
    missing = [k for k in SHARED_SIDECAR_KEYS if k not in sidecar]
    nd = sidecar.get("nodata")
    if isinstance(nd, dict):
        missing += [f"nodata.{k}" for k in ("value", "meaning") if k not in nd]
    sw = sidecar.get("software")
    if isinstance(sw, dict):
        missing += [f"software.{k}" for k in ("repo", "version") if k not in sw]
    return missing


_NODATA_MEANING = (
    "masked: land, cloud, poor QC, outside AOI polygon, or water temperature "
    "outside the model's valid domain (dropped)"
)


@runtime_checkable
class _Field(Protocol):
    data: Any
    crs: str
    transform: tuple[float, float, float, float, float, float]
    method: str
    diagnostics: dict[str, Any]


def _resolve_granule_id(d: dict[str, Any]) -> str | None:
    return d.get("granule_id") or d.get("source_granule_id")


def _resolve_acq_time(d: dict[str, Any]) -> str:
    return str(
        d.get("acquisition_time_utc") or d.get("source_acquisition_time_utc") or ""
    )


def _parse_tile_and_time(granule_id: str | None) -> tuple[str, str]:
    import re

    if not granule_id:
        return "UNKtile", "UNKtime"
    tile = re.search(r"_(\d{2}[A-Z]{3})_", granule_id)
    acq = re.search(r"_(\d{8}T\d{6})_", granule_id)
    return (
        tile.group(1) if tile else "UNKtile",
        acq.group(1) if acq else "UNKtime",
    )


def _build_filename(field: _Field, product: str) -> str:
    model_id = model_id_from_method(field.method)
    variant = variant_from_method(field.method)
    tile, acq = _parse_tile_and_time(_resolve_granule_id(field.diagnostics))
    name = f"yolo_{product}_{model_id}_T{tile}_{acq}Z"
    if variant:
        name += f"_{variant}"
    return name + ".tif"


def write_geotiff(
    data: np.ndarray,
    *,
    path: str | Path,
    like: Any | None = None,
    crs: Any | None = None,
    transform: Any | None = None,
    nodata: float | None = NODATA,
    dtype: str = "float32",
    band_name: str | None = None,
    units: str | None = None,
    tags: dict | None = None,
    overview_resampling: str = "average",
) -> str:
    """Write ``data`` as a single-band COG.

    Georef via ``like`` OR ``crs`` + ``transform``.

    ``like`` is any object exposing ``.data``/``.crs``/``.transform`` (e.g. the
    volume repo's ``Raster``); alternatively pass ``crs`` and ``transform``
    directly. If a finite ``nodata`` is given, non-finite pixels (NaN) are
    converted to it on write so the sentinel is consistent with the header.
    """
    import rasterio

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D grid, got shape {data.shape}")

    if like is not None:
        crs = like.crs
        transform = like.transform
        if data.shape != like.data.shape:
            raise ValueError(
                f"Grid shape {data.shape} does not match reference {like.data.shape}"
            )
    if crs is None or transform is None:
        raise ValueError("provide like= OR both crs= and transform=")

    if nodata is not None and np.isfinite(nodata):
        data = np.where(np.isfinite(data), data, nodata)

    if not hasattr(transform, "a"):
        transform = rasterio.Affine(*transform)

    profile = {
        "driver": "COG",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "DEFLATE",
        "overview_resampling": overview_resampling,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)
        if band_name is not None:
            dst.set_band_description(1, band_name)
        if units is not None:
            dst.units = (units,)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return str(path)


def _write_field(field: _Field, outdir: str | Path, product: str) -> tuple[Path, Path]:
    entry = registry_entry(model_id_from_method(field.method))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = _build_filename(field, product)
    cog_path = outdir / fname
    sidecar_path = cog_path.with_suffix(".json")

    band_name = {
        "WST": "water_surface_temperature",
        "DO": "dissolved_oxygen_saturation",
    }[product]

    sidecar = {
        "file": fname,
        "aoi": "yolo",
        "product": product,
        "description": entry["description"],
        "units": entry["units"],
        "model_id": model_id_from_method(field.method),
        "method": field.method,
        "variant": variant_from_method(field.method),
        "crs": field.crs,
        "nodata": {"value": NODATA, "meaning": _NODATA_MEANING},
        "source_id": str(_resolve_granule_id(field.diagnostics) or ""),
        "acquisition_time_utc": _resolve_acq_time(field.diagnostics),
        "software": {"repo": REPO_NAME, "version": __version__},
        "diagnostics": field.diagnostics,
    }

    # GeoTIFF tags mirror the scalar sidecar keys; diagnostics flattened to diag_*
    tags = {
        "aoi": "yolo",
        "product": product,
        "model_id": sidecar["model_id"],
        "method": field.method,
        "variant": sidecar["variant"],
        "units": entry["units"],
        "nodata_value": NODATA,
        "nodata_meaning": _NODATA_MEANING,
        "source_id": sidecar["source_id"],
        "acquisition_time_utc": sidecar["acquisition_time_utc"],
        "software_repo": REPO_NAME,
        "software_version": __version__,
    }
    for k, v in field.diagnostics.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            tags[f"diag_{k}"] = v

    write_geotiff(
        field.data,
        path=cog_path,
        crs=field.crs,
        transform=field.transform,
        nodata=NODATA,
        dtype="float32",
        band_name=band_name,
        units=entry["units"],
        tags=tags,
        overview_resampling="average",  # WST/DO are continuous
    )
    sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str))
    return cog_path, sidecar_path


def write_temperature(field: _Field, outdir: str | Path) -> tuple[Path, Path]:
    """Write a water-surface-temperature field (primary product) as COG + sidecar."""
    return _write_field(field, outdir, "WST")


def write_dissolved_oxygen(field: _Field, outdir: str | Path) -> tuple[Path, Path]:
    """Write a dissolved-oxygen field (secondary product) as COG + sidecar."""
    return _write_field(field, outdir, "DO")
