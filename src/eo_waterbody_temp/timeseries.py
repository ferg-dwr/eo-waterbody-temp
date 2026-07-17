"""Run the WST pipeline across many granules -> a JSON run manifest.

Consumes a set of already-downloaded ECOSTRESS granule directories, runs
reader -> (optional DO coupling) -> COG writer per scene, and aggregates the
per-scene diagnostics into a single JSON manifest: a run-level summary ("what
does the window look like") plus a chronological array of per-scene records.

WST is the default product; pass ``products=("WST", "DO")`` to also emit the DO
map (free from the same field). Reprocessed granules that share a sensing time
are de-duplicated with **latest-processing-wins** (highest processing build),
consistent with the writer's overwrite policy; the winning build's full id is
recorded as ``source_id``. Granules that cannot be read are skipped and recorded
(loud, not silent), so one bad granule never sinks a whole run.

Acquisition (download) is intentionally out of scope here -- this core takes
local directories so it stays hermetic and testable; a thin download helper
lives in examples/.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dissolved_oxygen import couple_do_saturation
from .ecostress import PRODUCT as WST_MODEL_ID
from .ecostress import find_granule_layers, read_ecostress_granule
from .outputs import write_dissolved_oxygen, write_temperature

_VALID_PRODUCTS = ("WST", "DO")


def _granule_stem(lst_path: Path) -> str:
    return lst_path.stem.rsplit("_LST", 1)[0]


def _sensing_and_build(stem: str) -> tuple[str, int]:
    """(sensing 'YYYYMMDDThhmmss', processing build int) parsed from a granule stem."""
    m = re.search(r"_(\d{8}T\d{6})_(\d+)", stem)
    if not m:
        return stem, 0
    return m.group(1), int(m.group(2))


def run_wst_timeseries(
    granule_dirs: Iterable[str | Path],
    aoi: Any,
    outdir: str | Path,
    *,
    products: Sequence[str] = ("WST",),
    thermal_layer: str = "LST",
    accept_degraded: bool = False,
    aoi_crs: str = "EPSG:4326",
    manifest_name: str | None = None,
) -> dict:
    """Run the pipeline over ``granule_dirs`` and write a JSON manifest.

    Returns the manifest dict (also written to ``outdir``).
    """
    bad = [p for p in products if p not in _VALID_PRODUCTS]
    if bad:
        raise ValueError(f"unknown products {bad}; valid: {list(_VALID_PRODUCTS)}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # resolve + dedup by sensing time (latest processing build wins)
    resolved: dict[str, dict] = {}
    skipped: list[dict] = []
    for gdir in granule_dirs:
        gdir = Path(gdir)
        try:
            paths = find_granule_layers(gdir)
        except FileNotFoundError as e:
            skipped.append({"dir": str(gdir), "error": str(e)})
            continue
        stem = _granule_stem(paths["LST"])
        sensing, build = _sensing_and_build(stem)
        prev = resolved.get(sensing)
        if prev is None or build > prev["build"]:
            resolved[sensing] = {"dir": gdir, "paths": paths, "build": build}

    scenes: list[dict] = []
    for sensing in sorted(resolved):
        item = resolved[sensing]
        try:
            wst = read_ecostress_granule(
                item["paths"],
                aoi,
                thermal_layer=thermal_layer,
                accept_degraded=accept_degraded,
                aoi_crs=aoi_crs,
            )
        except ValueError as e:  # e.g. AOI does not overlap tile
            skipped.append({"dir": str(item["dir"]), "error": str(e)})
            continue

        d = wst.diagnostics
        record: dict[str, Any] = {
            "acquisition_time_utc": d.get("acquisition_time_utc"),
            "source_id": d.get("granule_id"),
            "wst_file": None,
            "do_file": None,
            "aoi_pixel_count": d.get("aoi_pixel_count"),
            "valid_pixel_count": d.get("valid_pixel_count"),
            "water_pixel_count": d.get("water_pixel_count"),
            "aoi_valid_fraction": d.get("aoi_valid_fraction"),
            "cloud_fraction_of_observed": d.get("cloud_fraction_of_observed"),
            "temp_min_c": d.get("temp_min_c"),
            "temp_max_c": d.get("temp_max_c"),
        }
        if "WST" in products:
            cog, _ = write_temperature(wst, outdir)
            record["wst_file"] = cog.name
        if "DO" in products:
            do = couple_do_saturation(wst)
            cog, _ = write_dissolved_oxygen(do, outdir)
            record["do_file"] = cog.name
            record["do_mean_mgl"] = do.diagnostics.get("do_mean_mgl")
        scenes.append(record)

    # run-level summary
    acqs = [s["acquisition_time_utc"] for s in scenes if s["acquisition_time_utc"]]
    with_water = [s for s in scenes if (s.get("valid_pixel_count") or 0) > 0]
    temp_mins = [s["temp_min_c"] for s in with_water if s["temp_min_c"] is not None]
    temp_maxs = [s["temp_max_c"] for s in with_water if s["temp_max_c"] is not None]

    try:
        from . import __version__
    except Exception:
        __version__ = "unknown"

    manifest = {
        "run": {
            "aoi": "yolo",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "model_id": WST_MODEL_ID,
            "thermal_layer": thermal_layer,
            "products": list(products),
            "scene_count": len(scenes),
            "date_range_utc": [min(acqs), max(acqs)] if acqs else [None, None],
            "scenes_with_water": len(with_water),
            "window_temp_min_c": min(temp_mins) if temp_mins else None,
            "window_temp_max_c": max(temp_maxs) if temp_maxs else None,
            "skipped_count": len(skipped),
            "software": {"repo": "eo-waterbody-temp", "version": __version__},
        },
        "scenes": scenes,
        "skipped": skipped,
    }

    if manifest_name is None:
        if acqs:
            d0 = min(acqs)[:10].replace("-", "")
            d1 = max(acqs)[:10].replace("-", "")
            manifest_name = f"yolo_WST_timeseries_{thermal_layer}_{d0}_{d1}.json"
        else:
            manifest_name = f"yolo_WST_timeseries_{thermal_layer}_empty.json"
    (outdir / manifest_name).write_text(json.dumps(manifest, indent=2, default=str))
    manifest["_manifest_path"] = str(outdir / manifest_name)
    return manifest
