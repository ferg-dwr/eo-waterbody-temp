"""Carve a small real-granule fixture (water-rich band) + run the full chain.

Writes a trimmed multi-layer fixture to tests/fixtures/ecostress_real (committed)
and keeps the full strip in data/scratch. Auto-selects the densest water rows so
the fixture exercises every mask gate. Prints the diagnostics the fixture test
will assert against.
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import mapping, shape

from eo_waterbody_temp.dissolved_oxygen import couple_do_saturation
from eo_waterbody_temp.ecostress import (
    LAYER_SUFFIXES,
    find_granule_layers,
    read_ecostress_granule,
)

GRANULE_DIR = Path("data/scratch/granule")
AOI_PATH = Path("data/polys/yolo-bypass-boundary.geojson")
FIXTURE_DIR = Path("tests/fixtures/ecostress_real")
FIXTURE_ROWS = 250  # trimmed height

layer_paths = find_granule_layers(GRANULE_DIR)
aoi = shape(json.load(open(AOI_PATH))["features"][0]["geometry"])

# --- carve window around AOI (full strip, in granule CRS) --------------------
with rasterio.open(layer_paths["LST"]) as ref:
    dst_crs = ref.crs
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", dst_crs, *aoi.bounds)
    pad = 700.0
    full_win = from_bounds(
        minx - pad, miny - pad, maxx + pad, maxy + pad, ref.transform
    )
    full_win = full_win.round_offsets().round_lengths()
    full_win = full_win.intersection(Window(0, 0, ref.width, ref.height))

    # find the water-rich band: densest FIXTURE_ROWS-row run in the water layer
    with rasterio.open(layer_paths["water"]) as wds:
        water_strip = wds.read(1, window=full_win)
    water_per_row = (water_strip == 1).sum(axis=1)
    if water_strip.shape[0] > FIXTURE_ROWS:
        csum = np.concatenate([[0], np.cumsum(water_per_row)])
        band_counts = csum[FIXTURE_ROWS:] - csum[:-FIXTURE_ROWS]
        row_off = int(np.argmax(band_counts))
        rows = FIXTURE_ROWS
    else:
        row_off, rows = 0, water_strip.shape[0]

    fix_win = Window(full_win.col_off, full_win.row_off + row_off, full_win.width, rows)
    fix_transform = ref.window_transform(fix_win)
    water_px = int(water_per_row[row_off : row_off + rows].sum())
    print(
        f"full strip: {int(full_win.width)}x{int(full_win.height)} | "
        f"fixture: rows {row_off}..{row_off + rows} ({int(fix_win.width)}x{rows}) | "
        f"water px: {water_px}"
    )

# --- write trimmed fixture ---------------------------------------------------
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
for name in LAYER_SUFFIXES:
    src_path = layer_paths[name]
    with rasterio.open(src_path) as ds:
        data = ds.read(1, window=fix_win)
        profile = ds.profile.copy()
        profile.update(
            height=rows,
            width=int(fix_win.width),
            transform=fix_transform,
            driver="GTiff",
            compress="deflate",
        )
    out = FIXTURE_DIR / src_path.name
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data, 1)
total = sum(f.stat().st_size for f in FIXTURE_DIR.glob("*.tif"))
print(f"fixture total: {total // 1024} KB")

# --- run the full chain on the TRIMMED fixture (what the test will assert) ---
fix_paths = find_granule_layers(FIXTURE_DIR)
print("\n# --- diagnostics on the committed fixture (assert these in the test) ---")
for layer in ("LST", "SST"):
    wst = read_ecostress_granule(fix_paths, mapping(aoi), thermal_layer=layer)
    do = couple_do_saturation(wst)
    print(f"\n===== {layer} (fixture) =====")
    for k in (
        "aoi_pixel_count",
        "valid_pixel_count",
        "water_pixel_count",
        "qc_pass_fraction",
        "temp_min_c",
        "temp_max_c",
        "n_subzero_water_px",
    ):
        print(f"  wst.{k}: {wst.diagnostics[k]}")
    for k in (
        "n_dropped_subzero",
        "n_dropped_over_max",
        "valid_pixel_count",
        "do_min_mgl",
        "do_max_mgl",
        "do_mean_mgl",
    ):
        print(f"  do.{k}: {do.diagnostics[k]}")
