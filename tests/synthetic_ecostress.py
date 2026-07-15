"""Build a tiny synthetic ECOSTRESS-shaped granule with KNOWN geometry.

Encodes the real V003 conventions (float32 Kelvin LST with NaN fill; uint8 water
0/1; uint8 cloud 0/1/255; uint16 packed QC; EPSG:32610) at 10x10 / 70 m so every
masked pixel count is hand-countable in the tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping, shape

CRS = "EPSG:32610"
RES = 70.0
X0, Y0 = 535000.0, 4265000.0  # UTM 10N, near the Yolo Bypass
N = 10
GRANULE_STEM = "ECOv003_L2T_LSTE_42945_013_10SFH_20260201T014232_03"
TRANSFORM = from_origin(X0, Y0, RES, RES)

# Known-answer layout (row, col), 0-indexed from top-left:
#   water  : cols 0-4 water(1), cols 5-9 land(0)
#   cloud  : row0 cloudy(1), row1 no-obs(255), rows2-9 clear(0)
#   QC     : col0 not-produced(3), col1 degraded(1), else good(49856 -> bits 00)
#   LST    : 285.0 K everywhere, except (4,3)=NaN and (3,2)=272.0 K (sub-zero degC)


def _layers():
    lst = np.full((N, N), 285.0, dtype=np.float32)
    lst[4, 3] = np.nan
    lst[3, 2] = 272.0
    sst = np.full(
        (N, N), 286.0, dtype=np.float32
    )  # distinct so we can tell layers apart

    water = np.zeros((N, N), dtype=np.uint8)
    water[:, 0:5] = 1

    cloud = np.zeros((N, N), dtype=np.uint8)
    cloud[0, :] = 1
    cloud[1, :] = 255

    qc = np.full((N, N), 49856, dtype=np.uint16)  # 49856 & 0b11 == 0 -> good
    qc[:, 0] = 3  # not produced
    qc[:, 1] = 1  # degraded
    return {"LST": lst, "SST": sst, "water": water, "cloud": cloud, "QC": qc}


def write_granule(dirpath: str | Path) -> dict[str, Path]:
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, arr in _layers().items():
        path = dirpath / f"{GRANULE_STEM}_{name}.tif"
        nodata = (
            np.nan if arr.dtype == np.float32 else (255 if name == "cloud" else None)
        )
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=N,
            width=N,
            count=1,
            dtype=arr.dtype,
            crs=CRS,
            transform=TRANSFORM,
            nodata=nodata,
        ) as ds:
            ds.write(arr, 1)
        paths[name] = path
    return paths


def aoi_4326():
    """AOI polygon (EPSG:4326) covering exactly rows 2-6, cols 2-6 (a 5x5 block)."""
    minx = X0 + 2 * RES
    maxx = X0 + 7 * RES
    maxy = Y0 - 2 * RES
    miny = Y0 - 7 * RES
    utm_box = box(minx, miny, maxx, maxy)
    return shape(transform_geom(CRS, "EPSG:4326", mapping(utm_box)))
