"""Smoke test: the package imports and exposes a sane version string."""

import re

import eo_waterbody_temp


def test_version_is_a_semver_string():
    assert isinstance(eo_waterbody_temp.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", eo_waterbody_temp.__version__)


def test_top_level_import_is_numpy_only():
    """Guard the lazy-import contract: importing the package must not pull in
    heavy I/O deps. If this fails, something added an eager top-level import."""
    import sys

    forbidden = {"rasterio", "xarray", "earthaccess", "matplotlib", "pyproj"}
    leaked = forbidden & set(sys.modules)
    assert not leaked, f"top-level import leaked heavy deps: {sorted(leaked)}"
