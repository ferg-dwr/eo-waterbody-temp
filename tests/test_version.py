"""Smoke test: the package imports and exposes a sane version string."""

import re

import eo_waterbody_temp


def test_version_is_a_semver_string():
    assert isinstance(eo_waterbody_temp.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", eo_waterbody_temp.__version__)


def test_top_level_import_is_numpy_only():
    """Guard the lazy-import contract: importing the package must not pull in
    heavy I/O deps. Run in a FRESH interpreter -- within one pytest process other
    tests import rasterio et al. into sys.modules, so an in-process check would be
    a false positive. If this fails, something added an eager top-level import."""
    import subprocess
    import sys

    code = (
        "import eo_waterbody_temp, sys\n"
        "forbidden = {'rasterio', 'xarray', 'earthaccess', 'matplotlib', 'pyproj'}\n"
        "leaked = sorted(forbidden & set(sys.modules))\n"
        "assert not leaked, 'leaked: ' + repr(leaked)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
