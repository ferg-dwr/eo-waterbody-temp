"""eo-waterbody-temp: Earth-observation water-surface temperature over the Yolo Bypass.

This package retrieves water-surface temperature (WST) from thermal satellite
observations over the Yolo Bypass and derives a dissolved-oxygen-saturation
(DO_sat) diagnostic from it.

Two things to understand before trusting any output (see README):

1. Satellites measure a radiometric *skin* temperature (top microns-mm of water),
   not the *bulk* water column temperature a fish experiences. These diverge in
   calm, sunny, low-flow conditions. Bulk requires an in-situ offset calibration.

2. DO_sat is the theoretical *maximum* dissolved oxygen at equilibrium. Actual DO
   is almost always lower because of biological oxygen demand. This package does
   NOT estimate actual DO; it produces the saturation ceiling only.

The top-level package imports numpy only. Heavy I/O dependencies (rasterio,
xarray, earthaccess, ...) live behind the ``[io]`` extra and are imported lazily
inside the modules that need them, so ``import eo_waterbody_temp`` stays cheap.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
