"""Couple a water-surface-temperature field to a dissolved-oxygen-saturation field.

Given a :class:`~eo_waterbody_temp.temperature_source.TemperatureField` (masked
water-surface temperature in degrees Celsius), produce a standalone
:class:`DissolvedOxygenField` (DO saturation in mg/L) using the validated
Benson & Krause core.

Why sub-zero water is dropped (not clamped, not extrapolated)
------------------------------------------------------------
The Benson & Krause equation is defined for 0-40 deg C. On the Yolo Bypass, a
near-sea-level freshwater floodplaim, liquid water below 0 deg C is not
physical: a sub-zero radiometric reading is skin-effect noise, a thin-ice edge,
or retrieval error, none of which is a bulk temperature you would trust for
oxygen. So water pixels below 0 deg C are set to NaN (dropped) and counted.
The undropped view already exists the source TemperatureField, which keeps
those pixels as temperatures.

Pixels above 40 deg C are dropped symmetrically, but a non-zero
``n_dropped_over_max`` is a surprise on this geography and should be investigated
(unlike sub-zero drops, which are expected edge noise).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .do_saturation import MODEL_ID as DO_MODEL_ID
from .do_saturation import VALID_TEMP_C, do_saturation_freshwater
from .temperature_source import TemperatureField


@dataclass
class DissolvedOxygenField:
    """A dissolved-oxygen-saturation grid (mg/L) with its provenance.

    Mirrors :class:`TemperatureField` so the two products read the same way.
    ``data`` is DO saturation in **mg/L**; pixels that were masked upstream OR
    dropped for being outside the equation's 0-40 deg C domain are ``NaN``.
    """

    data: NDArray[np.float64]
    crs: str
    transform: tuple[float, float, float, float, float, float]
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def valid_mask(self) -> NDArray[np.bool_]:
        """Boolean array: True where a usable DO saturation value exists."""
        return np.isfinite(self.data)

    @property
    def valid_fraction(self) -> float:
        return float(self.diagnostics.get("aoi_valid_fraction", float("nan")))


def couple_do_saturation(field_in: TemperatureField) -> DissolvedOxygenField:
    """Convert a WST TemperatureField (deg C) to a DO-saturation field (mg/L).

    Water pixels outside the Benson & Krause 0-40 deg C domain are dropped to NaN
    and counted (see module docstring); the equation is only ever evaluated on
    in-range pixels, so its loud out-of-range guard stays a genuine guard.
    """
    wst = np.asarray(field_in.data, dtype=np.float64)
    lo, hi = VALID_TEMP_C
    finite = np.isfinite(wst)
    subzero = finite & (wst < lo)
    over_max = finite & (wst > hi)
    in_range = finite & (wst >= lo) & (wst <= hi)

    do = np.full(wst.shape, np.nan, dtype=np.float64)
    if in_range.any():
        do[in_range] = do_saturation_freshwater(wst[in_range])

    n_valid = int(in_range.sum())
    valid_do = do[in_range]
    src = field_in.diagnostics

    diagnostics: dict[str, Any] = {
        "do_model_id": DO_MODEL_ID,
        "source_method": field_in.method,
        "source_granule_id": src.get("granule_id"),
        "source_acquisition_time_utc": src.get("acquisition_time_utc"),
        "n_dropped_subzero": int(subzero.sum()),
        "n_dropped_over_max": int(over_max.sum()),
        "valid_pixel_count": n_valid,
        "aoi_valid_fraction": (
            (n_valid / src["aoi_pixel_count"])
            if src.get("aoi_pixel_count")
            else float("nan")
        ),
        "do_min_mgl": float(np.min(valid_do)) if n_valid else float("nan"),
        "do_max_mgl": float(np.max(valid_do)) if n_valid else float("nan"),
        "do_mean_mgl": float(np.mean(valid_do)) if n_valid else float("nan"),
    }

    return DissolvedOxygenField(
        data=do,
        crs=field_in.crs,
        transform=field_in.transform,
        method=f"{DO_MODEL_ID}<-{field_in.method}",
        diagnostics=diagnostics,
    )