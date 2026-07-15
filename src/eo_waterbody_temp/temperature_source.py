"""Ports for water-surface-temperature retrieval.

A ``TemperatureSource`` turns some satellite product into a ``TemperatureField``:
a masked water-surface-temperature grid in degrees Celsius plus a self-describing
``diagnostics`` dictionary explaining how it was produced. Every retrieval method
(ECOSTRESS today, Landsat next) implements this same port, so downstream code can
treat them interchangeably and always inspect *how* a field was made.

The self-describing result object (a ``method`` string + a ``diagnostics`` dict)
mirrors the shape that worked well in the sibling water-volume project: the output
carries its own provenance instead of relying on the caller to remember it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class TemperatureField:
    """A masked water-surface-temperature grid with its provenance.

    Attributes
    ----------
    data:
        2-D array of water-surface temperature in **degrees Celsius**. Pixels that
        were masked out (land, cloud, poor QC, outside the AOI, or no-data) are
        ``NaN`` -- they are never silently set to a number.
    crs:
        Coordinate reference system of ``data`` (e.g. ``"EPSG:32610"``), as a str.
    transform:
        Affine geotransform of ``data`` (a 6-tuple ``(a, b, c, d, e, f)``), so the
        grid can be written back out or aligned with other rasters.
    method:
        Provenance string identifying the product and thermal layer used, e.g.
        ``"ecostress_l2t_lste_v003:LST"``.
    diagnostics:
        How the field was produced and how much survived masking -- valid-pixel
        fraction within the AOI, water/cloud/QC fractions, temperature range, etc.
        Loud enough that a half-off-swath scene cannot masquerade as a full one.
    """

    data: NDArray[np.float64]
    crs: str
    transform: tuple[float, float, float, float, float, float]
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def valid_fraction(self) -> float:
        """Fraction of AOI pixels that carry a usable temperature (0-1)."""
        return float(self.diagnostics.get("aoi_valid_fraction", float("nan")))


class TemperatureSource(ABC):
    """A source that reads water-surface temperature over an area of interest."""

    @abstractmethod
    def read(self, aoi: Any, **kwargs: Any) -> TemperatureField:
        """Return a :class:`TemperatureField` over ``aoi`` (an EPSG:4326 geometry)."""
        raise NotImplementedError
