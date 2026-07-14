"""Dissolved-oxygen saturation from water temperature (Benson & Krause, 1984).

This module answers one question: *given a water temperature, what is the most
oxygen the water could possibly hold?* That ceiling is called dissolved-oxygen
**saturation** (DO_sat). Colder water can hold more oxygen, so DO_sat falls as
temperature rises.

Two things this module deliberately does NOT do
------------------------------------------------
1. It does not report the oxygen actually in the water. Real dissolved oxygen is
   almost always *below* saturation, because decomposition and respiration draw
   oxygen down (the "oxygen debt"). Estimating actual DO needs a separate model;
   this module produces only the saturation ceiling that such a model starts from.
2. It does not know skin from bulk. If you feed it a satellite water-*surface*
   (skin) temperature, you get the DO_sat of that skin temperature. The skin can
   differ from the bulk water a fish breathes (see the package README). Feed it
   bulk temperature if you have it.

Scope of this implementation
----------------------------
Fresh water (salinity = 0) at standard sea-level pressure (1 atm ~= 760 mm Hg).
The Yolo Bypass is a freshwater floodplain near sea level, so this is the right
default there. Barometric-pressure and salinity corrections are separate,
also-closed-form factors added in a later module (each with its own DOTABLES
reference table); they are intentionally out of scope here to keep this a clean,
single-variable, known-answer computation.

Validated against USGS DOTABLES output (Benson & Krause equations) to within the
0.01 mg/L rounding of that table across 0-40 deg C. See
``tests/fixtures/dotables_do_solubility_bk_freshwater.csv``.

References
----------
- Benson, B.B., & Krause, D. (1984). The concentration and isotopic
  fractionation of oxygen dissolved in freshwater and seawater in equilibrium
  with the atmosphere. Limnology and Oceanography, 29(3), 620-632.
  https://doi.org/10.4319/lo.1984.29.3.0620   (equations implemented here)
- USGS Office of Water Quality Technical Memorandum 2011.03 (the polynomial form
  used by DOTABLES): https://water.usgs.gov/water-resources/memos/documents/WQ.2011.03.pdf
- Garcia, H.E., & Gordon, L.I. (1992). Oxygen solubility in seawater: better
  fitting equations. Limnology and Oceanography, 37(6), 1307-1312.
  https://doi.org/10.4319/lo.1992.37.6.1307  (plus 1993 erratum, 38(3), 656) --
  an alternative refit, noted as a possible future method; NOT used here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Versioned identifier for this computation. Any change to the equation, its
#: coefficients, or its assumptions gets a new ID (registry / provenance pattern).
MODEL_ID = "do_sat_bk1984_freshwater_1atm_v1"

#: Temperature validity range of the fitted equation, in degrees Celsius.
VALID_TEMP_C = (0.0, 40.0)

# Benson & Krause (1984) freshwater coefficients, ln(DO_sat[mg/L]) as a function
# of absolute temperature T[K]. Form: sum of a0 + a1/T + a2/T^2 + a3/T^3 + a4/T^4.
_BK_A0 = -139.34411
_BK_A1 = 1.575701e5
_BK_A2 = -6.642308e7
_BK_A3 = 1.243800e10
_BK_A4 = -8.621949e11

_KELVIN = 273.15


def do_saturation_freshwater(temperature_c: ArrayLike) -> NDArray[np.float64]:
    """Dissolved-oxygen saturation of fresh water at 1 atm, in mg/L.

    Parameters
    ----------
    temperature_c
        Water temperature in degrees Celsius. Scalar or any array-like; the shape
        is preserved, so a whole temperature raster can be passed at once. NaN is
        allowed and passes through as NaN (masked / no-data pixels stay unknown,
        they are never silently turned into a number).

    Returns
    -------
    numpy.ndarray
        DO_sat in mg/L, same shape as the input (0-d array for a scalar input).

    Raises
    ------
    ValueError
        If any *finite* temperature lies outside the equation's validity range
        (0-40 deg C). The failure is loud on purpose: extrapolating this fit past
        its range would return a plausible-looking but unsupported number.
    """
    t_c = np.asarray(temperature_c, dtype=np.float64)

    finite = np.isfinite(t_c)
    if finite.any():
        lo, hi = VALID_TEMP_C
        vals = t_c[finite]
        if vals.min() < lo or vals.max() > hi:
            raise ValueError(
                f"temperature outside validity range {VALID_TEMP_C} deg C: "
                f"observed finite range [{vals.min():.3f}, {vals.max():.3f}]"
            )

    T = t_c + _KELVIN
    ln_do = _BK_A0 + _BK_A1 / T + _BK_A2 / T**2 + _BK_A3 / T**3 + _BK_A4 / T**4
    return np.exp(ln_do)
