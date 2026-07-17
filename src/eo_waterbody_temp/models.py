"""Model registry: every model_id that appears in a filename is defined here.

The shared cross-repo output contract puts a ``model_id`` in slot 3 of every
output filename, and requires each such id to have a registry entry AND a row in
``MODELS.md``. This is the ``eo-waterbody-temp`` analogue of the volume repo's
``MODEL_REGISTRY`` -- adapted to our function-based models (a registry of
metadata dicts) rather than a class hierarchy, but giving the same guarantee: a
model_id cannot reach a filename without being registered and documented.

Versioning rule (shared with the volume repo): **any behavior change is a new
id**. Old outputs stay interpretable forever; nothing is redefined in place.

The two model_ids are imported from where they are actually defined, so there is
a single source of truth and the registry cannot drift from the code.
"""

from __future__ import annotations

from .do_saturation import MODEL_ID as DO_SAT_MODEL_ID
from .ecostress import PRODUCT as ECOSTRESS_WST_MODEL_ID

#: Required metadata keys for every registry entry.
_REQUIRED = ("kind", "product", "units", "description")

MODEL_REGISTRY: dict[str, dict] = {
    ECOSTRESS_WST_MODEL_ID: {
        "kind": "temperature-retrieval",
        "product": "WST",
        "units": "degC",
        "description": (
            "Water-surface temperature from ECOSTRESS L2T LSTE V003, masked to "
            "clear, QC-good water pixels. Thermal layer (LST or SST) is recorded "
            "as the filename variant; the two layers can diverge over water "
            "(SST uses a water emissivity), which is a known, unresolved "
            "difference pending in-situ validation."
        ),
        "assumptions": (
            "Radiometric skin temperature approximates the bulk water a fish "
            "experiences; valid where the water column is well mixed. Skin<->bulk "
            "offset is not yet calibrated."
        ),
        "diagnostics": (
            "aoi/valid/water pixel counts, cloud & QC fractions, temp range, "
            "n_subzero_water_px"
        ),
        "failure_modes": (
            "skin != bulk in calm sunny low-flow conditions; LST vs SST diverge "
            "over water; coverage collapses when the bypass is mostly dry."
        ),
    },
    DO_SAT_MODEL_ID: {
        "kind": "dissolved-oxygen-saturation",
        "product": "DO",
        "units": "mg/L",
        "description": (
            "Dissolved-oxygen saturation from water-surface temperature via "
            "Benson & Krause (1984), freshwater at 1 atm."
        ),
        "assumptions": (
            "Freshwater (salinity 0) at sea-level pressure; valid 0-40 degC. "
            "Saturation is the equilibrium ceiling, NOT actual dissolved oxygen."
        ),
        "diagnostics": (
            "n_dropped_subzero, n_dropped_over_max, valid_pixel_count, "
            "do_min/max/mean_mgl"
        ),
        "failure_modes": (
            "sub-zero / >40 degC water dropped (out of equation domain); "
            "saturation overstates real DO wherever biological oxygen demand is high."
        ),
    },
}


def is_registered(model_id: str) -> bool:
    """True if ``model_id`` has a registry entry."""
    return model_id in MODEL_REGISTRY


def registry_entry(model_id: str) -> dict:
    """Return the registry entry for ``model_id`` or raise KeyError."""
    return MODEL_REGISTRY[model_id]


def model_id_from_method(method: str) -> str:
    """Extract the (registered) model_id from a field's ``method`` string.

    WST method ``"ecostress_l2t_lste_v003:LST"`` -> ``"ecostress_l2t_lste_v003"``.
    DO method  ``"do_sat_..._v1<-ecostress_l2t_lste_v003:LST"`` -> ``"do_sat_..._v1"``.
    """
    return method.split("<-")[0].split(":")[0]


def variant_from_method(method: str) -> str | None:
    """Thermal-layer variant (``"LST"``/``"SST"``) if present, else None."""
    return method.rsplit(":", 1)[-1] if ":" in method else None
