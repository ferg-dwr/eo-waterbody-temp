# Model registry — eo-waterbody-temp

Every model that can appear in an output filename's `model_id` slot is listed
here and in `MODEL_REGISTRY` (`src/eo_waterbody_temp/models.py`). This file is
the human-readable companion; the dict is the machine-readable source. A CI test
(`tests/test_models.py`) fails if the two drift or if a model_id used in code is
missing from either.

**Versioning rule (shared with `eo-water-volume-estimator`): any behavior change
is a new id** (e.g. `do_sat_bk1984_freshwater_1atm_v1` -> `..._v2`). Old outputs
stay interpretable forever; nothing is redefined in place.

The thermal-layer choice (LST vs SST) is recorded as the filename **variant**,
not as part of the model_id: it is an input-layer dimension, not a different
model. The two layers can diverge over water and which is correct is an open
in-situ-validation question.

## Models

| model_id | kind | product | units | key assumptions | diagnostics | measured / known failure modes |
|---|---|---|---|---|---|---|
| `ecostress_l2t_lste_v003` | temperature-retrieval | WST | degC | radiometric skin ≈ bulk water (uncalibrated); masked to clear, QC-good water | aoi/valid/water counts, cloud & QC fractions, temp range, `n_subzero_water_px` | skin ≠ bulk in calm/sunny/low-flow; LST vs SST diverge over water (SST floor ~3.4 K warmer on the 2026-02-01 scene); coverage collapses when bypass is mostly dry (8.1% valid on a 91%-dry Feb scene) |
| `do_sat_bk1984_freshwater_1atm_v1` | dissolved-oxygen-saturation | DO | mg/L | freshwater (S=0), 1 atm, valid 0–40 °C; saturation is the equilibrium ceiling, NOT actual DO | `n_dropped_subzero`, `n_dropped_over_max`, valid count, `do_min/max/mean_mgl` | sub-zero / >40 °C water dropped (out of domain); saturation overstates real DO where biological oxygen demand is high |

## Adding a model

1. Define the `MODEL_ID` constant next to the code that implements it.
2. Add an entry to `MODEL_REGISTRY` importing that constant (single source of truth).
3. Add a row here.
4. `pytest tests/test_models.py` must stay green (it checks registry ⇄ MODELS.md ⇄ code).