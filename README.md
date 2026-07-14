# eo-waterbody-temp

Water-surface temperature over the Yolo Bypass from thermal satellite imagery,
plus a dissolved-oxygen-saturation diagnostic derived from it.

This README assumes **no remote-sensing background**. If a term is unfamiliar,
it is explained here or it does not matter for using the outputs.

## What this does

The Yolo Bypass is a seasonally flooded floodplain west of Sacramento that is
important rearing habitat for juvenile salmon. Water temperature drives two
things DWR cares about:

1. **Where salmon are comfortable.** Juvenile salmon have a temperature range
   they tolerate; outside it they are stressed or die. A map of water
   temperature is a map of thermal comfort.
2. **How much oxygen the water can hold.** Colder water can hold more dissolved
   oxygen. This package turns temperature into the *maximum possible* dissolved
   oxygen (the saturation ceiling), which is an input to later oxygen modeling.

It reads thermal satellite scenes (ECOSTRESS ~70 m; Landsat ~100 m as a
cross-check), clips them to the bypass, keeps only water pixels, and produces a
water-surface-temperature map plus a dissolved-oxygen-saturation map.

## Two limitations to read before trusting any number

These are not edge cases. They shape how the outputs should be used.

1. **Skin is not bulk.** Satellites measure the temperature of the very top of
   the water (microns to millimeters) — the *skin*. Fish live in the water
   *column* — the *bulk*. On a windy, well-mixed flood pulse these are close.
   On a calm, sunny afternoon the skin can read meaningfully warmer than the
   water below it. Because comfort/lethality is threshold-driven, a skin bias of
   1-2 degrees can misclassify comfort zones. Converting skin to bulk requires
   comparison against in-water gauges; that calibration is planned, not yet done.

2. **Saturation is not actual oxygen.** Dissolved-oxygen *saturation* is the
   maximum oxygen the water could hold at equilibrium. The oxygen fish actually
   breathe is usually lower, because decomposing vegetation and respiration
   consume oxygen — exactly the process that drives fish kills on flooded
   fields. This package produces the saturation ceiling only. The gap between
   the ceiling and reality (the "oxygen debt") requires a separate model and is
   out of scope here.

## Install

```bash
# core only (numpy): enough to compute DO saturation from temperatures you supply
pip install .

# with satellite I/O (reading/clipping/masking scenes)
pip install .[io]

# development (tests, linting, formatting)
pip install .[dev]
```

## Status

Early development. See the milestone plan (local `ROADMAP.md`). The first
milestone delivers the DO-saturation core (verified against the USGS DOTABLES
reference) and one real ECOSTRESS scene read end-to-end over the bypass.