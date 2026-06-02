# Hand-off: ED global_baseline Model C inputs (1997–2016)

A single compressed NetCDF that bundles the inputs needed to continue refining
Model C (the burned-area module from `ed-autoresearch`, ranked #1 on ILAMB
TRENDY v14 burned-area when run offline). Everything in this file comes from
one ED run; no driver products were re-attached.

## File

`global_baseline_modelC_inputs_1997-2016.nc` (NetCDF4, zlib level 4, float32)

Dims: `time` (240, monthly midpoints), `lat`, `lon` (0.5°, global).
Calendar: `noleap`, time stamps `YYYY-MM-15`.

## How it was created

1. Source: `output/global_baseline_PARALLEL_gsapp*/global_baseline_PARALLEL_gsapp*.region.nc`
   on HPC — 11 latitude-banded files from a global run of ED (stock fire model)
   forced with the same drivers used in the GCB2025 S3 setup. Monthly,
   1990‑01..2024‑12.
2. Concatenated the 11 bands along latitude (sorted ascending), checked that
   the longitude axis is identical across bands.
3. Sliced to real years **1997..2016** to match the GFED4.1s window used in our
   first refit (`analysis/fit_modelC_optuna.py`).
4. Sentinel masked: any value ≤ −9990 → `NaN`.
5. Unit conversion: `P_month` was divided by 12 (ED stores monthly precip as
   `mm yr-1`; the refit treats it as `mm month-1`).
6. Cast to `float32`, written with `zlib=True, complevel=4, shuffle=True`,
   chunked `(12, lat, lon)` so a year of any field reads as one chunk.

Generator script: `scripts/package_global_baseline_for_handoff.py`. The git
revision and creation timestamp are stored in the file's global attributes.

## What each variable represents

All variables are dimensioned `(time, lat, lon)`. They are the inputs the
Model C refit consumes plus the model's own `area_burned` for sanity checking.

| variable | units | meaning |
|---|---|---|
| `D_bar` | mm | monthly mean dryness index used by ED's fire scheme |
| `T_air` | degC | monthly mean air temperature (driver) |
| `P_ann` | mm yr-1 | annual precipitation (driver, repeated within year) |
| `P_month` | mm month-1 | monthly precipitation (already divided by 12 here) |
| `GPP_month_ntrl` | kg m-2 yr-1 | monthly GPP, **natural** vegetation |
| `GPP_month_scnd` | kg m-2 yr-1 | monthly GPP, **secondary** vegetation |
| `GPP_month_past` | kg m-2 yr-1 | monthly GPP, **pasture** |
| `area_frac_ntrl` | 1 | grid-cell area fraction, natural |
| `area_frac_scnd` | 1 | grid-cell area fraction, secondary |
| `area_frac_past` | 1 | grid-cell area fraction, pasture |
| `area_burned` | km² yr-1 | ED stock-fire output. ED writes the annual value into all 12 monthly slots — values are identical within a year. To get a per-cell fraction-per-month, divide by `12 · cell_area_km²`. |

The full Model C ignition combiner (using `D_bar`, `T_air`, `P_ann`, `P_month`,
GPP per landuse, area fractions per landuse) is in
`analysis/fit_modelC_optuna.py::_modelC_predict`. Best-fit constants from the
first global refit (10k Optuna trials, GFED 1997‑2016) are in
`analysis/modelC_refit_global.json`.

## What is *not* in this file

- The GFED4.1s burned-area target itself (publicly available; ILAMB sample copy
  on the HPC at `…/ILAMB_sample/DATA/burntArea/GFED4.1S/burntArea.nc`).
- Anything from the **coupled** Model C run (only the **stock-fire** global
  baseline is here). The coupled global run is the next step on our side.
- Per-landuse `area_burned` — only the cell-aggregate is in the standard ED
  output.

## Suggestions for next steps

1. **Reproduce the offline-refit baseline.** Re-fit Model C against
   GFED4.1s using only this file as ED state input; you should land near
   loss ≈ 0.881 / Pearson r ≈ 0.63 (numbers from
   `analysis/modelC_refit_global.json`). If you do, the data hand-off is
   self-consistent and any improvement from there is genuine.
2. **Look at the regional residuals.** The current global refit undershoots
   Sahel + Congo magnitude (predicted ~1.7 %/yr vs GFED ~3.1 %/yr). Diagnose
   whether the sigmoid asymptotes saturate too soon, the suppression term is
   too aggressive at high `D_bar`, or the GPP-hump caps fuel.
3. **Try region-aware fits.** A single global parameter set has to compromise
   between fire-on-grass biomes (Sahel, Australia) and fire-in-forest biomes
   (Amazon edge, boreal). Fit per-biome or per-continent and compare against
   the global fit on held-out years (split 1997‑2010 train / 2011‑2016 test).
4. **Replace climatology IAV with year-by-year matching.** The first refit
   already runs year-by-year; if you switch to climatology you'll lose IAV
   skill — keep the per-year alignment.
5. **Beware data leakage.** Do not tune thresholds on the same years you
   report metrics on. Hold out a window.
6. **When ready for coupling**, the same parameter set goes into
   `ed_patched/fire.cc::ModelC` (constants compiled in). Coupling adds
   feedback through GPP — expect a different fit optimum than the offline one.

## Provenance pointers (in this repo)

- `analysis/fit_modelC_optuna.py` — offline refit (loads same variables this
  file exposes).
- `analysis/eval_burntArea_multimodel.py` — ILAMM-style multimodel evaluation
  including the GFED window and unit harmonization.
- `analysis/modelC_refit_global.json` — best parameters from the 10k-trial
  global refit.
- `doc/ED_fire_notes.md` — mapping of ED's stock fire path to Model C inputs
  and where each variable comes from inside ED.
