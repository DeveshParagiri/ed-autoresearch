# ED Autoresearch: Fire Module Replacement

Closed-form, mechanistic fire formulas for the ED v3 dynamic global vegetation model. Trained against GFED4.1s burned area using TRENDY v14 ED S3 inputs, scored via ILAMB's `ConfBurntArea` confrontation.

Three variants at different complexity tiers:

- **A** — full 7-mechanism formula, 19 params
- **B** — Shapley-reduced 4 mechanisms, 13 params (**best Overall**)
- **C** — minimal 2-mechanism physics-only, 10 params

## ILAMB Leaderboard (TRENDY v14 + our models)

GFED4.1s burned-area benchmark, 2001-2016 monthly. `ConfBurntArea` confrontation, `mass_weighting = True`, area-weighted cos-lat.

| Rank | Model | Bias | RMSE | Seasonal | Spatial Dist | **Overall** |
|-----:|-------|-----:|-----:|---------:|-------------:|------------:|
| 1 | CLM6.0 | 0.759 | 0.474 | 0.758 | 0.838 | **0.707** |
| 2 | CLASSIC | 0.738 | 0.507 | 0.782 | 0.797 | **0.706** |
| 3 | ELM-FATES | 0.724 | 0.512 | 0.860 | 0.676 | **0.693** |
| 4 | CLM-FATES | 0.725 | 0.525 | 0.802 | 0.707 | **0.690** |
| **5** | **ED-ModelB-v2** | **0.725** | **0.482** | **0.612** | **0.791** | **0.652** |
| **6** | **ED-ModelC-v2** | **0.701** | **0.476** | **0.662** | **0.730** | **0.642** |
| **7** | **ED-ModelA-v2** | **0.732** | **0.478** | **0.520** | **0.806** | **0.634** |
| 8 | JULES-ES | 0.709 | 0.506 | 0.784 | 0.447 | 0.611 |
| 9 | ELM | 0.687 | 0.492 | 0.778 | 0.333 | 0.572 |
| 10 | VISIT-UT | 0.636 | 0.488 | 0.695 | 0.466 | 0.571 |
| 11 | LPJmL | 0.693 | 0.489 | 0.459 | 0.612 | 0.563 |
| 12 | LPJ-GUESS | 0.671 | 0.489 | 0.459 | 0.288 | 0.477 |
| 13 | **EDv3 (stock)** | **0.681** | **0.489** | **0.439** | **0.290** | **0.475** |
| 14 | LPJ-EOSIM | 0.654 | 0.489 | 0.459 | 0.227 | 0.457 |

**B ranks #5** — beats JULES-ES, ELM, VISIT-UT, LPJmL, LPJ-GUESS, and ED's stock fire module (+0.18 over EDv3). All three of our models sit in the top half.

For the JASMIN-reproducer scorecard that the TRENDY leaderboard numbers come from, see [trendy-v14-fire-benchmark](https://github.com/DeveshParagiri/trendy-v14-fire-benchmark).

## What's in this repo

```
models/
├── A/params.json        19 params, full 7-mechanism formula
├── A/formula.md         math, mechanism table, citations
├── B/params.json        13 params, Shapley-reduced 4-mechanism
├── B/formula.md
├── C/params.json        10 params, 2-mechanism minimal
├── C/formula.md
└── shapley.json         exact Shapley decomposition of A (128 subsets)

scripts/
├── reproduce.py         regenerate burntArea NetCDFs for all 3 models
├── score.py             run ILAMB on generated NetCDFs
└── download_inputs.sh   fetch TRENDY v14 + GFED reference

WRITEUP.md               full experimental narrative
```

## Inputs required for replication

The three models all use the same input set. A uses all 8, B uses 5, C uses 3.

### TRENDY v14 ED S3 (downloaded from [gcb-2025-upload bucket](https://s3.eu-west-1.wasabisys.com/gcb-2025-upload/land/output/ED/S3/))

| File | Size | Used by | Purpose |
|------|-----:|:-------:|---------|
| `EDv3_S3_gpp.nc` | 4.0 GB | A | monthly GPP (coupled ED output) |
| `EDv3_S3_cLeaf.nc` | 337 MB | A/B/C | leaf carbon → AGB component |
| `EDv3_S3_cWood.nc` | 337 MB | A/B/C | wood carbon → AGB component |
| `EDv3_S3_cSoil.nc` | 337 MB | A | soil carbon pool |

AGB for the `fuel_hump` term = `cLeaf + cWood` (annual, kg C/m²).

### CRUJRA v3.5 climate (same as TRENDY models use for forcing)

Any CRUJRA 0.5° monthly product covering 2001-2016:
- Temperature (for PET via Thornthwaite)
- Precipitation
- Soil temperatures layers 1-6 (layers 3-6 averaged for `T_deep`)

### ED frozen-simulation output (no TRENDY v14 equivalent exists)

- `mean_height_natr` — canopy height, natural vegetation
- `mean_height_scnd` — canopy height, secondary vegetation (A only)
- `frac_scnd` — secondary vegetation land fraction (A only)

These PFT-level state variables aren't published in TRENDY v14. They come from a frozen ED simulation. For production coupled deployment, each ED run would use its own prognostic versions — the formula parameters are scale-aware enough that this should work, but hasn't been fully validated.

### Target (for validation/re-calibration only)

- `GFED4.1s_*.hdf5` — 2001-2016 monthly burned-area fraction

Not needed at inference time. Only used to recompute scores.

## Quick start

```bash
# Clone + setup
git clone https://github.com/DeveshParagiri/ed-autoresearch.git
cd ed-autoresearch
python -m venv .venv && source .venv/bin/activate
pip install numpy xarray cftime netCDF4 optuna h5py

# Download inputs (~5 GB for minimal set; ~10 GB for all)
bash scripts/download_inputs.sh

# Regenerate NetCDFs for all 3 models
python scripts/reproduce.py

# Score with ILAMB (see trendy-v14-fire-benchmark for full leaderboard)
python scripts/score.py
```

## Model selection guide

- **Best ILAMB score**: B (0.652). Use this for benchmark comparisons.
- **Best spatial pattern**: A (SpDist 0.806). Use if you need sharp geographic fidelity, e.g., regional attribution.
- **Minimum dependencies for a coupled ED port**: C. Only 3 input fields, 10 params, no GPP/landuse/soil_C coupling.
- **Best seasonal cycle**: C (Seas 0.662). Fewer mechanisms interfere less with monthly climate signal.

## Caveats

- Trained on TRENDY v14 offline inputs. Coupled-ED transferability validated in principle (rank-invariance under GPP/AGB scale shifts) but not in a full production deployment.
- Absolute Bias and Overall scores are ~2-3× higher than JASMIN's internal leaderboard because they use a private `BurnedAreaExtended` confrontation with a burnable-area mask (GFED4.1S16). Rankings match; absolute magnitudes don't.
- These models predict **burned area fraction**. Emissions, mortality, and coupling to vegetation state are handled by ED's existing machinery.

## References

- Bistinas et al. 2014 · Archibald et al. 2009, 2010, 2013 · Pausas & Ribeiro 2013 · Pausas & Keeley 2009 · van der Werf 2008, 2010 · Venevsky et al. 2002 · Krawchuk et al. 2009
- ILAMB: Collier et al. 2018
- GFED4.1s: van der Werf et al. 2017
- TRENDY v14: Global Carbon Project 2025

## License

MIT (see LICENSE).
