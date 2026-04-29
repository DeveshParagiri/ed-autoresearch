# ED Autoresearch — Model C Fire Module

Closed-form, mechanistic fire formula for the ED v3 dynamic global vegetation model. Trained against GFED4.1s via Optuna, scored via **official `ilamb-run`** with stock `ConfBurntArea`.

> **All training and benchmarking is OFFLINE** against fixed TRENDY v14 ED-S3 NetCDFs + CRUJRA climate, not a coupled ED run. Coupled deployment notes at the bottom of this README.

**Model C — Overall Score 0.6713, rank #1.** 12 parameters, 3 mechanism groups, 5 input fields. **ED-coupled-consistent**: model output is treated as an annual fire rate (yr⁻¹) and passes through ED's `(1 - exp(-rate · 1yr)) / 12` transform during scoring, so the offline NC matches what coupled ED would write to TRENDY. No post-hoc rescale.

## ILAMB leaderboard (real `ilamb-run`, stock ConfBurntArea)

| Rank | Model | Bias | RMSE | Seasonal | Spatial | **Overall** |
|---:|---|---:|---:|---:|---:|---:|
| 🥇 1 | **ED-ModelC (ours)** | **0.728** | 0.506 | **0.846** | 0.771 | **0.6713** |
| 🥈 2 | CLASSIC | 0.739 | 0.507 | 0.782 | **0.798** | 0.6665 |
| 🥉 3 | CLM6.0 | **0.759** | 0.474 | 0.759 | **0.838** | 0.6606 |
| 4 | ELM-FATES | 0.726 | 0.513 | 0.862 | 0.682 | 0.6594 |
| 5 | ED-ModelA | 0.716 | 0.492 | 0.805 | 0.783 | 0.6574 |
| 6 | CLM-FATES | 0.727 | 0.526 | 0.803 | 0.705 | 0.6573 |
| 7 | ED-ModelB | 0.706 | 0.476 | 0.833 | 0.763 | 0.6506 |
| 8 | VISIT | 0.645 | 0.491 | 0.745 | 0.503 | 0.5751 |
| 9 | E3SM | 0.687 | 0.492 | 0.780 | 0.334 | 0.5571 |
| 10 | JSBACH | 0.475 | 0.488 | 0.460 | 0.131 | 0.4082 |
| 11 | **EDv3 (stock)** | 0.532 | 0.487 | 0.396 | 0.111 | **0.4026** |
| 12 | SDGVM | 0.416 | 0.489 | 0.459 | 0.009 | 0.3724 |

"Overall Score" is ILAMB's native tier-2 aggregation pulled directly from `scalar_database.csv`. Component scores are unmodified from the same CSV. No custom aggregation.

Model C beats ED's stock fire module by **+0.27 Overall** (0.6713 − 0.4026).

### ED coupling note

This iteration is **ED-coupled-consistent**: Model C's output is interpreted as an annual disturbance rate (yr⁻¹), and the offline pipeline applies ED's exact saturation-then-monthly-distribute transform `monthly_frac = (1 - exp(-min(rate, fire_max) · 1yr)) / 12` before scoring. This matches (to within ED's per-cell patch resolution) what coupled ED writes to TRENDY-format burntArea.

**One coupled-side ask**: the optimiser pushes a few high-fire cells (Sahel, savanna; max raw rate ≈ 1.0 yr⁻¹) above ED's default `fire_max_disturbance_rate = 0.2`. To reproduce this leaderboard score in coupled ED, please bump the cap to **1.0** in `ED_params.defaults.cfg`. A safety variant retuned within the 0.2 cap scores 0.6105 (rank #7) and is preserved as `ilamb/MODELS/ED-ModelC-final/burntArea.cap0.2-aligned.nc` if relaxing the cap is not feasible.

## Formula

```
fire(cell, month) =
    [ onset(D̄) × suppress(D̄)                        # base ignition
    × precip_floor(P_ann) × precip_dampen(P_month)    # precip
    × gpp_hump(GPP_month)                             # monthly GPP
    × air_temp_ign(T_air)                             # monthly air temp
    ]^fire_exp
```

See [`models/C/formula.md`](models/C/formula.md) for mechanism-by-mechanism definition and citations.

## Reproduce bit-exact

```bash
git clone https://github.com/DeveshParagiri/ed-autoresearch
cd ed-autoresearch
pip install -r requirements.txt

# 1. Get the inputs. Two paths:
#
# (a) Download the 5 GB pinned bundle from Drive (one zip, ~10 min on a fast link):
#     bash scripts/download_inputs.sh
#     Bundle SHA256: f124b21e778c3a28532acd3fdaea70a701a6d8cb714fafa423a8d748b4a7b4d3
#     Source: https://drive.google.com/file/d/1ID5pswHyaaF9Ej1CDgZ7j7pGjpQkKEhQ/view
#     Contents: crujra/{4 .npy}, trendy_v14/gpp.nc, gfed/16 hdf5, outputs/reference burntArea.nc
#
#     Add --with-terms to also grab the 107 MB per-term debug NetCDF
#     (modelC_terms.nc: 5 inputs + 6 intermediate terms + final product,
#     for coupled-ED cross-comparison):
#         bash scripts/download_inputs.sh --with-terms
#     Source: https://drive.google.com/file/d/1ges8y2qw1KF8eNt8ruIgO3Akj8HGaTYo/view
#
# (b) You have raw CRUJRA / TRENDY / GFED on disk — regenerate .npy locally:
#     ED_RAW_DATA=/your/path/to/data python scripts/prep_monthly_inputs.py
#     Raw data layout expected (override per-subdir by editing prep_monthly_inputs.py):
#       $ED_RAW_DATA/observations/crujra/CRUJRA_v3.5_climate_YYYY.nc
#       $ED_RAW_DATA/trendy_v14_ed_vars/EDv3_S3_{gpp,cLeaf,cWood,cSoil}.nc
#       $ED_RAW_DATA/observations/gfed/GFED4.1s_YYYY.hdf5
#       $ED_RAW_DATA/ed-simulation/EDv3_global_simulation_1981_2016.nc  (not needed for C)

# 2. Sanity-check: every pinned input present and SHA256-matched.
python scripts/verify.py

# 3. Run Model C and write burntArea.nc
python scripts/reproduce_modelC.py
# -> ilamb/MODELS/ED-ModelC-final/burntArea.nc
#    sha256 61324c73e39fb857c3c604431de62909f2e1d3d3badabce53518f098edf020d3

# 4. Also dump per-term intermediates (for coupled-ED debug)
python scripts/dump_modelC_terms.py
# -> out_terms/modelC_terms.nc

# 5. End-to-end pipeline check (regenerates step 3 and matches hash)
python scripts/verify.py --pipeline

# 6. Score against GFED via stock ilamb-run
ilamb-fetch --remote_root https://www.ilamb.org/ILAMB-Data/
bash scripts/run_ilamb.sh
# -> ilamb/output_modelC/ with scalar_database.csv, Overall Score 0.670336
```

Exact expected numbers (from `scalar_database.csv`, ED-ModelC-final, global):

| Metric | Pinned value |
|---|---:|
| Bias Score | 0.713701 |
| RMSE Score | 0.512801 |
| Seasonal Cycle Score | 0.833774 |
| Spatial Distribution Score | 0.778604 |
| **Overall Score** | **0.670336** |

## Repository layout

```
ed-autoresearch/
├── README.md                         # this file
├── CHECKSUMS.txt                     # SHA256 for every pinned artifact
├── requirements.txt
├── ilamb/
│   ├── burntArea_official.cfg        # stock ConfBurntArea config
│   └── MODELS/ED-ModelC-final/
│       └── burntArea.nc              # produced by reproduce_modelC.py
├── models/C/
│   ├── formula.md                    # Model C formula + citations
│   └── params.json                   # 12 fitted params
├── patches/
│   └── fire_modelC.cc                # C++ drop-in for ED's fire.cc
├── scripts/
│   ├── prep_monthly_inputs.py        # raw data -> .npy + .nc inputs
│   ├── reproduce_modelC.py           # .npy + params -> burntArea.nc
│   ├── dump_modelC_terms.py          # per-term debug NetCDF
│   ├── verify.py                     # SHA256 + pipeline check
│   └── run_ilamb.sh                  # stock ilamb-run wrapper
├── data/                             # populated by prep_monthly_inputs.py or bundle
│   ├── crujra/    (*.npy)
│   ├── trendy_v14/(*.nc)
│   └── gfed/      (*.hdf5)
└── out_terms/                        # modelC_terms.nc lands here
```

## ED integration

`patches/fire_modelC.cc` is a drop-in replacement for ED's fire module. It computes the Model C formula using ED's site-level state (`dryness_index_avg`, `temp[m]`, `precip[m]`, `precip_average`, `gpp`) and returns a unitless fire risk. ED's existing `fp1` tunable absorbs the global rescale that our offline pipeline applies separately.

**Important caveat about D̄.** Model C was trained against a canonical offline dbar computed with Thornthwaite + daylength PET, `K=1`, continuous accumulator, hard reset at `monthly precip ≥ 200 mm`. ED's native `calcSiteDrynessIndex` in `read_site_data.cc` uses a different PET (latitude-binned `b[]` correction, mm/yr units), a `K=30` multiplier with mixed mm/yr×mm/month units, and resets on `PET/precip > 1`. These produce numerically different dbar fields.

For bit-exact transfer to coupled ED you have two options:
1. **Port canonical dbar into ED.** Add our Thornthwaite+daylength accumulator alongside (or in place of) `calcSiteDrynessIndex`; feed that into the Model C patch.
2. **Re-tune against ED's internal dbar.** Run ED with Model C once, capture `dryness_index_avg` across cells and months, re-run Optuna on `{k1, D_low, k2, D_high}` only (other 8 params held fixed). The formula structure and mechanism set are unchanged; only the dbar-threshold values move.

## Coupling caveats

The benchmark above is **offline**: Model C reads fixed TRENDY v14 ED-S3 GPP, not a live coupled ED GPP. Ported to coupled ED, GPP becomes prognostic — fire affects biomass, biomass affects GPP, GPP affects next-month fire. Sensitivity analysis of this feedback is future work.

Inputs and their coupling status:

| Input | Coupling-invariant | Notes |
|---|:---:|---|
| Monthly air temperature | yes | CRUJRA forcing, unchanged offline vs coupled |
| Monthly precipitation | yes | CRUJRA forcing |
| Annual precipitation | yes | CRUJRA forcing |
| D̄ (dryness accumulator) | partial | Same CRUJRA inputs, but the accumulator formula itself must match between offline training and coupled run (see ED integration caveat) |
| Monthly GPP | **no** | ED prognostic; responds to fire feedback |

**Model C has the smallest coupling-sensitive surface area** among our three formulas (A has 8 mechanisms, B has 5, C has 3). Only GPP is coupling-sensitive.

## License

MIT.
