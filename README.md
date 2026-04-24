# ED Autoresearch — Model C Fire Module

Closed-form, mechanistic fire formula for the ED v3 dynamic global vegetation model. Trained against GFED4.1s via Optuna, scored via **official `ilamb-run`** with stock `ConfBurntArea`.

> **All training and benchmarking is OFFLINE** against fixed TRENDY v14 ED-S3 NetCDFs + CRUJRA climate, not a coupled ED run. Coupled deployment notes at the bottom of this README.

**Model C — Overall Score 0.6703, rank #1 on TRENDY v14 + baseline leaderboard.** 12 parameters, 3 mechanism groups, 5 input fields.

## ILAMB leaderboard (real `ilamb-run`, stock ConfBurntArea)

| Rank | Model | Bias | RMSE | Seasonal | Spatial | **Overall** |
|---:|---|---:|---:|---:|---:|---:|
| 🥇 1 | **ED-ModelC (ours)** | **0.714** | 0.513 | **0.834** | 0.779 | **0.6703** |
| 🥈 2 | CLASSIC | 0.738 | 0.507 | 0.782 | 0.797 | 0.6660 |
| 🥉 3 | CLM6.0 | 0.759 | 0.474 | 0.758 | **0.838** | 0.6606 |
| 4 | CLM-FATES | 0.725 | 0.525 | 0.802 | 0.707 | 0.6568 |
| 5 | ELM-FATES | 0.724 | 0.512 | 0.860 | 0.676 | 0.6568 |
| 6 | JULES-ES | 0.709 | 0.506 | 0.784 | 0.447 | 0.5905 |
| 7 | ELM | 0.687 | 0.492 | 0.778 | 0.333 | 0.5564 |
| 8 | VISIT-UT | 0.636 | 0.488 | 0.695 | 0.466 | 0.5545 |
| 9 | LPJmL | 0.693 | 0.489 | 0.459 | 0.612 | 0.5485 |
| 10 | LPJ-GUESS | 0.671 | 0.489 | 0.459 | 0.288 | 0.4792 |
| 11 | **EDv3 (stock)** | 0.681 | 0.489 | 0.439 | 0.290 | **0.4774** |
| 12 | LPJ-EOSIM | 0.654 | 0.489 | 0.459 | 0.227 | 0.4635 |

"Overall Score" is ILAMB's native tier-2 aggregation `(2·Bias + 2·RMSE + Seasonal + Spatial) / 6` pulled directly from `scalar_database.csv`. Component scores are also unmodified from the same CSV. No custom aggregation.

Model C beats ED's stock fire module by **+0.19 Overall** (0.6703 − 0.4774).

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
