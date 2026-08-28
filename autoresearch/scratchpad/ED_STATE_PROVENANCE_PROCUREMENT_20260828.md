# Clean ED state procurement decision, 2026-08-28

## Verdict

No public global ED product I could verify is both reconstructible for the 2001–2016 offline evaluation and demonstrably free of prescribed-fire ancestry. The only credible score step-change is therefore a small, source-aggregated export from the researcher's clean endogenous-fire run, accompanied by its native 1850 restart and an auditable lineage manifest. Do not procure any of the public candidates as evaluation predictors.

This conclusion is narrower than “no ED data exist.” Public ED output exists and ED's state machinery contains the needed pools. What is missing is the conjunction of pre-fire timing, fire-clean ancestry, monthly 2001–2016 coverage, and a restart that can actually initialize the coupled model.

## Concrete availability and provenance

| Rank | Candidate | What is actually available | Provenance and coverage | Decision |
|---:|---|---|---|---|
| 1 | Researcher clean-run export | Not public; exact minimum request is below | Can be clean only if the restart chain, configuration, and snapshot order pass the stated gate | **PROCURE** |
| 2 | [GCB/TRENDY public ED browser](https://mdosullivan.github.io/GCB/) and [machine-readable index](https://raw.githubusercontent.com/mdosullivan/GCB/main/fileIndex_merged_v4_noLPJGUESS.json) | Current v14 S3 whole-file objects include annual [cSoil, 337 MB](https://s3.eu-west-1.wasabisys.com/gcb-2025-upload/land/output/ED/S3/EDv3_S3_cSoil.nc), annual [cVegpft, 2.36 GB](https://s3.eu-west-1.wasabisys.com/gcb-2025-upload/land/output/ED/S3/EDv3_S3_cVegpft.nc), monthly [fHarvest, 4.04 GB](https://s3.eu-west-1.wasabisys.com/gcb-2025-upload/land/output/ED/S3/EDv3_S3_fHarvest.nc), monthly [mrso, 4.04 GB](https://s3.eu-west-1.wasabisys.com/gcb-2025-upload/land/output/ED/S3/EDv3_S3_mrso.nc), and monthly [laipft, 28.30 GB](https://s3.eu-west-1.wasabisys.com/gcb-2025-upload/land/output/ED/S3/EDv3_S3_laipft.nc). The [v13 README](https://s3.eu-west-1.wasabisys.com/gcb-2024-trendy/ED/readme.txt) documents 1700–2023, monthly 0.5° fluxes/soil moisture/LAI and annual carbon pools. | The public index contains no ED restart, history file, namelist, or configuration; v14 does not even contain an ED README. cSoil combines soil carbon with below-ground litter, PFT products are coarse grid aggregates, and there is no pre-fire litter, exact fire-moisture driver, SWE, patch age, or cohort state. Whole objects are exposed, not an advertised server-side 2001–2016 subset. The lineage is therefore not certifiable even though the time axis crosses 1850. | **NO PROCURE** |
| 3 | [Ma et al. EDv3 archive](https://zenodo.org/records/6901510) | 16.9 GB total: 15.30 GB inputs and a 1.576 GB simulation ZIP. Byte-range inspection of the ZIP central directory found exactly one substantive file, `EDv3_global_simulation_1981_2016.nc`; there is no restart in it. | The [model paper](https://gmd.copernicus.org/articles/15/1971/2022/) states that the transient ran AD 851–2016 and prescribed GFED4 burned area, using the 1996–2016 mean before 1996 and annual GFED4 thereafter. Every resulting vegetation, litter, patch-age, and soil state is consequently descended from prescribed observed fire. | **NO PROCURE** |
| 4 | [Public ED2 source](https://github.com/EDmodel/ED2/tree/d971a62058f67c782557021f9a7397eb2492ef46) | Code and restart machinery, but no verified public global clean restart or monthly state cube | It proves that the state is technically exportable, not that a usable global dataset exists | **NO DATA TO PROCURE; use as export specification** |

The archived Ma run cannot be rehabilitated by selecting 2001–2016 or by omitting burned area: the candidate predictor states themselves were evolved under prescribed GFED4. The GCB time series cannot initialize ED in 1850 because gridded annual/monthly diagnostics are not a native patch/cohort restart.

## What ED exposes and what is relevant

Pinned ED2 source shows the exact state classes needed. It stores `fast_grnd_C` and `structural_grnd_C` as fast litter and woody debris pools ([state lines 1149–1162](https://github.com/EDmodel/ED2/blob/d971a62058f67c782557021f9a7397eb2492ef46/ED/src/memory/ed_state_vars.F90#L1149-L1162)). Its active fuel calculation can sum those pools with `nplant * agb` for grasses and cohorts below the configured fuel-height threshold ([fire lines 144–162](https://github.com/EDmodel/ED2/blob/d971a62058f67c782557021f9a7397eb2492ef46/ED/src/dynamics/fire.f90#L144-L162)). It stores `avg_monthly_gndwater`, explicitly described as used for fire ignition ([state lines 1333–1340](https://github.com/EDmodel/ED2/blob/d971a62058f67c782557021f9a7397eb2492ef46/ED/src/memory/ed_state_vars.F90#L1333-L1340)), and the fire routine compares that value with the ignition threshold ([fire lines 179–235](https://github.com/EDmodel/ED2/blob/d971a62058f67c782557021f9a7397eb2492ef46/ED/src/dynamics/fire.f90#L179-L235)).

Snow is represented by surface-water mass, depth, and liquid fraction ([state lines 1246–1262](https://github.com/EDmodel/ED2/blob/d971a62058f67c782557021f9a7397eb2492ef46/ED/src/memory/ed_state_vars.F90#L1246-L1262)), plus total surface-water depth and snow-covered vegetation fraction ([state lines 1709–1716](https://github.com/EDmodel/ED2/blob/d971a62058f67c782557021f9a7397eb2492ef46/ED/src/memory/ed_state_vars.F90#L1709-L1716)); SWE is therefore derivable as the summed solid surface-water mass. ED also holds patch area/age/disturbance type and cohort PFT, density, height, DBH, and biomass in its restart hierarchy ([history lines 3794–3809](https://github.com/EDmodel/ED2/blob/d971a62058f67c782557021f9a7397eb2492ef46/ED/src/io/ed_init_history.f90#L3794-L3809), [history lines 5080–5103](https://github.com/EDmodel/ED2/blob/d971a62058f67c782557021f9a7397eb2492ef46/ED/src/io/ed_init_history.f90#L5080-L5103)). Crop and management state includes agriculture stocking, harvest targets/memory, monthly crop yield, and crop harvest ([state lines 2438–2469](https://github.com/EDmodel/ED2/blob/d971a62058f67c782557021f9a7397eb2492ef46/ED/src/memory/ed_state_vars.F90#L2438-L2469), [state lines 2514–2526](https://github.com/EDmodel/ED2/blob/d971a62058f67c782557021f9a7397eb2492ef46/ED/src/memory/ed_state_vars.F90#L2514-L2526)).

This ED2 evidence must not be mistaken for a GlobalEDv3 name guarantee. The repository's prior GlobalED coupling inspection at commit `11ee714` verified `dryness_index_avg`, live GPP, and live AGB in `COUPLING_SPEC_for_Lei.md`; the researcher must map the remaining fields to the exact GlobalEDv3 build and export the values actually consumed by its fire routine.

## Exact smallest researcher request

Ask for one source-aggregated CF NetCDF4 file, `ed_clean_prefire_1deg_200012_201612.nc`, with float32 values, lossless deflate, evaluator-aligned 1° cells, and these dimensions and variables:

```text
time = 193 monthly snapshots: 2000-12, then 2001-01 through 2016-12
lat = 180; lon = 360
pft_group = 7: C3 grass/shrub, C4 grass/shrub, early/mid/late broadleaf,
                late-successional conifer, northern/southern pine
patch_age_bin = [0,1), [1,5), [5,20), [20,inf) years

fast_grnd_c(time,lat,lon)                    kg C m-2
structural_grnd_c(time,lat,lon)              kg C m-2
eligible_live_fuel_c_by_pft(time,pft_group,lat,lon)
                                               kg C m-2; exact cohort eligibility used by fire
fire_moisture_driver(time,lat,lon)            native units; exact scalar passed to fire
swe(time,lat,lon)                             kg H2O m-2; summed solid surface-water mass
patch_area_by_age_bin(time,patch_age_bin,lat,lon)
                                               fraction of cell, sums to vegetated patch area
crop_area_fraction(time,lat,lon)              fraction of cell
crop_harvest_c(time,lat,lon)                  kg C m-2 month-1
```

These are 17 scalar-equivalent channels. At 193 × 180 × 360 float32 values the cube is 850,435,200 bytes, or 811 MiB before compression. Raw cohorts and patches are unnecessary. The exporter should area-weight source patches, conserve extensive carbon and water, preserve missing ocean cells, include time bounds, and record the exact PFT mapping and aggregation formula in attributes.

Every snapshot must be taken at one explicitly documented instant immediately before that month's fire ignition/disturbance update. A monthly post-fire mean is not acceptable because it leaks same-month model fire consequences into the predictor. If GlobalEDv3 has no literal fuel-moisture pool, `fire_moisture_driver` must be its actual `dryness_index_avg` or successor, not a newly inferred surrogate.

The data file alone is not enough. Request `ed_clean_18500101_restart.*`, the complete native restart at 1850-01-01, plus a small `provenance.json` containing the code commit and executable hash, complete configuration, forcing names and SHA-256 hashes, restart parent chain back to bare-ground/equilibrium spin-up, export-script hash, snapshot/update order, grid mapping, and the run's fire configuration. The researcher must affirm that no ancestor used prescribed, assimilated, or spatially weighted GFED, FireCCI, or other observational burned area/fire emissions. Any generic fire-parameter calibration must be declared. Without the full native restart, the aggregate cube supports offline evaluation only and cannot support a coupled 1850 start.

## Acceptance decision

Procure only if the researcher supplies both artifacts and the lineage gate passes. Reject a “clean” export if its restart descends from the published GFED-prescribed transient, if the fields are post-fire monthly means, if only gridded TRENDY diagnostics are supplied, or if code/config/restart ancestry is missing. This request is smaller and more diagnostic than downloading even one monthly TRENDY variable, while directly representing the missing fuel amount, moisture gate, snow constraint, demographic structure, and harvest timing.

## Claude consultation status

The local Claude CLI was available, but its OAuth session had expired and no API-key credential was present, so Claude returned no scientific answer. The procurement decision above is independently verified from repository history and the linked primary/public sources.
