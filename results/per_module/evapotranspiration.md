# Phase 2 Deep Dive: Evapotranspiration
Started: 2026-03-20 01:03:07

## Diagnostic Summary
- GLEAM: 70766 cells, mean=436 mm/yr
- Key predictor: P*PET/(P+PET) has r=0.90 with GLEAM
- Weak spots: Arctic (-70 mm/yr bias), hyper-arid (r=0.36)
- Using hybrid vegetation fields (30% more land coverage)

## Axis 1: Potential ET Methods

Physical basis for each:
- **Penman-Monteith** (CRU TS): Energy balance + aerodynamic resistance.
  Gold standard but depends on radiation, humidity, wind data quality.
  CRU TS PM is precomputed from reanalysis — may have systematic biases.
  Allen et al. 1998 (FAO-56).
  
- **Thornthwaite**: Empirical T→PET relationship. Only needs temperature.
  Underestimates PET in arid regions (no humidity/wind info).
  Thornthwaite 1948.
  
- **Hargreaves**: Uses diurnal temperature range as radiation proxy.
  Better than Thornthwaite in arid regions. DTR captures cloud cover.
  Hargreaves & Samani 1985.
  
- **Priestley-Taylor** (new): Simplified energy balance.
  PET = alpha * (Delta/(Delta+gamma)) * Rn/lambda
  alpha=1.26 is the classic value (Priestley & Taylor 1972).
  We approximate Rn from latitude and temperature.


## Axis 2: Actual ET / Moisture Constraint

Physical basis for each:
- **Budyko** (Fu 1981): ET/P = 1 + PET/P - (1+(PET/P)^w)^(1/w)
  Classic water-energy balance. w controls curvature.
  w≈2.6 is the original Budyko curve (1974).
  
- **Zhang** (2001): ET/P = (1+w*PET/P)/(1+w*PET/P+P/PET)
  Vegetation-dependent: w varies by land cover type.
  Higher w = more evapotranspiration (forests > grasslands).
  
- **Turc-Pike**: ET = P / sqrt(1 + (P/PET)^2)
  Alternative to Budyko with different curvature (Turc 1954).
  
- **ED-style**: AET = PET * (P/PET)^alpha
  Simple power-law moisture constraint.
  
- **Choudhury** (1999): ET = P*PET / (P^n + PET^n)^(1/n)
  Generalized Budyko with parameter n controlling transition.
  n=1: harmonic mean; n→∞: min(P,PET).


## Axis 3: Canopy / Vegetation Effects

Physical basis:
- **None**: pure climate-driven ET
- **LAI partition**: Beer's law canopy interception.
  Fc = 1-exp(-k*LAI) partitions radiation between canopy and soil.
  Canopy transpiration > bare soil evaporation at same energy input.
  
- **GPP scaling**: water use efficiency (WUE) linkage.
  Stomatal conductance links CO2 uptake (GPP) and water loss (transpiration).
  Higher GPP → more open stomata → more transpiration (Ball-Berry 1987).
  
- **Interception loss** (new): canopy rainfall interception.
  Some precipitation is intercepted by leaves and evaporates directly.
  This ADDS to ET without reaching the soil.
  Interception ∝ LAI * rainfall frequency.
  Physical basis: Gash 1979 interception model.


## Axis 4: Cold-Season Limitation (new)

Physical basis: ET is near-zero when soil and canopy are frozen.
ED's current model doesn't explicitly account for this at the monthly scale.
Our diagnostic showed Arctic ET is underestimated by 70 mm/yr — but the
current model ALSO slightly underestimates, so frozen-season limitation
would make it worse. The Arctic error may be from summer underestimate instead.

We'll test a growing-season scaling:
- Count months where T > T_threshold
- Scale ET by growing season fraction
- Physical basis: frozen soil prevents water uptake; no transpiration in winter.


## Full Structural Search
Testing all meaningful combinations with 50 trials each

Total combinations: 96

### Top 15 Configurations
Config                                                  | comp  | gleam_r | modis_r | bias
-----------------------------------------------------------------------------------------------
hargreaves+zhang+gpp_scaling                            | 0.939 | 0.930   | 0.916   | -0.0027
hargreaves+zhang+interception                           | 0.937 | 0.926   | 0.914   | +0.0009
priestley_taylor+budyko+lai_partition                   | 0.936 | 0.924   | 0.915   | -0.0043
priestley_taylor+choudhury+lai_partition                | 0.936 | 0.927   | 0.909   | +0.0028
hargreaves+turc_pike+gpp_scaling                        | 0.935 | 0.922   | 0.915   | +0.0021
priestley_taylor+turc_pike+gpp_scaling                  | 0.935 | 0.924   | 0.912   | -0.0034
priestley_taylor+budyko+gpp_scaling                     | 0.935 | 0.926   | 0.905   | -0.0010
priestley_taylor+choudhury+gpp_scaling                  | 0.934 | 0.926   | 0.910   | -0.0076
priestley_taylor+ed_style+interception                  | 0.934 | 0.926   | 0.905   | -0.0018
priestley_taylor+zhang+lai_partition                    | 0.934 | 0.923   | 0.910   | +0.0031
hargreaves+budyko+gpp_scaling                           | 0.934 | 0.922   | 0.910   | +0.0005
hargreaves+zhang+lai_partition                          | 0.934 | 0.923   | 0.914   | +0.0078
hargreaves+linear_moisture+interception                 | 0.934 | 0.922   | 0.909   | +0.0005
hargreaves+budyko+interception                          | 0.933 | 0.923   | 0.910   | -0.0055
priestley_taylor+choudhury+interception                 | 0.933 | 0.924   | 0.906   | -0.0011

### Bottom 5 (what fails)
hargreaves+budyko+none                                  | 0.851 | 0.889   | 0.880   | -0.2879
hargreaves+ed_style+none                                | 0.851 | 0.895   | 0.893   | -0.3226
hargreaves+choudhury+none                               | 0.850 | 0.891   | 0.882   | -0.2979
penman_monteith+turc_pike+none                          | 0.850 | 0.873   | 0.871   | -0.2380
hargreaves+turc_pike+none                               | 0.839 | 0.894   | 0.888   | -0.3692

## Deep Parameter Sweep: hargreaves+zhang+gpp_scaling
1000 trials with Optuna TPE + CMA-ES comparison

### TPE (1000 trials): composite = 0.9396

### Final Results
  composite: 0.9396
  gleam_r: 0.9306
  gleam_bias: -0.0005
  gleam_nrmse: 0.3355
  modis_r: 0.9147
  modis_bias: 0.0980

### Optimized Parameters
  DTR: 14.7394
  wue_scale: 0.1574
  zhang_w: 10.6707

### Zonal Means (mm/yr)
Zone         | Model  | GLEAM  | MODIS  | Residual
Arctic       |    165 |    234 |    308 |    -69
Boreal       |    384 |    428 |    396 |    -44
N.Temp       |    412 |    392 |    436 |    +20
Tropical     |    861 |    816 |    830 |    +45
S.Temp       |    506 |    469 |    344 |    +37

### Performance by Aridity Class
Class        | r(GLEAM) | mean_residual | n_cells
Hyper-arid   | 0.353    |      -10 mm/yr | 10964
Arid         | 0.689    |       -9 mm/yr | 9288
Semi-arid    | 0.895    |      -47 mm/yr | 17386
Sub-humid    | 0.947    |       -2 mm/yr | 13685
Humid        | 0.967    |     +103 mm/yr | 9913

Completed: 2026-03-20 01:08:26

---

## Step 3: Multi-Metric Re-Optimization

### Revised Composite Objective

Previous: 0.5*gleam_r + 0.3*modis_r + 0.2*(1-|bias|)
Problem: Over-weights pattern correlation, under-weights magnitude accuracy.

New composite (hydrology-appropriate):
  score = 0.30 * gleam_r         # spatial pattern
        + 0.20 * modis_r         # independent pattern validation
        + 0.20 * (1 - |bias|)    # global water budget
        + 0.15 * (1 - NRMSE)     # magnitude accuracy (clipped to [0,1])
        + 0.15 * zonal_skill     # mean of per-zone correlations

Why these weights:
- Pattern (r) and magnitude (NRMSE, bias) are equally important for hydrology
- Zonal skill catches models that get global r right but fail in specific climate zones
- MODIS provides independent validation but gets lower weight (more uncertainty)


### Re-scoring top configs with hydrology composite
  hargreaves+zhang+gpp_scaling                       | comp=0.8874 | r=0.928 | nrmse=0.324 | zonal=0.839 | bias=-0.0002
  hargreaves+zhang+interception                      | comp=0.8880 | r=0.928 | nrmse=0.324 | zonal=0.840 | bias=-0.0005
  priestley_taylor+budyko+lai_partition              | comp=0.8799 | r=0.927 | nrmse=0.347 | zonal=0.819 | bias=+0.0016
  priestley_taylor+choudhury+lai_partition           | comp=0.8733 | r=0.923 | nrmse=0.383 | zonal=0.819 | bias=-0.0082
  hargreaves+zhang+lai_partition                     | comp=0.8892 | r=0.930 | nrmse=0.323 | zonal=0.846 | bias=-0.0044

### Best with hydrology composite:
  Config: hargreaves+zhang+lai_partition
  composite: 0.8892
  gleam_r: 0.9299
  modis_r: 0.9139
  bias: -0.0044
  nrmse: 0.3232
  zonal_skill: 0.8456
  Params: {'DTR': 14.12071410977687, 'zhang_w': 11.897946638151327, 'k_extinction': 0.49473914549727654, 'transp_scale': 1.1880717931589377}

### Final 1000-trial sweep with hydrology composite

### Final Results (hydrology composite)
  composite: 0.8913
  gleam_r: 0.9308
  modis_r: 0.9128
  bias: 0.0001
  nrmse: 0.3163
  zonal_skill: 0.8466

### Optimized Parameters
  DTR: 21.6566
  k_extinction: 1.5236
  transp_scale: 1.2443
  zhang_w: 5.2790

### Old metric vs New metric comparison
Metric          | Old composite | New composite
gleam_r         | 0.946         | 0.931
modis_r         | 0.915         | 0.913
bias            | +0.002        | +0.0001
nrmse           | 0.280         | 0.316
zonal_skill     | (not tracked) | 0.847

Saved to experiments/evapotranspiration/best_config_v2_hydrology.json
Completed: 2026-03-20 01:12:42
