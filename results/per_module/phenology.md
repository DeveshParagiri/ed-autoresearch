# Phase 5 Deep Dive: Phenology (LAI)
Started: 2026-03-20 01:34:09

MODIS LAI: 54776 cells, mean=1.39 m²/m²
  range: [0.000, 5.88]
ED LAI vs MODIS: r=0.7558 (on 47755 cells)

## Diagnostic: Predictor Correlations
Predictor                                | r(MODIS LAI)
----------------------------------------------------------
  Precipitation (annual)                   | 0.7621
  sqrt(P * GS_months)                      | 0.7439
  P / PET (wetness)                        | 0.5847
  min(GS, P/500)                           | 0.5434
  log(1 + P) * GS/12                       | 0.5154
  Temperature (annual)                     | 0.3381
  Growing season months (T>5)              | 0.3196
  Growing season months (T>0)              | 0.3001

## Phenology Composite Metric

score = 0.30 * r(MODIS)              # spatial pattern
      + 0.25 * (1 - |bias|)          # global mean LAI
      + 0.20 * (1 - NRMSE)           # magnitude accuracy
      + 0.25 * zonal_skill           # biome-level (tropical LAI >> boreal LAI)

Why: LAI feeds into radiation partitioning (ET), fire fuel load, and
soil carbon litter inputs. Getting the right LAI per biome is critical
for all downstream modules.


## Structural Search (5 models × 500 trials)

  **gsi**: comp=0.7615, r=0.8103, bias=+0.0009, nrmse=0.525, zonal=0.695
    Params: {'LAI_max': 7.8571, 'T_leaf': 0.2581, 'T_slope': 0.0957, 'W_half': 2.1701}
    Boreal: model=1.27, MODIS=1.50
    Tropical: model=2.13, MODIS=2.17

  **gs_length**: comp=0.7085, r=0.7185, bias=-0.0008, nrmse=0.625, zonal=0.673
    Params: {'LAI_max': 5.3944, 'T_min_lai': -0.4649, 'P_half_lai': 1358.6907}
    Boreal: model=0.99, MODIS=1.50
    Tropical: model=2.43, MODIS=2.17

  **budyko_veg**: comp=-0.1065, r=-0.5836, bias=+0.2110, nrmse=1.089, zonal=-0.515
    Params: {'LAI_max': 2.0005, 'bud_w': 5.9954}
    Boreal: model=1.67, MODIS=1.50
    Tropical: model=1.60, MODIS=2.17

  **colimit**: comp=0.7578, r=0.7870, bias=-0.0000, nrmse=0.558, zonal=0.733
    Params: {'LAI_max': 4.8479, 'T_opt_lai': 20.644, 'T_width_lai': 20.3535, 'P_half_lai': 1464.1325}
    Boreal: model=1.27, MODIS=1.50
    Tropical: model=2.11, MODIS=2.17

  **biome_blend**: comp=0.7661, r=0.7983, bias=+0.0004, nrmse=0.541, zonal=0.739
    Params: {'LAI_cold': 2.1515, 'LAI_tropical': 3.0492, 'LAI_dry': 2.7467, 'T_cold_lai': -3.702, 'AI_dry_lai': 1.1849, 'P_scale_lai': 0.0007}
    Boreal: model=1.43, MODIS=1.50
    Tropical: model=1.92, MODIS=2.17

### Winner: biome_blend (composite=0.7661)

## Deep Sweep: biome_blend (1500 trials)

### Final Results
  composite: 0.7670
  r: 0.8042
  bias: -0.0006
  nrmse: 0.5369
  zonal_skill: 0.7332

### Parameters
  AI_dry_lai: 1.0515
  LAI_cold: 2.9304
  LAI_dry: 2.9610
  LAI_tropical: 3.2454
  P_scale_lai: 0.0005
  T_cold_lai: -1.9825

### Zonal Means (m²/m²)
Zone         | Model | MODIS | ED sim | Ratio
Arctic       |  0.98 |  0.80 |  0.94  | 1.23
Boreal       |  1.39 |  1.50 |  1.78  | 0.93
N.Temp       |  1.20 |  0.94 |  1.68  | 1.28
Tropical     |  1.91 |  2.17 |  2.52  | 0.88
S.Temp       |  1.07 |  0.97 |  1.85  | 1.11

## Ecological Interpretation

Winner is the biome-blended model:
  LAI = w_cold × LAI_cold × GS_fraction
      + w_wet × LAI_tropical × moisture
      + w_dry × LAI_dry × (1 - exp(-P_scale × P))

Physical basis: LAI is controlled by different factors in different biomes.
Cold regime: growing season length limits total leaf area (short summers → low LAI)
Wet regime: year-round warmth allows high LAI, limited by self-shading
Dry regime: precipitation controls leaf display (drought deciduous phenology)


Completed: 2026-03-20 01:35:25
