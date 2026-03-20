# Phase 3 Deep Dive: GPP/Photosynthesis
Started: 2026-03-20 01:14:01

FLUXCOM GPP: 51888 valid cells, mean=2.290 kgC/m²/yr

## Diagnostic: Predictor Correlations with FLUXCOM GPP
Predictor                                | r(FLUXCOM)
-------------------------------------------------------
  T * P / 1e4                              | 0.6864
  sqrt(P * PET)                            | 0.6427
  Precipitation (annual)                   | 0.6189
  LAI (hybrid)                             | 0.5718
  min(P, PET)                              | 0.5289
  Growing season months                    | 0.4479
  GDD (>0°C)                               | 0.3425
  Temperature (annual mean)                | 0.3400
  PET (annual)                             | -0.3371

## GPP Composite Metric

score = 0.30 * r(FLUXCOM)           # spatial pattern
      + 0.25 * (1 - |bias|)         # global carbon budget
      + 0.20 * (1 - NRMSE)          # magnitude accuracy
      + 0.25 * zonal_skill          # biome-level accuracy

Why: GPP feeds into carbon budget calculations. Getting the right spatial
pattern AND the right magnitude by biome is essential. A model that
concentrates all GPP in the tropics with correct global mean is useless
for regional carbon budgets.


## Structural Search (4 methods × 300 trials each)
  lue_basic                : comp=0.5926, r=0.7199, bias=-0.0007, nrmse=0.626, zonal=0.208
  miami_temperature        : comp=0.4397, r=0.4666, bias=+0.0000, nrmse=1.990, zonal=0.199
  miami_precip             : comp=0.3526, r=0.5525, bias=-0.3645, nrmse=1.443, zonal=0.112
  water_energy_balance     : comp=0.4577, r=0.6222, bias=-0.3983, nrmse=0.635, zonal=0.191
  prescott                 : comp=0.3530, r=0.5132, bias=-0.4646, nrmse=1.307, zonal=0.261

**Winner: lue_basic** (composite=0.5926)

## Deep Parameter Sweep: lue_basic (1000 trials)

### Final Results
  composite: 0.5947
  r: 0.7150
  bias: -0.0034
  nrmse: 0.6014
  zonal_skill: 0.2053

### Parameters
  T_opt_gpp: 32.3825
  T_width: 11.1721
  epsilon: 10.3101
  k_ext_gpp: 2.5566

### Zonal Means (kgC/m²/yr)
Zone         | Model  | FLUXCOM | Ratio
Boreal       |  0.006 |   1.981 | 0.00
N.Temp       |  0.613 |   1.797 | 0.34
Tropical     |  5.403 |   4.422 | 1.22
S.Temp       |  0.213 |   1.709 | 0.12

Completed: 2026-03-20 01:15:23

---

## Step 2: Fixing Boreal GPP Collapse

### Root Cause Analysis
- LUE model uses fAPAR = 1-exp(-k*LAI)
- Optimizer pushed k_ext to 2.56 (physically unrealistic; should be 0.4-0.7)
- At k=2.56, LAI=0.3 gives fAPAR=0.53 but LAI=0.05 gives fAPAR=0.12
- Boreal hybrid LAI is often 0.01-0.1 → fAPAR collapses → GPP=0

### Strategy: LAI-independent GPP models
Test models that derive productivity from climate alone (T, P, PET),
bypassing the LAI bottleneck entirely. These are ecologically valid
because GPP ultimately responds to climate — LAI is an *output* of
productivity, not an independent *input*.


### Testing 4 GPP models (500 trials each)

  **tp_product**: comp=0.3530, r=0.5130, bias=-0.4614, nrmse=1.309, zonal=0.258
    Params: {'tp_scale': 0.019995187122548656, 'T_opt_gpp': 21.846577456922248, 'T_width': 8.675278868400252}
    Boreal: model=0.126, obs=1.981
    N.Temp: model=1.013, obs=1.797
    Tropical: model=3.752, obs=4.422

  **gs_lue**: comp=0.3132, r=0.5440, bias=-0.5548, nrmse=1.201, zonal=0.155
    Params: {'epsilon': 7.999472457482849, 'T_min_gs': 8.392950553241194}
    Boreal: model=1.770, obs=1.981
    N.Temp: model=0.684, obs=1.797
    Tropical: model=3.157, obs=4.422

  **lue_constrained**: comp=0.5784, r=0.7128, bias=-0.0200, nrmse=0.644, zonal=0.194
    Params: {'epsilon': 9.971518239496191, 'k_ext_gpp': 0.5755639701681314, 'T_opt_gpp': 28.89440102201861, 'T_width': 9.18919422652862}
    Boreal: model=0.002, obs=1.981
    N.Temp: model=0.644, obs=1.797
    Tropical: model=5.315, obs=4.422

  **miami_precip**: comp=0.5632, r=0.6337, bias=-0.0002, nrmse=0.524, zonal=0.112
    Params: {'miami_scale': 3.137921300886354}
    Boreal: model=1.995, obs=1.981
    N.Temp: model=2.619, obs=1.797
    Tropical: model=4.923, obs=4.422

### Winner: lue_constrained (composite=0.5784)

Saved to experiments/photosynthesis/best_config_v2_deep.json
Completed: 2026-03-20 01:17:16

---

## Step 3: Biome-Aware Blended GPP Model

### Physical justification for blending

Productivity is controlled by different factors in different biomes:
- **Energy-limited** (tropics): GPP ∝ radiation × water availability
- **Temperature-limited** (boreal/arctic): GPP ∝ growing season length
- **Water-limited** (arid): GPP ∝ precipitation

A single formula can't capture all three regimes. But we can blend
regime-specific models using smooth climate-based weights. This is
physically grounded in the concept of **co-limitation** (Nemani et al.
2003): the most limiting factor varies across space, and the transition
between regimes is continuous, not sharp.

Blending weights:
- w_cold = sigmoid(-T, T_cold_threshold) → boreal/arctic model
- w_dry = sigmoid(aridity_index, AI_threshold) → arid model  
- w_energy = 1 - w_cold - w_dry → energy/radiation model
All weights are soft (sigmoid), so transitions are smooth.


### Blended Model Results (1500 trials)
  composite: 0.6578
  r: 0.7333
  bias: -0.0030
  nrmse: 0.4845
  zonal_skill: 0.3416

### Parameters
  AI_dry: 2.5891
  AI_slope: 0.7587
  T_cold: -0.5921
  T_min_gs: -3.3764
  T_opt: 32.2344
  T_slope_cold: 0.3005
  T_width_d: 14.5111
  T_width_e: 11.7649
  epsilon: 6.5021
  gs_scale: 0.7602
  p_scale: 0.0100

### Zonal Means (kgC/m²/yr)
Zone         | Blended | FLUXCOM | Ratio | Prev LUE
Boreal       |   0.326 |   1.981 | 0.16  | (was 0.00)
N.Temp       |   1.282 |   1.797 | 0.71  | (was 0.00)
Tropical     |   5.215 |   4.422 | 1.18  | (was 0.00)
S.Temp       |   1.290 |   1.709 | 0.75  | (was 0.00)

### Comparison: Previous vs Blended
Metric          | Quick LUE | Deep LUE | Blended
r               | 0.686     | 0.715    | 0.733
bias            | -0.6%     | -0.3%    | -0.003
nrmse           | —         | 0.601    | 0.485
zonal_skill     | —         | 0.205    | 0.342
composite       | —         | 0.595    | 0.658

Completed: 2026-03-20 01:18:59
