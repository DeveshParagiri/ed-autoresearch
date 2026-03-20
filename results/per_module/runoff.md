# Phase 4 Deep Dive: Runoff
Started: 2026-03-20 01:20:46

LORA: 55561 cells, mean=299 mm/yr

## Diagnostic
P_annual vs LORA: r=0.7646 (upper bound for runoff correlation)
P - ET_best vs LORA: r=0.8367, bias=-0.2657

Water balance check (global land mean):
  P = 714 mm/yr
  ET = 497 mm/yr
  R (model) = 219 mm/yr
  R (LORA) = 299 mm/yr
  P - ET - R(LORA) = -81 mm/yr (should be ~0)

## Runoff Composite Metric

score = 0.30 * r(LORA)              # spatial pattern
      + 0.25 * (1 - |bias|)         # total continental discharge
      + 0.20 * (1 - NRMSE)          # magnitude accuracy
      + 0.25 * water_balance_score   # P ≈ ET + R (closure)

The water_balance_score checks that R = P - ET at the zonal level.
This is THE fundamental constraint — if water balance doesn't close,
the model is physically inconsistent.


## Full Search: ET configurations optimized for runoff

### Best: priestley_taylor+choudhury+none (composite=0.7989)

## Deep Sweep (1000 trials)

### Final Results
  composite: 0.7998
  r: 0.8446
  bias: 0.0000
  nrmse: 0.7678
  wb_score: 1.0000

### Parameters
  alpha_pt: 1.5159
  choudhury_n: 4.3848

### Zonal Means (mm/yr)
Zone         | P      | ET     | R(model) | R(LORA) | P-ET-R
Arctic       |    387 |    135 |      251 |     321 |    -70
Boreal       |    585 |    304 |      281 |     288 |     -7
N.Temp       |    517 |    350 |      167 |     182 |    -14
Tropical     |   1165 |    707 |      458 |     402 |    +56
S.Temp       |    621 |    444 |      177 |     165 |    +12

Completed: 2026-03-20 01:23:25
