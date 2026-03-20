# Phase 6: Cross-Module Joint Optimization
Started: 2026-03-20 01:36:54

## Current independent-module baseline
  Soil carbon: r=0.4805 (HWSD), ncscd_r=0.1937
  ET:          r=0.9308 (GLEAM)
  Runoff:      r=0.8446 (LORA)

## Cross-Module Consistency Check

Water balance (P - ET_model - R_model):
  Global mean residual: -81 mm/yr
  NOTE: ET and Runoff were optimized with DIFFERENT ET configs!
  ET config: hargreaves+zhang
  Runoff config: priestley_taylor+choudhury

Litter input consistency:
  GPP-implied litter: 371 gC/m2/yr
  Model Rh: 254 gC/m2/yr
  Hashimoto Rh: 375 gC/m2/yr

## Joint Optimization

### Strategy
1. Use ONE ET model for both ET prediction and runoff (P - ET)
2. Jointly optimize soil carbon params + ET params
3. Weighted objective:
   - 0.30 × soil_score (iLAMB worst tier)
   - 0.25 × et_score
   - 0.20 × runoff_score  
   - 0.15 × ncscd_score (high-lat soil C)
   - 0.10 × water_balance_closure


### Joint Optimization Results (2000 trials)
Best joint score: 0.7459

### Per-Module Comparison: Independent vs Joint
Module          | Independent r | Joint r    | Independent bias | Joint bias
--------------------------------------------------------------------------------
Soil Carbon     | 0.4805        | 0.4824     | -0.0001           | +0.0050
ET              | 0.9308        | 0.9295     | -0.0012           | -0.0012
Runoff          | 0.8446        | 0.8297     | +0.0000           | -0.2598
NCSCD           | 0.1937        | 0.1938     |                  |

### Water Balance Closure
  Global mean: P=722, ET=490, R=221
  Residual: -3 mm/yr (should be ~0)

### Joint Parameters
ET: {'DTR': 15.430722829053412, 'zhang_w': 5.187029584087068, 'k_extinction': 1.0301150278197397, 'transp_scale': 1.4368359039821808}
Soil Carbon: {'R0': 1.9257348528373872, 'E0': 243.49874561379545, 'T_freeze': 3.6090741814359166, 'theta_opt': 0.1948161142817928, 'moist_sigma': 3.7046582247320266, 'K_labile': 0.6081020683330864, 'K_stable': 0.008706572471905693, 'r_labile': 0.6644530200629973, 'r_stable': 0.1276792741068321, 'f_stable': 0.29237857444218207, 'f_litter': 0.5508263697897924}

Completed: 2026-03-20 01:42:16
