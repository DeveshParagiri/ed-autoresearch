# Model B — Shapley-Reduced 5-Mechanism Formula

**17 parameters, 5 mechanism groups, Official ILAMB Overall: 0.6943**
**Rank: #7 of 24 models on TRENDY v14 leaderboard**

Middle-tier simplification of Model A. Keeps the top-5 Shapley mechanisms, drops fuel, height, and soil_temp (the 3 lowest-φ mechanisms from the v8 Shapley decomposition).

## Formula

```
fire(cell, month) = 
    [ onset(D̄) × suppress(D̄)                          # ignition (always-on base)
    × precip_floor(P_ann) × precip_dampen(P_month)      # M3: precip
    × gpp_hump(GPP_month)                               # M5: gpp_monthly
    × gpp_anom_supp(GPP - cell_mean) × fuel_anom_boost  # M6: gpp_anom
    × surf_temp_gate(T_surf)                            # M7: t_surf
    × air_temp_ign(T_air)                               # M8: t_air_ign
    ]^fire_exp
```

## Why these 5 mechanisms

From Model A Shapley (see `../shapley.json`):

| Rank | Mechanism | φ | % | Kept? |
|------|-----------|---:|---:|:-----:|
| 1 | t_air_ign | +0.021 | 18.5% | ✓ |
| 2 | precip | +0.020 | 17.5% | ✓ |
| 3 | gpp_monthly | +0.019 | 16.7% | ✓ |
| 4 | gpp_anom | +0.015 | 13.2% | ✓ |
| 5 | t_surf | +0.014 | 12.6% | ✓ |
| 6 | soil_temp | +0.011 | 9.9% | ✗ |
| 7 | height | +0.007 | 6.4% | ✗ |
| 8 | fuel | +0.006 | 5.2% | ✗ |

B's 5 mechanisms cover **78% of Model A's total explained Shapley variance**. Dropping fuel was counter-intuitive — Shapley says the GPP-based mechanisms (M5, M6) already carry most of the biomass/productivity signal that fuel was encoding.

## Scores (Official ILAMB)

| Metric | Score | vs Model A | vs CLM6.0 |
|--------|------:|-----------:|----------:|
| Bias | 0.706 | −0.010 | −0.053 |
| RMSE | 0.476 | −0.016 | +0.002 |
| Seasonal | **0.833** | **+0.028** | **+0.075** |
| Spatial | 0.763 | −0.020 | −0.075 |
| **Overall** | **0.6943** | −0.005 | −0.013 |

**Model B actually improves Seasonal over A** by +0.028 — fewer mechanisms, less interference. Loses some Spatial (no explicit height or fuel) but within the ILAMB top 7.

## Inputs

Subset of Model A's: CRUJRA (D̄, T_surf, T_air, precip) + TRENDY v14 GPP + derived GPP anomaly. Does **not** need AGB, LAI, canopy height, soil temp, cSoil.

## Parameter values

See `params.json`.
