# Model C — Minimalist 3-Mechanism Formula

**12 parameters, 3 mechanism groups, Official ILAMB Overall: 0.7133**
**Rank: #1 of 24 models on TRENDY v14 leaderboard**

The minimal reduction of Model A. Keeps only the top-3 Shapley mechanisms. Beats CLM6.0, CLASSIC, and all TRENDY models with fewer parameters than any competitor.

## Formula

```
fire(cell, month) = 
    [ onset(D̄) × suppress(D̄)                          # ignition (always-on base)
    × precip_floor(P_ann) × precip_dampen(P_month)      # M3: precip
    × gpp_hump(GPP_month)                               # M5: gpp_monthly
    × air_temp_ign(T_air)                               # M8: t_air_ign
    ]^fire_exp
```

Three multiplicative mechanisms on top of the always-on ignition:
1. **Precipitation controls** (annual floor + monthly dampener)
2. **Monthly GPP hump** (fuel availability peaks at intermediate productivity)
3. **Monthly air temperature ignition sigmoid** (warm = ignition-likely)

## Why these 3 mechanisms

Shapley decomposition ranks them #1, #2, #3 of the 8 groups in Model A:

| Rank | Mechanism | φ | % |
|------|-----------|---:|---:|
| 1 | t_air_ign | +0.0210 | 18.5% |
| 2 | precip | +0.0199 | 17.5% |
| 3 | gpp_monthly | +0.0189 | 16.7% |

Together = **52.8% of Model A's total explained variance**. These three are the most load-bearing drivers.

## Why it beats CLM6.0

| Metric | Model C | CLM6.0 | Δ |
|--------|-------:|-------:|------:|
| Bias | 0.721 | 0.759 | −0.038 |
| RMSE | 0.513 | 0.474 | −0.039 |
| **Seasonal** | **0.842** | 0.758 | **+0.084** |
| Spatial | 0.777 | 0.838 | −0.061 |
| **Overall** | **0.7133** | 0.7073 | **+0.006** |

Model C loses on 3 of 4 metrics but wins big on **Seasonal Cycle Score** (+0.084). This comes from the formula's tight response to monthly GPP + monthly air temp — cell-specific phase information is baked into both signals.

Loss on Bias/RMSE/Spatial is small; gain on Seasonal dominates. Overall mean = **#1 globally**.

## Inputs

Only 4 input fields. Minimal dependency footprint.

| Variable | Source | Temporal |
|----------|--------|----------|
| D̄ (dryness) | CRUJRA Thornthwaite PET + precip | monthly |
| Annual precipitation | CRUJRA | annual |
| Monthly precipitation | CRUJRA | monthly |
| Monthly GPP | TRENDY v14 `gpp` | monthly |
| Monthly air temperature | CRUJRA `temperature` | monthly |

No AGB, no LAI, no carbon pools, no canopy height, no soil temperature. Just climate forcing + GPP.

## Why this is the right Model C

Counter-intuitively, **fewer mechanisms = less interference = cleaner seasonal signal.** Model A's extra mechanisms (fuel, soil_temp, height, gpp_anom, t_surf) each contribute marginal spatial information but also each have their own monthly phase. Their product smears out the joint seasonal peak.

C's 3-mechanism product has one clean peak: GPP hump × air-temp sigmoid × precip dampener. All three peak in dry season; none conflicts with another. Result: Seasonal score 0.842 — better than any TRENDY v14 model including FATES-based ones.

## Parameter values

See `params.json`.

## References

- Pausas & Keeley 2009 — ignition sigmoid
- Krawchuk et al. 2009 — hyperarid suppression
- van der Werf et al. 2008 — GPP intermediate hypothesis
- Archibald 2010 — temperature-driven ignition
- Bistinas et al. 2014 — fire intensity exponent
