# Model C — Minimalist 3-Mechanism Formula

**12 parameters, 3 mechanism groups, ILAMB Overall Score: 0.6703** (native tier-2 aggregation from `scalar_database.csv`)
**Rank: #1** on TRENDY v14 + baseline offline leaderboard (beats CLASSIC 0.6660, CLM6.0 0.6606).

> **Offline benchmark.** Training and scoring use fixed TRENDY v14 / CRUJRA / ED-sim NetCDFs. See "Coupling caveats" in the root `README.md`.

## Formula

```
fire(cell, month) =
    [ onset(D̄) × suppress(D̄)                        # ignition (base, always on)
    × precip_floor(P_ann) × precip_dampen(P_month)    # M3: precip
    × gpp_hump(GPP_month)                             # M5: gpp_monthly
    × air_temp_ign(T_air)                             # M8: t_air_ign
    ]^fire_exp
```

Three multiplicative mechanisms on top of the always-on ignition:
1. **Precipitation controls** (annual floor + monthly dampener)
2. **Monthly GPP hump** (fuel availability peaks at intermediate productivity)
3. **Monthly air temperature ignition sigmoid** (warm = ignition-likely)

## Why these 3 mechanisms

Shapley decomposition on Model A v8 ranked `t_air_ign`, `precip`, `gpp_monthly` as the top-3 of 8 groups (52.8% of explained variance). Model C keeps only these; fuel, height, soil_temp, gpp_anom, t_surf are dropped.

## Scores (real ilamb-run, stock `ConfBurntArea`, canonical inputs)

| Metric | Model C | CLASSIC | CLM6.0 |
|---|---:|---:|---:|
| Bias Score | **0.714** | 0.738 | 0.759 |
| RMSE Score | 0.513 | 0.507 | 0.474 |
| Seasonal Cycle | **0.834** | 0.782 | 0.758 |
| Spatial Distribution | 0.779 | 0.797 | **0.838** |
| **Overall (native)** | **0.6703** | 0.6660 | 0.6606 |

ILAMB "Overall Score" here is `(2·Bias + 2·RMSE + Seasonal + Spatial) / 6`, the tier-2 value ILAMB writes directly to `scalar_database.csv`. The component scores are also pulled unmodified from the same file.

Model C's win is driven by **Seasonal Cycle** (+0.05 over CLASSIC, +0.08 over CLM6.0). Loses on Spatial to CLM6.0 but Bias + Seasonal make up the difference under the tier-2 weighting.

## Inputs (offline)

| Variable | Source | Temporal |
|---|---|---|
| D̄ (dryness) | CRUJRA Thornthwaite + daylength PET, continuous accumulator | monthly |
| Annual precipitation | CRUJRA | annual |
| Monthly precipitation | CRUJRA | monthly |
| Monthly GPP | TRENDY v14 `EDv3_S3_gpp.nc` | monthly |
| Monthly air temperature | CRUJRA `temperature` | monthly |

Only 5 input fields. No AGB, LAI, carbon pools, canopy height, soil temperature.

The D̄ accumulator formula is specific: Thornthwaite PET with latitude-dependent daylength correction, monthly precip units, `K=1` multiplier, continuous (no annual reset), hard reset at `precip ≥ 200 mm/month`. See `scripts/prep_monthly_inputs.py` for the exact Python reference.

## Parameter values

See `params.json` (the 12 retuned values, full 12-parameter Optuna fit on canonical dbar, 2500 trials warm-started).

## References

- Pausas & Keeley 2009 — ignition sigmoid
- Krawchuk et al. 2009 — hyperarid suppression
- van der Werf et al. 2008 — GPP intermediate hypothesis
- Archibald 2010 — temperature-driven ignition
- Bistinas et al. 2014 — fire intensity exponent
