# Model B — Middle-Ground Fire Formula

**4 mechanisms, 13 parameters, Overall ILAMB score 0.652 (best of the three)**

Shapley-informed reduction: drops the 3 mechanisms with lowest contribution (`gpp`, `landuse`, `soil_c`).

## Formula

```
fire(cell, month) =
    [ ignition_sigmoid(D̄)                 # k1, D_low
    × hyperarid_suppression(D̄)            # k2, D_high
    × fuel_hump(af · AGB)                  # af, fb, fd
    × soil_temp_gate(T_deep)               # sc2, ss2
    × precip_floor(P_ann)                  # P_half
    × height_suppression(H_natr)           # h_k, h_crit
    ]^fire_exp                             # fire_exp
```

## Why these 4 mechanisms

From the exact Shapley decomposition of Model A (128 subsets, 500 trials each):

| Rank | Mechanism | Shapley φ | % of explained | In B? |
|------|-----------|----------:|---------------:|:-----:|
| 1 | fuel | +0.0446 | 41.4% | ✓ |
| 2 | soil_temp | +0.0368 | 34.1% | ✓ |
| 3 | precip | +0.0139 | 12.9% | ✓ |
| 4 | height | +0.0077 | 7.1% | ✓ |
| 5 | soil_c | +0.0028 | 2.6% | ✗ |
| 6 | landuse | +0.0022 | 2.1% | ✗ |
| 7 | gpp | −0.0003 | −0.3% | ✗ |

B keeps the top 4. These contribute 95.5% of total explained variance.

## Mechanisms

Same as Model A minus `gpp_gaussian`, `landuse_mod`, and `soil_C_mult`. See `../A/formula.md` for full descriptions.

## Inputs

Same as Model A except doesn't need: GPP, frac_scnd, H_scnd, soil_C. That's 4 fewer input fields.

| Variable | Source |
|----------|--------|
| D̄ | CRUJRA |
| AGB | TRENDY v14 `cLeaf + cWood` |
| T_deep | CRUJRA soil temps |
| P_ann | CRUJRA precipitation |
| H_natr | ED frozen-sim `mean_height_natr` |

## Scores (ILAMB, GFED4.1s 2001-2016)

| Metric | Score |
|--------|------:|
| Bias | 0.725 |
| RMSE | 0.482 |
| Seasonal | 0.612 |
| Spatial Distribution | 0.791 |
| **Overall** | **0.652** |

**Beats Model A** (0.634) — fewer mechanisms, less interference, better score. Confirms Shapley decomposition: those 3 dropped mechanisms were net-neutral or harmful.

## Parameter Values

See `params.json`.
