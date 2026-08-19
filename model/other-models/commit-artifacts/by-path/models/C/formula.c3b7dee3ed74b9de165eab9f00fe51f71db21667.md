# Model C — Minimal Fire Formula

**2 mechanisms, 10 parameters, Overall ILAMB score 0.642**

Skeletal physics-only formula: only the two highest-Shapley mechanisms (`fuel` + `soil_temp`) on top of the ignition gate.

## Formula

```
fire(cell, month) =
    [ ignition_sigmoid(D̄)              # k1, D_low
    × hyperarid_suppression(D̄)         # k2, D_high
    × fuel_hump(af · AGB)               # af, fb, fd
    × soil_temp_gate(T_deep)            # sc2, ss2
    ]^fire_exp                          # fire_exp
```

## Why these 2 mechanisms

fuel (41.4% of Shapley φ) + soil_temp (34.1%) = **75.5% of total explained variance**. No human/lightning, no landuse, no soil carbon. Just climate × fuel.

## Inputs

Minimal. 3 input fields total.

| Variable | Source |
|----------|--------|
| D̄ | CRUJRA |
| AGB | TRENDY v14 `cLeaf + cWood` |
| T_deep | CRUJRA soil temps layers 3-6 |

## Scores (ILAMB, GFED4.1s 2001-2016)

| Metric | Score |
|--------|------:|
| Bias | 0.701 |
| RMSE | 0.476 |
| Seasonal | 0.662 |
| Spatial Distribution | 0.730 |
| **Overall** | **0.642** |

## Parameter Values

See `params.json`.

## Trade-off Summary

| Model | Params | Overall | Spatial | What it buys |
|-------|-------:|--------:|--------:|------|
| A | 19 | 0.634 | 0.806 | All 7 mechanisms, highest spatial |
| B | 13 | **0.652** | 0.791 | Best balance: drops redundant mechanisms |
| C | 10 | 0.642 | 0.730 | Minimal physics-only |

C sacrifices spatial (−0.08 vs B) but gains +0.05 seasonal. Parameter count is half of A.
