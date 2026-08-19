# Model A — Full Mechanistic Fire Formula

**7 mechanisms, 19 parameters, Overall ILAMB score 0.634**

Full mechanism library with all terms active. Most complex of the three.

## Formula

```
fire(cell, month) =
    [ ignition_sigmoid(D̄)                          # k1, D_low
    × hyperarid_suppression(D̄)                     # k2, D_high
    × fuel_hump(af · AGB)                           # af, fb, fd
    × gpp_gaussian(GPP)                             # gpp_opt, gpp_sigma
    × soil_temp_gate(T_deep)                        # sc2, ss2
    × precip_floor(P_ann)                           # P_half
    × height_suppression(H_natr)                    # h_k, h_crit
    × landuse_modifier(f_scnd, H_scnd)              # lu_c, lu_tau
    × soil_C_multiplier(soil_C)                     # sc_max, sc_half
    ]^fire_exp                                      # fire_exp
```

## Mechanisms

| Term | Form | Function |
|------|------|----------|
| `onset × suppress` | `σ(D̄; k1, D_low) · σ_supp(D̄; k2, D_high)` | Ignition threshold + hyperarid cutoff. Pausas & Keeley 2009, Krawchuk 2009. |
| `fuel_hump` | `(1 − exp(−af·AGB/fb)) · exp(−af·AGB/fd)` | Intermediate-biomass peak (savanna maximum). Pausas & Ribeiro 2013. |
| `gpp_gaussian` | `exp(−((GPP − gpp_opt)/gpp_sigma)²)` | Productivity sweet spot. van der Werf 2008. |
| `soil_temp_gate` | `σ(T_deep; ss2, sc2)` | Permafrost gate — deep soil must be warm enough for combustion. Venevsky 2002. |
| `precip_floor` | `P / (P + P_half)` | Annual precipitation minimum. Standard pyrogeography boundary. |
| `height_suppress` | `1 / (1 + exp(h_k · (H − h_crit)))` | Closed-canopy shading. Archibald 2009 grass/tree threshold. |
| `landuse_mod` | `(1 + lu_c · f_scnd) · exp(−H_scnd / lu_tau)` | Secondary vegetation burns differently. Archibald 2010, Bistinas 2014. |
| `soil_C_mult` | `1 + (sc_max − 1) · C / (C + sc_half)` | Organic-rich soils sustain smoldering. Krawchuk 2009. |
| `fire_exp` | `^fire_exp` | Global intensity exponent. Bistinas 2014. |

## Inputs

| Variable | Source | Monthly? |
|----------|--------|----------|
| D̄ (accumulated dryness) | CRUJRA (Thornthwaite PET, monthly accumulation) | yes |
| AGB | TRENDY v14 ED S3 `cLeaf + cWood` (annual, tiled) | no |
| GPP | TRENDY v14 ED S3 `gpp` (monthly, s⁻¹ → yr⁻¹) | yes |
| T_deep | CRUJRA soil temps layers 3-6 mean | yes |
| P_ann | CRUJRA annual precipitation | tiled |
| H_natr, H_scnd | ED frozen-sim `mean_height_natr/scnd` | tiled |
| f_scnd | ED frozen-sim `frac_scnd` | tiled |
| soil_C | TRENDY v14 ED S3 `cSoil` (annual, tiled) | no |

Heights and fractions fall back to frozen-sim because TRENDY v14 doesn't ship PFT-level state. Everything else is TRENDY v14.

## Scores (ILAMB, GFED4.1s 2001-2016)

| Metric | Score |
|--------|------:|
| Bias | 0.732 |
| RMSE | 0.478 |
| Seasonal | 0.520 |
| Spatial Distribution | 0.806 |
| **Overall** | **0.634** |

## Parameter Values

See `params.json`.
