# Model A — Full 8-Mechanism Fire Formula

**27 parameters, 8 mechanism groups, ILAMB Overall: 0.6574** (native tier-2 aggregation from `scalar_database.csv`)
**Rank: #4 on TRENDY v14 offline leaderboard**

Full mechanistic formula. The starting point for Shapley analysis — Models B and C are derived by dropping the least-contributing mechanisms.

## Formula

```
fire(cell, month) = 
    [ onset(D̄) × suppress(D̄)                          # ignition
    × fuel_hump(af·AGB + af_lai·LAI_ann)                # M1: fuel
    × soil_temp_gate(T_deep) × warming_rate(dT_deep/dt) # M2: soil_temp
    × precip_floor(P_ann) × precip_dampen(P_month)      # M3: precip
    × height_suppress(H_natr)                           # M4: height
    × gpp_hump(GPP_month)                               # M5: gpp_monthly
    × gpp_anom_supp(GPP - cell_mean) × fuel_anom_boost  # M6: gpp_anom
    × surf_temp_gate(T_surf)                            # M7: t_surf
    × air_temp_ign(T_air)                               # M8: t_air_ign
    ]^fire_exp
```

## Mechanisms

| # | Mechanism | Form | Inputs | Reference |
|---|-----------|------|--------|-----------|
| base | Ignition onset + hyperarid suppression | double sigmoid on D̄ | CRUJRA accumulated dryness | Pausas & Keeley 2009; Krawchuk 2009 |
| **M1** | Fuel availability hump | `(1 - exp(-a/fb))·exp(-a/fd)` where `a = af·AGB + af_lai·LAI_ann` | TRENDY v14 `cLeaf + cWood`, ED `LAI` | Pausas & Ribeiro 2013 |
| **M2** | Soil thermal state | deep sigmoid + month-over-month warming rate sigmoid | CRUJRA `soil_temp3-6` | Venevsky 2002 |
| **M3** | Precipitation control | annual floor `P/(P+P_half)` + monthly dampener `1/(1+P_m/half)` | CRUJRA `pre` | pyrogeography boundary |
| **M4** | Canopy height suppression | `1/(1 + exp(h_k·(H - h_crit)))` | ED `mean_height_natr` | Archibald 2009 |
| **M5** | Monthly GPP hump | `(1 - exp(-g/gb))·exp(-g/gd)` on monthly GPP | TRENDY v14 `gpp` | van der Werf 2008 |
| **M6** | GPP anomaly + fuel×anom cross-term | suppressing sigmoid on (GPP − cell_mean) + saturating on max(0, −anom) | derived per-cell climatology | Archibald 2013 pyrome phase |
| **M7** | Surface soil temperature gate | sigmoid on monthly `soil_temp1` | CRUJRA | Venevsky 2002 |
| **M8** | Monthly air temperature ignition | sigmoid on monthly `temperature` | CRUJRA | Archibald 2010; Abatzoglou & Williams 2016 |
| global | Fire intensity exponent | `[Π]^fire_exp` | — | Bistinas 2014 |

## Shapley Decomposition

From 2⁸ = 256-subset exact Shapley on Model A (500 trials per subset, 6 workers, 219 min):

| Rank | Mechanism | φ (Overall) | % of explained variance |
|------|-----------|-------------:|------------------------:|
| 1 | t_air_ign | +0.0210 | 18.5% |
| 2 | precip | +0.0199 | 17.5% |
| 3 | gpp_monthly | +0.0189 | 16.7% |
| 4 | gpp_anom | +0.0151 | 13.2% |
| 5 | t_surf | +0.0143 | 12.6% |
| 6 | soil_temp | +0.0112 | 9.9% |
| 7 | height | +0.0073 | 6.4% |
| 8 | fuel | +0.0059 | 5.2% |

Empty subset (base ignition only): 0.6225. Full subset: 0.7362. ΣΦ = 0.1137 (verified).

> Scores above are the Optuna training objective (equal-weighted mean of 4 component scores), not ILAMB native Overall. Mechanism *rankings* are what drive B and C construction and are expected to be stable under either aggregation. See README "Shapley results" section.

**Key finding**: monthly seasonal mechanisms (t_air_ign, gpp_monthly, gpp_anom, t_surf = 61% of explained variance) dominate. Fuel's Shapley contribution is small here because GPP-based mechanisms already carry most biomass/productivity signal.

## Scores (Official ILAMB)

| Metric | Score | vs CLM6.0 |
|--------|------:|----------:|
| Bias Score | 0.716 | −0.043 |
| RMSE Score | 0.492 | +0.018 |
| Seasonal Cycle | **0.805** | **+0.047** |
| Spatial Distribution | 0.783 | −0.055 |
| **Overall** | **0.6574** | **−0.003** |

## Inputs

| Variable | Source | Temporal |
|----------|--------|----------|
| D̄ | CRUJRA Thornthwaite PET + precip | monthly |
| AGB | TRENDY v14 `cLeaf + cWood` | annual |
| LAI (annual mean) | ED frozen-sim `LAI` averaged | static per cell |
| T_deep + d/dt | CRUJRA `soil_temp3-6` | monthly |
| T_surf | CRUJRA `soil_temp1` | monthly |
| T_air | CRUJRA `temperature` | monthly |
| Precip (annual + monthly) | CRUJRA | both |
| H_natr | ED frozen-sim `mean_height_natr` | static |
| GPP (monthly) | TRENDY v14 `gpp` | monthly |
| GPP anomaly | derived per-cell from monthly GPP | monthly |

## Parameter values

See `params.json`.
