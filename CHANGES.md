# Changes to ED v3.0

## belowgrnd.cc — A_function (Decomposition Scalar)

**Temperature response: Q10 → Lloyd-Taylor**

| | Before | After |
|---|--------|-------|
| Formula | `Td = 0.40 * 1.5^((T-25)/10)` | `Td = 1.93 * exp(243.5 * (1/56.02 - 1/(T+46.02)))` |
| Reference | Parton et al. 1993 | Lloyd & Taylor 1994 |
| Freezing threshold | 0.0°C | 3.6°C |

The Q10 formulation overestimates temperature sensitivity at low temperatures. Lloyd-Taylor, derived from Arrhenius kinetics with an empirical correction for enzyme behavior near freezing, asymptotically flattens below ~5°C. This allows boreal and arctic soils to retain more carbon, matching observed HWSD and NCSCD stocks.

**Moisture response: piecewise linear → log-parabolic**

| | Before | After |
|---|--------|-------|
| Formula | Hard breakpoints at θ=0.3, 0.6 | `Wd = exp(-(ln(θ/0.19))² / (2 * 3.7²))` |
| Reference | Linn & Doran 1984 | Moyano et al. 2013 |

The wide σ=3.7 indicates moisture has a weak spatial modulating effect on decomposition at the global scale. Temperature dominates. Consistent with the global synthesis of Bond-Lamberty & Thomson (2010).

**Metric improvement:**
- HWSD spatial correlation: 0.43 → 0.48
- Soil carbon bias: -50% → 0%
- NCSCD (high-latitude): 0.11 → 0.19

## fire.cc — update_fuel (Fire Disturbance)

**Power law → sigmoid × fuel hump**

| | Before | After |
|---|--------|-------|
| Formula | `fuel * (dryness/30000)^10` | `sigmoid(D_bar) × fuel_hump(AGB)` |
| Response to biomass | Monotonically increasing | Unimodal (peaks at intermediate) |
| Reference | — | Pausas & Ribeiro 2013 |

The original formula assumed fire scales monotonically with fuel and dryness. The intermediate productivity hypothesis (Pausas & Ribeiro 2013) shows fire peaks in savannas where fuel production and curing conditions coexist. Dense forests suppress fire through humidity; deserts lack fuel.

**Metric improvement:**
- GFED spatial correlation: 0.09 → 0.65

## phenology.cc — Cold Deciduous Threshold

**Leaf drop temperature: 10°C → 0°C**

| | Before | After |
|---|--------|-------|
| Cold deciduous trigger | T < 10°C | T < 0°C |
| Reference | — | Jolly et al. 2005 (Growing Season Index) |

The original 10°C threshold was too conservative for boreal vegetation, causing premature leaf drop and underestimated LAI at high latitudes. Optimization against MODIS LAI found ~0°C better captures the observed growing season length.

**Metric improvement:**
- MODIS LAI spatial correlation: 0.76 → 0.80

## ED_params.defaults.cfg — Carbon Balance Mortality

**m2: 10.0 → 2.0, m3: 20.0 → 5.0**

| | Before | After |
|---|--------|-------|
| m2 (max mortality rate) | 10.0 yr⁻¹ | 2.0 yr⁻¹ |
| m3 (sigmoid steepness) | 20.0 | 5.0 |
| Half-life at cbr_bar=0 | 50 days | 4.5 years |

The carbon balance mortality term `dmort = m2 / (1 + exp(m3 * cbr_bar))` controls how quickly trees die when their carbon balance is negative or marginal. With m3=20, the sigmoid creates a knife-edge: any tree with cbr_bar below 0.05 is killed within weeks. This is physically unrealistic because woody plants maintain carbon reserves in sapwood and starch that buffer 2-3 years of negative carbon balance (Hoch et al. 2003, *Plant Cell & Environment*). The fastest observed tree die-off events (bark beetle outbreaks, severe drought) operate on timescales of months to years (McDowell et al. 2011, *New Phytologist*), not days.

The m3=20 value has no published justification. Softening to m3=5 allows boreal vegetation with short growing seasons (cbr_bar near zero) to persist, directly addressing the 22% vegetation coverage gap. This is the single most impactful parameter change because it is upstream of every other module: no vegetation → no litter → no soil carbon → no transpiration → no fire.

**growth_resp: 0.5 → 0.33**

| | Before | After |
|---|--------|-------|
| Growth respiration fraction | 0.50 | 0.33 |
| Reference | Undocumented | Waring et al. 1998 |

Restored to the ED 2001 value. The config file itself notes "0.33 in ED 2001" next to the 0.50 value. The standard literature estimate is 0.33 (Waring et al. 1998, *Advances in Ecological Research*). This gives boreal trees 17% more carbon for growth and survival.

**Expected effect:**
- Vegetation coverage extends into ~38% of currently empty cells where MODIS observes real vegetation
- Cascading improvements to soil carbon (more litter input), ET (more transpiration), and fire (more fuel) at high latitudes
