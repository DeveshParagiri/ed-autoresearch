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
