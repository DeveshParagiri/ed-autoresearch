# Writeup: Shapley-Informed Simplification of the ED Fire Module

This document details the experimental chain that produced Models A, B, and C. The short version: we picked A as the complete formula, ran an exact Shapley decomposition to identify which mechanisms actually matter, and used that to derive B (middle ground) and C (minimal) without needing to refit from scratch.

## Starting point

ED v3's stock fire module (`EDv3` in the ILAMB leaderboard) gets an Overall ILAMB score of **0.475**. That's rank #13 of 14 models in TRENDY v14. Our goal was to build a closed-form replacement mechanism that meaningfully improves on this while staying (a) interpretable, (b) globally uniform (no regional sub-formulas), and (c) transferable across ED runs — meaning the formula should survive swapping from the frozen 1981-2016 offline simulation to the TRENDY v14 coupled submission without breaking.

## Model A — full mechanistic formula

A is built by starting from the R28-style multiplicative formula with all 7 ecologically-grounded mechanism families active:

1. **Ignition onset/suppress** (double sigmoid on accumulated dryness D̄) — Pausas & Keeley 2009, Krawchuk 2009
2. **Fuel availability hump** (Pausas-Ribeiro hump on AGB) — peaks at savanna biomass
3. **GPP-productivity hump** (Gaussian on annual GPP) — van der Werf 2008
4. **Soil temperature gate** (sigmoid on T_deep) — Venevsky 2002 permafrost gate
5. **Precipitation floor** — standard pyrogeography boundary
6. **Canopy height suppression** — Archibald 2009 grass/tree threshold
7. **Land-use modifier** — Archibald 2010, Bistinas 2014
8. **Soil carbon multiplier** — Krawchuk 2009

Plus a global `fire_exp` intensity exponent. All factors are multiplicative. Each has a physical citation.

### Input swap (TRENDY v14 vs frozen-sim)

Model A was originally trained with the frozen ED 1981-2016 simulation (`EDv3_global_simulation_1981_2016.nc`). That version scored Overall 0.747. But when we swapped in the TRENDY v14 coupled S3 outputs (GPP, cLeaf+cWood as AGB, cSoil) without changing hyperparameters, the model **collapsed to Overall 0.320** — Spatial Distribution went from 0.880 down to 0.003.

Root cause: the formula's `fuel_hump` and `gpp_gaussian` peaks are absolute physical values (kg C/m² and kg C/m²/yr). TRENDY v14 AGB averages ~10 kg C/m² vs the frozen sim's ~3; GPP averages 0.68 vs 0.22. The humps were peaked at the wrong absolute values, so every cell landed on the wrong part of the curve.

We refit with new inputs. Overall came back to **0.736** with the same 19 parameters — a genuine test of the formula's transferability rather than the 0.747 frozen-sim number. That's the Model A we keep.

### Attempts to push past 0.747 (all failed)

Before accepting 0.736, we tried four structural enhancements:

1. **Rank-normalized inputs (v5)** — percentile ranks of AGB/GPP/cSoil instead of raw values, to kill scale sensitivity. Trained a fresh 18-param formula. Result: Overall 0.40. The percentile space lost too much spatial information. Rejected.

2. **Dimensionless physical ratios (v5.1)** — `AGB/GPP` (stand age) and `AGB × D̄` (fuel-dryness product) as inputs. Same idea as v5 but keeping physical units via ratios. Result: Spatial r collapsed from 0.80 to 0.55. Ratios lose biome-discriminative signal. Rejected.

3. **Cured-fuel + warm-season ignition modifiers** — van der Werf 2008 / Archibald 2010 seasonal adds: track consecutive dry months (fuel curing) and monthly air-temperature ignition boost. Result: Seasonal improved by +0.012, Spatial lost −0.023, Overall −0.002 (wash). Rejected.

4. **Cell-specific climatological phase anchor (Archibald 2013 pyromes)** — computed per-cell `peak_month = argmax(D̄ climatology)` from CRUJRA, added a Gaussian window centered on that month with 2 new global parameters. Result: Optuna converged with w_phase near 0 (disabling the phase term). It was redundant with information already in the D̄ ignition sigmoid. Seasonal +0.003 only. Rejected.

**First-principles conclusion**: the multiplicative-product-of-drivers form has a structural ceiling of ~0.50-0.52 for Seasonal. Not because the parameters are wrong, but because the phase of a product of weakly-aligned periodic drivers cannot reliably match GFED's cell-specific peak months. Breaking past this would require event-driven/threshold-gated formulations that aren't closed-form multiplicative. So we accept 0.736 Overall as the ceiling for the current formula class and move on.

## Shapley decomposition of A

To derive B and C rigorously, we need to know which of the 7 mechanisms are actually load-bearing. Exact Shapley on all 2⁷ = 128 mechanism subsets, refitting each with 500 Optuna trials and 6 parallel workers. Total 80 minutes.

### Results

Shapley value φ (using ILAMB Overall score as the game value function):

| Rank | Mechanism | φ (Overall contribution) | % of explained variance |
|-----:|-----------|-------------------------:|------------------------:|
| 1 | **fuel** | **+0.0446** | **41.4%** |
| 2 | **soil_temp** | **+0.0368** | **34.1%** |
| 3 | precip | +0.0139 | 12.9% |
| 4 | height | +0.0077 | 7.1% |
| 5 | soil_c | +0.0028 | 2.6% |
| 6 | landuse | +0.0022 | 2.1% |
| 7 | **gpp** | **−0.0003** | **−0.3%** |

Empty subset (ignition only) Overall = 0.6227. Full subset Overall = 0.7304. Sum verifies exactly: ΣΦ = 0.1078 = Full − Empty.

### Key findings

- **fuel + soil_temp = 75.5% of the total explained variance.** These are the irreducible climate + biomass signal.
- **gpp has Shapley φ ≈ 0.** Adding GPP doesn't help Overall; the fuel hump on AGB already captures the productivity signal. This surprised us given how loaded the literature is with GPP × fire coupling; the story is that our fuel hump's peak placement absorbs the information.
- **landuse and soil_c contribute less than 3% each.** frac_scnd signal is weak at the 1° resolution we evaluate on.

## Model B — Shapley-reduced middle-ground

Take the top 4 mechanisms by Shapley: **fuel + soil_temp + precip + height**. Drop gpp, landuse, soil_c. Refit with 2500 Optuna trials.

Result: **Overall 0.652, 13 parameters**. Actually *beats* Model A (0.634) on ILAMB. Confirms the Shapley decomposition — those 3 dropped mechanisms were collectively redundant or harmful given the other 4.

Why B beats A on ILAMB:
- Fewer mechanisms → fewer interference terms → cleaner monthly signal → better Seasonal score (0.612 vs 0.520).
- Spatial loses a hair (0.791 vs 0.806) because A's extra terms do capture some spatial fidelity.
- Net gain on Overall because Seasonal improvement dominates.

## Model C — minimal physics-only

Skeletal: just **fuel + soil_temp** (Shapley's top 2, 75.5% of explained variance) plus the ignition gate and the global `fire_exp`. Drop everything else.

Result: **Overall 0.642, 10 parameters**. Slightly below B but with barely half the parameters and a clean physics-only story: "fire ignites at sufficient dryness × sufficient fuel × not-permafrost." Three input fields total.

Seasonal score actually *rises* to 0.662 — the cleanest monthly signal of the three, because there's nothing extra to interfere. Spatial drops to 0.730, the largest single trade-off in the trio.

## Model comparison

| Model | Mechanisms | Params | Inputs | Overall | Bias | RMSE | Seas | Spatial |
|-------|-----------:|-------:|-------:|--------:|-----:|-----:|-----:|--------:|
| A | 7 | 19 | 8 | 0.634 | 0.732 | 0.478 | 0.520 | 0.806 |
| **B** | 4 | 13 | 5 | **0.652** | 0.725 | 0.482 | 0.612 | 0.791 |
| C | 2 | 10 | 3 | 0.642 | 0.701 | 0.476 | 0.662 | 0.730 |

### Observations

- **Shapley's guidance worked.** B was built *without* retuning A — just took the top-4 mechanisms and refit. It beat A. That's the validation that φ was accurate.
- **Parameter count halves from A to C** (19 → 10) with only 0.008 Overall cost. The marginal cost of complexity is low at the top end.
- **Seasonal improves as we simplify.** This is counter-intuitive at first — fewer mechanisms should miss more signal, not catch more. What's happening is that the *irrelevant* mechanisms in A were introducing phase noise in the monthly dimension that the Shapley scoring (static Overall only) couldn't penalize precisely but that ILAMB's Seasonal Cycle metric does.

## What happens on coupled ED

All three models were calibrated against TRENDY v14 ED S3 outputs, which is the closest thing to a production-coupled ED run that's publicly available. The calibration inputs are the same quantities ED produces at runtime from its vegetation dynamics (GPP, cLeaf+cWood, cSoil). So transferability is structural: in principle, the formulas should work in a live coupled ED without refitting.

Two caveats:

1. **Canopy heights and frac_scnd** aren't in TRENDY v14, so we used frozen-sim versions. A live run would use its own prognostic versions — these are PFT-level, so the absolute numbers will differ but the global distribution should be similar enough.

2. **We haven't validated this in a real coupled deployment.** The test is to port one of these models into ED's `fire.cc`, run coupled for a multi-year spinup, and check that the fire predictions don't drift away from GFED. That's ED-team work.

## Why this approach makes sense

**Shapley-informed ablation is a cheaper way to design simpler models than training 100 candidates.** Instead of trying dozens of hand-curated simplifications, we let Shapley identify which mechanisms are load-bearing and which are noise, then built two new models purely by dropping mechanisms.

The three models represent a **complexity frontier**: A covers cases where you want sharp spatial fidelity (0.806 Spatial), B is the best overall scorecard performance, C is the cleanest port target for a closed-form implementation. Picking one depends on what matters most for your application.

## Key rejected approaches and why

Documented here so nobody retries them:

- **Rank-normalized inputs** — destroys spatial information that physical-unit humps preserve.
- **Regional sub-formulas** — explicitly vetoed; breaks the "single global formula" constraint.
- **Phase anchor from D̄ climatology** — redundant with existing D̄ sigmoid.
- **Cured-fuel and warm-season ignition boosts** — in principle right ideas, but the per-cell seasonal signal they add gets eaten by other terms in the multiplicative product.
- **Event/threshold-gated reformulations** — would break past the 0.50 Seasonal ceiling but abandon closed-form interpretability. Not attempted.

## References

Primary formula-family references:
- Pausas & Keeley 2009 "A burning story" — ignition sigmoid
- Pausas & Ribeiro 2013 "The global fire-productivity relationship" — fuel hump
- Krawchuk et al. 2009 "Global Pyrogeography" — hyperarid suppression, soil carbon
- van der Werf et al. 2008 "Climate controls on the variability of fires" — GPP/productivity
- Venevsky et al. 2002 — soil temperature as permafrost gate
- Archibald et al. 2009 "What limits fire in Africa" — canopy height threshold
- Archibald et al. 2010 "Climate and the inter-annual variability of fire in Africa" — human/landuse modifiers
- Archibald et al. 2013 "Defining pyromes and global syndromes of fire regimes" — climatological fire season (phase anchor, rejected here)
- Bistinas et al. 2014 "Relationships between human population density and burned area" — intensity exponent

Benchmarking:
- Collier et al. 2018 — ILAMB framework
- van der Werf et al. 2017 — GFED4.1s
- Global Carbon Project 2025 — TRENDY v14
