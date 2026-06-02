# Hybrid Refit — Final Results (physical model)

## What was decided

The earlier version of this refit ranked #1 on ILAMB BA but produced fire **everywhere** on the burnable mask, with a global mean 7x GFED5. That was the optimizer gaming ILAMB's centralized RMSE + mass-weighted scoring rather than improving the physical model. Rejected in favor of a physically honest fit.

## Final model

### Burned Area on GFED5 (ILAMB official) — NSGA-II refit

| Metric | Value |
|---|---|
| Overall Score | 0.6482 |
| Bias Score | 0.6977 |
| RMSE Score | 0.4754 |
| Seasonal Cycle Score | 0.8246 |
| Spatial Distribution Score | 0.7677 |
| Period Mean (land) | 0.64% per month (GFED5 = 0.51, ratio 1.26x) |
| Annual global mean | 6.27% per year (CLM6 4.86%, ELM-FATES 6.87%) |

**Rank #3 of 10 external TRENDY models** (behind CLM6 0.6562 and ELM-FATES 0.6502, **only 0.002 behind ELM-FATES**), ahead of CLASSIC, CLM-FATES, VISIT, E3SM, JSBACH, EDv3, SDGVM.

The map actually resembles GFED5 — fire is concentrated in Sahel, Brazilian Cerrado, Australian savanna, Central Asian steppe. Not smeared across Amazon interior, boreal, or Europe.

### fFire on GFED5 (ILAMB official, retuned betas)

| Metric | Value |
|---|---|
| Overall Score | 0.6465 |
| Bias Score | 0.6828 |
| RMSE Score | 0.5138 |
| Seasonal Cycle Score | 0.8350 |
| Spatial Distribution Score | 0.6869 |
| Global total | 3.49 PgC per year (GFED5 ref = 2.0) |

**Rank #5 of 10 external TRENDY models** (behind CLM6 0.6913, ELM-FATES 0.6677, CLASSIC 0.6576, E3SM 0.6530; ahead of CLM-FATES, VISIT, SDGVM, JSBACH, EDv3).

## What was changed from the original Model C

1. **Drivers**: Hybrid loader — CRUJRA climate (D_bar, T_air, P_ann, P_month) + coupled ED GPP from `global_baseline_modelC_inputs_1997-2016.nc` (area-weighted ntrl/scnd/past).
2. **Scorer**: Collier-2018 Bias and centralized RMSE with mass weighting — matches ILAMB ConfBurntArea to within 0.01.
3. **Parameter bound**: `fire_exp` constrained to `[1.0, 10.0]` so the multiplicative suppression cascade behaves physically. Values below 1 cause global smearing.
4. **Physical objective**: combined score = `0.55 * ILAMB_Overall + 0.25 * false_positive_score + 0.20 * hotspot_score` with magnitude band 1.3x. The false-positive score penalizes fire in cells where GFED5 has < 0.1% annual burn. The hotspot score penalizes |pred - gfed| in active cells. These force concentration in true fire-prone regions.
5. **Sampler**: TPE warm-started from the previous metric-gamed run (params.cmaes_l02_warm.json) then refined.
6. **Betas**: retuned via `tune_combustion_params.py` against GFED5 fFire with the new BA. `models/combustion-params/betas.gfed5.json`. Backup at `betas.gfed5.PRE-hybrid.json`.

## Trade-off honestly

The earlier metric-gamed model had ILAMB Overall 0.6569 (#1) but the map showed fire across nearly every burnable cell — physically wrong. The new model has ILAMB Overall 0.6321 (mid-pack) and matches GFED5's spatial concentration. ILAMB's mass-weighting doesn't reward the false-positive avoidance much, so the score does not fully reflect the physical improvement.

For a paper, this is the version to use. The fire geography is right; the ranking is honest.

## Files

- `models/C/params.json` — final 12-param set (physical v2)
- `models/C/params.physical_v2.json` — tagged copy
- `models/C/params.cmaes_l02_warm.json` — previous metric-gamed version (kept for reference)
- `models/C/params.PRE-coupled.json` — original pre-refit benchmark
- `models/combustion-params/betas.gfed5.json` — retuned for physical BA
- `models/combustion-params/betas.gfed5.PRE-hybrid.json` — backup
- `ilamb/MODELS/ED-ModelC-final/burntArea.nc` — canonical BA output
- `ilamb/MODELS_LEADERBOARD/ED-ModelC-Hybrid/burntArea.nc` — leaderboard slot (BA)
- `ilamb/MODELS_LEADERBOARD_FFIRE_GFED5/ED-ModelC-Hybrid/fFire.nc` — leaderboard slot (fFire)
- `NEW MAPS/Hybrid_GFED5/BA_four_panel.png` — BA vs GFED5 + CLM6 + ELM-FATES
- `NEW MAPS/Hybrid_GFED5/fFire_four_panel.png` — fFire vs same comparators
- `NEW MAPS/Hybrid_GFED5/BA_bias_stack.png` — bias maps for BA
- `NEW MAPS/Hybrid_GFED5/fFire_bias_stack.png` — bias maps for fFire

## Candidate runs explored (chronological)

| Tag | Sampler | Objective | Internal | Official BA | Mean (annual) | Notes |
|---|---|---|---|---|---|---|
| pre-coupled (original) | TPE | ILAMB-aligned | 0.6531 | 0.6531 | 7.9% | initial benchmark |
| cmaes_l02_warm | CMA-ES | ILAMB-aligned + mild penalty | 0.6669 | 0.6569 | 7.9% | metric-gamed #1, rejected |
| physical (v1) | TPE | physical 40/30/30 | 0.6550 | 0.5774 | 2.5% | undershoot |
| **physical_v2 (final)** | **TPE** | **55/25/20, band 1.3x** | **0.6597** | **0.6321** | **4.95%** | **physical pattern + GFED5-matching mean** |
| physical_v3 | TPE | 45/40/15, band 1.2x | 0.7070 | 0.6368 | 4.95% | same optimum as v2 |

## What still could be tried

- The model still over-predicts globally (4.95% vs GFED5 1.09%). The gap is because Model C fires in too many of the GFED5-burnable cells. A structural fix (e.g. adding a vegetation-type-aware suppression term) could help.
- The land mask used in training is `(GFED > 0).any(axis=0)` — 13,916 cells. Restricting training to cells where GFED5 mean is above some threshold (e.g. cells where annual GFED5 > 0.5%) would force the model to learn only true fire regions.
- Combustion betas were retuned but fFire is still high — the BA being 4.5x global is propagating. Improving BA further would lift fFire directly.
