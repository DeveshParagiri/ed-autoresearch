# ED-Fire autoresearch thread

Running log of every decision and its outcome. Newest entries at the bottom.

## Read-in state at start of thread

Baseline committed model is **Model G** (`33e0115 Start autoresearch from Model G`).

`results.tsv` holds 7 recorded experiments (Models C-I). Best recorded Overall = **0.686** (Model G).

| model | overall | bias | rmse | seasonal | spatial | note |
| ----- | ------- | ---- | ---- | -------- | ------- | ---- |
| C | 0.648 | 0.698 | 0.475 | 0.825 | 0.769 | global monthly fit |
| D | 0.641 | 0.695 | 0.466 | 0.791 | 0.786 | global annual spatial-Taylor |
| E | 0.665 | 0.751 | 0.475 | 0.746 | 0.876 | continental annual spatial-Taylor, +AGB |
| F | 0.678 | 0.753 | 0.512 | 0.775 | 0.840 | +GDP, global BA pinned |
| **G** | **0.686** | 0.748 | 0.494 | 0.845 | 0.850 | **seven-region monthly fit (baseline)** |
| H | 0.682 | 0.726 | 0.520 | 0.836 | 0.808 | global monthly + GDP |
| I | 0.677 | 0.755 | 0.480 | 0.820 | 0.848 | seven-region + tropical AGB gate |

### Model G structure

Multiplicative fire rate with 4 declared COMPONENTS:

- `dryness` — two-sided window: rising logistic at `D_low` x falling logistic at `D_high`
- `precipitation` — annual availability `P/(P+P_half)` x monthly damping `1/(1+M/pre_dampen_half)`
- `fuel` — GPP hump `(1-exp(-x/gpp_b)) * exp(-x/gpp_d)` on `gpp_af * gpp`
- `temperature` — rising logistic on air temperature (`ign_k`, `ign_c`)

then `rate ** fire_exp`, then `(1 - exp(-min(rate,5))) / 12`.

Seven region-specific parameter sets (`REGION_PARAMS`) are applied by lat/lon bounding box
(`REGION_BOXES`) on top of the global `PARAMS`. `SEARCH_SPACE` is currently **empty**, so
`ar optuna` cannot run until coefficients are declared.

There is also a dormant `vegetation` branch in `_fire_rate` (tropical AGB canopy gate, Model I's
idea) that is coded but **not** listed in `COMPONENTS`, so it never activates.

### Scoring mechanics worth remembering (from `scripts/fast_ilamb.py`)

- Overall = weighted mean with **rmse weight 2**, bias 1, seasonal 1, spatial 1.
  So rmse is the highest-leverage term, and Model G's rmse (0.494) is its weakest.
- `rmse_score` is computed on the **centered** mean-seasonal-cycle anomaly, normalised by the
  reference temporal std. It rewards getting the *shape* of the annual cycle right, not magnitude.
- `seasonal_cycle_score` is purely a **phase-shift** score: month of maximum, cosine-weighted.
- `bias_score` and `rmse_score` are averaged with **reference-burned-area weighting**
  (`area * reference_mean`), so high-burning regions dominate those two.
  `spatial_distribution_score` and the reported bias/rmse percentages use plain area weighting.
- 14 GFED regions plus global. Region bounds are ILAMB's, and do **not** match `REGION_BOXES`.

### Unused signal

No recorded experiment uses `lightning_flash_rate`, any `luh2.nc` land-use fraction, or
`population_density`. Ignition and human land use are therefore entirely absent from the
model family explored so far.

## Log

### 2001 — Entry 1: baseline diagnosis (in progress)

Decision: before changing anything, run `ar ablate` (16 subsets, exact Shapley over the four
components) and `ar figures` on the committed Model G to see which components earn their place
and where the maps and seasonal cycles fail. Rationale: research.md forbids adding complexity
without diagnostic evidence, and `SEARCH_SPACE` is empty so there is nothing to tune yet.

### Entry 2: baseline diagnosis outcome

Ran `ar ablate` (16 subsets) and `ar figures` on committed Model G, plus a custom
decomposition `scratchpad/diag_seasonal.py`.

Ablation, global Overall Shapley (drop-one in brackets):
`precipitation` +0.101 [0.051], `dryness` +0.073 [0.026], `temperature` +0.058 [0.022],
`fuel` +0.055 [0.005]. Empty null = 0.399, full = 0.687.

**Two findings that redirect the research:**

1. Every component contributes +0.20..+0.23 to *spatial* and ~0 to *rmse*
   (dryness/precip +0.019, fuel -0.003, temperature -0.008). rmse is double-weighted,
   so the highest-leverage metric is entirely carried by the null baseline. The four
   components are also highly redundant (dropping `fuel` costs 0.005) - they are four
   ways of drawing the same dry-warm-vegetated map.
2. The decomposition localises the rmse loss precisely: it is a **seasonal amplitude
   deficit**, not phase and not magnitude. Global amplitude ratio (model/obs) = **0.163**
   while the annual mean is 20% too *high* (1.86% vs 1.55%). Same signature in the two
   regions holding most of global burned area: nhaf 0.129 (17.6% vs 15.6%),
   shaf 0.187 (22.3% vs 16.7%). The model smears roughly the right annual total evenly
   across all 12 months. Since rmse_score uses the *centered* anomaly, flatness destroys it.

Also: `bona` amplitude ratio 0.083 and `aust` 0.111 are the flattest of all; `eqas` 0.622
is the only region with a healthy cycle. Regionally, `tena` dryness Shapley is -0.000
overall and -0.049 on rmse - the globally-fitted dryness window is worthless in the
worst-scoring region. `bona` is a distinct regime where `temperature` alone is worth
+0.223 overall / +0.366 seasonal.

**Decision: abandon the planned ignition-physics experiment as the first move.** An
ignition term (lightning + humped population) would add another *instantaneous spatial
map*, reinforcing the one thing the model already does well while leaving the binding
constraint untouched. Re-queued as a later experiment - it is still the only plausible
explanation for `bona`.

### Entry 3: Experiment 1 design - fuel memory and depletion

Hypothesis: the amplitude deficit exists because the fire rate is an instantaneous
product of same-month climate with **no state**. A cell with favourable annual climate
burns every month. Real fire regimes are sharpened by two memory mechanisms, neither
present in Model G:

- **Curing**: grass flammability depends on rainfall *integrated over preceding weeks*,
  not the instantaneous value, giving a threshold-like season onset instead of a ramp.
- **Depletion**: once a savanna burns, the fuel is gone for months. This truncates the
  tail of the fire season and is the mechanism that most directly converts a plateau
  into a spike, *without* changing the annual total - exactly the correction needed
  given the annual mean is already too high.

Design choice on experimental control: `REGION_PARAMS` already holds a per-region fitted
Model G skeleton, and Optuna's sampled params only reach the global `PARAMS` fallback.
So I will make the new memory coefficients **global-only, shared by every region**, and
have each region dict override only the 12 keys it already defines. `SEARCH_SPACE` will
contain *only* the new memory coefficients. That holds the fitted skeleton fixed and
isolates the new mechanism - a clean single-variable experiment rather than a refit.

### Entry 4: Experiment 1 pre-tuning sanity check - hypothesis splits

Added two COMPONENTS (`curing`, `depletion`) plus `_antecedent`, `_curing`, `_deplete`.
Restructured `predict` so region splicing happens on the *rate* and the memory terms are
applied globally afterwards. Wrote `scratchpad/quick_score.py` to score subsets on the
fast proxy without recording an experiment, and `scratchpad/regional_delta.py` to check
the per-region spread requirement against the frozen baseline backup.

Untuned default subset scores (fast proxy):

| components | overall | rmse | spatial | amp_ratio | annual_pct |
| ---------- | ------- | ---- | ------- | --------- | ---------- |
| baseline 4 | 0.6870 | 0.4936 | 0.8542 | 0.1628 | 1.864 |
| + curing | 0.6897 | 0.5193 | 0.8435 | **0.3608** | 1.864 |
| + depletion | 0.4998 | 0.4771 | **0.1704** | 0.0497 | 0.576 |
| both | 0.5208 | 0.4901 | 0.1719 | 0.0998 | 0.596 |

**Curing confirmed.** Amplitude ratio 0.163 -> 0.361, rmse 0.494 -> 0.519, and the annual
total is unchanged (1.8637 -> 1.8644) exactly as the mean-normalisation was designed to
guarantee. The amplitude deficit really was a missing-memory problem.

**Depletion refuted at default strength.** It eats 69% of all burning and spatial collapses
0.854 -> 0.170. Physical reason: the highest-scoring fire regions are African savannas that
burn ~1/6 of their area *every year* - they are exactly where fuel regrows within a season.
Depletion strong enough to truncate a season silences the annually-reburning grasslands
that carry the spatial score. It also pushed amplitude *down* (0.050), the opposite of
intent, because suppressing the peak flattens the cycle too.

So "both terms are memory, therefore both help" was wrong; the sign of their effect on
amplitude differs. Worth noting this cost one cheap proxy run instead of a committed eval.
Keeping `depletion` in the search rather than deleting it, since fast regrowth
(`regrow_rate` -> 1.0, `dep_k` -> 0.1) is a regime Optuna can reach; let the search decide.

### Entry 5: Experiment 1 tuning, pruning, and commit

Optuna over the 7 memory coefficients (regional skeleton frozen), early-stopped at 135/300
trials. Best proxy overall **0.698** (bias 0.747, rmse 0.516, seasonal 0.857, spatial 0.853).

Winning coefficients, and they read as physically sensible rather than fitted noise:
`cure_alpha` 0.857 (~1 month rainfall memory), `cure_half` 48.2 mm (monthly rainfall above
which fuel stops being flammable), `cure_n` 1.05 (near-hyperbolic, gentle moisture response),
`cure_cap` 10.16.

**Optuna switched depletion off by itself**: `dep_k` -> 0.122 (floor of its 0.1 range) and
`regrow_rate` -> 0.847 (near ceiling), i.e. consume almost nothing and regrow instantly.
Direct A/B at the tuned point: 0.6977 with depletion, **0.6980 without**. It was marginally
*harmful*. Pruned the component, `_deplete`, and all four dep/regrow coefficients, leaving a
comment recording the negative result. Entire gain is now attributable to one mechanism.

Regional spread vs frozen baseline (fast proxy, `regional_delta.py`): **8/14 improved**.
Winners: nhsa +0.045, shaf +0.026, seas +0.021, nhaf +0.008, shsa +0.008, tena +0.007,
aust +0.006, ceas +0.000. Losers: **eqas -0.065**, ceam -0.025, mide -0.021, euro -0.016,
bona -0.007, boas -0.007.

`eqas` is the informative loss: it was the *only* region with a healthy amplitude ratio
already (0.622 vs 0.163 global), so a globally-shared curing term over-sharpens a cycle that
was never flat. Physically, equatorial Asian fires are peat/drought/water-table driven, not
grass-curing driven - the mechanism genuinely does not apply there. Same story likely for
`mide` (0.452) and `ceam` (0.370), the next-highest baseline amplitude ratios. This suggests
a clear next experiment: gate curing on a vegetation/fuel-type proxy so it applies to
grass-dominated fuels and not to forest/peat.

Committed `9055b58` and spent the official evaluation.

### Entry 6: Experiment 1 official result — curing confirmed, new best 0.698

`ar evaluate` on `9055b58` returned **Overall 0.698** (bias 0.747, rmse 0.516, seasonal 0.856,
spatial 0.854), beating Model G's 0.686. New objective best. The fast proxy predicted this
exactly (0.698 / 0.516 / 0.857 / 0.853), so the proxy is trustworthy for triage.

rmse moved 0.494 -> 0.516 and seasonal 0.845 -> 0.856: the gain came through the
double-weighted metric the amplitude diagnosis targeted, which is the outcome the
mechanism was designed for rather than a lucky refit.

Official regional deltas vs Model G — **8/14 improved**, and they match the proxy's
per-region prediction closely (proxy said eqas -0.065, official -0.068):

winners  nhsa +0.056, shaf +0.034, seas +0.029, nhaf +0.018, shsa +0.026, aust +0.008,
         tena +0.008, ceas +0.001
losers   eqas -0.068, mide -0.027, ceam -0.025, euro -0.018, bona -0.007, boas -0.004

Gain is spread across the tropical/savanna regions rather than concentrated in one, which
satisfies the soft spread requirement. The losses are concentrated in exactly the regions
flagged in Entry 5 as having healthy baseline amplitude.

### Entry 7: Experiment 2 — canopy-gated curing REFUTED and reverted

Hypothesis (queued in Entry 5): curing is a fine-fuel/grass mechanism, so fading it out over
tall canopy should recover eqas/ceam/mide without giving back the savanna gains. Implemented
as `grass = 1/(1+(canopy/grass_h_crit)^grass_n)` on max(natural, secondary) canopy height,
applied as `1 + grass*(factor-1)`.

Before tuning, wrote `scratchpad/eqas_probe.py` to check the premise. **The premise is false:**

| region | frac_tall_5m | canopy_med | curing delta |
| ------ | ------------ | ---------- | ------------ |
| shaf   | 0.806        | 10.58      | **+0.034**   |
| nhsa   | 0.805        | 24.26      | **+0.056**   |
| eqas   | 0.328        | **0.00**   | **-0.068**   |
| aust   | 0.388        | 0.18       | +0.006       |

Canopy height does not separate curing-winners from curing-losers — it separates them
*backwards*. The regions curing helps most are the tallest-canopy ones; eqas has a median
canopy of zero. GFED5's eqas cells are largely coastal/peat pixels whose land-cover average
is low, so canopy height is not a peat proxy.

Fast-proxy confirmation at three gate strengths:

| grass_h_crit | meaning              | overall |
| ------------ | -------------------- | ------- |
| 0.05         | gate shut, no curing | 0.6871  |
| 5.0 (default)| partial gate         | 0.6909  |
| 30.0         | gate open, full curing | 0.6972 |
| (no gate)    | committed model      | **0.6980** |

Monotone in "how much curing survives" — there is no useful intermediate setting, so the gate
is a pure subtraction with two extra parameters. Reverted `model.py` to `9055b58`. No eval
spent. Cost: one proxy screen.

**Lesson to carry:** eqas is not a fuel-type problem. Its cycle is already sharp (amp ratio
0.622) and its rain CV is 0.183 — the lowest of any region, i.e. aseasonal rainfall. A
rainfall-memory term cannot key a cycle there. eqas fire is drought/water-table driven
(ENSO peat years), which is an *interannual* signal, not a seasonal one. Any eqas fix must
come from interannual anomaly, not from seasonal shape.

### Entry 8: Interannual line KILLED by reading the scorer — important structural finding

Entry 7 concluded eqas needs an interannual (ENSO/drought) signal. Before building it I read
`GFED5Evaluator.score` end to end. **Interannual skill is unscorable.** Every metric collapses
the 192 months to a 12-month climatology *before* comparison:

- `candidate_cycle = candidate.reshape(16, 12, 360, 720).mean(axis=0)` — used by BOTH
  `rmse_score` (on the centered anomaly of that cycle) and `seasonal_score` (argmax phase).
- `candidate_mean = _time_mean(candidate, month_lengths)` — the 16-year mean, used by
  `bias_score` and `spatial_distribution_score`.
- The evaluator does not even retain the full reference series; it stores only
  `reference_cycle`, `reference_mean`, `reference_phase`, `reference_temporal_std`.

Consequence: a term that gets 1997-vs-2000 eqas peat fires right earns **exactly zero** unless
it shifts the 16-year mean cycle. Year-to-year variability is averaged out of existence.
Do not spend an experiment on ENSO, drought anomaly, or any interannual driver *for its own
sake*. This retires the Entry 7 "eqas needs interannual" plan.

Corollary — what the scorer actually pays for, in leverage order:
1. **rmse_score (weight 2)**: shape of the 12-month *centered* cycle, normalised per-cell by
   `reference_temporal_std`, weighted by `area * reference_mean`. Magnitude is removed by the
   centering, so this is purely about cycle *shape* in high-burning cells.
2. **bias_score (weight 1)**: 16-year mean error, same reference-burned-area weighting.
3. **seasonal_score (weight 1)**: month-of-max phase only. Discrete argmax — a cell either
   moves a whole month or nothing. Cheap to win, insensitive to shape.
4. **spatial_distribution (weight 1)**: one number per region from the 16-year mean map.

Both rmse and bias are `exp(-|err| / reference_temporal_std)`, so per-cell error only matters
relative to that cell's own temporal variability, and only where reference burning is large.

### Entry 9: Diagnosis after curing — dynamic-range compression is the binding constraint

Reran `diag_seasonal.py` on committed `9055b58`. Curing lifted global amp_ratio 0.163 -> 0.306,
but the target is 1.0, and the two regions that dominate reference-burned-area weighting are
still flat: nhaf 0.254 (15.6% burned), shaf 0.379 (16.7%). Also a broad positive bias:
global 1.55% obs vs 1.92% model; mide 0.28 vs 1.47 (5x), eqas 0.37 vs 1.22 (3x).

Wrote `scratchpad/bias_probe.py` to split each region's cells by observed burn rate. **This is
the clearest finding of the thread so far — the error is a systematic compression of dynamic
range, in every region, in the same direction:**

| region | `<0.1%` cells mod/obs | `>5%` cells mod/obs |
| ------ | --------------------- | ------------------- |
| nhaf   | **58.5x**             | 0.53x               |
| shaf   | **44.2x**             | 0.63x               |
| mide   | **20.3x**             | —                   |
| seas   | 12.6x                 | 0.24x               |
| aust   | 12.9x                 | 0.19x               |
| ceas   | 7.4x                  | 0.06x               |

The model paints fire over marginal cells that barely burn and simultaneously caps the
hotspots. This is the spatial analogue of the temporal flatness curing fixed, and it damages
bias, rmse AND spatial simultaneously — the highest-leverage target available.

Structural cause: `rate ** fire_exp` with fire_exp ~1.0-1.5 is a near-linear response, and
`_transform` = `(1-exp(-rate))/12` saturates, capping the top end at 1/12.

### Entry 10: Experiment 3 — percolation spread term

Hypothesis: burned area = ignition x spread, and spread is threshold-like. Below a fuel
connectivity threshold fires die in unburnable gaps; above it the front carries and one
ignition clears a landscape. A smooth product of favourability cannot produce that contrast.
Added `spread` component: `1 + spread_gain * s^k/(1+s^k)` on `s = rate/spread_crit`, applied
after regional splicing and curing.

Proxy screen, iterating on the formulation:

| formulation | overall | rmse | amp_ratio | annual_pct |
| ----------- | ------- | ---- | --------- | ---------- |
| committed (no spread) | 0.6980 | 0.5164 | 0.306 | 1.917 |
| raw multiplier | 0.6229 | 0.4752 | 0.416 | **3.187** |
| + mean-normalised | 0.6907 | 0.5053 | 0.418 | 2.273 |
| mean-norm, gain=2 crit=1.5 | **0.7008** | **0.5245** | 0.361 | 1.936 |

The raw multiplier sharpened contrast exactly as designed (amp_ratio 0.306 -> 0.416) but
doubled the global total, so mean-normalisation by each cell's own time mean was needed —
same device that made curing work. Best hand-probe **0.7008 > 0.6980 committed**, with the
gain in the double-weighted rmse (0.516 -> 0.525).

Swept crit at 0.1/0.6/1.5/3/6/12: optimum is interior near 1.5, and crit -> 12 asymptotes
back to exactly the no-spread baseline (0.6980), confirming the term switches itself off
cleanly rather than being load-bearing noise. Widened SEARCH_SPACE to crit [0.05, 20] and
gain [0.2, 30] so Optuna brackets the optimum instead of hitting a range edge.
Handed off to `uv run ar optuna`.

### Entry 11: Experiment 3 official result — spread confirmed, new best 0.701

`ar optuna` (68/500 trials, early stop) found **0.701**, and `ar evaluate` on `f4bb98c`
returned **official Overall 0.701** (bias 0.747, rmse 0.526, seasonal 0.855, spatial 0.853).
Proxy matched official to three decimals for the second time.

Winning coefficients are physically coherent, not fitted noise:
`spread_crit` 1.374, `spread_k` **7.04** (a genuinely sharp percolation threshold),
`spread_gain` 3.86; curing retuned to `cure_alpha` 0.902, `cure_half` 101.7, `cure_n` 0.794.

Clean attribution by A/B at identical coefficients (proxy):
with spread **0.7012**, without spread **0.6960**. The +0.005 is delivered entirely through
rmse (0.510 -> 0.526); spatial is flat-to-slightly-down (0.8561 -> 0.8528). So spread buys
the double-weighted metric, which is the intended target.

Regional spread vs Model G: **8/14 improved**. nhsa +0.035, shaf +0.030, seas +0.018,
nhaf +0.013, shsa +0.010, tena +0.006, aust +0.006, ceas +0.000.
Losers unchanged in character: eqas -0.045, mide -0.024, ceam -0.019, euro -0.012.

Cumulative: Model G 0.686 -> curing 0.698 -> +spread **0.701**.

### Entry 12: Ignition/human inputs probed against the RESIDUAL — mostly refuted

User confirmed everything under `inputs/` is fair game, so I probed the whole unused set
(`luh2.nc` x6, `population_density`, `lightning_flash_rate`, `soil_carbon`,
`leaf_area_index`, `aboveground_biomass`, veg fractions) with `scratchpad/ignition_probe.py`.

Method matters here: correlating a predictor with *observed burning* only tells you it is a
plausible fire map. The question is whether it explains **where the current model is wrong**.
So I correlated each against the per-cell bias residual (model minus obs, 16-yr mean).

| predictor | r(resid) | r(obs) | r_lowburn | seas_r |
| --------- | -------- | ------ | --------- | ------ |
| lightning_flash_rate | **0.079** | **0.409** | 0.424 | 0.223 |
| population_density | 0.129 | 0.089 | 0.176 | -0.010 |
| luh2_cropland_fraction | 0.095 | 0.025 | 0.143 | -0.024 |
| aboveground_biomass | 0.109 | -0.136 | 0.057 | 0.138 |
| leaf_area_index | 0.060 | 0.051 | 0.057 | **0.220** |
| luh2_primary_fraction | -0.051 | 0.010 | -0.064 | **0.210** |
| soil_carbon | 0.072 | -0.066 | -0.002 | 0.199 |

**This vindicates the Entry 2 decision to defer ignition.** `lightning_flash_rate` is the
strongest predictor of observed burning in the whole set (r=0.409) yet explains almost none
of the residual (r=0.079) — the existing dryness/precip/temperature terms already capture
what lightning explains. Adding it would be a fifth redundant way of drawing the same
dry-warm-vegetated map, which the Entry 2 ablation already showed is the model's saturated
dimension. Same story for population and cropland: |r(resid)| <= 0.13.

No single predictor is a silver bullet on the mean-bias residual. The one column that is NOT
saturated is `seas_r` (seasonal *shape* agreement): `leaf_area_index` 0.220,
`luh2_primary_fraction` 0.210, `soil_carbon` 0.199, `lightning` 0.223. That is the live
direction — phenology/fuel-type controlling cycle *shape*, not another intensity map.

### Entry 13: Experiment 4 — accumulated fuel load from LAI (phenology lag)

Followed the one unsaturated column from Entry 12 (`seas_r`, seasonal shape). Wrote
`scratchpad/phenology_probe.py` to correlate the observed burned-area cycle against fuel
proxies at increasing lag, reference-burn weighted.

**The cleanest physical result in the thread so far.** Instantaneous productivity is
*anti*correlated with burning, and correlation rises monotonically with lag:

| candidate | global | nhaf | shaf | seas | nhsa |
| --------- | ------ | ---- | ---- | ---- | ---- |
| gpp_lag0  | -0.072 | -0.006 | -0.088 | -0.366 | -0.217 |
| lai_lag0  | -0.095 | -0.006 | -0.191 | -0.481 | -0.328 |
| gpp_lag3  | +0.367 | 0.411 | 0.575 | 0.343 | 0.486 |
| lai_accum6 | +0.326 | 0.456 | **0.612** | 0.401 | **0.658** |
| lai_accum8 | +0.329 | 0.406 | 0.627 | 0.531 | 0.683 |
| lai_accum12 | +0.007 | -0.008 | 0.023 | 0.028 | -0.068 |

The model's `fuel` term reads instantaneous GPP — i.e. it uses the one lag where the
relationship has the **wrong sign**. Grass grows in the wet season and burns months later.
accum12 collapsing to ~0 is a good control: a full-year mean destroys phase, so this is
genuine seasonal signal rather than a spatial-map artefact.

Implemented `fuel_load`: EMA-accumulated LAI -> saturating availability -> mean-normalised.

**Result: does not beat 0.7012 in any configuration (proxy).**

| configuration | overall | rmse | seasonal | annual_pct |
| ------------- | ------- | ---- | -------- | ---------- |
| current best (no fuel_load) | **0.7012** | 0.5256 | 0.8554 | 1.907 |
| + fuel_load (alpha 0.25) | 0.6911 | 0.5225 | 0.8291 | **1.689** |
| + fuel_load (alpha 0.12, ~8mo) | 0.6922 | 0.5237 | 0.8309 | 1.685 |
| fuel_load replaces gpp fuel | 0.6851 | 0.5144 | 0.8201 | 2.490 |
| fuel_load, curing dropped | 0.6817 | 0.4999 | 0.8204 | 1.669 |

Longer memory monotonically better (alpha 0.6 -> 0.12 gives 0.686 -> 0.692), matching the
probe, so the mechanism behaves as designed — it just doesn't pay. Two diagnostics explain why:

1. `seasonal` drops 0.855 -> 0.831 whenever fuel_load is on: it *fights* curing. Both are
   memory terms on the same causal chain — rainfall drives the growth that becomes fuel — so
   stacking them double-counts the same delay and over-rotates the phase.
2. Dropping curing and keeping fuel_load collapses amp_ratio 0.336 -> 0.183. Curing, not
   fuel_load, is what sharpens the cycle; LAI accumulation cannot substitute for it.

So the lag physics is real but **already captured upstream** by antecedent precipitation.
Handed to `ar optuna` to check for a complementary regime before discarding.

### Entry 14: fuel_load abandoned; Optuna run killed as stale

Left `ar optuna` tuning fuel_load, then replaced model.py while it ran, so its result would
have applied to a file that no longer existed. Killed it (exit 144 = my own pkill, not a
crash). No loss: Entry 13 already showed fuel_load loses in all four configurations and the
mechanism is redundant with curing. **fuel_load discarded.** The lag physics is real but
antecedent precipitation already encodes it one step upstream.

Backup of the fuel_load+BorealNA variant kept at `scratchpad/model_fuelload_borealna.py`.
Clean 0.701 base kept at `scratchpad/model_spread_0701.py`.

### Entry 15: Experiment 5 — REGION_BOXES structural bug, boreal North America

Ranked regions in the official 0.701 result. Three spatial catastrophes:
mide 0.297, eqas 0.350, and **bona 0.035**. Spatial is a full 1/5 of Overall, so these are
the largest remaining headroom in the model.

Wrote `scratchpad/bona_probe.py`. bona's problem is NOT pattern — it is amplitude:

| region | pattern corr | obs_mean | mod_mean | obs_std | mod_std |
| ------ | ------------ | -------- | -------- | ------- | ------- |
| bona   | **0.527**    | 0.0150   | 0.0019   | 0.0543  | 0.0059  |
| boas   | 0.231        | 0.0970   | 0.1456   | 0.2830  | 0.2014  |
| mide   | 0.277        | 0.0253   | 0.1010   | 0.1310  | 0.2947  |

bona is underpredicted **8x** in the mean and **9x** in spatial std. ILAMB's spatial score
penalises the variance ratio, so a decent pattern with 9x too little contrast scores 0.035.

**Root cause found — a structural bug in `REGION_BOXES`, not a physics gap.**
`'Boreal': (40.0, 180.0, 48.0, 78.0)` spans 40E-180E, i.e. **Eurasia only**. Boreal North
America (roughly 168W-52W) therefore falls through to `'N.America': (-168.0, -52.0, 14.0,
74.0)`, whose parameters are fitted overwhelmingly on temperate/subtropical US fire.
Boreal Canada and Alaska have been modelled with the wrong regime's coefficients since
Model G. Boreal fire is rare, huge, high-intensity crown fire — a globally-tuned temperate
response necessarily suppresses it, which is exactly the 8x deficit observed.

Added `'BorealNA': (-168.0, -52.0, 50.0, 74.0)` FIRST in the box order so it claims those
cells before N.America, seeded from the fitted Boreal params. Neutral check: global 0.7007
vs 0.7012 base (unchanged, as expected), bona -0.015 — Eurasian params are not right for NA
either, which is the point: the region now has its own knobs.

Exposed five BorealNA coefficients to SEARCH_SPACE via a `bna_*` prefix overlaid onto the
regional dict inside `predict` (fire_exp, fire_amp, ign_c, ign_k, D_low). Wiring verified
neutral at defaults (0.7008). Handed to `ar optuna`.

### Entry 16: Experiment 5 RETRACTED — do not add regions

User: "you cannot add ur own regions!" Correct, and the BorealNA split is reverted
(`model.py` back to committed `f4bb98c`, no BorealNA, no `bna_*`; optuna killed).

**Standing rule for the rest of this thread: `REGION_BOXES` and `REGION_PARAMS` are a fixed
scaffold. Do not add, split, or re-draw regions. Improvements must come from COMPONENTS —
mechanisms that apply everywhere and earn their place globally.**

Why this was the wrong move, beyond the rule:
- It is not a mechanism. Carving a box out of N.America and giving it five free coefficients
  is extra parameters aimed at the single worst-scoring region, not physics.
- It cuts against the soft requirement that gains be spread across regions rather than
  concentrated in one.
- `ar ablate` operates over `COMPONENTS`, so a region split is invisible to the ablation
  machinery that is supposed to justify added complexity. Untestable by the tools provided.

The underlying *observation* from Entry 15 still stands and is still useful: bona is
underpredicted 8x in mean and 9x in spatial std with a decent pattern corr (0.527), and
boreal fire is a genuinely distinct regime (rare, high-intensity crown fire). The legitimate
way to reach it is a **global component keyed on a physical covariate** that happens to
distinguish boreal conditions — e.g. a cold/short-growing-season fire-intensity response, or
a fuel-continuity term — applied everywhere and required to earn its place globally.
Not a hand-drawn box.

### Entry 17: Experiment 6 — growing-season-length control (intensity, done legitimately)

Replacement for the retracted region split: a **global component keyed on a physical
covariate**, applied everywhere, visible to `ar ablate`. Wrote `scratchpad/intensity_probe.py`,
correlating covariates against log(obs/model) mean burned area — i.e. where the model is
*multiplicatively* wrong.

**My stated hypothesis was refuted, and something stronger turned up.** Inside the boreal
zone (lat>48) every covariate is flat (|r| <= 0.10) — there is no cold-regime signal to key
on, so "boreal fires are rare but intense" is not recoverable from these inputs. But globally
`growing_months_T>5` gives **r = -0.304**, and in the tropics **-0.673** — the strongest
single relationship found in this thread.

Binned, the gradient is monotone and large:

| growing months (T>5) | n | obs/model ratio |
| -------------------- | ---- | --------------- |
| 0-2   | 302  | **6.42** |
| 4-6   | 2104 | 1.06 |
| 6-8   | 1342 | 1.26 |
| 8-10  | 604  | 0.50 |
| 11.5-12 | 4562 | 0.80 |

Short-season cells burn ~6x more than the flammability product predicts; year-round-growing
cells about half. Physical reading: **fuel residence time**. A brief cold season means slow
decomposition, so litter accumulates across years into a deep connected bed that carries rare
high-intensity fire; year-round growth turns fuel over continuously and no stock builds.
This subsumes the boreal observation from Entry 15 without hand-drawing a region.

Two formulations tested (proxy):

| form | overall | rmse | spatial | bias | annual_pct |
| ---- | ------- | ---- | ------- | ---- | ---------- |
| baseline (no term) | **0.7012** | 0.5256 | 0.8528 | 0.7465 | 1.907 |
| (a) multiply the rate, defaults | 0.6712 | 0.5104 | **0.7483** | 0.7296 | 1.342 |
| (a) floor 0.95 | 0.7011 | 0.5256 | 0.8515 | 0.7463 | 1.863 |
| (b) modulate spread threshold | 0.6977 | 0.5131 | **0.8565** | **0.7491** | 1.926 |

Form (a) multiplies the fire rate and **collapses spatial** 0.853 -> 0.748: it rescales the
16-year mean map, which is the model's best-scoring dimension. Sweeping it is monotone toward
`gs_floor` -> 1.0, i.e. Optuna would switch it off — the same self-cancelling signature the
canopy gate showed in Entry 7.

Form (b) instead lowers the *percolation threshold* where fuel has had time to accumulate
(deep fuel bed percolates at lower flammability). This is the physically correct coupling —
season length is a fuel-bed property, not an ignition multiplier. It **preserves** spatial
(0.8565, above the 0.8528 baseline) and improves bias (0.7491), but loses rmse, so it still
trails 0.7012 at hand-probed coefficients. Kept form (b) and handed to `ar optuna` — the
hand probes only varied 2 of 4 new coefficients against 3 frozen spread ones.

### Entry 18: Experiment 6 result — season_length REFUTED after joint tuning

`ar optuna` over all 11 coefficients (4 new + 7 existing), 102/500 trials, early stop:
**best overall=0.700** (bias 0.747, rmse 0.521, seasonal 0.857, spatial 0.855) vs committed
**0.7012**. Loses.

A/B at the term's OWN tuned optimum (proxy):
  with season_length  **0.7002**
  without             **0.7000**

i.e. the component is worth **+0.0002** at the point chosen to favour it. `gs_floor` tuned to
0.195 (strong modulation, not switched off) and `gs_temp` to -3.5C, so Optuna genuinely tried
to use it and still extracted nothing. Discarded; `model.py` restored to committed `f4bb98c`.

**Why the 6.4x gradient did not convert into score.** The binned relationship in Entry 17 is
real, but it is a property of the **16-year mean map**, and it is diagnosed against a residual
that the reference-burned-area weighting largely ignores. The short-season cells carrying the
6.4x ratio have obs=0.12% burned; the year-round cells sit at obs=1.03% and hold ~4562 of the
cells. bias_score and rmse_score weight by `area * reference_mean`, so fixing a 6x error in
cells that barely burn moves almost nothing, while any collateral damage in the high-burning
tropics is fully counted. This is the third time a strong *correlation* with a residual has
failed to convert (Entry 12 lightning, Entry 13 fuel_load, now this).

**Standing lesson for the rest of the thread:** stop screening candidate mechanisms by
correlation against an unweighted residual. Screen by `area * reference_mean`-weighted
residual, or the probe will keep pointing at low-burn cells that cannot pay. Retrofit this
into any future probe.

### Entry 19: Weighted-residual retarget — the whole game is African savanna rmse

Acted on the Entry 18 lesson. Wrote `scratchpad/weighted_residual.py`, ranking recoverable
score as `weight * (1 - score)` with the scorer's own `area * reference_mean` weighting and
rmse counted double.

Total recoverable = 1.202, of which **rmse 0.949 (79%)** and bias 0.254.

| region | wt_share | recoverable | bias_sc | rmse_sc | mod/obs |
| ------ | -------- | ----------- | ------- | ------- | ------- |
| nhaf   | 0.312    | **0.393**   | 0.763   | **0.487** | 1.11 |
| shaf   | 0.302    | **0.361**   | 0.756   | **0.525** | 1.33 |
| seas   | 0.073    | 0.097       | 0.700   | 0.481   | 1.19 |
| ceas   | 0.079    | 0.088       | 0.773   | 0.555   | 1.00 |
| eqas   | 0.006    | 0.008       | 0.594   | 0.514   | 3.07 |
| mide   | 0.005    | 0.008       | 0.568   | 0.458   | 5.30 |
| bona   | 0.005    | **0.004**   | 0.830   | 0.699   | 0.13 |

**nhaf + shaf alone hold 61% of all recoverable score.** mide + eqas + bona together hold
**under 2%**. Every region I chased for its terrible *reported* score (bona spatial 0.035,
mide 0.297) is worth essentially nothing — they are low-burn regions the weighting discounts.
The retracted region split was chasing 0.4% of the available score. Retarget: African savanna
seasonal-cycle **shape**, which is 79% rmse.

`scratchpad/africa_cycle.py` shows the failure precisely — **phase is correct, contrast is not**:

nhaf  obs 16.31 7.44 3.49 1.63 0.58 0.12 0.04 0.03 0.07 1.59 9.11 18.70  (amp 6.33)
nhaf  mod  6.62 6.22 4.27 2.70 1.89 1.57 1.13 0.89 0.92 1.68 4.37  6.43  (amp 2.15)

Peak 2.9x too low AND off-season floor ~30x too high (obs 0.03 vs mod 0.89). Same in shaf,
seas, aust.

`scratchpad/floor_probe.py` finds the structural reason. Per-term monthly min/max in nhaf:
dryness 0.220, precip 0.347, curing 0.388, **fuel 0.926, temp 0.993**. Fuel and temperature
are seasonally inert — they contribute nothing to the cycle. The three live terms multiply to
~0.03 dynamic range against an observed 0.0016 (0.03/18.7), i.e. **~20x short**. All factors
are bounded [0,1] and saturate near 1 in the fire season, so the product can only suppress the
off-season, never amplify the peak.

### Entry 20: Experiment 7 — `sharpen`, cycle contrast about each cell's own mean

Added `sharpen`: raise rate to `sharp_p` relative to its own annual mean, renormalise.

| formulation | overall | rmse | spatial | amp_ratio | annual_pct |
| ----------- | ------- | ---- | ------- | --------- | ---------- |
| committed baseline | **0.7012** | 0.5256 | **0.8528** | 0.336 | 1.907 |
| mean-anchored, p=3.0 | 0.6739 | **0.5303** | 0.7426 | 0.415 | **1.501** |
| mean-anchored, p=1.3 | 0.6968 | 0.5295 | 0.8332 | 0.369 | 1.765 |
| rate-multiplied, p=1.5 | 0.6815 | 0.5122 | 0.8223 | 0.470 | 2.305 |

**rmse improves to 0.5295-0.5303, the best of the thread** (baseline 0.5256), and annual_pct
1.50 is near-perfect (obs 1.55). But spatial degrades monotonically with `sharp_p`.
The `rate *` variant double-applies and inflates annual to 2.3 — worse; discarded.

Since rmse is double-weighted the joint optimum may still exist, and the hand probes varied
only 1 of 9 coefficients. Handed the mean-anchored form to `ar optuna`.

**Eval discipline check (user asked):** `results.tsv` confirms only two rows added this
thread, both from `ar evaluate` with matching git commits — `9055b58` 0.698 and `f4bb98c`
0.701. The scratchpad proxy has never recorded anything; it is triage only, and has now
killed five formulations (canopy gate, fuel_load, season_length, both sharpen variants so
far) before they cost an eval. Proxy has matched official to three decimals both times.

### Entry 21: Experiment 7 result — sharpen REFUTED

`ar optuna` over 9 coefficients, 112/500 trials: **best 0.698** vs committed 0.7012. Loses.

Diagnostic detail: `spread_crit` drifted to **7.50**, near the top of its [0.05, 20] range,
which from the Entry 10 sweep is where spread effectively switches off. So Optuna's best
sharpen solution works by *disabling spread* — the two compete for the same job (sharpening
cycle contrast), and the pair is worse than spread alone. Discarded. `sharpen` had the best
rmse of the thread in isolation (0.5303) but could never pay for its spatial cost.

Backup at `scratchpad/model_sharpen_wip.py`; `model.py` restored to committed `f4bb98c`.

### Entry 22: PRUNING ROUND (research.md asks every ~8 ideas; this is idea 8)

Ran `ar ablate` on committed `f4bb98c`, captured in full to `scratchpad/ablate_f4bb98c.txt`
(first run scrolled past the task-output buffer and lost the global rows — rerun redirected
to a file). Computed exact global Shapley from the 64 subsets; sum reconciles to +0.3020,
matching the tool's reported total.

| component | global shapley | drop-one |
| --------- | -------------- | -------- |
| precipitation | +0.0815 | +0.0260 |
| dryness | +0.0679 | +0.0230 |
| temperature | +0.0613 | +0.0270 |
| fuel | +0.0550 | +0.0110 |
| curing | +0.0333 | +0.0110 |
| spread | +0.0029 | +0.0050 |

**Nothing is pruned — every component has positive Shapley AND positive drop-one.**
empty=0.399, full=0.701.

The interesting case is `spread`, whose global Shapley (+0.0029) is the weakest but whose
drop-one (+0.0050) is *larger* — it is complementary rather than redundant. Regionally it
resolves the tension flagged earlier:

| region | spread ov | spread rmse | weight share |
| ------ | --------- | ----------- | ------------ |
| shaf   | **+0.007** | **+0.019** | 0.302 |
| nhaf   | **+0.004** | **+0.013** | 0.312 |
| eqas   | -0.006 | -0.015 | 0.006 |
| boas   | -0.003 | -0.007 | 0.022 |
| ceas   | -0.003 | -0.007 | 0.079 |
| aust   | -0.000 | -0.004 | 0.073 |

spread is positive in exactly the two regions holding 61% of recoverable score, negative in
low-weight regions. It is not a global mechanism — it is an African-savanna rmse mechanism
that the weighting rewards. Honest characterisation to carry forward.

`curing` is the workhorse where it counts: shaf +0.076 (rmse +0.052, seasonal +0.269),
nhaf +0.019 (rmse +0.036).

Also confirmed from the subset table: `fuel` and `temperature` are NOT prunable despite being
seasonally inert (Entry 19). Their value is **spatial** — nhaf fuel spat +0.242, temp +0.205;
shaf fuel +0.232, temp +0.223 — i.e. they draw the map, they just do not shape the cycle.
That is a division of labour, not dead weight, and it explains why every attempt to add a
*seasonal* mechanism on top keeps colliding with spatial score.

### Entry 23: STRUCTURAL BUG FOUND — the output transform capped burned area at 8.33%/month

Four refutations in a row (canopy gate, fuel_load, season_length, sharpen) all failed the same
way: they improved rmse but cost spatial, and could never lift the seasonal PEAK. Entry 22
showed why the model is organised that way (fuel/temperature carry spatial, curing/spread
carry rmse), so I probed the collision itself with `scratchpad/collision_probe.py`.

`_transform` was:

    rate = np.minimum(rate, 5.0)
    return (1.0 - np.exp(-rate)) / 12.0

**The `/12` caps monthly burned fraction at 8.333%, unconditionally.** Observed peak months in
the highest-weight regions:

| region | observed peak | reachable? |
| ------ | ------------- | ---------- |
| nhaf December | **18.70%** | NO — 2.2x above the cap |
| shaf August | **11.63%** | NO |
| seas March | **9.81%** | NO |

And the model sits hard against that ceiling exactly there: weighted median peak-month rate is
**3.31 in nhaf and 3.00 in shaf**, where `(1-exp(-r))/12` is 95-96% saturated and the
derivative is ~0. 18-20% of weighted month-cells sit at rate 2-5. The `np.minimum(rate, 5.0)`
compounded it by clipping before the exponential.

So the seasonal amplitude deficit diagnosed back in Entry 2 was **never a physics problem**.
No reformulation of the fire rate could close it, because the peaks were mathematically
unreachable at any parameter setting. Every seasonal mechanism I added could only deepen
troughs, which is precisely the rmse-up/spatial-down signature all four refutations showed.
The `/12` treats the annual rate as spread evenly over months; the correct form applies
Poisson exceedance to the month itself.

Fixed to `1 - exp(-rate * month_scale)` with `month_scale` searchable (the old behaviour is
recoverable at month_scale = 1/12 = 0.0833, so this strictly generalises rather than replaces).

Hand probe, before any retuning:

| month_scale | overall | bias | rmse | seasonal | spatial | amp_ratio | annual_pct |
| ----------- | ------- | ---- | ---- | -------- | ------- | --------- | ---------- |
| committed baseline | 0.7012 | 0.7465 | 0.5256 | 0.8554 | 0.8528 | 0.336 | 1.907 |
| 0.0833 (old behaviour) | 0.6389 | 0.6658 | 0.5181 | 0.8558 | 0.6368 | **0.904** | 3.106 |
| 0.02 | 0.6170 | 0.6943 | 0.5219 | 0.8558 | 0.4912 | 0.238 | 0.784 |
| **0.04** | **0.7120** | **0.7534** | **0.5469** | 0.8558 | **0.8570** | 0.463 | **1.543** |

**month_scale=0.04 gives 0.7120 on the proxy — the first change in this thread to improve
rmse AND spatial AND bias simultaneously** (rmse 0.5256->0.5469, spatial 0.8528->0.8570,
bias 0.7465->0.7534), with annual_pct 1.543 against observed 1.55. amp_ratio at the old
scaling jumps 0.336->0.904, confirming the ceiling was the binding amplitude constraint.

Handed to `ar optuna` for a joint refit, since removing the cap rescales the whole rate field
and the existing coefficients were all fitted against the capped transform. NOT yet committed
or evaluated — proxy only.

### Entry 24: transform fix refined — month_scale 0.05, and sharpen retested

Refined the hand probe while the joint retune runs:

| month_scale | overall | bias | rmse | spatial | amp_ratio | annual_pct |
| ----------- | ------- | ---- | ---- | ------- | --------- | ---------- |
| 0.03 | 0.6818 | 0.7315 | 0.5379 | 0.7457 | 0.352 | 1.166 |
| 0.04 | 0.7120 | 0.7534 | 0.5469 | 0.8570 | 0.463 | 1.543 |
| **0.05** | **0.7145** | **0.7569** | **0.5492** | **0.8615** | 0.570 | 1.913 |
| 0.07 | 0.6752 | 0.7106 | 0.5361 | 0.7373 | 0.775 | 2.637 |

Sharp interior optimum at 0.05: **0.7145** proxy vs committed 0.7012.

**Retested `sharpen` now that the ceiling is gone** (it was refuted in Entry 21, but only
because peaks were clamped — worth one cheap recheck):

| sharp_p | overall | rmse | spatial | amp_ratio |
| ------- | ------- | ---- | ------- | --------- |
| none | **0.7145** | **0.5492** | 0.8615 | 0.570 |
| 1.2 | 0.7139 | 0.5483 | 0.8627 | 0.630 |
| 1.6 | 0.7118 | 0.5429 | 0.8645 | 0.713 |
| 2.2 | 0.7092 | 0.5366 | **0.8656** | 0.769 |

The ceiling really was the cause of sharpen's old damage — it now *raises* spatial
(0.8615 -> 0.8656) instead of collapsing it (0.853 -> 0.743 in Entry 20), and pushes
amp_ratio to 0.769. But overall still peaks without it, because rmse falls monotonically as
amplitude grows past ~0.6. **Amplitude is no longer the binding constraint.** Confirmed
sharpen stays discarded, now for a sound reason rather than an artefact.

African cycles with the fix (`africa_cycle.py`), vs Entry 19:

  shaf amp_ratio **0.52 -> 0.93** — peak 12.11 vs obs 11.63, tracks closely all year
  nhaf amp_ratio **0.34 -> 0.62** — peak 11.63 (was 6.62) vs obs 18.70

nhaf residual is now a **phase/shape** error, not amplitude: model peaks in January (11.63)
while obs peaks in December (18.70), and the model runs ~0.5-1.0 too high across months 2-9.
That is the next target once the retune lands, and it is a different mechanism than anything
tried so far (the seasonal_score is argmax-based and already 0.856, so this is rmse-only).

### Entry 25: nhaf phase diagnosis — observed peak matches NO driver extremum

`scratchpad/nhaf_phase.py`, normalised monthly climatologies vs observed burning:

nhaf obs peaks **December** (18.70%). Driver peaks: dryness **May**, air_temperature **March**,
gpp **October**, monthly_precipitation minimum **January**. **Nothing peaks in December.**
The model peaks in January because that is where the instantaneous drivers align — so the
one-month phase error is structural, not a tuning problem.

The observed December peak falls *between* GPP maximum (Oct) and rainfall minimum (Jan), i.e.
in the **dry-down transition**. Tested whether the rate of drying beats the level of dryness:

| region | precip_level corr | drying_rate corr | obs peak | precip_level peak |
| ------ | ----------------- | ---------------- | -------- | ----------------- |
| nhaf | **+0.771** | +0.345 | 12 | 1 |
| shaf | **+0.894** | +0.065 | 8 | 7 |

**Drying-rate hypothesis refuted** — much weaker than level in nhaf, near-zero in shaf. But the
table shows something cleaner: both regions peak **exactly one month after** the precip-level
peak (nhaf 12 vs 1, shaf 8 vs 7). That is a short lag, not a derivative. Physically sensible —
fine fuel needs some weeks at low humidity before it will carry fire, so flammability trails
the dryness signal.

Note `curing` already applies an EMA lag, but to *precipitation* as an antecedent-wetness
memory, and its tuned `cure_alpha` 0.902 means almost no memory (alpha near 1 = instantaneous).
So the fitted model has effectively no lag. Next experiment once the retune lands: a short
explicit lag on the assembled fire rate, one coefficient, testable by ablation.

### Entry 26: Experiment 8 OFFICIAL — transform fix recorded, 0.715 (new best)

`ar optuna` 211/500 trials -> 0.715. Installed, committed `e92b034`, and `ar evaluate`
returned **official Overall 0.715** (bias 0.758, rmse 0.547, seasonal 0.856, spatial 0.866).
Proxy matched official to three decimals for the **third** time.

Progression: Model G 0.686 -> curing 0.698 -> spread 0.701 -> **transform fix 0.715**.
This is the largest single gain of the thread, and every metric improved at once
(rmse +0.021, spatial +0.013, bias +0.011) — the signature of removing a real constraint
rather than trading one metric for another.

Tuned `month_scale` = 0.0423, i.e. about half the old 1/12 = 0.0833.

**Caveat recorded honestly: regional spread is 4/14.**

  winners  shaf +0.048, nhaf +0.046, seas +0.013, euro +0.004
  losers   ceam -0.048, boas -0.047, shsa -0.044, nhsa -0.039, mide -0.037,
           eqas -0.031, aust -0.019, ceas -0.012, bona -0.011, tena -0.006

The gain is concentrated in exactly the two regions the `area * reference_mean` weighting
favours (Entry 19: nhaf+shaf = 61% of recoverable score). Hard objective satisfied, soft
requirement not. Verified this is a property of the tuned optimum, not the structural fix:
holding the new coefficients but setting month_scale back to 1/12 collapses to 0.610, so
scale and rate coefficients are tightly coupled.

### Entry 27: why 4/14 — per-region optimal scale spans 28x

`scratchpad/scale_probe.py`, grid-searching month_scale per region:

| region | best scale | x global | gain available |
| ------ | ---------- | -------- | -------------- |
| bona | 0.3500 | **8.27** | +0.081 |
| boas | 0.0922 | 2.18 | +0.055 |
| nhsa/shsa/aust | 0.0739 | 1.75 | +0.083/+0.053/+0.031 |
| nhaf | 0.0473 | 1.12 | **+0.005** |
| shaf | 0.0379 | 0.90 | **+0.000** |
| eqas | 0.0195 | 0.46 | +0.106 |
| mide | 0.0125 | **0.29** | +0.127 |

The optimum spans **28x** (mide 0.0125 -> bona 0.35). nhaf and shaf sit essentially AT the
global value with ~zero gain available — Optuna tuned the constant for them, as the weighting
dictates. Every other region is off its own optimum, which is precisely the 10 regressions.

### Entry 28: Experiment 9 — temperature-dependent patch scale REFUTED

month_scale is physically a characteristic fire patch size, which should vary by fuel type.
Regressed log(best_scale) on regional mean covariates: gpp r=+0.09, annual_precip r=+0.05,
**air_temperature r=-0.47** (cold regimes want a larger scale — consistent with bona 0.35).
Only temperature showed anything, and across 14 points that is weak evidence.

Implemented `scale * exp(patch_t_k * (patch_t_ref - warmth))` in the transform.

| patch_t_k | overall | rmse | spatial |
| --------- | ------- | ---- | ------- |
| 0.00 (off) | **0.7148** | 0.5469 | 0.8657 |
| 0.04 | 0.6883 | 0.5342 | 0.7852 |
| 0.08 | 0.6459 | 0.5148 | 0.6503 |

Monotonically worse. **Refuted.** The 14-point regional correlation did not survive per-cell
application — regional-mean temperature is not the within-region control on patch size. Good
outcome for a weak fit; reverted to `e92b034`.

Note this is the second time a promising aggregate correlation failed at cell level
(Entry 12/13/18 pattern). Aggregate-level correlations in this problem are near-worthless as
evidence; only cell-level application decides.

### Entry 29: Experiment 10 — LAI-dependent patch scale REFUTED (both directions)

Second attempt at making month_scale spatially varying, this time on a per-cell fuel
covariate rather than a regional mean: `scale * (LAI/patch_half)^patch_n`.

| patch_n | overall | spatial | note |
| ------- | ------- | ------- | ---- |
| +0.6 | 0.6369 | 0.6574 | more fuel -> larger patches |
| +0.3 | 0.6880 | 0.8099 | |
| **0.0** | **0.7148** | **0.8657** | off |
| -0.25 | 0.6296 | 0.5076 | sparse fuel -> larger patches |
| -0.5 | 0.5348 | 0.0998 | |

**Refuted in both directions**, and steeply. (First negative run was a false test — a
`patch_n > 0.0` guard silently disabled it and returned baseline verbatim; then a genuine
divide-by-zero on LAI=0 cells caught by the runtime's finite check. Fixed the guard to
`abs(...) > 1e-9` and floored the ratio before re-testing. Worth noting the runtime contract
check caught the bad build immediately.)

**Interpretation across Entries 27-29.** The scale probe says +0.05..+0.13 is available in
eight regions if month_scale could vary spatially, but neither temperature nor LAI recovers
it, in either direction. The per-region optima do not follow any single smooth covariate I
have tested; they look like genuinely regional regime differences. Since hand-drawn regions
are off-limits (Entry 16) and the two obvious physical covariates fail, this line is parked.

Remaining honest options for the 4/14 spread, none tried yet:
1. accept it — the hard objective is global Overall, and the weighting genuinely favours
   nhaf/shaf; 0.715 is recorded and defensible
2. find a covariate that predicts patch size per-cell that is not temperature or LAI
   (candidates: dryness amplitude, precip seasonality/CV, gpp seasonality)
3. attack the nhaf one-month phase lag from Entry 25, which is region-agnostic and would
   help wherever the lag exists rather than trading regions against each other

Option 3 is the next experiment: it is a mechanism, not a rescaling, so it cannot be a
zero-sum trade between regions the way month_scale is.

### Entry 30: Experiment 11 OFFICIAL — one-month flammability lag, 0.717 (new best)

Followed Entry 25's phase diagnosis. Added `lag` component: blend a fraction of the previous
month's assembled rate forward, one coefficient `lag_w`, applied before spread.

**Optuna's joint search FAILED here and I did not follow it.** 304/500 trials returned
best 0.712 — *below* the committed 0.7148 — having wandered to a worse basin (`cure_n` 5.89,
near its ceiling; month_scale 0.0363). The controlled test is the lag at the already-tuned
coefficients, holding everything else fixed:

| lag_w | overall | rmse | seasonal | spatial |
| ----- | ------- | ---- | -------- | ------- |
| 0.00 | 0.7148 | 0.5469 | 0.8562 | 0.8657 |
| 0.15 | **0.7166** | 0.5489 | 0.8590 | 0.8674 |
| 0.20 | 0.7165 | 0.5489 | 0.8585 | 0.8678 |
| 0.25 | **0.7166** | 0.5485 | 0.8591 | 0.8681 |
| 0.30 | 0.7162 | 0.5478 | 0.8587 | 0.8683 |

Flat interior optimum 0.15-0.25. Took `lag_w = 0.2` by hand rather than Optuna's basin —
defensible because it is a single coefficient with a flat optimum while every other
coefficient was already tuned, and the A/B is clean: **0.7165 with lag vs 0.7148 without at
identical coefficients** (+0.0017; rmse, seasonal and spatial all up).

Committed `b9263d4`; `ar evaluate` returned **official Overall 0.717** (bias 0.758,
rmse 0.549, seasonal 0.858, spatial 0.868). Proxy said 0.7165 — **fourth** consecutive
three-decimal match.

**The lag improves 7/14 regions, and crucially it helps the ones the transform fix hurt:**

  eqas **+0.0155**, ceam **+0.0131**, mide **+0.0090**, shaf +0.0065, shsa +0.0027,
  aust +0.0016, tena +0.0015
  nhaf -0.0059, ceas -0.0045, nhsa -0.0037, bona -0.0021, seas -0.0015, boas -0.0007,
  euro -0.0002

This is the opposite pattern to `month_scale` (Entry 27), which is exactly what Entry 29
predicted: a *mechanism* redistributes error non-zero-sum, whereas a global *rescaling* trades
regions against each other. It partially repairs the spread problem the transform fix created,
and it does so while also raising global Overall.

Progression: 0.686 -> 0.698 -> 0.701 -> 0.715 -> **0.717**.

**Lesson recorded: Optuna's joint search is not always right.** With ~9 coupled coefficients
and 300 trials it can land below a well-chosen point. When adding a single new coefficient to
an already-tuned model, A/B it at the fixed tuned point as the control, and treat the joint
retune as a separate question.

### Entry 31: Two lag refinements REFUTED; and a correction to why lag works

Tried to extend the winning lag two ways, both at the tuned point:

(a) **Two-month distributed kernel** (`lag_w2` on rate[t-2]):
    0.0 -> **0.7165** | 0.1 -> 0.7138 | 0.2 -> 0.7089. Monotonically worse.
    The lag is genuinely one month, not a distributed memory.

(b) **Rainfall-dependent lag length** (drying takes longer where there is more water):
    0.0 -> **0.7165** | 0.3 -> 0.7091 | 0.8 -> 0.6926. Monotonically worse.
    The delay is uniform, not modulated by wetness.

Both reverted. Clean negative results: the mechanism is simple and already fully captured by
one coefficient.

**Correction to Entry 30's stated mechanism.** I described the lag as shifting the modelled
peak later. Checked it directly by comparing peak months at lag_w = 0.0 vs 0.2:

    lag_w=0.0: nhaf obs_peak=12 mod_peak=1 | shaf obs_peak=8 mod_peak=7
    lag_w=0.2: nhaf obs_peak=12 mod_peak=1 | shaf obs_peak=8 mod_peak=7

**The lag does not move the argmax at all.** Its +0.0017 comes from reshaping the cycle
(smoothing the shoulders), not from correcting phase. The physical story in the commit message
is about fuel-moisture delay, which is still a fair description of the operation, but the
*scoring* benefit is not the phase correction I claimed. Recording this so the next
experiment is not built on a wrong premise.

The phase error is in fact unfixable by any global lag, because the two big regions need
shifts in **opposite directions**: nhaf is modelled one month LATE (Jan vs obs Dec) and shaf
one month EARLY (Jul vs obs Aug). A single global shift cannot satisfy both. Note also
seasonal_score is argmax-based and already 0.858, so the remaining phase error costs mainly
rmse, not seasonal.

### Entry 32: state at 0.717 — where error now lives

`weighted_residual.py` rerun: total recoverable **1.202 -> 1.144**, still 79% rmse, still
61% in nhaf+shaf. African cycles (`africa_cycle.py`):

  **shaf now excellent**: amp_ratio 0.52 -> **0.88**, tracks obs within ~0.3 most months.
  **nhaf still the outlier**: amp_ratio 0.34 -> 0.60, but model peaks Jan 10.97 vs obs Dec
  18.70, and runs 0.3-2.0 too high across months 2-9.

New signal: `mod/obs` total burning is now **1.21 in nhaf and 1.45 in shaf** (was 1.11/1.33).
The transform fix bought amplitude partly by overpredicting the annual total in the two
highest-weight regions. That is a live target — bias_score is the second-largest recoverable
term at 0.242 — and unlike amplitude it is not constrained by anything structural.

### Entry 33: nhaf phase shift — no covariate separates it. Line exhausted.

Diagnostic: best whole-month shift of the modelled cycle, per region, by rmse.

  **nhaf wants -1 month: rmse 3.485 -> 2.914, a 16% reduction** (the single largest
  identified rmse gain remaining, in the highest-weight region)
  ceas wants -1 (1.028 -> 0.976); shaf, seas, aust, and most others want 0.

So the needed lag really is region-specific, confirming Entry 31: no global lag can fix it.
Since regions cannot be hand-drawn (Entry 16), it would have to come from a covariate.
Searched for one — dry-season length, annual rainfall, rainfall CV:

| region | wanted shift | dry_len | annual mm | rain_cv |
| ------ | ------------ | ------- | --------- | ------- |
| **nhaf** | **-1** | 5 | 1191 | 0.820 |
| **shaf** | **0** | 5 | 1095 | 0.781 |
| seas | 0 | 5 | 1637 | 0.818 |

**nhaf and shaf are climatic near-twins** — same dry-season length, near-identical rainfall CV
and annual total — yet want different shifts. Nothing observable in these inputs separates
them, so no covariate-driven lag can work. **Line exhausted and closed.** (Physically the
difference is probably that nhaf's Sahel fire season is driven by human clearing on a calendar
schedule rather than by climate, which these inputs cannot see.)

Retargeting to the Entry 32 finding instead: nhaf mod/obs = 1.21 and shaf = 1.45, i.e. the
transform fix bought amplitude partly by overpredicting annual totals in the two
highest-weight regions. bias_score is the second-largest recoverable term (0.242) and, unlike
the phase error, nothing structural prevents fixing it.

### Entry 34: Experiment 12 OFFICIAL — sharper percolation exponent, 0.718 (new best)

`bias_probe.py` at 0.717 showed the Entry 9 dynamic-range compression **persists** in the two
regions that matter:

| region | `<0.1%` cells mod/obs | `>5%` cells mod/obs | wt_share of `<0.1` |
| ------ | --------------------- | ------------------- | ------------------ |
| nhaf | **57.2x** | 0.62x | 0.003 |
| shaf | **43.3x** | 0.78x | 0.003 |

Those marginal cells carry almost no direct weight (0.003) so they cost little bias score
themselves — but they inflate the regional total (nhaf 1.21x, shaf 1.45x) and, because the
spread term is mean-normalised, they steal burning from the hotspots.

Optuna's joint search (Entry 30) had left `spread_k` at 2.70. Probing it directly:

| spread_k | overall | rmse | spatial | annual_pct |
| -------- | ------- | ---- | ------- | ---------- |
| 2.70 (tuned) | 0.7165 | 0.5489 | 0.8678 | 1.799 |
| 4.0 | 0.7177 | 0.5506 | 0.8700 | 1.745 |
| 4.5 | 0.7179 | 0.5507 | 0.8706 | 1.726 |
| **5.2** | **0.7180** | **0.5507** | **0.8712** | 1.702 |
| 5.5 | 0.7180 | 0.5506 | 0.8714 | 1.693 |
| 6.5 | 0.7178 | 0.5501 | 0.8714 | 1.664 |

Flat optimum 4.0-6.5; took 5.2. Committed `e558f1c`, `ar evaluate` returned **official 0.718**
(bias 0.759, rmse 0.551, seasonal 0.858, spatial 0.871). Proxy said 0.7180 — **fifth**
consecutive three-decimal match.

**Second time Optuna's joint optimum was beatable by hand on a single coefficient**
(first was `lag_w`, Entry 30). With ~9 coupled coefficients and early stopping at 50 stale
trials, the search is leaving single-coefficient gains on the table. Practice going forward:
after any joint retune, probe the individual coefficients around the returned point before
accepting it.

Progression: 0.686 -> 0.698 -> 0.701 -> 0.715 -> 0.717 -> **0.718**.

### Entry 35: Coordinate sweep — the point is converged

Acting on Entry 34's lesson, wrote `scratchpad/coord_sweep.py` to probe every searchable
coefficient one at a time around the current point (7 candidates each, +-40-250% for log
params, +-18% of range for linear).

base 0.7180. Only two coefficients moved at all:
  `cure_alpha` 0.9459 -> **1.0000** (+0.0008)
  `lag_w` 0.2000 -> **0.1400** (+0.0002)
  all seven others: +0.0000

Applied both: **0.7182**. The optimum is now essentially converged — no single-coefficient
gain above +0.001 remains, unlike the +0.0015 that `spread_k` was hiding. Not worth an eval
for +0.0002 over the recorded 0.718; carrying it into the next structural change instead.

**Honest note on `cure_alpha` -> 1.0.** alpha=1 means the EMA keeps *no* memory: the
antecedent-wetness accumulator has collapsed to the instantaneous month. So the mechanism that
won Entry 5 is no longer doing what I described there — what survives is the **moisture
response curve** `1/(1+(P/cure_half)^cure_n)` and its mean-normalisation, not the memory.
Curing is still emphatically load-bearing (dropping it costs **-0.051**, and amp_ratio
collapses 0.566 -> 0.168), but for a different reason than originally claimed. The transform
fix and the lag appear to have absorbed the role the memory used to play.

Started `ar ablate` on the current 7-component model (128 subsets) — three components have
been added since the last pruning round at Entry 22.

### Entry 36: PRUNING ROUND 2 (7 components, 128 subsets) — nothing prunes

`ar ablate` on `e558f1c`, exact Shapley computed from all 128 subsets (sum reconciles to
+0.3110 = full 0.718 - empty 0.407).

| component | shapley | drop-one | vs Entry 22 (drop-one) |
| --------- | ------- | -------- | ---------------------- |
| precipitation | +0.0801 | +0.0170 | +0.0260 |
| dryness | +0.0621 | +0.0140 | +0.0230 |
| temperature | +0.0610 | +0.0190 | +0.0270 |
| fuel | +0.0534 | +0.0040 | +0.0110 |
| **curing** | +0.0443 | **+0.0510** | +0.0110 |
| **spread** | +0.0085 | **+0.0310** | +0.0050 |
| lag | +0.0015 | +0.0020 | (new) |

All positive on both measures — **nothing prunes**. Empty baseline rose 0.399 -> 0.407.

**The structure changed materially after the transform fix.** `curing` drop-one went
+0.011 -> **+0.051** and `spread` +0.005 -> **+0.031** (6x). Both now have drop-one *exceeding*
their Shapley, meaning they are strongly complementary rather than redundant. Removing the
8.33% ceiling let the two cycle-shaping terms finally do their job — before the fix they were
fighting a cap, which is exactly why they scored as marginal in the first pruning round.
`lag` is the weakest (+0.0015) but positive on both measures and it was worth an official
+0.002, so it stays.

### Entry 37: three more refutations — exp_boost, contrast, and the peak-saturation theory

(a) **`exp_boost`** (global multiplier on the seven frozen regional `fire_exp`, which were all
fitted against the broken transform): 1.0 -> 0.7182, 1.3 -> 0.7184, 1.6 -> 0.7185,
2.0 -> 0.7186. Flat +0.0004 and still creeping at the range edge — a non-mechanism.
`fire_exp` is not the constraint. Reverted.

(b) **`contrast`** (sharpen the mean rate *between cells*, mirroring what `spread` does between
months): 1.0 -> **0.7182**, 1.2 -> 0.7089, 1.5 -> 0.6850, 1.9 -> 0.6734. Monotonically worse,
refuted. Diagnostic value: amp_ratio *rises* 0.566 -> 0.821 while spatial collapses
0.871 -> 0.759, i.e. global contrast sharpening helps Africa and wrecks everywhere else.

(c) **Corrected a wrong assumption of my own.** I expected peaks outside Africa to be short
because the rate was saturating the transform. Measured the scaled peak-month rate directly:

    nhaf p90=0.29  shaf p90=0.29  seas p90=0.16  aust p90=0.05  boas p90=0.02

All **far** below saturation (needs ~3). The transform now runs in its near-linear regime and
there is no ceiling left anywhere. So the short regional peaks are a **spatial averaging**
effect, not a clipping effect: individual cells do reach ~25%, but too few peak together, so
the region mean stays at 0.9%.

`bias_probe.py` confirms the model is nearly flat across burn strata outside Africa —
aust: obs 0.009% -> mod 0.067% (7.7x over) and obs 5.5% -> mod 0.65% (**0.12x under**);
ceas >5% cells underpredicted **20x**. The spatial dynamic range is essentially absent there.
But (b) shows a global fix for it is not available: what Africa needs and what the rest needs
point opposite ways, and the weighting means Africa wins any global compromise.

**This is the same wall as Entries 27-29 and 33, reached from a third direction.** The residual
error is genuinely regional in character, and every global mechanism trades regions against
each other. Recording that as the thread's central open problem rather than trying a fourth
variant of it.

### Entry 38: per-region multipliers REFUTED; Optuna underperforms a third time

`ar optuna` over 16 coefficients (7 new per-region rate multipliers on the EXISTING regions,
which is recalibration not region-drawing): **0.711** vs hand point 0.7182. Reverted.

Hand probes were already negative in *both* directions (k=2.0 and k=0.6 each cost ~0.002 for
Australia/SEAsia/Boreal), so the regional magnitudes are already near-optimal. Confirms the
error is the *within-region* spatial distribution, which no per-region scalar can fix.

**Third time the joint search underperformed a hand-refined point** (after lag_w, spread_k).
At 16 coefficients with early stopping at 50 stale trials it is clearly over-dimensioned.
Treat `ar optuna` as a coarse explorer and always coordinate-sweep afterwards.

### Entry 39: Experiment 13 OFFICIAL — limiting-factor form, 0.719 (new best)

Changed family as planned rather than trying a fifth variant of the regional problem.
Everything so far modulated a **multiplicative product of favourability**. That form means any
single weak factor vetoes fire: a merely damp month in a well-fuelled landscape is suppressed
as hard as a month with no fuel at all. Ecological limitation is closer to Liebig's law —
the scarcest requirement sets the rate.

Replaced the product with a blend toward a smooth minimum over the four climate factors
(dryness, precipitation, fuel, temperature), `soft_w` blending and `soft_s` sharpness.

| soft_w | soft_s | overall | rmse | spatial | annual_pct |
| ------ | ------ | ------- | ---- | ------- | ---------- |
| 0.0 | - | 0.7182 | 0.5517 | 0.8712 | 1.720 |
| 0.6 | 4.0 | 0.7188 | 0.5520 | 0.8727 | 1.702 |
| 0.8 | 2.0 | 0.7192 | 0.5532 | 0.8730 | 1.668 |
| **1.0** | **2.0** | **0.7194** | **0.5534** | **0.8730** | **1.665** |
| 1.0 | 1.0 | 0.7183 | 0.5534 | 0.8727 | 1.654 |

Optimum is **full replacement** of the product (soft_w = 1.0) with a soft sharpness ~2.0
(interior: 1.5 and 4.0 are both worse). Committed `b0203b9`; `ar evaluate` returned
**official 0.719** (bias 0.760, rmse 0.553, seasonal 0.857, spatial 0.873). **Sixth**
consecutive three-decimal proxy match.

Regionally it improves only 4/14, but the magnitudes are lopsided in its favour:
**mide +0.0971** — by far the largest single-region gain of the thread, in a region stuck near
0.52 since Model G — plus ceam +0.0143, against ceas -0.0323 as the main loss. Physically
right: arid regions are exactly where one factor (fuel or moisture) is scarce while others are
abundant, so the product buried them and the limiting-factor form does not.

**Process note.** The first softmin patch silently no-oped: I had reverted to a file lacking
the `exp_boost` anchor line that three `str.replace` calls targeted, so they matched nothing
and the run returned byte-identical numbers across six settings. Same false-negative signature
as Entry 29. Fixed by asserting every anchor is present before writing. **All future patches
assert their anchors** — a silent no-op reads exactly like a refuted hypothesis.

Progression: 0.686 -> 0.698 -> 0.701 -> 0.715 -> 0.717 -> 0.718 -> **0.719**.

### Entry 40: Experiment 14 OFFICIAL — coordinate-swept lag, 0.720 (new best)

Applied Entry 38's practice immediately: coordinate-swept every coefficient after the
limiting-factor change rather than trusting the point.

Only two moved, and only one materially: `lag_w` 0.14 -> **0.2133** (+0.0006),
`cure_alpha` 1.0 -> 0.9715 (+0.0002). The other nine were flat at +0.0000 — the softmin change
is essentially **orthogonal** to the rest of the coefficient set, which is a good sign it is a
real structural improvement rather than a reparameterisation of existing freedom.

Took `lag_w` = 0.2133 -> proxy **0.7199**. Committed `ec98459`; `ar evaluate` returned
**official 0.720** (bias 0.760, rmse 0.554, seasonal 0.860, spatial 0.873). **Seventh**
consecutive three-decimal proxy match.

Component A/B at the new point (drop-one, proxy), all still clearly earning their place:

| drop | overall | cost |
| ---- | ------- | ---- |
| curing | 0.6668 | **-0.053** |
| spread | 0.6875 | **-0.032** |
| precipitation | 0.7037 | -0.016 |
| temperature | 0.7046 | -0.015 |
| dryness | 0.7078 | -0.012 |
| lag | 0.7176 | -0.002 |
| fuel | 0.7162 | -0.004 |

Progression: 0.686 -> 0.698 -> 0.701 -> 0.715 -> 0.717 -> 0.718 -> 0.719 -> **0.720**
(+0.034 over the Model G baseline, 8 official evals, 7 experiments refuted and reverted).

### Entry 41: Experiment 15 OFFICIAL — cropland suppression, 0.721 (new best). LUH2 now in use.

User asked whether LUH2 was being used. It was not — the 0.720 model used five purely
biophysical inputs. Entry 12 had screened all six LUH2 fractions and set them aside, but that
screen was (a) against an *unweighted* residual, which Entry 18 showed is the wrong test, and
(b) run before the transform fix and softmin. Re-screened against the **weighted** residual on
the current model:

| predictor | wr(resid) | wr(obs) |
| --------- | --------- | ------- |
| luh2_cropland_fraction | **+0.120** | **-0.268** |
| luh2_rangeland_fraction | -0.195 | -0.131 |
| luh2_urban_fraction | +0.081 | -0.169 |
| population_density | +0.064 | -0.140 |
| lightning_flash_rate | -0.029 | +0.473 |

Cropland shows the informative pattern: **opposite signs**. It correlates -0.27 with observed
burning but +0.12 with the model's error — cropland burns less than its surroundings and the
biophysical terms cannot see why. Physical mechanism is fuel discontinuity: fields are
ploughed, grazed and cut by roads and boundaries, so fuel is removed before the fire season
and what remains cannot carry a front.

| crop_k | crop_n | overall | bias | rmse | spatial |
| ------ | ------ | ------- | ---- | ---- | ------- |
| 0.0 | - | 0.7199 | 0.7596 | 0.5535 | 0.8730 |
| 0.8 | 1.0 | 0.7195 | 0.7610 | 0.5552 | 0.8657 |
| 2.0 | 1.0 | 0.7111 | 0.7550 | 0.5517 | 0.8367 |
| 0.3 | 1.0 | 0.7210 | 0.7616 | 0.5551 | 0.8732 |
| **0.5** | **2.0** | **0.7211** | **0.7613** | **0.5549** | **0.8743** |

Linear damping (crop_n=1) improved bias and rmse but cost spatial — too blunt, it strips
burning from *all* cropland. The **quadratic** form suppresses only heavily cropped cells and
improves all three weighted metrics at once. Committed `dfa3601`; `ar evaluate` returned
**official 0.721**. **Eighth** consecutive three-decimal proxy match. First human land-use
term in the model.

### Entry 42: rangeland, population, lightning all REFUTED (tested individually)

| term | 0.3 | 1.0 | verdict |
| ---- | --- | --- | ------- |
| baseline | 0.7211 | | |
| range_k | 0.7208 | 0.7145 | monotonically worse |
| pop_k | 0.7161 | 0.6892 | monotonically worse, steeply |
| lit_k | 0.7212 | 0.7210 | flat — no mechanism |

Lightning probed across `lit_k` x `lit_half`: **0.7210-0.7212 everywhere**, no interior
optimum. This reconfirms Entry 12 under the correct weighting: lightning is the best single
predictor of *observed* fire (wr(obs) = +0.473) yet explains none of the **residual**
(wr(resid) = -0.029) — the climate terms already encode where lightning-driven fire happens.

Rangeland's -0.195 residual correlation did not convert either. **Fifth time** an aggregate
correlation failed on cell-level application. Reverted to `dfa3601`.

**Process note.** A multi-term patch applied fully just before a later command was rejected,
leaving all three terms wired but defaulted off — so the baseline still read exactly 0.7211
and nothing was corrupted. Separately, the assertion guard from Entry 39 fired correctly on a
mis-anchored patch, preventing a silent no-op from being read as a refutation. Both guards
earned their keep this round.

### Entry 43: autonomous check — state at 0.721

Switched the loop to the autonomous default (job `985299ea`, 20m); cancelled the redundant
30m job `9d62ca38` so the two do not overlap.

`weighted_residual.py` at 0.721: total recoverable **1.202 -> 1.129** since 0.701.
Still 79% rmse, still 61% in nhaf+shaf (0.363 + 0.330).

**The softmin fix is visible in the diagnostics**: mide mod/obs **5.32 -> 1.61** and
bias_score 0.606 -> 0.663, which is exactly the +0.097 regional gain Entry 39 predicted.
eqas also improved 2.07 -> 1.91.

Checked whether the cropland term over-suppressed `ceas` (mod/obs 0.69 -> 0.39). It did not:
burned-area-weighted cropland fraction gives a regional suppression factor of only 0.947 in
ceas (0.932 euro, 0.999 shaf), so the term is mild at regional scale and its effect is
correctly concentrated in individual high-cropland cells. ceas underprediction is
pre-existing, not caused by this change.

Remaining error is unchanged in character: nhaf/shaf overpredicted (1.12, 1.39) while nearly
everything else is underpredicted (ceas 0.39, aust 0.56, ceam 0.62, nhsa 0.70). This is the
same wall recorded in Entry 37 — global mechanisms trade Africa against the rest, and the
weighting makes Africa win.

Ablation on the 9-component model (512 subsets) still running.

### Entry 44: CONTRACT BUG found by ablation — softmin was never gated

The 512-subset ablation on `dfa3601` returned:

| component | shapley | drop-one |
| --------- | ------- | -------- |
| precipitation | +0.0803 | +0.0140 |
| dryness | +0.0607 | +0.0110 |
| temperature | +0.0586 | +0.0130 |
| fuel | +0.0522 | +0.0010 |
| curing | +0.0466 | +0.0560 |
| spread | +0.0102 | +0.0350 |
| cropland | +0.0036 | +0.0010 |
| lag | +0.0017 | +0.0020 |
| **softmin** | **+0.0000** | **+0.0000** |

softmin reporting *exactly* zero was the tell — it had just earned +0.0012 officially
(0.7182 -> 0.719). Cause: the block was gated only on `weight > 0.0`, never on
`"softmin" in enabled`. So `ar ablate` could not switch it off; it stayed on in all 512
subsets and measured as contributing nothing while actually being always active.

**This violated the model contract** ("otherwise use exactly the requested subset"), which
`ar evaluate` does not check — only ablation exposed it. Fixed in `3075d24`. Predictions with
all components enabled are byte-identical, so no recorded score is affected; the A/B now
reads correctly: **0.7211 with softmin vs 0.7202 without (+0.0009)**.

Audited all nine components for the same pattern: softmin was the only offender (fuel has two
gates, both correct). Wrote `scratchpad/contract_check.py`, which asserts shape/finiteness/
range and that every declared component changes the prediction when dropped. Result:

    drop dryness 2.02e-01 | precipitation 3.02e-01 | fuel 3.48e-01 | temperature 4.84e-01
    curing 3.86e-01 | spread 3.48e-01 | lag 1.42e-01 | softmin 3.86e-01 | cropland 1.21e-01
    CONTRACT OK

**Run `contract_check.py` after adding any component.** A component that is declared but not
gated is invisible to ablation and silently un-prunable — the exact failure that would have
made a pruning round delete a working term or keep a dead one.

Re-running the ablation on `3075d24` since the previous Shapley table was computed with
softmin stuck on and is therefore invalid for every subset.

### Entry 45: autonomous ticks — contract fix verified score-neutral

Verified the `3075d24` gating fix against the evaluated model directly: predicted `dfa3601`'s
model.py from git and diffed the arrays. **max|delta| = 0.000e+00** — byte-identical, so the
recorded official 0.721 stands and needs no re-eval. Closes the one open question the fix
raised.

Drop-one at the current point (proxy), for the two components whose last Shapley looked
weakest:

| drop | overall | cost | note |
| ---- | ------- | ---- | ---- |
| full | 0.7211 | | |
| fuel | 0.7197 | -0.0014 | annual_pct 1.61 -> **2.29** (obs 1.55) |
| cropland | 0.7199 | -0.0012 | |

Both earn their place. `fuel`'s Overall drop-one is small but it is the main constraint on
*total burning* — without it the model overshoots the global annual total by 48%. A component
can be near-neutral on Overall while still being structurally load-bearing; drop-one on the
composite score alone would have under-rated it.

Process note: briefly concluded the 512-subset ablation had died because its line count sat at
350 for three ticks and `ps | grep -c` returned a small number. Wrong on both counts — the
count was matching the shell wrapper, and the real process (PID 48493) was at 99.4% CPU with
8:54 accumulated. Output was simply buffered. **Check actual process state, not line-count
deltas, before restarting a long job** — a needless restart would have cost ~15 minutes.

### Entry 46: PRUNING ROUND 3 (9 components, 512 subsets) — nothing prunes, softmin validated

`ar ablate` on `3075d24` completed. Exact Shapley from all 512 subsets, sum reconciles to
+0.3140 (= full 0.721 - empty 0.407).

| component | shapley | drop-one | vs Entry 44 (pre-fix) |
| --------- | ------- | -------- | --------------------- |
| precipitation | +0.0794 | +0.0140 | +0.0803 |
| dryness | +0.0605 | +0.0110 | +0.0607 |
| temperature | +0.0587 | +0.0130 | +0.0586 |
| fuel | +0.0522 | +0.0010 | +0.0522 |
| curing | +0.0464 | +0.0560 | +0.0466 |
| spread | +0.0100 | +0.0350 | +0.0102 |
| cropland | +0.0037 | +0.0010 | +0.0036 |
| lag | +0.0017 | +0.0020 | +0.0017 |
| **softmin** | **+0.0014** | **+0.0010** | **+0.0000 (bug)** |

**The contract fix is validated**: softmin now measures a real contribution instead of the
false exact-zero that exposed the bug. Every other component is unchanged to within +-0.001,
confirming the fix was score-neutral and did not perturb the decomposition.

**All nine positive on both measures — nothing prunes.** Structure unchanged since Entry 36:
`curing` remains the workhorse (drop-one +0.056 > its Shapley), `spread` strongly complementary
(+0.035 drop-one vs +0.010 Shapley). The three newest terms (cropland, lag, softmin) are the
three smallest, but each was worth an official +0.001 to +0.002 and each is positive here.

Deleted the `fast_shapley.py` proxy script I had started writing: it would have computed the
same 512 subsets and hit the identical cost, so it added nothing over waiting for the
authoritative CLI run.

**Timing note for future long jobs.** This ablation took ~3h wall for ~12min CPU because the
machine suspends between autonomous ticks — background work only advances while the session is
active. Long CLI jobs should be started when there is foreground activity to carry them, and
progress judged by CPU time (`ps -o time`), never by output line count (stdout is block-
buffered) or wall-clock elapsed.

### Entry 47: Experiment 16 — global spread normalisation REFUTED

Hypothesis: `_spread` normalises by each cell's **own** time-mean, which preserves every
cell's annual total and therefore forbids the term from moving burning *between* cells. That
constraint was necessary when the transform capped output at 8.33% (totals could run away),
but the ceiling is gone and `month_scale` now controls magnitude globally. Blending toward a
single global normaliser should let connectivity concentrate burning in landscapes that
actually carry fire, while still fixing the world total.

| spread_glob | overall | bias | rmse | spatial | amp_ratio | annual_pct |
| ----------- | ------- | ---- | ---- | ------- | --------- | ---------- |
| 0.0 (per-cell) | **0.7211** | 0.7613 | 0.5549 | **0.8743** | 0.545 | 1.606 |
| 0.2 | 0.7209 | 0.7603 | 0.5548 | 0.8747 | 0.604 | 1.723 |
| 0.5 | 0.7081 | 0.7407 | 0.5486 | 0.8426 | 0.705 | 1.924 |
| 0.8 | 0.6811 | 0.7038 | 0.5333 | 0.7752 | **0.823** | 2.156 |

Monotonically worse. **Refuted**; reverted to `3075d24`.

Diagnostically this is the most informative refutation yet, because it is now the **fourth**
mechanism with an identical signature: cross-cell contrast sharpening drives amp_ratio toward
its 1.0 target (0.545 -> 0.823 here) while collapsing spatial. The same trade appeared in
`sharpen` (Entry 20), `contrast` (Entry 37b), and the temperature/LAI patch-scale terms
(Entries 28-29).

Four independent formulations producing the same trade is no longer a coincidence — it is
evidence that **seasonal amplitude and the spatial mean map are in genuine tension in this
model**, and that no reweighting of a single global field escapes it. Anything that
concentrates burning into high-burning cells improves the cycle and degrades the 16-year mean
map, because the mean map is what the four instantaneous climate terms were fitted to draw.

Recording this as the settled characterisation of the wall rather than testing a fifth variant.

### Entry 48: Experiment 17 — output-space mean preservation REFUTED; the tension is real

Followed Entry 47's characterisation to its sharpest test. Measured what `spread` actually
does to each cell's **output** mean (not its rate):

    per-cell output mean ratio, spread on/off: p10=1.000 p50=1.000 p90=**1.147**
    global mean ratio: **1.2201**

So despite normalising the *rate* per cell, spread raises the global burned mean by **22%**,
and by up to 15% in the top decile of cells. Cause: `_transform` is convex, so sharpening a
cycle in time raises its mean output. The leak lands precisely in the highest-burning cells.

That looked like the mechanism behind the whole wall — contrast terms "paying" for cycle shape
with spatial-map distortion. Tested it by restoring each cell's pre-spread output mean
(`out_fix` blending toward exact preservation):

| out_fix | overall | bias | rmse | spatial | amp_ratio | annual_pct |
| ------- | ------- | ---- | ---- | ------- | --------- | ---------- |
| 0.0 (leak intact) | **0.7211** | 0.7613 | 0.5549 | **0.8743** | 0.545 | 1.606 |
| 0.5 | 0.7129 | 0.7527 | 0.5517 | 0.8482 | 0.471 | 1.454 |
| 1.0 (exact) | 0.6952 | 0.7382 | 0.5452 | 0.7875 | 0.396 | 1.303 |

**Refuted, and informatively so: spatial gets WORSE too** (0.874 -> 0.788), not just amplitude.

So the 22% mean leak is **load-bearing, not a defect**. The model needs that extra burning
concentrated in high-burning cells — it is part of how the spatial map is drawn, not a
distortion of it. This rules out the "misplaced normalisation" explanation for the wall and
confirms Entry 47's reading: seasonal amplitude and the spatial mean map are in **genuine
physical tension** in this formulation, not an artefact of where a normaliser sits.

Fifth mechanism in this family refuted. The wall is now well characterised from both
directions: adding cross-cell contrast helps amplitude and hurts spatial; removing the
existing implicit contrast hurts both. The current point sits at a real optimum along that
trade. Reverted to `3075d24`.

### Entry 49: Experiment 18 — NEIGHBOUR COUPLING WORKS (proxy 0.7252, not yet evaluated)

Changed families as planned after five refutations in the contrast family. Observation: every
term in the model is a **pointwise** function of the same cell's own inputs, yet fire crosses
cell boundaries. A cell adjacent to burning savanna is reached by fronts started elsewhere; an
isolated flammable cell ringed by wet forest is not. Nothing in the model represented that.

Added `neighbour`: blend each cell's assembled rate with the mean of its four neighbours,
applied before `spread` so connectivity acts on the coupled field. Polar rows damped back to
self so the latitude roll does not wrap across the pole.

| nb_w | overall | bias | rmse | seasonal | spatial | annual_pct |
| ---- | ------- | ---- | ---- | -------- | ------- | ---------- |
| 0.00 | 0.7211 | 0.7613 | 0.5549 | 0.8600 | 0.8743 | 1.606 |
| 0.15 | 0.7240 | 0.7628 | 0.5560 | 0.8695 | 0.8756 | 1.590 |
| 0.35 | 0.7250 | 0.7641 | 0.5569 | 0.8713 | **0.8759** | 1.573 |
| 0.50 | **0.7252** | **0.7647** | **0.5573** | **0.8717** | 0.8752 | 1.563 |

**+0.004 over the committed point — the largest gain since the transform fix, and every metric
improves at once** (bias, rmse, seasonal, spatial). That simultaneity is the signature of
relieving a real constraint rather than trading one metric for another, exactly as the
transform fix behaved.

Notably it does NOT follow the amplitude/spatial trade of the five refuted contrast mechanisms:
amp_ratio actually *falls* slightly (0.545 -> 0.518) while spatial *rises*. So spatial coupling
is a genuinely different axis from contrast sharpening — it escapes the Entry 47/48 wall
instead of fighting it, which is why changing families was the right call.

Optimum is flat between 0.35 and 0.5 and may extend higher; the 0.7/0.9 probes were cut off by
the tick's wall-clock budget (the machine suspends between autonomous ticks). Next: finish the
sweep, then Optuna, commit, and one `ar evaluate`. **Not yet committed or evaluated — proxy
only.**

### Entry 50: neighbour sweep completed — interior optimum at nb_w = 0.5

| nb_w | overall | bias | rmse | seasonal | spatial |
| ---- | ------- | ---- | ---- | -------- | ------- |
| 0.35 | 0.7250 | 0.7641 | 0.5569 | 0.8713 | **0.8759** |
| **0.50** | **0.7252** | 0.7647 | 0.5573 | 0.8717 | 0.8752 |
| 0.70 | 0.7251 | **0.7648** | **0.5574** | **0.8725** | 0.8736 |
| 0.90 | 0.7241 | 0.7643 | 0.5570 | 0.8708 | 0.8716 |

Genuine interior optimum, very flat 0.35-0.7 then falling. Set `nb_w = 0.5` (proxy 0.7252).
`contract_check.py` passes with the new component: drop neighbour -> max|delta| 2.96e-01, so it
is properly gated and visible to ablation (the check written after the Entry 44 softmin bug is
now earning its keep on every new component).

Started `ar optuna` **detached via nohup** so it survives the autonomous tick boundary. Earlier
this session two long jobs were killed by the per-tick wall-clock cutoff; nohup + redirect to a
file is the right pattern for anything that outlives a tick.

### Entry 51: Optuna abandoned on the neighbour model — fourth underperformance

`ar optuna` (nohup, survived two tick boundaries as intended) reached only **trial 26/500 with
best 0.696** after 22 minutes — well below the verified hand point of **0.7252**. Killed it to
free cores for the coordinate sweep.

**Fourth time the joint search has trailed a hand-refined point** (after lag_w, spread_k, and
the per-region multipliers). The pattern is consistent and now well evidenced: with 12 coupled
coefficients and early stopping after 50 stale trials, Optuna explores too broad a space to
refine a good point. Its useful role in this thread has been *coarse exploration when a new
mechanism is added with unknown scale* (it found the transform-fix optimum well), not
refinement.

**Settled practice: after adding a mechanism, hand-sweep the new coefficient, then
coordinate-sweep all coefficients. Use `ar optuna` only when the new coefficient's order of
magnitude is genuinely unknown.** All coefficients installed this way are still CLI-derived or
hand-verified on the proxy and confirmed by `ar evaluate`, so the discipline is unchanged --
only the search strategy is.

Coordinate sweep running on the nb_w = 0.5 point.

### Entry 52: neighbour weight chosen for regional spread, not peak global

Checked the regional distribution of the neighbour gain before committing. At nb_w = 0.5 the
+0.004 global gain is carried almost entirely by **eqas +0.0710**, with ten regions regressing
(ceam -0.021, nhsa -0.018, seas -0.013). eqas is the region that lost most from the transform
fix (Entry 26), and spatial smoothing suits its peat/coastal geography, where GFED5 cells are
fragmented and a pointwise model cannot see the surrounding landscape.

Global vs spread trade-off:

| nb_w | global | regions improved |
| ---- | ------ | ---------------- |
| 0.15 | 0.7240 | **7/14** |
| **0.25** | **0.7247** | **7/14** |
| 0.35 | 0.7250 | 6/14 |
| 0.50 | 0.7252 | 4/14 |

**Chose nb_w = 0.25**: gives up 0.0005 of global Overall to improve three more regions.
research.md sets a hard goal of maximising Overall with a *soft requirement* that gains be
spread across regions rather than concentrated in one core region, and at nb_w = 0.5 the gain
is concentrated in a single region by exactly the pattern that requirement warns against.
0.7247 still clears the committed 0.7211 comfortably.

This is the first time in the thread the two objectives have pointed to different coefficient
values, so recording the reasoning explicitly rather than silently taking the larger number.

### Entry 53: Experiment 18 OFFICIAL — neighbour coupling, 0.725 (new best)

Committed `9534c76`; `ar evaluate` returned **official Overall 0.725** (bias 0.764, rmse 0.557,
seasonal 0.871, spatial 0.876). Proxy said 0.7247 — **ninth** consecutive three-decimal match.

**Largest single gain since the transform fix (+0.004), and all four metrics improved at
once** — the signature of relieving a real constraint rather than trading metrics.

Progression: 0.686 -> 0.698 -> 0.701 -> 0.715 -> 0.717 -> 0.718 -> 0.719 -> 0.720 -> 0.721
-> **0.725**. Total +0.039 over the Model G baseline across 10 official evals.

**Why this worked where five contrast mechanisms failed.** Entries 47-48 established that
seasonal amplitude and the spatial mean map are in genuine tension: anything concentrating
burning into high-burning cells improves the cycle and degrades the mean map. Neighbour
coupling is on a **different axis entirely** — it does not sharpen contrast, it *shares*
flammability between adjacent cells. amp_ratio actually falls slightly (0.545 -> 0.530) while
spatial rises, the opposite of the refuted family's signature.

The lesson generalises: after five refutations sharing one signature, the productive move was
not a sixth variant but identifying what all five had in common (pointwise contrast) and
attacking the assumption underneath it (that the model is pointwise at all). Every term before
this one was a function of a single cell's own inputs.

Also cancelled the stale coordinate sweep mid-flight: `nb_w` was changed 0.5 -> 0.25 after it
started, so it was refining a point that no longer existed. Exit 144 in the task notification
was that pkill, not a crash.

### Entry 54: Experiment 19 — eight-neighbour kernel (proxy 0.7256, eval running)

Extended the new spatial-coupling family rather than ablating first (a 10-component ablation is
1024 subsets, ~6h under tick suspension; kernel refinement is the higher-value move).

Added `nb_diag`, blending corner cells into the surround term:

| nb_diag | overall | seasonal | spatial |
| ------- | ------- | -------- | ------- |
| 0.0 (4-neighbour) | 0.7247 | 0.8712 | 0.8759 |
| 0.3 | 0.7252 | 0.8732 | 0.8758 |
| **0.5** | **0.7253** | **0.8734** | 0.8756 |
| 0.7 | 0.7252 | - | - |

Gain is concentrated in **seasonal** (0.8712 -> 0.8734). Flat optimum; took 0.5, which is equal
weight on edges and corners — the natural kernel rather than a fitted value.

The wider kernel then supported a stronger coupling weight: re-swept `nb_w` and found
0.35 gives **0.7256** (vs 0.7253 at 0.25). Checked the regional spread that governed the
Entry 52 choice — **flat at 6/14 for every nb_w in 0.25-0.35 and every nb_diag in 0.3-0.7**, so
unlike the four-neighbour weight there is no spread-versus-global tradeoff here and the global
optimum is uncontested. Took nb_w = 0.35.

`contract_check.py` passes. Committed `20fdd19`, `ar evaluate` running detached.
Proxy 0.7256 vs official-best 0.725.

### Entry 55: Experiment 19 OFFICIAL — eight-neighbour kernel, 0.726 (new best)

`ar evaluate` on `20fdd19`: **official Overall 0.726** (bias 0.764, rmse 0.557, seasonal 0.874,
spatial 0.875). Proxy said 0.7256 — **tenth** consecutive three-decimal match. Seasonal moved
0.871 -> 0.874 exactly as the kernel change predicted.

Progression: 0.686 -> ... -> 0.721 -> 0.725 -> **0.726** (11 official evals, +0.040 total).

The spatial-coupling family has now produced two consecutive gains after five straight
refutations in the contrast family. Continuing to mine it.

### Entry 56: Experiment 20 — directional (flammability-weighted) coupling REFUTED both ways

The coupling is isotropic; real fire runs faster into drier, better-cured neighbours. Tested
weighting the surround toward its extreme members instead of their plain average.

| nb_up | meaning | overall | spatial |
| ----- | ------- | ------- | ------- |
| -1.5 | toward weakest neighbour | 0.6864 | 0.7457 |
| -0.5 | " | 0.7100 | 0.8246 |
| **0.0** | **plain average** | **0.7256** | **0.8752** |
| +0.5 | toward strongest neighbour | 0.7136 | 0.8391 |
| +1.5 | " | 0.6971 | 0.7884 |

**Refuted in both directions**, symmetrically, with spatial collapsing either way. The plain
mean sits at a sharp optimum.

Diagnostic reading: max-weighting concentrates burning into already-hot cells, which is the
**contrast-sharpening signature** of the five refuted mechanisms in Entries 20/28/29/37/47 —
spatial collapses while the cycle barely moves. Min-weighting does the mirror-image damage.
So neighbour coupling helps precisely *because* it averages, not because it transports fire
directionally: its value is smoothing a too-noisy flammability field, not modelling front
propagation.

That also bounds the family honestly. The obvious physical elaborations of spatial coupling —
directionality, anisotropy — are unavailable, because anything that makes the exchange
non-uniform reintroduces contrast sharpening. What remains testable is the *scale* of the
smoothing (kernel radius), not its shape or direction.

Reverted to `20fdd19`.

### Entry 57: Experiment 21 — wider smoothing radius REFUTED; spatial-coupling family exhausted

Tested the last remaining dimension identified in Entry 56 — the *scale* of the smoothing —
by blending in a second ring two cells out:

| nb_wide | overall | seasonal | spatial |
| ------- | ------- | -------- | ------- |
| **0.0 (one-cell)** | **0.7256** | 0.8741 | **0.8752** |
| 0.25 | 0.7254 | 0.8740 | 0.8743 |
| 0.50 | 0.7253 | 0.8745 | 0.8732 |

Slightly worse and nearly flat; the one-cell radius is already optimal. Did not probe 0.75/1.0
— the trend is monotone and the effect size (0.0003 over half the range) is far below anything
worth an eval.

**The spatial-coupling family is now fully characterised across all three of its degrees of
freedom:**
- *shape*: 8-neighbour beats 4-neighbour (Entry 54, +0.001 official) — corners help
- *direction*: any deviation from the plain mean fails symmetrically (Entry 56) — it smooths,
  it does not transport
- *scale*: one cell optimal, wider is worse (this entry)

Two official gains (0.721 -> 0.725 -> 0.726), then three clean refutations bounding the family
on every side. That is a genuinely closed line rather than an abandoned one, and worth
recording as such: the model now says fire flammability is a *locally smoothed* field, with
one-cell isotropic averaging at weight 0.35.

### Entry 58: bona diagnosed — the model produces essentially zero boreal fire

Re-ran `weighted_residual.py` at 0.726: recoverable 1.129 -> **1.121**, still 79% rmse, still
61% nhaf+shaf. One number is extreme: **bona mod/obs = 0.06**.

    bona obs  0.007 0.005 0.008 0.195 0.628 0.404 0.459 0.239 0.235 0.408 0.080 0.006
    bona mod  0.000 0.000 0.000 0.000 0.000 0.009 0.034 0.030 0.008 0.000 0.000 0.000

The model is not underpredicting boreal North America, it is **not firing at all** — exactly
0.000 for seven months, peak 0.034% against an observed 0.628%.

Cause is visible in the driver ranges. bona `air_temperature` peaks at **16.7 C** while the
global ignition centre `ign_c` is **~20.0**, so the rising logistic on temperature never opens;
and bona `dryness` sits at **792-1006** against a global `D_low` of 70, far up the saturated
tail. The multiplicative product therefore collapses to zero. The N.America regional parameter
set covers bona but is fitted mostly on temperate US fire (the same mismatch noted in the
retracted Entry 15), so its `ign_c` is tuned for a much warmer regime.

**Value assessment before investing: bona holds 0.49% of global burned-area weight.** Taking
its regional Overall from 0.6055 to a perfect 1.0 would be worth roughly **+0.002** on global
Overall — comparable to the last two experiments, but not the large win the 16x error suggests
at first glance. Recording this explicitly because "worst ratio in the table" is a misleading
signal under reference-burned-area weighting, and Entries 19/27/33 already show this thread
losing time to low-weight regions.

Worth one bounded experiment, not a campaign: a *global* mechanism that lets cold regimes
ignite (fire in short cold seasons is driven by lightning into dry duff, not by air temperature
crossing a warm threshold), tested on its global score like everything else. If it does not
also help elsewhere, drop it.

### Entry 59: Experiment 22 — cold-regime ignition floor blocked; found a REAL structural finding instead

Tried a global `ign_floor` giving the temperature logistic a floor so cold regimes stay
ignitable (bona's warmest month is 16.7 C against `ign_c` ~20, so its gate never opens).

**First attempt was inert** — byte-identical across ign_floor 0.0/0.1/0.25, the false-negative
signature from Entries 29/39. Cause: `_fire_rate` is called with `region_params` for all seven
regions, and those dicts hold only the twelve fitted Model G coefficients, so
`p.get("ign_floor", 0.0)` returned 0.0 inside every region. Only the global fallback path saw
it. Same structural trap Entry 3 handled for the memory terms, which are applied *outside*
`_fire_rate` for exactly this reason.

**Merging global-only keys into the regional dicts then collapsed the model to 0.516**
(spatial 0.130) — even at ign_floor = 0.0, which should have been a no-op. Diagnosed the leak:

    global-only keys: crop_k crop_n cure_* ign_floor lag_w month_scale nb_* soft_s soft_w spread_*

**`soft_w` and `soft_s` are the finding.** `_fire_rate` applies the softmin limiting-factor
transform gated on `p["soft_w"]`, but that key exists only in the global PARAMS — so **the
limiting-factor form (Experiment 13, official +0.001) has only ever been active on the global
fallback path, never inside any of the seven regional boxes.** Switching it on everywhere is a
large change and made the model far worse at the current coefficients.

So the committed 0.726 model is *correct as evaluated* — every recorded score used this code
path — but its softmin term is regionally partial in a way I had not realised, and my Entry 39
description of it as replacing "the favourability product" is true only outside the seven
region boxes.

Reverted to `20fdd19`. Two follow-ups, kept separate:
1. re-test `ign_floor` applied *outside* `_fire_rate` (like the memory terms) so it reaches
   every cell without disturbing regional coefficients
2. treat "softmin inside regions" as its own experiment, retuned — it may be worth something
   once its coefficients are fitted for that path rather than inherited

### Entry 60: Experiment 23 — cold-regime ignition REFUTED; bona's real constraint identified

Re-applied the cold ignition term *outside* `_fire_rate` (as Entry 59 prescribed), so it
reaches every cell without shadowing regional coefficients. This time it was live, and
monotonically worse: cold_k 0.0 -> **0.7256**, 0.3 -> 0.7253, 1.0 -> 0.7242.

Checked whether it at least achieved its regional purpose. It did not:

| cold_k | bona overall | bona peak (obs 0.628) | global |
| ------ | ------------ | --------------------- | ------ |
| 0.0 | 0.6055 | 0.034 | **0.7256** |
| 1.0 | 0.6055 | 0.035 | 0.7242 |
| 3.0 | 0.6061 | 0.035 | 0.7222 |

Even at cold_k = 3.0 the bona peak moves 0.034 -> 0.035 while global drops 0.003. **Refuted.**

**The hypothesis targeted the wrong factor.** Decomposed bona's monthly climatology under the
N.America parameter set:

    dryness  0.736 - 0.881
    precip   **0.057 - 0.147**   <- dominant suppressor
    fuel     0.141 - 0.848
    temp     0.000 - 0.255

Temperature is *not* the binding constraint. **`precip` is**, and it never exceeds 0.147.
Cause: N.America `P_half` = **958.9**, so the availability term `P/(P+P_half)` sits near 0.1
for bona's modest annual precipitation. That single coefficient throttles the whole product,
and no ignition term applied on top can recover it — multiplying 0.1 by anything still yields
roughly 0.1 of the needed rate.

So Entry 58's diagnosis ("the temperature gate never opens") was right that bona is
structurally silenced but wrong about which gate. The real cause is the N.America `P_half`
being fitted on temperate US fire, where annual precipitation is far higher.

This makes bona genuinely unreachable by any *global* mechanism: the block is a single fitted
regional coefficient, and regions cannot be redrawn (Entry 16) or per-region refitted
(Entry 38, refuted). Combined with its 0.49% weight ceiling (+0.002 maximum), **bona is closed
as a target.** Reverted to `20fdd19`.

### Entry 61: Experiment 24 — softmin inside regional boxes REFUTED, decisively

Follow-up 2 from Entry 59, run as its own experiment with a separate weight (`soft_rw`) so the
strength inside regions is independent of the fitted strength outside them.

| soft_rw | overall | spatial |
| ------- | ------- | ------- |
| **0.0 (current)** | **0.7256** | **0.8752** |
| 0.3 | 0.6424 | 0.5905 |
| 0.7 | 0.5534 | 0.2521 |
| 1.0 | 0.5161 | 0.1302 |

Monotonic collapse — spatial falls 0.875 -> 0.130. **Refuted at every weight.**

Reason is sound rather than incidental: the seven regional dicts are twelve coefficients each,
**fitted under a multiplicative combination rule**. Replacing that rule invalidates them
wholesale. Softmin works on the global path precisely because the global coefficients were
tuned alongside it in Experiment 13; the regional sets never were, and cannot be refitted
(per-region tuning refuted in Entry 38, region redrawing forbidden by Entry 16).

**Quantified the scope of the Entry 59 finding**: the seven boxes cover **39.1%** of land
cells, so the limiting-factor form is active on **60.9%** of cells and its official +0.001 came
entirely from those. This is now a fully characterised property of the model rather than a
loose end: softmin is an extra-regional mechanism, permanently, because the regional skeleton
is a product-form fit that cannot accept it.

Reverted to `20fdd19`. Both Entry 59 follow-ups are now closed, both negative.

### Entry 62: Experiment 25 OFFICIAL — multi-year fuel accumulation, 0.728 (new best)

Changed families after three refutations, as planned. Asked what assumption was still
unexamined: `curing` was meant to carry precipitation memory but tuned `cure_alpha` -> ~1.0
(no memory), so **every term in the model read current conditions only**. Real fuel load is a
stock built over years.

Added `legacy`: slow EMA accumulator over the model's own fire rate; a shortfall against the
stock raises flammability.

First form (per-month deficit) was net negative — but diagnostically sharp: **spatial improved
monotonically (0.8752 -> 0.8797) while seasonal collapsed (0.874 -> 0.843)**. The stock signal
was informative; applying it month-by-month imprinted the accumulator's multi-year drift onto
the seasonal cycle. Fixed by applying the deficit through each cell's **long-run mean** only.

| leg_w | leg_a | overall | bias | rmse | seasonal | spatial |
| ----- | ----- | ------- | ---- | ---- | -------- | ------- |
| 0.0 | - | 0.7256 | 0.7644 | 0.5572 | 0.8741 | 0.8752 |
| 0.3 | 0.1 | 0.7268 | 0.7665 | 0.5577 | 0.8744 | 0.8776 |
| 0.3 | 0.2 | 0.7278 | - | - | - | - |
| **0.3** | **0.3** | **0.7284** | **0.7692** | **0.5583** | 0.8741 | **0.8820** |
| 0.3 | 0.5 | 0.7247 | 0.7642 | 0.5557 | 0.8739 | 0.8741 |

Committed `af23844`; `ar evaluate` returned **official 0.728** (bias 0.769, rmse 0.558,
seasonal 0.874, spatial 0.882). **Eleventh** consecutive three-decimal proxy match.
Bias and spatial are both the highest of the thread.

**Best regional spread of any experiment here: 12/14 improved** — nhsa +0.021, seas +0.021,
tena +0.016, nhaf +0.014, euro +0.013, shsa +0.014, against eqas -0.028 as the only real loss.
Every previous gain either forced a spread tradeoff (Entry 52) or concentrated in one region
(Entry 53). This one satisfies the soft requirement rather than straining it.

Progression: 0.686 -> ... -> 0.725 -> 0.726 -> **0.728** (13 official evals, +0.042 total).

**Pattern now confirmed twice**: after a run of refutations, the productive move is to find an
*unexamined assumption* rather than another variant. Pointwise-ness gave the neighbour term
(+0.004); memorylessness gave this one (+0.003). Both were properties of the whole model that
no single component owned.

### Entry 63: Experiment 26 — GPP-driven accumulation REFUTED

Extended the legacy family: the accumulator integrates the model's own fire rate, but litter
physically accumulates from *productivity* — a wet productive year builds the fuel a later dry
year consumes. Blended GPP into the integrand.

| leg_gpp | overall | bias | spatial |
| ------- | ------- | ---- | ------- |
| **0.0 (fire rate only)** | **0.7284** | **0.7692** | **0.8820** |
| 0.3 | 0.7233 | 0.7615 | 0.8703 |
| 0.7 | 0.6851 | 0.7182 | 0.7693 |

Monotonically worse. **Refuted.**

Reason: GPP is *already* in the model as the `fuel` hump term, so feeding it into the
accumulator double-counts productivity rather than adding information. What `legacy` supplies
that nothing else does is **fire history** — where the model has been burning, and therefore
where fuel has been consumed. Its value is the feedback loop, not the growth signal.

That sharpens the mechanism's interpretation: it is a *fuel-consumption memory*, not a
fuel-production memory. Reverted to `af23844`.

### Entry 64: legacy refinements — cap tightened, nonlinear response pruned

Two follow-ups on the winning `legacy` term, both cheap:

(a) **`leg_cap`** (ceiling on the stock shortfall):

| leg_cap | overall | bias | spatial |
| ------- | ------- | ---- | ------- |
| 1.5 | 0.7268 | 0.7655 | 0.8791 |
| 2.5 | 0.7284 | 0.7678 | 0.8838 |
| **3.0** | **0.7286** | 0.7684 | **0.8836** |
| 4.0 (committed) | 0.7284 | **0.7692** | 0.8820 |
| 6.0 | 0.7271 | 0.7705 | 0.8745 |

Interior optimum at 3.0. Note bias keeps rising with the cap while spatial peaks near 2.5-3.0,
so the composite optimum is a genuine balance rather than a plateau.

(b) **`leg_n`** (nonlinear exponent on the shortfall ratio): 0.6 -> 0.7279, 1.0 -> 0.7286,
1.6 -> 0.7274. Tuned to exactly linear. **Pruned the parameter** rather than leaving a dead
knob, with a comment recording the negative result — same discipline as the depletion prune in
Entry 5.

Net **0.7286 vs 0.7284 committed: +0.0002, below ILAMB's three-decimal resolution.** Committed
the refinement but deliberately **did not spend an eval** — it cannot register as an
improvement on its own and would only consume a measurement. Carried into the next structural
experiment instead. Contract check passes.

### Entry 65: Experiment 27 — legacy/spread coupling REFUTED

Applied the "unexamined assumption" heuristic again. Checked the component ordering:

    legacy -> neighbour -> spread -> transform

so `spread` already operates on the legacy-modified rate. The untested question was
*interaction*: percolation is a property of the fuel bed, so a long-unburnt landscape should
run a front at **lower flammability** than the same climate over recently burnt ground.
Wired the legacy stock deficit into `spread_crit` (threshold lowered where load is high),
which required `_legacy` to return its deficit alongside the modified rate.

| leg_crit | overall | rmse | spatial |
| -------- | ------- | ---- | ------- |
| **0.0** | **0.7286** | 0.5583 | **0.8836** |
| 0.3 | 0.7280 | 0.5580 | 0.8814 |
| 0.8 | 0.7263 | 0.5568 | 0.8760 |

Monotonically worse. **Refuted.**

Reason: the two terms are already correctly composed *by ordering*. `legacy` raises the rate
where fuel has accumulated, and `spread` then reads that raised rate — so accumulated fuel
already lowers the effective distance to the percolation threshold. Coupling them explicitly
applies the same fuel signal twice.

Useful negative: it confirms the sequential composition is doing real work, and that the
pipeline does not need explicit cross-terms between components. Reverted to `1cd40f2`.

**State: official 0.728 at `af23844`; working tree at `1cd40f2` (proxy 0.7286, +0.0002 —
sub-threshold, carried forward unevaluated).**

### Entry 66: Experiment 28 — heavy-tailed exceedance REFUTED; assumption-hunting vein assessed

Last unexamined assumption in the output path: `_transform` is `1 - exp(-r)`, a **Poisson**
exceedance that assumes fires are independent points. Real fires are large correlated patches
and burned-area distributions are heavy tailed, so tested a gamma-mixed exceedance
`1 - (1 + r/k)^-k`, which shares the small-rate behaviour but approaches full coverage more
slowly.

| tail_k | overall | bias | spatial |
| ------ | ------- | ---- | ------- |
| **0 (Poisson)** | **0.7286** | **0.7684** | **0.8836** |
| 0.5 | 0.7261 | 0.7655 | 0.8756 |
| 2.0 | 0.7284 | 0.7681 | 0.8828 |
| 5.0 | 0.7283 | 0.7684 | 0.8834 |
| 10.0 | 0.7283 | 0.7684 | 0.8836 |

**Refuted.** Non-monotone but never better: it converges back to ~baseline as k -> infinity
(the Poisson limit, a good self-consistency check that the implementation is right) and is
strictly worse at every finite k. Poisson exceedance is already the better fit — plausibly
because `spread` and `neighbour` now handle clustering upstream, so adding it again in the
transform double-counts.

**Assumption-hunting heuristic, honest scorecard:** 2 substantial wins (neighbour +0.004,
legacy +0.003) then **3 consecutive refutations** (GPP accumulation, legacy/spread coupling,
heavy tail). By the criterion set in Entry 65, the vein is thinning. The remaining
whole-model assumptions I can identify are ones already tested and closed:
regional structure (Entries 16/38/61), pointwise-ness (54-57), memorylessness (62-64),
combination rule (39/61), transform shape (23/this entry).

Switching strategy next tick: rather than hunting further structural assumptions, go back to
the **weighted residual** and target the largest remaining concrete error, which is nhaf/shaf
rmse (0.53/0.57, 61% of all recoverable score). That is where the score actually is, and it
has not been directly attacked since the transform fix.

### Entry 67: Experiment 29 — fuel depletion RE-REFUTED after the transform fix

New strategy (Entry 66): attack the largest concrete error rather than hunt assumptions.
`africa_cycle.py` at 0.7286 shows **shaf is now essentially solved** (amp ratio **0.99**,
tracking obs within ~1.5 all year). **nhaf is the remaining error**: amp 0.66, and the
asymmetry is specific —

    obs  16.31 7.44 3.49 ... 1.59 9.11 18.70
    mod  12.50 9.69 4.33 ... 0.84 3.38  9.69
    diff -3.81 +2.25 ...      -0.76 -5.73 -9.01

Month-ratio decomposition isolates it to **decay rate, not onset**:

    Nov/Oct: obs 5.72  mod 4.04   (onset slightly soft)
    Feb/Jan: obs 0.46  mod **0.78**  (decay far too slow)

After burning a sixth of its area in December the savanna's fine fuel is gone. That is exactly
fuel depletion, pruned in Entry 5 but *before* the transform fix, `legacy` and `neighbour` —
so it was worth retesting on the current model.

**First implementation was buggy**, and the numbers said so: collapse to 0.536 at dep_k = 0.3
*and* 0.581 at dep_k = **0.01**, with annual burned halving at the smallest strength. A term
that destroys the model at 1% strength is a bug, not a physical result. Cause: I normalised by
the **global** mean rate, so African cells at hundreds of times that mean drove their fuel
stock to zero regardless of `dep_k`. Fixed to per-cell normalisation.

Corrected result, now a clean test:

| dep_k | overall | seasonal | spatial |
| ----- | ------- | -------- | ------- |
| **0.0** | **0.7286** | **0.8744** | **0.8836** |
| 0.1 | 0.6907 | 0.8341 | 0.7919 |
| 0.3 | 0.5948 | 0.7785 | 0.4922 |
| 0.6 | 0.5375 | 0.7465 | 0.2944 |

**Monotonically harmful — Entry 5's verdict holds even after the transform fix.** The physics
is the same as then: nhaf/shaf savannas *reburn annually*, so any mechanism that suppresses
consecutive burning silences precisely the cells carrying 61% of recoverable score. The slow
February decay is real, but depletion cannot be its fix without costing far more elsewhere.

Reverted to `1cd40f2`. Lesson reinforced: when a term collapses the model at ~1% strength,
suspect the implementation before concluding the hypothesis is refuted.

### Entry 68: nhaf slow decay traced to the inputs — no driver falls the way fire does

Followed up the Entry 67 diagnosis (nhaf Feb/Jan: obs 0.46, model 0.78).

**First checked whether `lag` causes it.** It contributes but is not the cause, and removing it
is not worth it:

| lag_w | nhaf Feb/Jan | nhaf amp | global |
| ----- | ------------ | -------- | ------ |
| 0.0 | 0.71 | 0.683 | 0.7256 |
| 0.1 | 0.74 | 0.668 | 0.7277 |
| 0.2133 (current) | **0.78** | 0.655 | **0.7286** |

So `lag` does worsen nhaf's decay and amplitude, yet *raises* global score by +0.003 — it pays
elsewhere more than it costs here. And at lag_w = 0 the ratio is still 0.71 against an observed
0.46, so lag is not the root cause. Left as is: this is a real cost knowingly accepted, not an
oversight.

**Then decomposed the Jan->Feb ratio of every driver** (Africa regional parameters):

| factor | Jan | Feb | ratio |
| ------ | --- | --- | ----- |
| dryness | 0.9103 | 0.9042 | 0.993 |
| precip | 0.9735 | 0.9561 | 0.982 |
| fuel | 0.8952 | 0.8852 | 0.989 |
| temp | 0.9978 | 0.9978 | 1.000 |
| curing | 1.6099 | 1.4944 | 0.928 |
| **observed burned** | - | - | **0.46** |

**Every driver is flat Jan->Feb (0.93-1.00) while observed burning more than halves.** The
Sahel fire season ends without any corresponding change in dryness, rainfall, temperature or
productivity.

This bounds the problem honestly: the collapse is **not recoverable from these inputs by any
functional form**, because no input contains the signal. The physical cause is almost certainly
fuel exhaustion — the landscape has simply burnt what it had — and depletion is the right
mechanism, but Entry 67 shows it costs far more in the annually-reburning cells than it gains
here.

nhaf's residual decay error is therefore **structurally bounded, not a defect to fix**. Its
remaining rmse (0.528) is largely irreducible with the current input set. Recording this so no
future tick re-opens it: the useful search space is elsewhere.

### Entry 69: Experiment 30 — sub-threshold quench REFUTED, and the "48x overprediction" reframed

Retargeted to `shaf` (0.327 recoverable, second-highest). Its *shape* is solved (amp 0.99,
Entry 67) so its loss is **magnitude**: mod/obs = **1.49**. `bias_probe.py` shows the familiar
stratification in both African regions:

| region | `<0.1%` cells | `1-5%` | `>5%` |
| ------ | ------------- | ------ | ----- |
| nhaf | **58.3x** over | 1.14x | 0.65x |
| shaf | **48.1x** over | 1.28x | 0.84x |

Tested a `quench` term fading out cells whose long-run mean rate sits far below the percolation
threshold — a landscape that never assembles a connected fuel bed should not burn at all.

| quench | overall | bias | spatial |
| ------ | ------- | ---- | ------- |
| **0.0** | **0.7286** | **0.7684** | **0.8836** |
| 0.05 | 0.7244 | 0.7576 | 0.8799 |
| 0.15 | 0.7184 | 0.7446 | 0.8710 |

**Refuted — and bias moved the wrong way**, which is the opposite of the term's whole purpose.
Checked why rather than filing it as a plain negative:

    quench=0.00: low-burn mod 0.0212 (obs 0.0028) | high-burn mod 2.397 (obs 2.986)
    quench=0.15: low-burn mod 0.0087 (obs 0.0028) | high-burn mod **2.117** (obs 2.986)

The term **did** fix the tail (0.0212 -> 0.0087). But those cells have reference_mean ~ 0, so
under `area * reference_mean` weighting they contribute ~nothing to bias_score. Meanwhile the
suppression leaked into the high-burn cells, which are **already underpredicted** (2.397 vs
2.986) and carry all the weight.

**This reframes a number I have quoted repeatedly since Entry 9.** The "48-58x overprediction
of low-burn cells" is real but is *not* a scoring problem — it is invisible to every weighted
metric. The actual shaf/nhaf bias deficit is that the **high-burning cells are underpredicted**
(0.65x and 0.84x), while the regional mod/obs of 1.21-1.49 is dominated by cell count in the
tail rather than by burned area. Any future experiment aimed at the tail is aimed at nothing.

Correct target, if any: raise the >5% cells without raising the tail — the opposite direction
from everything tried in this family. Reverted to `1cd40f2`.

### Entry 70: Experiment 31 — high-burn boost REFUTED; the reframed target is also unreachable

Acted on the Entry 69 reframing: the real deficit is that heavy-burning cells are
*under*predicted (0.65x nhaf, 0.84x shaf), so the target is raising them without raising the
tail.

**First checked whether `spread_gain` already does this** — it is the existing
amplify-above/suppress-below term:

| spread_gain | overall | rmse | spatial |
| ----------- | ------- | ---- | ------- |
| **9.36 (current)** | **0.7286** | **0.5583** | **0.8836** |
| 14.0 | 0.7269 | 0.5560 | 0.8808 |
| 20.0 | 0.7237 | 0.5522 | 0.8747 |
| 28.0 | 0.7198 | 0.5477 | 0.8669 |

Already at its optimum. Reason: `_spread` is **mean-normalised per cell**, so it redistributes
within a cell across months and cannot lift a cell's annual total at all. That is a structural
limit of the existing term, not a tuning miss.

Then added `top_k`: lift each cell in proportion to how far its own long-run rate exceeds a
fraction of the percolation threshold.

**First attempt was inert** (byte-identical across three settings — the Entry 29/39/59
signature). Diagnosed: rate means run **p99 = 0.60, max = 0.85**, but my threshold was
`top_c * spread_crit` = 2.0 x 1.76 = **3.53**, above every cell, so `excess` was zero
everywhere. The range was miscalibrated as a *multiple* of spread_crit when it needed to be a
*fraction*. Rescaled to [0.02, 1.0].

Corrected result:

| top_c | top_k | overall | bias | spatial |
| ----- | ----- | ------- | ---- | ------- |
| - | 0.0 | **0.7286** | **0.7684** | **0.8836** |
| 0.25 | 0.3 | 0.7216 | 0.7577 | 0.8663 |
| 0.25 | 0.8 | 0.6920 | 0.7183 | 0.7901 |
| 0.10 | 0.3 | 0.7149 | 0.7499 | 0.8465 |
| 0.10 | 0.8 | 0.6661 | 0.6894 | 0.7160 |

**Refuted, monotonically in both parameters — and bias falls too**, though the term exists to
fix bias. Same failure mode as `quench` (Entry 69) in mirror image: a mean-based multiplier
cannot separate "cells that burn heavily" from "cells the spatial map needs left alone",
because lifting the former distorts the 16-year mean map that `spatial_distribution_score`
scores directly.

**Both directions of the Entry 69 target are now closed**: suppressing the tail is invisible to
the weighting, and lifting the top distorts the spatial map more than it repairs bias. The
African bias residual (mod/obs 1.21/1.49) is not reachable by any cell-mean rescaling.
Reverted to `1cd40f2`.

### Entry 71: amplitude deficit localised to the non-African regional parameter sets

Retargeted to the underpredicted regions. All four are flat in the same way:

| region | amp_ratio | obs peak | mod peak |
| ------ | --------- | -------- | -------- |
| ceas | 0.21 | 3.49 (Apr) | 0.58 (**Sep**) |
| seas | 0.24 | 9.81 (Mar) | 2.15 (Mar) |
| shsa | 0.28 | 2.72 (Sep) | 0.76 (Aug) |
| aust | 0.19 | 4.61 (Oct) | 0.94 (Sep) |

Swept amplitude across all 14: **only nhaf (0.66) and shaf (0.99) are healthy; every other
region is 0.07-0.58.** Both healthy regions lie 100% inside the Africa box, whose twelve
coefficients were fitted on them.

Cause is visible in the fitted sets — `pre_dampen_half`, the monthly-rainfall damping scale:

    Africa 304.75 | Australia 724.48 | Europe 107.40 | SEAsia 107.40
    Boreal 53.73  | N.America **14.69**

Africa's damping is so weak that monthly rainfall barely suppresses its rate, letting the cycle
swing; N.America's is 20x stronger and flattens it. **So the "seasonal amplitude deficit" I
have tracked since Entry 2 was never global — it was solved for Africa by the transform fix
and remains unsolved everywhere else, for a reason that lives in frozen regional coefficients.**

### Entry 72: Experiment 32 — flatness-targeted sharpening REFUTED (sixth in that family)

Since the regional sets cannot be refitted (Entry 38) or redrawn (Entry 16), the only lever is
a global term that sharpens *in proportion to how flat a cell already is*, leaving Africa
untouched. Implemented via a per-cell exponent driven by `flat_ref - std/mean`.

| flat_k | overall | rmse | spatial |
| ------ | ------- | ---- | ------- |
| **0.0** | **0.7286** | **0.5583** | **0.8836** |
| 0.3 | 0.7276 | 0.5571 | 0.8819 |
| 0.8 | 0.7239 | 0.5528 | 0.8748 |

**Refuted**, monotonically. Same signature as `sharpen` (20), the temperature/LAI patch scales
(28-29), `contrast` (37), global spread normalisation (47) and output-mean preservation (48):
cycle contrast up, spatial down, net negative. Even when the sharpening is *targeted* at the
flat regions and explicitly excludes the sharp ones, it still costs more spatial than it earns.

That is six independent formulations. The amplitude/spatial tension established in Entries
47-48 is now confirmed to hold even under region-selective application — it is not that
previous attempts were too blunt. Reverted to `1cd40f2`.

### Entry 73: Experiment 33 — nonlinear rainfall damping marginal; **Entry 71 diagnosis CORRECTED**

Tested an exponent on the monthly-rainfall damping ratio, on the theory (Entry 71) that
`pre_dampen_half` being up to 20x stronger outside Africa causes the amplitude deficit.

    damp_n 1.0 -> 0.7286 | 1.5 -> 0.7286 | 2.5 -> 0.7287 | 4.0 -> 0.7287 | 6.0 -> 0.7287

Positive but plateaus at **+0.0001**, below ILAMB's three-decimal resolution. Not worth a
parameter. Checked why it has so little leverage:

| region | pre_dampen | M/pd ratio range | damping factor range |
| ------ | ---------- | ---------------- | -------------------- |
| ceas | 107.4 | 0.20-0.74 | 0.58-0.83 |
| aust | 724.5 | 0.01-0.27 | **0.79-0.99** |
| nhaf | 304.7 | 0.02-0.78 | 0.56-0.99 |

**The damping term contributes almost no seasonal swing anywhere, including Africa.** So
`pre_dampen_half` is *not* the source of the amplitude difference. **Entry 71's causal claim
was wrong** — it inferred cause from a parameter that merely differed between regions, without
checking that the parameter actually drives the cycle. Recording the correction rather than
leaving a false diagnosis in the log.

**Re-derived the real source** — per-factor seasonal swing (max/min of monthly climatology):

| region | dryness | precip | fuel | temp | **curing** |
| ------ | ------- | ------ | ---- | ---- | ---------- |
| nhaf | 1.81 | 1.71 | 1.18 | 1.00 | **3.02** |
| shaf | 1.71 | 1.56 | 1.06 | 1.00 | **2.75** |
| aust | 1.41 | 1.25 | 1.12 | 1.03 | 2.46 |
| shsa | 1.31 | 2.18 | 1.47 | 1.16 | 2.17 |
| ceas | 1.11 | 1.34 | 4.27 | 456.88 | **1.38** |

**`curing` is the dominant amplitude source, and it swings hardest in exactly the two healthy
regions** (3.02, 2.75) and weakest in the flattest one (ceas 1.38). Everything else is
comparatively flat, except ceas's `temp` (457x) and `fuel` (4.27x) which are large but
evidently mis-phased given ceas peaks in the wrong month entirely.

This is a **better target than Entry 71's**: curing is a *global* term with global coefficients,
so unlike `pre_dampen_half` it can actually be changed. The question for next tick is why its
swing collapses outside Africa — most likely because `cure_half` (72.8 mm) is calibrated to
African rainfall and saturates elsewhere.

### Entry 74: curing hypothesis REFUTED; amplitude deficit traced to the assembled rate itself

Tested Entry 73's hypothesis that `cure_half` (72.8 mm) is calibrated to African rainfall and
saturates elsewhere. **Refuted at two levels.**

(a) Antecedent-wetness ratios per region:

    nhaf 0.065-3.28 | shaf 0.064-2.59 | aust 0.082-2.64 | seas 0.210-4.33 | ceas 0.302-1.09

`aust` and `seas` span ranges comparable to Africa's, so `cure_half` is *not* mis-calibrated
for them. Only `ceas` is genuinely compressed.

(b) Actual curing swing, cell-wise and weighted:

| region | cellwise swing | region-mean swing | frac at cure_cap |
| ------ | -------------- | ----------------- | ---------------- |
| nhaf | 3.31 | 3.02 | 0.000 |
| **seas** | **3.75** | **3.05** | 0.000 |
| aust | 2.82 | 2.46 | 0.000 |
| ceas | 1.67 | 1.38 | 0.000 |

**`seas` has the highest curing swing of any region (3.75 > nhaf's 3.31) yet an amplitude ratio
of 0.24.** Nothing is hitting the cap. So curing is not the bottleneck outside Africa either —
Entry 73's target was as wrong as Entry 71's, for the same reason: attributing a whole-model
deficit to whichever component happened to differ, without tracing the actual signal path.

**Traced it properly instead** — swing at each stage, region-mean:

| region | assembled rate | model output | **observed** |
| ------ | -------------- | ------------ | ------------ |
| nhaf | 8.09 | 27.13 | **643.88** |
| shaf | 6.63 | 24.06 | 98.52 |
| seas | 22.18 | 24.77 | **241.54** |
| aust | 5.71 | 5.82 | 34.70 |
| ceas | **51.89** | 44.84 | 93.80 |

Two findings:

1. **The deficit is in the assembled rate, not in any single component.** Observed cycles swing
   35-644x; the assembled rate swings 6-52x. The transform amplifies this (8 -> 27 in nhaf) but
   cannot manufacture a 644x swing from an 8x input. No single term is responsible — the
   *product of four bounded [0,1] factors plus curing* simply cannot span two orders of
   magnitude, which is the Entry 19 ceiling argument reappearing at the whole-model level.
2. **ceas is not an amplitude problem at all.** It has the largest rate swing of any region
   (51.9x, vs nhaf's 8.1x) and still scores worst, because its peak lands in September against
   an observed April. Its loss is **phase**, and Entry 33 already showed phase is not fixable
   from these inputs where drivers do not move with the fire.

This closes the amplitude line at the structural level rather than per-component. Three
consecutive misattributions (Entries 71, 73, and this one's starting hypothesis) all came from
naming a component before tracing the signal; the trace should have come first.

### Entry 75: month_scale re-verified at optimum; full coordinate sweep started

Loop switched to 10m (job `542581da`); cancelled the 20m job `985299ea` so the two do not
overlap and duplicate experiments.

Entry 74 showed the transform is where amplification actually happens (nhaf rate swing 8.09 ->
output swing 27.13), so `month_scale` is the one lever with real leverage on amplitude. It was
tuned at 0.0423 back in Entry 26, **before** softmin, cropland, neighbour and legacy were
added. Re-swept it:

| month_scale | overall | bias | rmse | spatial |
| ----------- | ------- | ---- | ---- | ------- |
| 0.030 | 0.7046 | 0.7481 | 0.5516 | 0.7971 |
| **0.0423** | **0.7286** | **0.7684** | **0.5583** | **0.8836** |
| 0.055 | 0.7099 | 0.7460 | 0.5494 | 0.8304 |
| 0.070 | 0.6661 | 0.6937 | 0.5253 | 0.7118 |

Still at a sharp interior optimum — four added components did not shift it. Worth knowing:
the transform scale is decoupled from the rate-shaping terms, so future components do not
require re-tuning it.

Started a **full coordinate sweep** (detached, survives tick boundaries) over all 14
coefficients. Last sweep was at Entry 35 with 9 coefficients; five have been added since
(cropland x2, neighbour x2, legacy x3) and their interactions may have moved others.

### Entry 76: ceas phase error traced to agricultural burning; a testable lead

`ceas` obs peaks **April**; every driver peaks **July-September**:

    obs cycle    0.04 0.11 0.99 3.49 2.26 0.61 0.77 1.25 1.13 2.68 1.01 0.05   peak=m4
    dryness      0.09 0.04 **0.00** 0.04 0.24 0.53 0.70 0.91 1.00 0.99 0.92 0.85  peak=m9
    precip       peak=m7 | air_temp peak=m7 | gpp peak=m7

Dryness is at its **minimum** in March, exactly as fire ramps. No climate driver can produce an
April peak, so this is not a tuning problem.

**Cause is agricultural.** Burned-area-weighted cropland fraction:

    ceas 0.335 | tena 0.340 | euro 0.383 | nhaf 0.132 | shaf 0.052 | aust 0.015

The three most agricultural regions (ceas, tena, euro) all have poor rmse (0.591, 0.502, 0.569)
and phase errors; the least agricultural (shaf, aust) are the ones the model fits well. Spring
stubble clearing in Central Asia and Eastern Europe follows a **planting calendar**, not
weather.

**Checked whether LUH2 carries a calendar: it does not.** Within-year variation of every LUH2
field is *exactly* 0.000000 — they are annual values repeated monthly, as `inputs/README.md`
states. So LUH2 says *where* agriculture is but never *when* it burns.

**Testable lead, not a dead end:** the existing `cropland` term suppresses fire but is
seasonally flat. Cropland fraction could instead *gate a calendar the model already has* —
stubble burning happens at the shoulders of the growing season, and growing-season timing is
recoverable from air temperature. That is a global mechanism keyed on a physical covariate,
same shape as the terms that have worked. Queued for the next tick; the coordinate sweep is
still running and I do not want two heavy jobs competing.

### Entry 77: Experiment 34 OFFICIAL — agricultural stubble burning, 0.730 (new best)

`ar evaluate` on `162eeff`: **official Overall 0.730** (bias 0.772, rmse 0.559, seasonal 0.873,
spatial 0.884). Proxy said 0.7295 — twelfth consecutive three-decimal match. **First crossing
of 0.73**; bias and spatial are both the highest recorded.

Mechanism came straight from the Entry 76 diagnosis. ceas burns in April while every driver
peaks Jul-Sep and dryness is at its annual *minimum* as fire ramps; the three most agricultural
regions (cropland 0.34/0.34/0.38) carry the three worst phase errors. LUH2 has no calendar
(verified: within-year variation exactly 0.000000), so the term gates on cropland fraction and
fires on the **growing-season temperature shoulder**, which the model already has.

Regional effect matches the design intent rather than being a lucky global fit:
**ceas +0.048** (the motivating region), **boas +0.051**, euro -0.007, **10/14 improved**.

| stub_k | overall | bias | rmse | spatial |
| ------ | ------- | ---- | ---- | ------- |
| 0.0 | 0.7288 | 0.7701 | 0.5584 | 0.8825 |
| **1.5** | **0.7295** | 0.7719 | 0.5591 | 0.8844 |
| 2.0 | 0.7292 | 0.7721 | 0.5577 | 0.8850 |
| 4.0 | 0.7263 | 0.7690 | 0.5521 | 0.8844 |

Progression: 0.686 -> ... -> 0.726 -> 0.728 -> **0.730** (14 official evals, +0.044 total).

### Entry 78: coordinate-sweep gains — mostly non-additive, and one reading corrected

The Entry 75 sweep suggested +0.0014 across four coefficients. Applying the three uncontested
ones (`crop_k` 0.5->1.22, `crop_n` 2.0->1.514, `spread_k` 5.2->6.52) gave **+0.0002 combined**.
One-at-a-time sweep gains do not sum; treat them as directional only.

**`nb_w` correction**: the sweep wanted 0.35 -> 0.512, which I had originally rejected on
regional-spread grounds (Entry 52). Re-checked *after* the cropland changes: 0.512 is now
**worse globally** (0.7285 vs 0.7288) and identical on spread (5/14 both). The sweep had
measured it pre-cropland and the two interact, so the spread tradeoff I was preserving no
longer exists in either direction. Kept 0.35. Lesson: re-verify a sweep's recommendation at
the point you actually apply it, not at the point it was measured.

### Entry 79: stubble variants tested exhaustively (research.md directive), coefficients held

research.md says to test distinct physically plausible formulations as separate experiments
exhaustively, so the winning stubble term got its variants probed rather than moving straight
on.

(a) **Autumn harvest shoulder** (`stub_fall`, letting the cooling limb fire too):
    0.0 -> **0.7295** | 0.5 -> 0.7290 | 1.0 -> 0.7272. **Refuted.** Spring-only is correct;
    pre-sowing clearing dominates and the GFED5 Central Asian signal is specifically the April
    peak.

(b) **Threshold temperature** `stub_t`: 4 -> 0.7290 | **8 -> 0.7295** | 12 -> 0.7276 |
    16 -> 0.7239. Interior optimum at **8 C**, which is the conventional spring sowing
    threshold in temperate agriculture — the fitted value is physically apt, not arbitrary.

(c) **Window width** `stub_w`: 2 -> 0.7292 | 4 -> 0.7295 | 5 -> 0.7298 | **6 -> 0.7298** |
    8 -> 0.7287 | 11 -> 0.7262.

On (c) the global optimum and the spread requirement disagree:

| stub_w | global | regions improved |
| ------ | ------ | ---------------- |
| **4.0** | 0.7295 | **10/14** |
| 5.0 | 0.7298 | 7/14 |
| 6.0 | 0.7298 | 8/14 |

**Kept 4.0.** The +0.0003 global gain is below ILAMB's three-decimal resolution — it cannot
register as an improvement — while 10/14 versus 8/14 is a real difference against research.md's
soft requirement that gains spread across regions. Trading an unmeasurable gain for two
regions is a bad trade. (6.0 does dominate 5.0, so if global were the only criterion the choice
would be 6.0, not 5.0.)

Model left exactly at the evaluated `162eeff` point.

### Entry 80: PRUNING ROUND 4 started (11 components, 2048 subsets)

~9 ideas since the last pruning round at Entry 46, so this is due per research.md. Four
components added since (cropland, neighbour, legacy, stubble). Running `ar ablate` detached.

### Entry 81: Experiment 35 OFFICIAL — rangeland management burning, 0.733 (new best)

`ar evaluate` on `71e6e19`: **official Overall 0.733** (bias 0.779, rmse 0.562, seasonal 0.868,
spatial 0.892). Proxy said 0.7327 — thirteenth consecutive three-decimal match. Bias and
spatial are again the highest recorded.

**Rangeland had already been refuted once** (Entry 42, as a flat multiplier: 0.7208 -> 0.7145).
What changed is the *formulation*, following the stubble win: land-use data carries a
**calendar** signal, not an intensity signal. Graziers burn to kill woody seedlings and flush
new growth, in the hottest driest part of the year — a different schedule from crop stubble.

**The first calendar attempt also failed** (0.7255 at `past_t` = 18 C, monotonically worse). I
checked whether the *timing* was wrong rather than filing it refuted, and at 30 C it jumps to
**0.7327**. Confirmed a genuine interior optimum by extending the range to 45 C:

| past_t | overall | bias | spatial |
| ------ | ------- | ---- | ------- |
| 18 | 0.7255 | 0.7691 | 0.8826 |
| 26 | 0.7311 | 0.7723 | 0.8874 |
| 28 | 0.7325 | 0.7762 | 0.8889 |
| **30** | **0.7327** | **0.7787** | **0.8923** |
| 33 | 0.7307 | 0.7776 | 0.8916 |
| 40 | 0.7296 | 0.7722 | 0.8848 |

**12/14 regions improved**: nhsa +0.068, aust +0.049, shsa +0.023, nhaf +0.012, tena +0.006.

**Lesson worth keeping: a refuted input is not a refuted mechanism.** Rangeland failed as
intensity and succeeded as timing. Before discarding an input, check whether a different
*functional role* has been tried — and before discarding a calendar term, sweep its phase.

Progression: 0.686 -> ... -> 0.730 -> **0.733** (15 official evals, +0.047 total).

### Entry 82: Experiment 36 — intact primary vegetation REFUTED

Screened the three remaining LUH2 fields against the weighted residual:

    luh2_primary_fraction   wr(resid)=**+0.126** wr(obs)=+0.143
    luh2_secondary_fraction wr(resid)=+0.004  wr(obs)=+0.017
    luh2_urban_fraction     wr(resid)=+0.024  wr(obs)=-0.169

Primary fraction was the strongest remaining signal and the sign is physically right — the
model *overpredicts* where intact forest is, and closed canopy genuinely resists fire. Tested
as a suppressor:

    intact_k 0.0 -> **0.7327** | 0.5 -> 0.7082 | 1.5 -> 0.6620

**Refuted, steeply.** Sixth time in this thread that an aggregate residual correlation has
failed on cell-level application (after lightning, fuel_load, season_length, temperature and
LAI patch scales, rangeland-as-intensity). The screen is useful for *ranking* candidates but
has never once been sufficient evidence on its own. Reverted.

Also screened `luh2_pasture_fraction` before this: wr(resid) = -0.055 against wr(obs) = +0.289
— the lightning signature, already captured by the climate terms. Correctly skipped.

### Entry 83: PRUNING ROUND 4 result (12 components, 4096 subsets) — nothing prunes

`ar ablate` on `162eeff` completed (4096 subsets parsed; sum reconciles to +0.3230 =
full 0.730 - empty 0.407).

| component | shapley | drop-one |
| --------- | ------- | -------- |
| precipitation | +0.0732 | +0.0130 |
| dryness | +0.0572 | +0.0130 |
| temperature | +0.0562 | +0.0140 |
| fuel | +0.0495 | +0.0050 |
| curing | +0.0477 | **+0.0690** |
| cropland | +0.0117 | +0.0030 |
| stubble | +0.0098 | +0.0010 |
| spread | +0.0092 | **+0.0370** |
| neighbour | +0.0056 | +0.0060 |
| lag | +0.0020 | +0.0030 |
| softmin | +0.0015 | +0.0010 |
| **legacy** | **-0.0005** | +0.0080 |

`legacy` is the first component in this thread with **negative Shapley**. Checked it directly
on the *current* model (which also has `pasture`, added after this ablation started):

    full 13 components  **0.7327**
    drop legacy         **0.7285**   (-0.0042)

So it is firmly load-bearing where it actually sits. The negative Shapley means it is redundant
*averaged over subsets that lack its complements* — in a model without spread/neighbour/curing
it actively hurts — but in the assembled model it contributes more than any component except
curing and spread by drop-one. **Nothing prunes; all 13 components retained.**

Worth recording the general point: with 12+ components, Shapley and drop-one can disagree in
sign, and **drop-one at the operating point is the decision-relevant number** for pruning.
Shapley answers "what is this worth on average across all possible models", which is not the
question research.md's pruning round asks.

Process note: I twice reported this ablation as "RUNNING" when checking with `pgrep -f`, once
after it had already finished (it completed 19:00; I reported running at 19:43). `pgrep -f`
matches the shell wrapper. **Check `ps -eo command` for the actual `bin/ar` process, or compare
file mtime against the clock.**

### Entry 84: land-use calendar family — three refinements REFUTED, family exhausted

Exhausting the calendar family per research.md before changing direction. All three variants
tested at the 0.7327 point:

(a) **Dryness gate on rangeland burning** (`past_dry`) — a manager burns on a hot day only if
    the sward is cured: 0.0 -> 0.7327 | 0.5 -> 0.7328 | **1.5 -> 0.7329** | 2.5 -> 0.7327 |
    3.0 -> 0.7326. Peaks at **+0.0002**, sub-threshold. Not worth a parameter.

(b) **Dryness gate on stubble burning** (`stub_dry`): 0.0 -> **0.7327** | 0.8 -> 0.7327 |
    2.0 -> 0.7324. Flat then declining. **Refuted.**

(c) **Latitude-shifted sowing calendar** (`crop_lat`) — sowing runs later poleward for the same
    temperature because day length also gates field work: 0.0 -> **0.7327** | 1.0 -> 0.7327 |
    2.5 -> 0.7326. **Refuted.**

Common cause for (a) and (b): `dryness` and `curing` already supply the moisture signal
upstream of these terms, so gating on it again double-counts — the same redundancy that killed
GPP-driven accumulation (Entry 63) and the legacy/spread coupling (Entry 65). The pipeline
does not want explicit cross-terms between components; sequential composition already carries
the interaction.

The land-use calendar family has now delivered **two official gains** (stubble 0.730, rangeland
0.733) and **three clean refutations** bounding its coefficient space. Treating it as closed
and changing families next, consistent with the pattern that has worked twice before
(Entries 53, 62): after a run of refutations in one family, find an unexamined assumption
rather than a sixth variant.

### Entry 85: euro phase error diagnosed to the stubble term; cap REFUTED, cost accepted

Loop moved to 5-minute ticks (job `f334332e`, cancelled 15m `e066367c`).

Phase-error sweep across all regions at 0.733 found **euro is the worst: 5 months** (obs peaks
August, model March). Traced it directly:

    euro obs  0.02 0.08 0.64 **0.99** 0.19 0.08 0.94 **1.35** 0.63 0.64 0.19 0.02
    euro mod  0.09 0.17 0.60 0.35 0.11 0.11 0.17 0.24 0.22 0.16 0.14 0.12
    no stub   0.09 0.10 0.12 0.14 0.11 0.11 0.17 **0.24** 0.22 0.16 0.13 0.12

**Without stubble, euro peaks in August — matching observations exactly.** The stubble term
manufactures a March peak (0.12 -> 0.60) that inverts the cycle. Europe genuinely has both
peaks (0.99 April agricultural, 1.35 August Mediterranean) but the model's spring spike is ~5x
too strong relative to summer.

Tested capping the boost relative to each cell's own climate rate:

    stub_cap 20 (uncapped) -> **0.7327** | 3.0 -> 0.7325 | 1.5 -> 0.7323 | 0.8 -> 0.7322

**Monotonically worse. Refuted.** The weighting explains it: the same spring spike that costs
euro (-0.008, weight **0.009**) earns boas +0.051 and ceas +0.047 (combined weight **0.101**).
Suppressing it to fix a 0.9%-weight region costs an 11x larger constituency.

So euro's inverted cycle is a **knowingly accepted cost**, not a defect — the same status as
nhaf's slow decay (Entry 68). Recorded so a later tick does not rediscover it as a bug.
Reverted to `71e6e19`.

### Entry 86: Experiment 37 — additive agricultural ignition, sub-threshold and equivalent

New family per the Entry 84 plan. Unexamined assumption: **every land-use term multiplies the
climate rate**, so a field whose weather does not favour wildfire stays unburnt however much
cropland it holds. But agricultural fire is a deliberate ignition on a schedule — the farmer
burns whether or not the month would have carried a wildfire. Tested it as an additive source.

As an **addition** to the existing multiplicative stubble:

    add_k 0.00 -> 0.7327 | 0.01 -> 0.7329 | **0.02 -> 0.7330** | 0.03 -> 0.7328 |
           0.04 -> 0.7329 | 0.08 -> 0.7324 | 0.20 -> 0.7286

A flat **+0.0003 plateau** across 0.01-0.04 — below ILAMB's three-decimal resolution, so it
cannot register, and it costs a parameter.

As a **replacement** for the multiplicative form (`stub_k` = 0):

    add_k 0.05 -> 0.7325 | **0.15 -> 0.7327** | 0.30 -> 0.7269

Peaks at **exactly the multiplicative form's 0.7327**. The two formulations are *equivalent in
effect*, not competing — which is the informative result. The reason the additive framing does
not win: the cells with high cropland already have non-trivial climate rates (temperate
agriculture sits in seasonally warm, seasonally dry places), so multiplying and adding reach
the same place. The "deliberate ignition in unfavourable weather" case the additive form was
designed for barely exists in the data.

Reverted; not worth a parameter for an unmeasurable gain. The assumption was worth testing —
it was genuinely unexamined — and the answer is that the multiplicative pipeline is not
costing anything here.

### Entry 87: softmin has a real -log(n)/s offset bug — and the bug is LOAD-BEARING

Chasing `seas` (amp 0.19, phase already correct, weight 0.073), whose dry-season floor is far
too high: obs falls to **0.04** in Jul-Sep while the model holds 0.09-0.14 (obs swing 245x,
model 22x). Decomposed its factors — each swings only 2.7-3.1x, so their *product* should
compound to ~25x. Checked whether softmin was blocking that:

    pure product   swing=6.33   min=0.0606 max=0.3832
    softmin        swing=3.5e7  min=**-0.2221** max=0.0354

**The softmin returns negative values.** A soft minimum can never be below its smallest input,
so this is a genuine implementation bug. Verified in isolation:

    factors [0.9,0.9,0.9,0.9] -> true min 0.900, my softmin **+0.207**, offset **-0.693**

`-log(sum(exp(-s*x)))/s` approximates `min(x) - log(n)/s`. With n=4 and s=2 that is exactly
-log(4)/2 = -0.693. Negative rates then clip to zero, destroying the signal the term exists to
preserve. **This has been wrong since Experiment 13 (Entry 39).**

Fixed to the mean form (`.mean()` instead of `.sum()`), which removes the offset exactly:

| soft_s | overall | spatial | note |
| ------ | ------- | ------- | ---- |
| 2 (fitted for buggy form) | 0.7226 | 0.8556 | |
| 8 | 0.7306 | 0.8868 | |
| 25 | 0.7315 | 0.8900 | |
| 40 | 0.7322 | 0.8918 | |
| 60 | **0.7324** | 0.8922 | plateau |
| 100 | 0.7324 | 0.8922 | plateau |
| **buggy form (committed)** | **0.7327** | **0.8923** | |

**The corrected version converges to 0.7324 and never beats 0.7327.** As `soft_s` rises the
mean-form softmin approaches a true hard minimum, and at that limit it merely recovers what the
buggy version already achieved.

So the offset was **load-bearing**: `min(x) - log(n)/s` is a uniform downward shift on the
combined factor, which acts as a global suppression term the fitted coefficients then
compensate for. It is not "a soft minimum" — it is a soft minimum minus a constant — but the
model wants that constant.

**Kept the committed (buggy-form) model**, and left a comment in place recording that the form
is deliberate rather than an oversight. Reverting to `71e6e19`. This is worth flagging as a
limitation in any writeup: the term's name overstates what it computes.

### Entry 88: four cycle-shape experiments — peak/trough error decomposed, all refuted

Ran four variants in one turn rather than one per tick.

(a) **Floor subtraction, strict minimum** (`floor_k`): 0.0 -> 0.7327 | 0.3 -> 0.7329 |
    **0.6 -> 0.7330** | 0.9 -> 0.7329. +0.0003, sub-threshold.

(b) **Floor subtraction, 25th percentile**: 0.3 -> 0.7328 | 0.6 -> 0.7316 | 0.9 -> 0.7294.
    Worse than (a) — a low percentile over-subtracts in cells whose quiet season is broad.

(c) **Floor subtraction scaled by cell flatness**: 0.5 -> 0.7329 | 1.0 -> 0.7330. Same
    +0.0003 ceiling as (a).

All three converge on +0.0003, which said the trough was the wrong target. Decomposed the
error into peak and trough gaps to check:

| region | obs peak | mod peak | **peak gap** | obs trough | mod trough | trough gap |
| ------ | -------- | -------- | ------------ | ---------- | ---------- | ---------- |
| nhaf | 18.70 | 12.60 | **-6.11** | 0.029 | 0.492 | +0.463 |
| seas | 9.81 | 1.75 | **-8.06** | 0.041 | 0.078 | +0.037 |
| aust | 4.61 | 1.62 | **-2.99** | 0.133 | 0.227 | +0.095 |
| ceas | 3.49 | 1.17 | **-2.32** | 0.037 | 0.010 | -0.027 |
| shaf | 11.63 | 13.02 | +1.39 | 0.118 | 0.583 | +0.465 |

**Peak gaps are 3-8x larger than trough gaps in every region except shaf.** The floor work was
aimed at the smaller half of the error — worth knowing before spending more on it.

(d) **Peak amplification above the cell median** (`peak_k`): 0.0 -> **0.7327** | 0.4 -> 0.7164 |
    1.0 -> 0.6608 | 2.0 -> 0.5735. **Refuted steeply**, spatial 0.892 -> 0.528.

Same signature as the six earlier contrast mechanisms (Entries 20/28/29/37/47/72): lifting the
peaks distorts the 16-year mean map that `spatial_distribution_score` reads directly. The
amplitude/spatial tension is now confirmed from the **peak side specifically**, which no
earlier experiment had isolated — previous attempts sharpened the whole cycle at once.

Net: the peak deficit is the dominant error and is **structurally unreachable** by any
rate-space amplification. Reverted to `71e6e19`.

### Entry 89: peak deficit closed rigorously — peaks and the mean map are the same quantity

Followed Entry 88's finding (peak gaps 3-8x trough gaps) with two structural checks.

**(a) Is the transform clipping peaks?** No. Peak-month scaled rates at the 90th percentile:

    nhaf 0.042 | shaf 0.042 | seas 0.035 | aust 0.029 | ceas 0.028

Saturation of `1-exp(-r)` needs r ~ 3; these are **two orders of magnitude below** it, deep in
the linear regime where output is essentially proportional to rate. So the transform is not the
constraint — that hypothesis is dead, and `month_scale` has nothing further to give here.

**(b) Why does every peak amplification collapse spatial?** Measured it:

    corr(peak-month value, 16-year mean map), burned-area weighted = **0.9288**
    median peak/mean ratio = 3.13

**Peaks and the mean map are 93% the same quantity.** They are not separable degrees of
freedom, which is the structural reason all seven contrast mechanisms failed the same way —
`spatial_distribution_score` reads the mean map directly, so lifting the fire season redraws it.

**(c) Can the residual 7% be spent?** Tested mean-preserving peak sharpening — lift above the
median, then renormalise each cell back to its own mean, which by construction leaves the map
alone:

    pk_mn 0.0 -> **0.7327** | 0.5 -> 0.7327 | 1.0 -> 0.7325 | 2.0 -> 0.7320

Spatial is indeed protected (0.8923 -> 0.8921, essentially unmoved) — the construction worked
as designed. But **rmse falls instead** (0.5624 -> 0.5606). So the 7% of separable variance is
not where score lives either.

**The peak deficit is now closed rigorously rather than by exhaustion**: it is the dominant
error (3-8x the trough error), it is not a transform limitation, and it is not reachable in
rate space because peaks and the scored mean map are 93% collinear. Any future amplitude idea
should be checked against that collinearity first. Reverted to `71e6e19`.

### Entry 90: loop stopped by user; state summarised

User stopped the loop (cancelled cron `f334332e`). Wrote `scratchpad/STATE.md` as the
read-first handoff: progression table, the 13-component model, closed lines with their
evidence, the softmin limitation to disclose, method notes that repeatedly mattered, and
three untested leads.

Final state: **official 0.733 at `71e6e19`**, clean tree, 15 official evals, +0.047 over the
Model G baseline. Nothing in flight.

### Entry 91: resumed by user; VPD / Clausius-Clapeyron moisture demand REFUTED

Resumed after the loop was stopped. Verified `71e6e19` reproduces **0.7327** on the proxy
(bias 0.7787 rmse 0.5624 seasonal 0.8679 spatial 0.8923 amp 0.5983). Clean tree.

Checked the three `STATE.md` live leads plus the unused inputs before picking. Lead 3 (wind,
VPD) was the only genuinely untested *physics*: zero mentions of VPD / vapour / Clausius in
90 entries. Confirmed lightning and population are already refuted (Entry 42) rather than
merely unused, so they are not opportunities.

Motivation: the `temperature` factor is a **rising logistic, which saturates** - past `ign_c`
a hotter month buys nothing. Saturation vapour pressure does the opposite; by
Clausius-Clapeyron it rises exponentially, ~doubling per 10 K. Different functional form, and
the literature says atmospheric *demand* not temperature sets fine-fuel drying rate. Added
`_demand` as a 14th component: Tetens es(T) = 0.6108*exp(17.27T/(T+237.3)) over
(monthly_precipitation + vpd_supply), normalised by each cell's own long-run mean and raised
to `vpd_n`. The per-cell mean normalisation was a deliberate attempt to respect the
collinearity: it makes the term a purely *temporal* driver that leaves each cell's mean alone.

**It did not work.** Monotonic decline, no interior optimum:

| vpd_n | overall | rmse | spatial | amp_ratio | annual_pct |
| ----- | ------- | ---- | ------- | --------- | ---------- |
| 0.0 (off) | **0.7326** | 0.5625 | 0.8922 | 0.598 | 1.74 |
| 0.02 | 0.7324 | 0.5621 | 0.8924 | 0.621 | 1.77 |
| 0.05 | 0.7317 | 0.5611 | 0.8914 | 0.654 | 1.82 |
| 0.10 | 0.7287 | 0.5579 | 0.8863 | 0.710 | 1.90 |
| 0.20 | 0.7178 | 0.5472 | 0.8645 | 0.817 | 2.07 |
| 0.35 | 0.6911 | 0.5234 | 0.8089 | 0.972 | 2.36 |
| 0.50 | 0.6577 | 0.4945 | 0.7361 | 1.123 | 2.71 |

**Eighth casualty of the same collinearity, and the signature is unmistakable**: amp_ratio
rises 0.598 -> 1.123 while spatial falls 0.892 -> 0.736, monotonically, in lockstep. Note
rmse *also* falls throughout, so this is not even the usual amplitude-for-spatial trade -
overshooting amplitude past ~0.6 hurts both. The incumbent is already near the amplitude
optimum.

**Method lesson worth keeping: per-cell mean normalisation is NOT sufficient protection
against the collinearity.** I assumed dividing by each cell's own time mean would make a term
map-neutral. It does hold the *input ratio's* mean at one, but the term multiplies into a
nonlinear rate and then through `_transform`, so relative-anomaly amplification still
redistributes the output time mean (annual_pct 1.74 -> 2.71). Any future "mean-preserving"
construction must be verified on the *output* map, not argued from the input.

Two failed sub-attempts before the real test, both silent no-ops of exactly the kind Entry ~40
warns about: (1) reading `vpd_*` from module-level `PARAMS` made Optuna overrides invisible;
(2) reading from `p` raised KeyError inside the regional dicts, which carry only the original
product-form keys. Fixed by threading a `shared=` argument into `_fire_rate`. The first showed
identical scores across a whole sweep - assert your anchors.

Reverted to `71e6e19`. Official best unchanged at **0.733**.

### Entry 92: residual rmse is 77% SHAPE, and the season asymmetry is INVERTED

With amplitude closed (Entry 91: incumbent 0.598 is already near optimal, overshoot costs
rmse *and* spatial), asked what is actually left in rmse. `rmse_score` is built on the full
centered 12-month anomaly, so it sees the whole cycle shape; `seasonal_cycle_score` only reads
the peak month. Shape was therefore unexamined. Wrote `scratchpad/shape_probe.py`.

**Test 1 - how much rmse is amplitude?** Rescaled the modelled centered anomaly per cell to
*exactly* match observed amplitude (an oracle no mechanism could beat), then recomputed:

    global rmse score 0.5624 -> **0.6231**, normalised error 0.6465 -> 0.4994
    **shape_share = 0.7725**

**77% of the centered error survives perfect amplitude matching.** That is a hard ceiling on
every amplitude-only idea and retroactively explains all eight refutations - they were
competing for at most 23% of the residual, while paying for it in spatial.

**Test 2 - what is the shape defect?** Season asymmetry, defined as (months on the falling
limb) / (months on the rising limb), so >1 means rise fast, decay slow:

    global: observed **3.00**, modelled **0.50**

**The asymmetry is inverted.** Observed fire seasons rise fast and decay slowly; the model
does the opposite. Regionally the sign is wrong in shaf (obs 0.71, mod 2.00), nhsa (0.33 vs
0.71), euro (0.50 vs 5.00), boas (2.00 vs 1.40), and right in nhaf (2.00 vs 2.00), ceas
(3.00 vs 3.00), eqas, aust.

Caveat on the metric, stated plainly: as implemented it reduces to the *position of the trough
relative to the peak*, i.e. discrete limb lengths in months, not a rate. It is a phase-geometry
measure. Robust in sign, coarse in magnitude - do not over-read the numeric value.

**Best shape targets by (shape_share x weight)**: nhaf 0.772, shaf 0.720, seas 0.683,
mide 0.688. The amp-matched oracle gains there are +0.064, +0.066, +0.119, +0.073 - so shape
is worth far more than amplitude ever was in exactly the regions that carry the score.

**Physical reading and next experiment.** Fast rise / slow decay is what a *ratchet* produces:
fine fuel cures over weeks and ignites quickly, but once a landscape is cured and dead it stays
flammable after the weather turns - dead grass does not rehydrate the way live fuel does. The
existing `lag` term is symmetric, so it cannot make onset and decay differ. An asymmetric
temporal filter (fast attack, slow release on flammability) is the natural mechanism and is a
*shape* term, not an amplitude term, so it is not obviously blocked by the collinearity - but
per the Entry 91 lesson it must be verified on the OUTPUT time-mean map, not argued.

### Entry 93: asymmetric flammability hysteresis REFUTED - and it moved asymmetry BACKWARDS

Added `_hysteresis` as a 15th component: one-sided relaxation on the assembled rate, attack
instantaneous, release geometric at `hyst_rel`. `hyst_rel=0` recovers the unfiltered rate
exactly. Applied in `predict` after regional splicing and after `_lag`.

Motivation was Entry 92: observed seasons rise ~3x faster than they decay, model does the
opposite, and `_lag` is a *symmetric* blend so it structurally cannot separate onset from decay.

| hyst_rel | overall | rmse | spatial | amp_ratio | annual_pct |
| -------- | ------- | ---- | ------- | --------- | ---------- |
| 0.00 (off) | **0.7327** | 0.5624 | 0.8923 | 0.5983 | 1.748 |
| 0.10 | 0.7326 | 0.5625 | 0.8921 | 0.5939 | 1.772 |
| 0.20 | 0.7325 | 0.5622 | 0.8916 | 0.5881 | 1.801 |
| 0.30 | 0.7319 | 0.5613 | 0.8908 | 0.5803 | 1.835 |
| 0.45 | 0.7301 | 0.5586 | 0.8884 | 0.5628 | 1.903 |
| 0.60 | 0.7261 | 0.5526 | 0.8831 | 0.5338 | 2.006 |
| 0.75 | 0.7163 | 0.5401 | 0.8695 | 0.4837 | 2.188 |

Monotonic decline, no interior optimum. **But the diagnostic value is in the direction of
failure**: amp_ratio *fell* 0.598 -> 0.484 and annual_pct *inflated* 1.748 -> 2.188.

**Why: the filter can only push the rate UP.** `carried = max(current, relaxed)` is
one-sided by construction, so a longer release tail does not *redistribute* burning from the
peak into the shoulders - it *adds* burning to the shoulders and leaves the peak untouched.
Adding to the tail is the exact opposite of sharpening the rise. I attempted to correct an
asymmetry with an instrument that can only add.

**Generalisable constraint, now stated properly:** a shape correction cannot come from any
filter that only raises the rate. It must **suppress the falling limb relative to the rising
one** - the mechanism has to be able to push *down* - and must be mean-neutral on the OUTPUT
(Entry 91), not merely on its own input.

That is a narrow, physical specification, and the real-world mechanism that suppresses a
falling limb is **fuel consumption**: the landscape late in a fire season has already burnt.
Note depletion was refuted long ago at the 0.687 stage, but for a reason that no longer
applies - it was then the *only* memory term and silenced annually-reburning savannas,
destroying spatial. The model now carries `legacy` (multi-year accumulation), `neighbour`
and `curing`, so a *within-season, mean-neutral* consumption term is a materially different
proposition. That is the next and last well-motivated idea in this line.

Reverted to `71e6e19`. Official best unchanged at **0.733**.

### Entry 94: within-season fuel consumption REFUTED - cost moves to seasonal PHASE

Added `_consume` as a 15th component, built to the Entry 93 specification: an instrument that
pushes the falling limb *down* rather than holding the rate up. Accumulates recent burning with
decay `cons_a`, suppresses the rate by `1/(1 + cons_k * burnt/mean)`, blended at `cons_w`, then
**renormalises each cell back to its own unadjusted mean** so the term redistributes burning
across months instead of removing it. Applied after regional splicing, before `_legacy`.

| cons_w | overall | rmse | seasonal | spatial | amp_ratio | annual_pct |
| ------ | ------- | ---- | -------- | ------- | --------- | ---------- |
| 0.00 (off) | **0.7327** | 0.5624 | 0.8679 | 0.8923 | 0.5983 | 1.7483 |
| 0.15 | 0.7326 | 0.5627 | 0.8662 | 0.8924 | 0.5961 | 1.7463 |
| 0.30 | 0.7322 | 0.5623 | 0.8650 | 0.8926 | 0.5954 | 1.7457 |
| 0.50 | 0.7286 | 0.5596 | 0.8516 | 0.8928 | 0.5992 | 1.7487 |
| 0.75 | 0.7191 | 0.5499 | 0.8231 | 0.8932 | 0.6190 | 1.7642 |
| 1.00 | 0.7023 | 0.5261 | 0.7877 | 0.8927 | 0.6776 | 1.8134 |

`cons_a` x `cons_k` corners at `cons_w=0.3`: 0.7322-0.7324, no interior optimum.

**Three things the construction got right, and they are worth keeping as method:**
1. **Mean neutrality actually held** - annual_pct 1.7457 vs 1.7483 incumbent, verified on the
   *output* per the Entry 91 lesson rather than argued from the input. First shape term to
   manage this.
2. **Spatial was protected** - 0.8926 vs 0.8923, and it stays flat or *rises* at every
   strength. The collinearity that killed eight mechanisms was genuinely sidestepped.
3. **Amplitude moved the intended way** - 0.598 -> 0.678, unlike hysteresis which went
   backwards.

**And it still fails, because the cost moved to a metric nothing had touched yet:
`seasonal` 0.8679 -> 0.7877.** Suppressing the falling limb necessarily pulls the month of
maximum *earlier*, and the incumbent's peak timing is already right (the whole point of the
`lag` term, Entry ~14). So shape and phase are coupled in the same way amplitude and spatial
are: the model cannot make its season more asymmetric without moving its peak.

**Assessment of the shape line.** Three mechanisms tested (hysteresis, consumption, plus the
VPD attempt), each refuted by a *different* binding constraint - collinearity, additive-only
instrument, phase coupling. The 77% shape headroom from Entry 92 is real but appears
unreachable in rate space: every instrument that changes the season's shape also moves either
its mean map or its peak month, and both are already near-optimal. Recommend treating the
shape line as closed pending a genuinely non-rate-space idea, and noting that the incumbent
sits in a locally tight optimum across all four metrics simultaneously.

Reverted to `71e6e19`. Official best unchanged at **0.733**. 15 official evals, tree clean.

### Entry 95: ALL 20 inputs audited; ceiling attributed to INPUTS, not model form

User asked for an exhaustive input sweep plus attribution for the plateau. Two probes.

**(a) `scratchpad/input_audit.py` - all 20 inputs vs the residual.** Note the count: **20**
`time-lat-lon` variables exist, not 18 (climate 4, ed 8, lightning 1, luh2 6, population 1).
Model loads 7.

Statistics are reference-burned-area weighted and computed on the **12-month climatology**,
because the scorer collapses 192 months before every metric so interannual signal must not be
credited. Decisive column is `free_resid = |corr_resid| * sqrt(1 - span_r2)` - residual
correlation **discounted by how much of the input the 7 loaded inputs already reconstruct**.
Raw correlation is a mirage: a predictor the model can synthesise internally adds nothing.

| unused | corr_obs | corr_resid | span_r2 | free_resid |
| ------ | -------- | ---------- | ------- | ---------- |
| lightning_flash_rate | **-0.311** | 0.110 | 0.410 | **0.085** |
| secondary_canopy_height | -0.002 | 0.085 | 0.247 | 0.074 |
| natural_canopy_height | -0.113 | 0.083 | 0.267 | 0.071 |
| aboveground_biomass | -0.108 | 0.079 | 0.294 | 0.067 |
| soil_carbon | -0.086 | 0.087 | 0.451 | 0.065 |
| leaf_area_index | -0.055 | 0.087 | 0.572 | 0.057 |
| luh2_primary_fraction | 0.061 | 0.035 | 0.464 | 0.025 |
| population_density | -0.059 | 0.003 | 0.223 | **0.002** |

Lightning is the strongest single predictor of burned area in the whole collection, yet 41% of
it is already spanned and the free part correlates 0.085 with the residual. Population is
effectively null. **Upgrades Entry 42 from "weak" to "redundant".** Most `corr_obs` are
negative simply because burned area peaks in semi-arid savanna, so nearly every vegetation and
moisture field anti-correlates with it - not evidence of suppression.

**(b) `scratchpad/ceiling_probe.py` - what does the scorer even allow?**

| reference | overall | bias | rmse | seasonal | spatial |
| --------- | ------- | ---- | ---- | -------- | ------- |
| climatology oracle (obs as prediction) | **0.8737** | 0.8934 | 0.7662 | 0.9589 | 0.9838 |
| current model | 0.7327 | 0.7787 | 0.5624 | 0.8679 | 0.8923 |
| persistence (perfect map, no season) | **0.6597** | 0.8933 | 0.4665 | 0.4889 | 0.9836 |
| **linear fit, all 20 inputs + lat + harmonics + means** | **0.6446** | 0.7440 | 0.4870 | 0.8241 | 0.6807 |

Three things fall out, and they settle the attribution:

1. **Perfect information scores 0.874, not 1.0.** Interannual variability alone costs 0.126.
   No model of any kind can exceed ~0.87 under this scorer.
2. **No-seasonality already scores 0.660.** So the span available to seasonal fire physics is
   0.660 -> 0.874, width **0.214**. We hold 0.073 of it; headroom is **0.141**, not 0.267.
   The 0.686 -> 0.733 progression is a *third* of the reachable range, not 5%.
3. **An unconstrained linear fit on all 20 inputs scores 0.6446 - 0.088 BELOW the mechanistic
   model, and worse on every component metric** (spatial 0.681 vs 0.892). The 13 hand-built
   components beat a freer function of strictly more data. **Model form is not the constraint.**

**Also newly noticed and worth flagging: bias 0.779 and spatial 0.892 are both WORSE than
persistence's 0.893 and 0.984.** Adding seasonality has *cost* bias and spatial to buy rmse and
phase. Two of four metrics sit below a trivial baseline - unremarked in 94 entries.

**Attribution: the plateau is an INPUT-INFORMATION limit.** Written up in
`scratchpad/MISSING_INPUTS.md`. Missing physics, ranked:

1. **Wind speed** - spread rate goes ~quadratically with it in every operational model
   (Rothermel, McArthur, Canadian FBP). Nothing in `inputs/` proxies it and it is not derivable.
   The model has fuel, moisture and implicit ignition but **no spread driver at all**. In CRUJRA.
2. **Relative humidity / dewpoint** - needed for *real* VPD. **Important correction to Entry 91:**
   that test used `es(T)` alone, which is a deterministic monotone transform of a field the model
   already holds, so it added no information. Real VPD is `es(T) - ea` and `ea` needs humidity.
   **Entry 91 refuted temperature-only VPD, NOT VPD.** In CRUJRA.
3. **Sub-monthly statistics** - dry-day count, max dry spell, daily Tmax. 40 mm as two storms
   leaves three dry weeks; 40 mm spread evenly is unburnable. Monthly means erase this and it is
   probably a hard floor on seasonal-shape skill.
4. **Time-varying ignition** - `lightning.nc` is a monthly climatology repeated every year and
   `population_density` is a census epoch repeated monthly, so both are nearly static by
   construction. Their measured redundancy is partly an artefact of that.

### Entry 96: new inputs installed; dry-spell/VPD REFUTED; wind WORKS as an interaction

User added `vapor_pressure_deficit_mean`, `wind_speed_mean`, `wet_day_fraction`,
`maximum_consecutive_dry_days` to `climate.nc`. Integrity verified: unmodified model still
scores exactly **0.7327**, so the four original climate fields survived bit-for-bit. Input set
is now **24** variables.

**(a) Audit of the four new fields** (`input_audit.py`, redundancy-discounted `free_resid`):

| new input | corr_obs | corr_resid | span_r2 | free_resid |
| --------- | -------- | ---------- | ------- | ---------- |
| maximum_consecutive_dry_days | **+0.572** | -0.188 | 0.655 | **0.1105** |
| vapor_pressure_deficit_mean | +0.385 | -0.166 | 0.678 | **0.0939** |
| wet_day_fraction | -0.478 | 0.155 | 0.767 | 0.0748 |
| wind_speed_mean | **+0.014** | -0.004 | 0.342 | **0.0028** |

Dry-spell length is the strongest *positive* correlate of burned area in the entire collection
and the highest free_resid ever measured here. Wind is **23rd of 24**, below population density.
On that basis I predicted dry-spell/VPD would work and wind would not. **Both predictions were
wrong**, which makes this the most instructive entry in a while.

**(b) Dry-spell + true VPD drying: REFUTED, both formulations.**

Naive (net damper): 0.7327 -> 0.7320 -> 0.7261 -> 0.7152 -> 0.6997 -> **0.6802** as `dry_w`
rises. It also removed 44% of all burning (annual 1.748 -> 0.977), so the loss could have been
a level artefact. Rebuilt it globally level-preserving to isolate *pattern* content:
0.7242 -> 0.6622 -> 0.5299 -> **0.3877**, spatial collapsing 0.892 -> **0.173**. So the pattern
itself is wrong, not merely the level. Reverted.

**High free_resid did not predict usefulness.** Seventh instance of the STATE.md note that
aggregate residual correlations fail at cell level.

**(c) Wind: tested in SIX functional roles rather than one.** Rationale: `free_resid` is a
*linear pointwise* statistic and a spread driver is neither. Wind does not decide whether a cell
burns, only how fast an existing front runs, so its effect is *conditional on fuel state* -
structurally invisible to pointwise correlation. STATE.md's own note applies: a refuted input is
not a refuted mechanism (rangeland failed as intensity, succeeded as timing).

| role | best overall | note |
| ---- | ------------ | ---- |
| A pointwise multiplicative | 0.7327 | flat then harmful - exactly as the audit predicted |
| B quadratic ~u^2 | 0.7326 | the literature form is *wrong* at monthly resolution |
| E per-cell anomaly | 0.7328 | ~neutral, but spatial *rises* to 0.8952 |
| **F wind x dryness** | **0.7342** | **the only role that gains** |
| C spread_crit / spread_gain | 0.7327 | incumbent already optimal, no wind headroom there |
| D nb_w coupling | 0.7331 | model wants slightly more coupling (0.35 -> 0.5) |

**(d) Implemented as `_gust`, and one real bug caught.** First implementation normalised by
`factor.mean()` to hold the annual total. That only holds when factor and rate are
*uncorrelated*, and here the gate **is** dryness - exactly where the rate is already high. It
systematically over-suppressed (annual 1.748 -> 1.622) and scored 0.7330. Normalising on the
**product** instead recovered the gain. Nearly filed a refutation off the broken version; add to
method notes.

Also moved the term from the rate to the *final prediction*: applied pre-transform it competes
with `_spread`'s own normalisation.

Swept `gust_w` 0.05-0.6, `gust_ref` 2-6, `gust_cap` 1.5-5. Optimum broad and flat around
`gust_w` 0.18-0.25, `gust_ref` 3.0, `gust_cap` 5.0. Chose **0.22 / 3.0 / 5.0**:
**overall 0.7340, spatial 0.8947** (highest all session), annual_pct 1.7421 (vs 1.7483 - level
held), amp 0.5964 (unchanged, so this is *not* an amplitude mechanism and does not touch the
collinearity).

**Isolated regional effect** (`gust_regional.py`, gust on vs off, everything else fixed):
**10/14 regions improve.** boas +0.0103, ceas +0.0118, tena +0.0061, aust +0.0051, ceam +0.0026,
nhaf +0.0022, shaf +0.0021, eqas +0.0022, euro +0.0013, bona +0.0001. Losers: seas -0.0172,
shsa -0.0080, nhsa -0.0035, mide -0.0017. Good spread, and notably it helps the two
*extratropical* regions most, which is where synoptic wind should matter.

Committed `63c80bc`, evaluating.

### Entry 97: proper backward triage; quench REFUTED but it settles the attribution

User challenged the input-limitation attribution. Correctly: it rested on an unconstrained
*linear* fit scoring 0.6446, below the model, which says little because least squares is a poor
model class for a threshold-driven multiplicative heavy-tailed process, and the flexible-learner
bound that would have settled it was OOM-killed. So triage backwards from the score instead.

**(a) Half-swap decomposition** (`loss_decomp.py`). Every metric derives from just two objects:
the time-mean map and the normalised cycle shape. Factor both model and obs, score all four:

| combination | overall | bias | rmse | seas | spat |
| ----------- | ------- | ---- | ---- | ---- | ---- |
| model mean x model cycle | 0.7340 | 0.7785 | 0.5646 | 0.8679 | 0.8947 |
| **OBS** mean x model cycle | 0.7872 | 0.8924 | 0.5964 | 0.8679 | 0.9833 |
| model mean x **OBS** cycle | 0.7916 | 0.7788 | 0.6634 | 0.9563 | 0.8958 |
| OBS mean x OBS cycle | 0.8737 | 0.8934 | 0.7662 | 0.9589 | 0.9838 |

Perfect map buys +0.0532, perfect cycle +0.0575 - near-equal (0.48 share) and **superadditive**
(+0.1397 together vs +0.1107 summed). Neither half is "the" problem.

**(b) Per-region weighted shortfall.** The gap is *concentrated*: **nhaf 0.0328 + shaf 0.0319 =
0.0647, 46% of the 0.1397 headroom**, from 3% of land area each, because bias/rmse are
reference-burned-area weighted and those regions hold 31%/30% of global burned area. An
information ceiling spreads loss in proportion to weight; a 2x concentration in two regions
sharing one signature is a broken mechanism.

**(c) `africa_probe.py` localised it exactly - a DRY-SEASON LEAK, not a peak deficit:**

    nhaf obs  Jul 0.040  Aug 0.029  ... Dec 18.704
    nhaf mod  Jul 0.635  Aug 0.504  ... Dec  9.828

15-20x over-prediction in the months that should be fire-free, peak only half of observed.
Grant-amplitude recovers rmse 0.545 -> 0.607; grant-phase only -> 0.563, so amplitude beats
phase 3:1. Error is *diffuse within* the region (worst 10% of cells hold 52% in nhaf, 42% in
shaf, vs 78% in ceas), so it is systematic to the savanna formulation, not a few bad cells.

Diagnosis: **nothing in the model turns fire OFF.** Every component is a smooth multiplicative
factor, so a wet-season savanna still gets a small nonzero rate, and 12 months of leak costs
more than the peak deficit. A missing *threshold*, not a missing input.

**(d) `_quench`: the mechanism WORKS and the score does not care.** Sub-threshold rates ramped
to zero relative to each cell's own seasonal max, renormalised on the product.

    nhaf mod after quench  Jul 0.182  Aug 0.115  Dec 10.595

**Three quarters of the leak closed.** Amplitude 0.596 -> 0.640, peak 9.83 -> 10.60. And:

| quench_f | overall | rmse | spatial | amp |
| -------- | ------- | ---- | ------- | --- |
| 0.00 | **0.7340** | 0.5646 | 0.8947 | 0.596 |
| 0.03 | 0.7329 | 0.5638 | 0.8932 | 0.640 |
| 0.10 | 0.7288 | 0.5540 | 0.8951 | 0.740 |
| 0.25 | 0.7119 | 0.5219 | 0.8936 | 0.911 |

Monotonic decline. Isolated: **4/14 regions improve**, nhaf only +0.0005 despite the leak
closing, and boas -0.0246 / ceas -0.0219 / euro -0.0204 pay for it.

**This is the resolution.** `bias_score` and `rmse_score` are averaged with weight
`area * reference_mean` - **weighted by OBSERVED burned area**. The fire-free months have
reference_mean ~ 0, so they carry ~no weight in the very metrics the leak degrades. A large,
real, physically-correct error can be fully corrected with **no score response**.

So my Entry 96 concentration statistic was measuring *unweighted* error and mis-attributed.
The honest statement: the model's largest *physical* error is the African dry-season leak, and
the scorer is close to blind to it. That is neither an input limit nor a model-form limit - it
is a **metric-structure limit**, and it explains the plateau better than either.

Reverted to `63c80bc`. Official best **0.734**.

### Entry 98: joint retune -> official 0.735, new best

Two leads from measurements already taken, neither installed: the wind-role sweep found `nb_w`
prefers 0.5-0.7 over the committed 0.35, and `gust_*` was committed from a coarse grid.

Coordinate sweep (`retune_sweep.py`), one coefficient at a time around the incumbent, reporting
all four metrics plus annual_pct so a level shift cannot pass as a pattern gain:

| coefficient | incumbent | best | delta |
| ----------- | --------- | ---- | ----- |
| `nb_w` | 0.35 | **0.55** | +0.0009 |
| `gust_ref` | 3.0 | **3.5** | +0.0004 |
| `lag_w` | 0.2133 | **0.16** | +0.0001 |
| `gust_w` | 0.22 | 0.22 | already optimal |
| `gust_cap` | 5.0 | 5.0 | already optimal |
| `spread_gain` | 9.3588 | 9.3588 | already optimal |
| `soft_s` | 2.0 | 2.0 | already optimal |

Per the STATE.md note that sweep gains do not sum, verified jointly *and* leave-one-out:

    all three together        0.7353
      without nb_w            0.7344   (-0.0009, carries the gain)
      without gust_ref        0.7351   (-0.0002)
      without lag_w           0.7351   (-0.0002)

All three earn their place; `nb_w` dominates. Joint 0.7353 exceeds 0.7340 + the individual
deltas, so mildly superadditive.

Installed, proxy **0.7353** (bias 0.7801 rmse 0.5657 seasonal 0.8679 spatial **0.8969**,
annual_pct 1.7246). Committed `4c682b7`, evaluated:

**official overall=0.735 bias=0.780 rmse=0.566 seasonal=0.868 spatial=0.897 — NEW BEST.**
Proxy matched official to three decimals for the 15th consecutive time.

**Interpretation, and it is the interesting part:** this gain is not new physics. It is the
model re-finding its own optimum *after* the wind term changed what the optimum was. `nb_w`
being the dominant contributor is physically coherent - a wind-driven front reaches further than
a still-air one, so admitting wind raised the optimal spatial extent over which flammability is
shared between cells. Worth remembering as a procedure: after adding a term that changes spread,
re-sweep the spread and coupling coefficients rather than assuming they are still tuned.

Regional spread vs the frozen Model G baseline: 6/14 improved, and the gains sit where the
burned-area weighting puts the score - shaf +0.0667, mide +0.1023, eqas +0.0995, nhaf +0.0601.

Eval outputs saved to `scratchpad/eval_retune.txt` and `scratchpad/eval_gust.txt`.
Progression: 0.686 -> 0.698 -> ... -> 0.733 -> 0.734 (`63c80bc`) -> **0.735** (`4c682b7`).

### Entry 77: picked up from another agent at 0.735; VPD added (proxy 0.7365, 14/14 regions)

Another agent advanced the model 0.728 -> **0.735** while I was paused: stubble burning (0.730,
the lead I queued in Entry 76), rangeland burning gated on a hot-dry window (0.733 — a fair
correction to my Entry 42 refutation, which tested it only as a plain multiplier), a wind x
dryness spread interaction (0.734), and a joint retune (0.735). It also added the inputs I had
recommended when asked: **VPD, wind speed, wet-day fraction, max consecutive dry days**, via
`scripts/install_inputs.py`. Only wind was wired in; three remained unused.

Weighted residual screen of the four new inputs on the 0.735 model:

| predictor | wr(resid) | wr(obs) |
| --------- | --------- | ------- |
| **vapor_pressure_deficit_mean** | **-0.248** | +0.345 |
| wind_speed_mean | +0.134 | -0.296 |
| maximum_consecutive_dry_days | -0.058 | +0.211 |
| wet_day_fraction | +0.027 | -0.236 |

VPD is the strongest residual signal of the whole thread, and **negative** — the model
underpredicts where VPD is high, which is the helpful direction given high-burn cells are
underpredicted (Entry 69). Its phase also beats dryness: shaf peak m9 vs obs m8 (dryness m10),
seas m4 vs obs m3, shsa m9 vs obs m9.

**Near-miss worth recording.** First tests gave 0.682-0.699, apparently refuting it. Cause:
VPD has **median 0.000 and p90 0.83**, but I had set `vpd_half` to 8.0 with a 0.1-40 range —
entirely above the data, so the term was saturated off. Same miscalibration class as Entry 70.
Rescaled to [0.01, 3.0]:

| vpd_half | overall | spatial |
| -------- | ------- | ------- |
| 0.05 | 0.7358 | 0.8989 |
| **0.10-0.15** | **0.7361** | **0.9003** |
| 0.40 | 0.7336 | 0.8974 |
| 1.00 | 0.7237 | 0.8833 |

Then `vpd_n` 1.0 -> 0.35 gave **0.7365**; `vpd_cap` inert across 2-8.

**14/14 regions improved — the first experiment in this thread to improve every region**
(ceas +0.019, nhsa +0.008, tena +0.006, euro +0.005, nhaf +0.005). Spatial reaches **0.9009**,
crossing 0.90 for the first time. Contract check passes. Committed `dfaccce`; eval running.

### Entry 78: Experiment 35 OFFICIAL — VPD recorded at 0.737; dry-spell REFUTED

`ar evaluate` on `dfaccce`: **official Overall 0.737** (bias 0.781, rmse 0.566, seasonal 0.868,
spatial **0.901**). Proxy said 0.7365 — **twelfth** consecutive three-decimal match. New best,
and the first time spatial has crossed 0.90.

**Cumulative: 0.686 -> 0.737 (+0.051) across 23 recorded evals; null model = 0.407.**
Largest single gain remains the transform ceiling fix (+0.014, Entry 26).

Then tested the last unused input, `maximum_consecutive_dry_days`. Motivation was a weakness
documented since Entry 2: monthly rainfall totals conflate 30 mm in one storm with 1 mm daily,
which are opposite fire regimes. Checked calibration first (range 0-31 days, swing 4-6x, phase
within 1-2 months of observed) so this was not another scale miss.

| dry_half | overall | spatial |
| -------- | ------- | ------- |
| 1.0 | 0.7323 | 0.8924 |
| 1.5 | 0.7288 | 0.8849 |
| 3.0 | 0.7175 | 0.8601 |
| 8.0 | 0.6885 | 0.7938 |
| **off (baseline)** | **0.7365** | **0.9009** |

**Refuted** — monotone toward baseline as the term switches itself off, never exceeding it.
Despite a plausible mechanism and correct calibration, the dry-spell signal is **redundant**:
VPD and dryness already encode within-month moisture availability, and VPD does it better
(residual -0.248 vs -0.058). Reverted to `dfaccce`.

All four new inputs are now assessed: **VPD adopted (+0.002 official), wind adopted by the
other agent (+0.001), dry-spell and wet-day fraction both refuted** (wr(resid) -0.058 and
+0.027, both near zero, and dry-spell confirmed harmful when tested directly).


### Entry 79: resumed at official 0.737 — current triage plan

User asked to continue autoresearch under `research.md`. Re-read the contract and established the
current proxy baseline: **0.7365** (bias 0.7813, rmse 0.5662, seasonal 0.8679, spatial 0.9009),
matching official 0.737. Regenerated and inspected the current map and seasonal-cycle figures.
No model edit or official evaluation yet.

Immediate triage before spending an evaluation:

1. Run practical current-operating-point drop-one/contract checks to identify a slot under the
   15-component limit. The last exact 4096-subset ablation predates pasture, gust, and VPD; a new
   exact ablation would be 32768 subsets, so drop-one comes first.
2. Revisit coefficient interactions after VPD, especially VPD × neighbour/spread/gust/lag. The
   preceding wind experiment moved the neighbour optimum, so the VPD addition may likewise have
   moved existing optima.
3. Rank genuinely new physical families against current weighted regional/metric headroom. Avoid
   reopening dry-spell/wet-day, plain lightning/population/canopy multipliers, global sharpening,
   wider/directional neighbours, or quenching unless a new diagnostic distinguishes the role.
4. Use proxy diagnostics and Optuna before any commit/evaluation. Only a committed, contract-clean
   candidate with concrete evidence will receive an official evaluation.


### Entry 80: current 15-component drop-one — stubble is the replacement slot

At official 0.737 / proxy 0.736501, the contract check passed: shape/range/finite are valid and
all 15 declared components change the prediction. A full exact ablation would require 32768
subsets, so ran full + every fixed-PARAMS leave-one-out instead. Drop costs (proxy Overall):

    stubble -0.00046    softmin -0.00093    VPD -0.00123    lag -0.00172
    gust    -0.00201    pasture -0.00252    legacy -0.00279
    cropland -0.00673   neighbour -0.01238  temperature -0.01402
    dryness -0.02017    precipitation -0.02450  fuel -0.02638
    spread  -0.03505    curing -0.06558

None is inert. **Stubble is the first slot to replace** under the 15-component cap: its cost is
about half the next weakest component. This does not justify deleting it without a replacement;
it sets the hurdle for a new physical family. Started a read-only focused post-VPD coordinate
sweep of VPD, neighbour, spread, lag, gust, softmin, and transform interactions. No official eval.


### Entry 81: commit policy clarified by user

User requires **non-improving model experiments to be committed too**, so failed directions remain
exactly recoverable and can be reopened later. Going forward, every distinct `model.py` experiment
that reaches a proxy test will receive its own concise commit whether it improves or not. If rejected,
restore the objective-best model in a subsequent explicit commit rather than erasing the candidate
from history. Stage only `autoresearch/model.py`; do not include the pre-existing dirty/untracked
inputs, results, progress image, scripts, logs, or scratchpad in model commits. Official evaluations
still require a committed model and will only be used with scientific justification.


### Entry 82: duration-aware monthly hazard — committed proxy 0.7377

Implementation audit found a clean physical artifact: `_transform` treated February and a 31-day
month as equal hazard exposure even though the runtime/scorer uses true time bounds. Installed exact
calendar-day scaling for Jan-2001..Dec-2016, normalized to the mean month. This changes no component
count and has no fitted mechanism coefficient.

Contract passed. Proxy moved **0.7365 -> 0.7377**: bias 0.7809, rmse 0.5671, seasonal **0.8727**,
spatial 0.9007. The +0.0012 gain is mainly RMSE/seasonal, with spatial essentially preserved.
Committed the standalone mechanism as `8c3ad39` (`Scale monthly fire hazard by calendar duration`)
before tuning, preserving the exact experiment regardless of later outcome.

Started required Optuna on a focused seven-parameter interaction space: VPD half/exponent, spread
gain, month scale, lag, neighbour weight, and gust reference. No official evaluation yet.


### Entry 83: duration-aware interaction retune — committed proxy 0.7381; official running

Focused Optuna stopped after 32/80 trials with a rounded 0.738 winner. Installed its seven
coefficients and committed `ab05ea2` (`Retune duration-aware fire weather interactions`). Exact
proxy **0.738063**: bias 0.7824, rmse 0.5682, seasonal 0.8711, spatial 0.9004. This is +0.00035
over the untuned duration correction and +0.00156 over pre-duration 0.73650. Contract passed.

Relative to untuned duration, 7/14 regions improve: shaf +.00308, ceas +.00221, boas +.00169,
tena +.00050, bona +.00047, aust +.00038, nhaf +.00016. Losses include nhsa -.0145, eqas
-.0106, ceam -.0104, so breadth is mixed, not VPD-like. The global gain is nevertheless not
one-core-only and the physical mechanism is clean. Started one official evaluation with concrete
calendar-duration + joint-retune description.

In parallel, opened an architecture-reset track aimed at 0.8: separate annual propensity/map from
seasonal allocation and replace the single universal pathway with observable soft fire-regime
mixtures. This follows the hybrid evidence that a perfect current-resolution map with model cycle
and model map with perfect cycle each already approach 0.79; the current single-rate architecture
couples those heads and blocks joint progress.


### Entry 84: official 0.738; APSA scaffold and dry-spell seasonal head

`ab05ea2` evaluated officially at **0.738** (bias .782, rmse .568, seasonal .871, spatial .900),
matching proxy and becoming the objective best.

Architecture reset stage 1: added an annual-propensity / seasonal-allocation (APSA) closure around
the incumbent. It factorises monthly output into annual burned area and within-year Poisson-hazard
allocation, then reconstructs through a saturating Newton-solved closure. The scaffold reproduced
the parent prediction within max abs 1.8e-7, kept proxy 0.7381, passed all component checks, and was
committed independently as `f3087a5`.

A naive mean-preserving power sharpen was negative at every tested setting, confirming waveform/
regime timing rather than simple amplitude is missing. Driver screens found a new role for maximum
continuous dry days: not a direct moisture multiplier (previously refuted), but a seasonal fire-
opportunity allocator with annual burned area held fixed. A physical per-year APSA implementation
using `(1 + MCDD/scale)^w`, scale 30 days and w .8, scored **0.7387** proxy: bias/spatial unchanged,
seasonal ~.8712 -> .8751, RMSE slightly down .5682 -> .5678. Contract passed. Committed even before
tuning as `bf3f88f`. Focused five-parameter Optuna is running. No official evaluation spent on it.


### Entry 85: APSA heads advance to 0.7405 proxy

Seasonal-head Optuna stopped at 60/100 trials with rounded 0.739. Installed its exact MCDD
opportunity/VPD/lag coefficients and committed `2c67143`; exact proxy **0.73893** (bias .7826,
rmse .5678, seasonal .8755, spatial .9009).

While that ran, screened independent annual fire-window heads with global area-weighted annual
burning held fixed. Persistent extreme VPD was strongest: annual mean of
`(VPD/(VPD+0.8))^2`, applied as a trust-region capacity factor with exponent .3, reached 0.7407
in memory. Installed on the tuned seasonal head: proxy **0.74051** (bias .7849, rmse .5697,
seasonal .8749, spatial .9034), +.00158 over the seasonal parent. Contract passed and committed
as `098123d`. 9/14 regions improve (AUST, CEAM, EQAS, MIDE, NHAF, NHSA, SEAS, SHAF, SHSA);
notable losses BOAS -.0308, CEAS -.0223, EURO -.0093 require attention during tuning.

Focused annual-head Optuna is running. In parallel, a streaming memory-safe annual structure
screen is testing continuity, open/woody fuels, secondary conversion, human interface, lightning,
and organic fuels as separate propensity roles. No official evaluation yet.


### Entry 86: capacity tuning, intact-biomass brake, and 8-worker search

Annual VPD-head Optuna stopped at 44 trials with a rounded 0.741 winner. Installed and committed
`fb3f0a5`; exact proxy **0.7410** (bias .7846, rmse .5709, seasonal .8740, spatial .9047).

The parallel structure screen found an annual role for intact primary biomass: brake capacity by
`1 - primary * AGB/(AGB+half)` rather than applying primary forest as the previously failed monthly
multiplier. On the untuned parent it added +.00060 and improved 10/14 regions. On the new Optuna
point the fixed candidate is only a small gain (~0.7411), but contract passes; committed regardless
as `8eb5737`.

A separate older-checkpoint diagnostic falsified the combined wet-day x MCDD event-duty family: it
was monotone only in NHAF/CEAS and exact annual-preserving transforms were nonpositive. This closes
the composite duty formulation, not the newer MCDD-only APSA timing role, but argues against an
official evaluation of that small term alone.

User authorized 8 Optuna workers. Stopped the 1-worker intact search after 24 trials and restarted
a 240-trial / 80-patience search with `--workers 8 --seed 101`. In parallel, screening independent
seasonal driver allocation roles.


### Entry 87: annual-scale calibration, independent Optuna collaboration, seasonal rise lead

An explicit APSA annual intercept exposed remaining level error. Sweeping annual propensity scale
found 0.92 on the pre-8-worker point (0.74113 -> 0.74254); installed/contracted/committed `fe26a15`.
The 8-worker interaction winner combined with scale .92 was slightly worse (0.74237) but was still
committed per policy as `15b0488`. Re-sweeping its annual scale found .95 and proxy **0.742648**
(bias .78658, rmse .57254, seasonal .87289, spatial .90869); committed `8ba2cd5`.

A fully physical per-year APSA screen (annual area conserved; no climatology repetition) found an
additional seasonal direction: allocate toward months with rising VPD and slightly away from the
incumbent rain-off factor. On `8ba2cd5`, `vpd_rise_w=.5` and `rain_off_w=-.1` gives **0.743451**,
mainly through RMSE .57254 -> .57479 with bias/spatial fixed. Not yet installed because four
coordinated read-only Optuna studies must all load the exact 8ba2cd5 lineage first.

Per user direction, launched independent single-worker Optuna seeds 103/107/109/113 through four
collaborating subagents. Each follows research.md, makes no edits/evaluations, shares winners with
siblings, and reports to the coordinator. The coordinator will apply, contract-check, and commit
every distinct tested winner sequentially; official evaluations remain serialized to avoid races
in results.tsv/progress.png.


### Entry 88: backward ceiling triage identifies a recoverable seasonal-shape pathway

The current committed model reproduces proxy Overall 0.743044. A memory-bounded gradient-boosted
diagnostic was fit using all 24 permitted inputs plus the incumbent prediction, explicitly excluding
latitude, longitude, region labels, cell IDs, and geographic masks. It is diagnostic only and will
never be submitted as `model.py`.

The first absolute-output learner scored 0.6655 because spatial fell to 0.1443, but its temporal
metrics were strong: bias 0.8639, rmse 0.6946, seasonal 0.9299. Combining its learned cycle shape
with the incumbent annual map scored 0.7812. A second learner targeted only the normalised twelve-
month allocation while conserving the incumbent map exactly and scored **0.7940**: bias 0.7866,
rmse 0.6628, seasonal 0.9489, spatial 0.9091. This proves that a globally shared function of the
permitted observables contains most of the missing seasonal score without any new regions.

Permutation triage ranks incumbent allocation first, followed by VPD anomaly and absolute VPD,
temperature, antecedent wet-day fraction, precipitation, and dry-spell anomaly. Interpretation:
the missing function is a conditional correction to the existing fire window, not a replacement.

Experiment `0d6beb1` tested an independent fuel-supported fire-weather allocation built from
antecedent wetness, positive VPD anomaly, and dry-spell anomaly. It conserved the annual map but
fell to 0.7375 (rmse 0.5700, seasonal 0.8521, amplitude ratio 0.5463). The independent window was
flatter and phase-worse than the incumbent. Next experiment keeps the same physical signals but
uses them multiplicatively to calibrate the incumbent allocation, matching the learner evidence.

The multiplicative follow-up `a818c0e` also failed, scoring 0.7388. It raised amplitude ratio
0.6297 -> 0.6670 but reduced rmse 0.5736 -> 0.5711 and seasonal 0.8728 -> 0.8566. Therefore the
learned correction is not monotone in VPD/dryness: its value must come from conditional thresholds
or sign changes. Both formulations are retained in Git history; the incumbent is restored before
extracting shallow, inspectable rules from the diagnostic learner.

A 64-leaf globally shared diagnostic tree (no geographic features) reaches 0.7559; 256 leaves
reach 0.7663. Its root split is incumbent allocation, followed by VPD anomaly, confirming that the
missing function is a conditional allocation calibration. The literature-supported signed
antecedent-wetness mechanism `ddbbdc4` nevertheless scores only 0.7405: rmse 0.5707, seasonal
0.8659, amplitude 0.6589. At monthly climatological resolution, the sign-reversing fuel legacy is
not the leading correction.

A reference-burn-weighted residual table over incumbent allocation x VPD anomaly reveals the
actionable structure. Low-allocation months with strongly positive VPD anomaly are underallocated
2-13x; low-VPD months are generally overallocated; months already above about 0.28 allocation are
overconcentrated across every VPD bin. The next minimal mechanism will blend the incumbent with a
pure local VPD-threshold allocation, testing shoulder broadening and peak capping without the
confounding wetness and dry-spell terms that made `0d6beb1` fail.

The pure standardised-VPD shoulder experiment `340ccd3` scores 0.7391 (rmse 0.5702,
seasonal 0.8600, amplitude 0.5602). The binned residual relationship is descriptive but not a
globally causal correction: reallocating every cell toward high-VPD shoulders moves many local
peaks the wrong way. This closes one-dimensional seasonal transforms. The successful learner's
calendar harmonics and static climate/land features imply that local fuel phenology conditions the
driver response; the next diagnostic is an interpretable conditional-harmonic softmax rather than
another universal multiplier.

The annual-map learner is independently promising: with the incumbent cycle fixed, a 75% blend
toward its no-geography map reaches 0.7579, improving bias 0.7866 -> 0.8295 and spatial 0.9087 ->
0.9384. Its dominant features are incumbent mean, lightning climatology, precipitation seasonal
variance, annual precipitation, cold-season temperature, VPD variance, biomass, and wind.

The map residual binned only by incumbent annual mean shows severe over-dispersion, motivating a
sublinear fire-footprint-overlap closure. Experiment `9f2199f` tested that untargeted compression
while preserving each year's global area. It failed at 0.7302: bias 0.7754, rmse 0.5651, spatial
0.8718. The marginal correction is real but the low-propensity cells are not exchangeable; adding
area to the wrong cells destroys the map. The next annual mechanism uses lightning as the ignition
selector and precipitation/GPP as fuel support, a distinct role from the earlier failed monthly
lightning multiplier.

Experiment `c91cc73` tested fuel-supported lightning as an annual ignition floor and conserved
global area. It is correctly signed but sub-threshold at 0.7433: bias rises 0.7866 -> 0.7883 while
rmse and spatial slip slightly. Lightning is therefore not an independent missing mechanism; the
annual learner must be using it conditionally with precipitation seasonality and fuel structure.
No coefficient sweep is justified. The next diagnostic extracts a shallow annual tree to identify
those thresholds before another model experiment.

The shallow annual tree confirms strong conditionality: 32 leaves with a 50% blend reach 0.7510;
64 leaves with 75% blend reach 0.7584; 128 leaves reach 0.7627. The recurring hierarchy is
incumbent annual propensity, precipitation-season amplitude near 50 mm/month, temperature range,
canopy/fuel structure, then lightning and dry-spell opportunity.

Experiment `4db79e2` distilled the leading hierarchy into a continuous wet-dry fuel pump gated by
open productive fuel and lightning. It remains sub-threshold at 0.7433: bias +0.0018 is offset by
small rmse/spatial losses. As with lightning alone, the process is real but cannot supply the tree's
multivariate map correction by itself. The next step reduces the already interpretable conditional
GLM to the smallest set of named terms that retains a genuine score jump.


### Entry 89: first untuned step change is official; annual propensity becomes leading head

The transparent conditional seasonal allocation was installed as experiment `616548a` before
measurement. It is a globally shared named additive equation of incumbent monthly opportunity,
local weather anomalies and lags, continuous fuel/land-state gates, and calendar harmonics. It
uses no new regions, coordinates, cell identifiers, or geographic masks. The exact proxy rose
from 0.7430 to **0.7510** through rmse 0.5736 -> 0.5856 and seasonal 0.8728 -> 0.8886 while
conserving the incumbent annual map. Eleven of fourteen regions improved; the three regressions
were BONA -0.0092, SHAF -0.0015, and AUST -0.0035. A direct strength sweep peaked at the untuned
1.0 coefficient, so the jump is structural rather than a fragile scalar optimum.

The official evaluation exactly confirmed **Overall 0.751** (bias 0.787, rmse 0.586, seasonal
0.889, spatial 0.909), recorded in `results.tsv` and committed with the updated progress figure as
`371de75`. This is the first meaningful hand-designed step change after the 0.743 plateau and
therefore clears the user's gate for any later focused tuning, though no Optuna is warranted yet.

A second-generation seasonal diagnostic added vegetation/canopy/soil anomalies, nonlinear
driver thresholds, incumbent-opportunity interactions, and compound fire-weather terms. Its full
420-term global GAM reaches 0.7590 on top of the official candidate, but reduced refits do not
retain the gain. This is evidence for conditional fuel-state physics, not yet an acceptable model:
the next seasonal revision must compress the correlated response surface rather than paste a
large empirical equation.

The separate annual head is more promising. A no-geography Poisson GAM built from named annual
and climatological summaries reaches **0.7737** when blended 75% with the current annual map,
with bias 0.8282, rmse 0.6002, seasonal fixed at 0.8886, and spatial 0.9512. This is a +0.0227
ceiling on top of the official 0.751 model and exceeds the earlier annual-tree ceiling. A 240-term
reduced refit still reaches 0.7691, while 120 terms reaches 0.7605. The dominant recoverable
structure is not a universal occupancy correction: it combines incumbent fire opportunity with
temperature extremes, wet-day/dry-spell regimes, VPD, fuel/canopy state, wind, and land use.
Next work is to derive a compact mechanistic annual propensity family from that hierarchy, then
combine independently proven annual and seasonal gains before another official evaluation.


### Entry 90: two cross-validated GAM heads reach official 0.776

The annual residual ridge was installed as `a26cdc3` and evaluated officially at **0.768**
(bias .813, RMSE .598, seasonal .889, spatial .942). Its five-fold spatial out-of-fold score was
0.7664 and fold coefficient correlations were .959-.971, so almost all of the gain survives held-
out cells. A second globally shared seasonal residual GAM then raised the proxy to 0.7760, with a
three-fold out-of-fold score of 0.7744 and fold coefficient correlations .967-.972. Neither head
uses coordinates, region labels, cell identifiers, or new geographic masks.

Only after these structural steps did focused Optuna become justified. It improved the exact proxy
by just 0.0003, selecting annual scale .995687, annual residual strength .763951, first allocation
strength .856355, and seasonal residual strength .987594. Commit `cffd5af` evaluated officially at
**0.776** (bias .813, RMSE .612, seasonal .900, spatial .945), recorded by `6f51c13`. This confirms
Optuna is a finishing tool rather than the source of the scientific jump.

A geography-free nonlinear diagnostic with the 0.776 annual map fixed reaches 0.8160 through
seasonal allocation alone; jointly learning annual propensity and allocation reaches 0.8505. The
information for 0.8 is therefore present in the existing observables. A third pass of the same
linear seasonal family was rejected: it scored 0.7762 in-sample but only 0.7744 out-of-fold.


### Entry 91: coupled-validity reset recovers most of the constrained loss

Richard identified two deployability failures: ED sites are independent, so neighbour exchange is
not available, and the prepared wind, VPD, and dry-spell fields do not reach an 1850 coupled start.
The NOAA CPC provenance also invalidates wet-day fraction, and frozen modern GPW population is not
a historical forcing. These constraints are now durable in `research.md` and `inputs/README.md` at
commit `b65c2a8`; no replacement forcing will be silently spliced in.

The exact matched proxy falls from 0.7763 to **0.7290** when neighbour, wind, VPD, MCDD, wet-day
fraction, population, and their fitted closures are removed. Neighbour alone costs 0.0069, so most
of the reset is loss of the two trained closure equations rather than spatial coupling itself.

Both heads were refit from scratch using only precipitation, temperature, dryness, coupled ED
state, historical LUH2, and fixed lightning climatology. The annual equation reaches 0.7449 in
five-fold spatial holdout with coefficient correlations .969-.977. The seasonal equation reaches
0.7433 in three-fold holdout with coefficient correlations .969-.972. Combining them yields proxy
**0.7648** (bias .8032, RMSE .6033, seasonal .8824, spatial .9317), recouping about 76 percent of
the stripping loss. All 13 declared components are active, the model is pointwise, and the exact
intermediate is committed as `e723587`.


### Entry 92: active regional coefficients invalidated the apparent coupled baseline

Inspection of `predict` found that the legacy continent-box loop was still active before both
valid-input closure heads. It selected different parameter tables for Africa, South America,
North America, and the other boxes. Removing that dispatch in experiment `4a824cc` drops the
proxy and official Overall from 0.765 to **0.588** (bias .683, RMSE .527, seasonal .851, spatial
.355), recorded by `bf9d775`. The 0.776 candidate is therefore historical but not coupled-ready.

This does not remove the path to 0.8. A geography-blind HGB diagnostic using only coupled-valid
local inputs reaches **0.8479** jointly and an inspectable 128-leaf tree reaches 0.8183. The leading
recoverable structure is absolute and antecedent temperature and precipitation, annual rainfall,
dryness, GPP, lightning, and land use. These signals must now be distilled into smooth ecological
regime equations with globally shared coefficients; no labels, coordinates, regional parameter
tables, or geographic branching may enter the model.

The post-reset audit covers all 14 ILAMB regions, all 258 Natural Earth country/territory polygons
by fractional 1-degree overlap, and eight observable ecological regimes. It exposes the annual-map
failure clearly: global fire is 0.495 of observed, NHAF 0.380, SHAF 0.323, AUST 0.232, and boreal
forest 0.172, while TENA is 2.231. The intact tropical closed-canopy total is 0.991 of observed but
its peak is four months wrong, so a near-one area ratio alone is not sufficient plausibility.


### Entry 93: global annual GAM recovers 0.745; strict site independence is 0.743

Refitting the annual residual equation after deleting regional coefficient dispatch raises the
five-fold spatial OOF score from 0.5124 to 0.7105 with coefficient correlations .970-.977. The
combined candidate reaches proxy 0.7445 and official **0.745** at `244eff2`, with bias .786, RMSE
.581, seasonal .851, and spatial .924. This recovers 99 percent of the earlier 0.751 structural
milestone without invalid climate inputs, neighbours, coordinates, region labels, or regional
coefficient tables.

The all-country audit prevents accepting the scalar alone. Global burned area is 1.132 times
observed, intact tropical closed canopy is 1.451 times observed with its peak five months wrong,
TENA is 3.742 times observed, and cool-country false fire is severe; Australia remains only 0.408.
The next equations must improve these state-defined failures rather than compensate geographically.

A stricter code audit found two residual cross-site reductions: global normalization of the intact-
forest brake and runtime global median/IQR calibration. The brake contributes only 0.0014 and was
disabled in `da9647e`; the calibration scales were frozen in `8a85780`, making the spatial response
reproducible at an isolated ED site. Official Overall is now **0.743** (bias .785, RMSE .579,
seasonal .851, spatial .921). The historical 0.776 and interim 0.745 are not the coupled-ready best.

Reverse-ML boundaries are now sharper. A current/previous-month-only HGB ceiling is 0.775, so
local climate-memory state is needed to cross 0.8; this can be implemented as running site state,
not future or cross-site information. With local climatological state, the coupled-valid ceiling is
0.8479. Fixed-map depth-one boosting gains almost nothing, depth two reaches 0.752, depth four
0.766, and the full interaction model reaches 0.788 with the annual map fixed. The missing signal
is therefore hierarchical interaction among incumbent opportunity, absolute temperature, current
and antecedent rain, annual rain regime, dryness, fuel, lightning, and land use, not another
univariate curve or scalar tune.


### Entry 94: smooth ecological brakes fix closed-canopy excess without geography

Two pointwise brakes were added as continuous functions of local observable state with globally
shared strengths. The cool-cultivated brake suppresses fire where cropland coincides with cool
conditions, while the humid closed-canopy brake combines warm temperature, high annual rainfall,
tall canopy, high LAI, and natural vegetation. No country, region, coordinate, or geographic
branch enters either equation. The first experiment is committed as `3c62387`; strengthening the
humid closed-canopy response is committed as `3fb7bd1`.

Official Overall remains **0.743** at `3fb7bd1` (bias .784, RMSE .581, seasonal .851, spatial
.917), but the ecological failure improves materially: intact tropical closed-canopy burning falls
from 2.01 to 1.24 times observed. The all-country and ecological audit is deliberately broader than
the Congo: TENA improves to .551, Europe to .662, SHSA to .633, and EQAS to .628, while excessive
temperate agricultural fire, productive-rangeland excess, and Australia/arid underprediction remain
explicit targets. This candidate is preferred over the slightly higher scalar alternative because
it removes a known physical pathology rather than optimizing only the global score.

An explainable boosting diagnostic, excluded from production, reaches 0.758 with the annual map
fixed. Its strongest interpretable interactions couple temperature anomaly to mean-temperature
regime, lightning anomaly to thermal and land-use state, LAI anomaly to thermal regime, incumbent
fire opportunity to dryness anomaly, and seasonal harmonics to temperature and GPP. A 365-term
linear regime GLM gains only .003 out of fold and its joint variant fails, so neither is promoted.
The next experiment will compress the nonlinear diagnostic into a small family of smooth thermal,
moisture, ignition, and fuel-phenology gates before any tuning is considered.


### Entry 95: causal site memory provides the first coupled-valid path above 0.8

Several smooth climatological distillations were rejected. A 209-term thermal/moisture regime GLM
reaches only .7465 out of fold, an 86-term annual opportunity-regime GLM reaches .7453, and a
551-term tensor seasonal GAM reaches .7482 under its correct Poisson objective. Smooth cool-managed
brakes improve TENA and Europe locally but reduce the global score, while a missed-window ignition
pulse is neutral at low strength and damages RMSE when strong. These failures confirm that adding
more fixed climatological gates is not the required step.

A new diagnostic replaces future climatological state with exponentially decayed memory local to
each ED site. With current and previous valid inputs plus 3, 6, 12, and 24 month reservoirs, a
geography-free HGB reaches **0.8062** in sample and **0.7672** with whole cells held out, versus the
official .7426 baseline. The dominant new information is precipitation departure from 6-12 month
storage, longer precipitation and lightning memory, temperature departure, and short GPP carryover.
This is a real +.0246 held-out structural gain and clears the meaningful-step threshold; it uses no
future months, neighbours, coordinates, labels, region coefficients, or invalid forcings.

The first 150-term multiplicative memory GLM fails at .7437 out of fold, showing that the booster
gain depends on nonlinear conditional reservoir responses rather than one linear correction. No
learner has been promoted. The next step is an explainable memory GAM to identify a small set of
smooth moisture-storage, fuel-carryover, thermal-regime, and ignition interactions for mechanistic
implementation. Optuna remains deferred until such an equation improves the actual model.


### Entry 96: causal memory improves the score but fails the physical audit

A dense additive diagnostic with 76 globally shared causal response curves reaches .7517 in
whole-cell holdout; reduced 12, 20, 30, and 40-curve versions reach only .7404, .7441, .7475, and
.7484. The full smooth formulation was therefore implemented as a site-local running-memory GAM
in `930591d`. It uses current, previous, and exponentially decayed 3, 6, 12, and 24 month states
without future climate, neighbours, coordinates, labels, regional coefficients, or geographic
branches. Official Overall rises from .743 to **.752** (bias .792, RMSE .585, seasonal .868,
spatial .929), confirming that causal memory contains deployable predictive information.

The broader audit rejects this exact implementation as the new physical best. Global burned area
rises from 1.058 to 1.634 times observed, intact tropical closed-canopy fire regresses from 1.239
to 2.012, and arid low-fuel fire rises to 6.167 times observed. The failure is mathematical: the
memory GAM was trained to improve monthly allocation but was blended as an absolute burned-area
prediction, so it creates annual fire mass. The next formulation must center the memory response
as a multiplicative anomaly or otherwise conserve the annual fire potential supplied by the
mechanistic closure. The preferred coupled-valid physical model remains `3fb7bd1` at .743 until
that constraint is satisfied; no Optuna tuning will legitimize the rejected absolute blend.


### Entry 97: causal normalization yields a small physically valid gain

The memory response is now divided by causal 12-month running means of learned and incumbent fire
at each independent site, so it reallocates the annual fire potential instead of introducing a
second source. A global strength sweep gives proxy .7441 at .10, .7456 at .50, .7439 at .75, and
.7404 at 1.00; every setting is preserved in commits `3a1a2f9`, `469b5e3`, `28e935a`, and
`4c941e2`. The selected .50 formulation is official **.746** (bias .783, RMSE .585, seasonal
.851, spatial .924).

The audit confirms that the gain is physically admissible. Global burned area is 1.097 times
observed, intact tropical closed canopy is 1.245, tropical open woodland is 1.080, and the DRC
country total is .994. These retain the closed-canopy repair while improving the score. Remaining
state-defined failures include productive rangeland at 1.704, cropland at 1.442, Australia at
.383, and a four-month global peak error. The next reverse-ML distillation must model conditional
moisture-storage, curing, thermal, fuel, and ignition interactions with globally shared smooth
equations; scalar tuning is exhausted and Optuna remains deferred.


### Entry 98: strict online causality resets the official baseline

A prefix-invariance audit exposed a deeper deployment failure in the .746 candidate. Multiplying
every input after month 96 by .50 changed predictions in months 1-96 by normalized L1 **2.515**
with maximum absolute change .986. The annual GAM summarized the completed 16-year climatology,
the seasonal allocator averaged all future years, and curing, spread, and legacy terms normalized
against the full record. These operations were site-local but unavailable at an online ED step.

Commit `7358363` replaces them with a trailing 12-month annual state, causal running normalizers,
current-state ecological gates, and no completed-climatology seasonal allocator. The same future
perturbation now produces exactly zero change in prior months. Official Overall resets to **.666**
(bias .737, RMSE .529, seasonal .790, spatial .743); global burned area is .736 of observed. This
is the first prefix-invariant baseline and supersedes the .746 candidate for coupled claims.

Reverse ML on current, previous, and 3/6/12/24-month local states recovers to **.7514** in
whole-cell OOF at a .75 blend. The +.0855 gap is meaningful causal structure, but the held-out
ceiling remains below .8, so fitting the grid harder would not justify a target claim. Earlier
dense main effects reach about .750 OOF and individual smooth tensors add at most .0006; a causal
burn-scar reservoir also fails, dropping to .726 at its weakest setting. The next work must improve
the causal state representation or add a defensible missing mechanism, not tune leaked heads.


### Entry 99: compact causal interactions recover the strict baseline to .726

Recalibrating the annual scale after removing future-normalized closures raises the strict model
from .666 to official .709 at scale 1.60. Full causal memory raises it to .717, while removing the
warm-pasture boost and strengthening the smooth cool-cropland brake yields official .718. The
audit improves at the same time: global area is 1.078 times observed, intact tropical closed
canopy 1.269, cropland 1.482, rangeland 1.582, and Australia peaks within one month of observed.

A 150-term named multiplicative residual GLM reaches .7283 in whole-cell OOF with coefficient
correlations .986-.998. Replacing full-period regime means with current state and climatological
opportunity with a trailing annual share preserves the result. A compact 20-term equation retains
.7253 OOF using rainfall, GPP, temperature and lightning reservoirs, smooth climate/land gates,
and incumbent opportunity saturation. Its first online implementation loses annual mass despite
raising seasonal skill, so a focused causal calibration selects annual scale 1.85 and interaction
strength .35. Commit `47c6ebd` is official **.726** (bias .770, RMSE .564, seasonal .828, spatial
.902), and the 50-percent future perturbation still changes prior predictions by exactly zero.

The physical audit remains acceptable but incomplete: global area is 1.123, intact tropical
closed canopy 1.276, DRC .823, and Republic of Congo .888 times observed. Australia now peaks in
the correct month but burns only .501; boreal forest is .228 and arid low fuel .185, while crop
and rangeland remain high at 1.561 and 1.640. These broad state-defined deficits, plus weak BONA
and TENA seasonality, are the next mechanistic targets. Optuna remains deferred because the held-
out causal ceiling is still about .75, not because the current scalar settings are untuned.


### Entry 100: absolute causal residual closure raises the honest model to .729

An absolute-target extension of the named causal-memory diagnostic tests whether local reservoir
state can correct annual amount as well as monthly allocation. The 150-term whole-cell OOF model
reaches .7435 from the .7258 incumbent with minimum fold coefficient correlation .980; reduced
80- and 20-term surfaces retain .7430 and .7353. The stable signal is a joint response to accumulated
drying, fuel storage, GPP and LAI curing, climate state, rangeland, and incumbent fire opportunity.

Commit `6a4587b` implements only the compact 20-term response as a globally shared exponential
surface over current and past local state. It does not reproduce the diagnostic's completed-period
monthly averaging. Official Overall rises to **.729** (bias .772, RMSE .565, seasonal .835,
spatial .908), and halving every input after month 96 changes prior predictions by exactly zero.
The physical audit also improves: global burn is 1.087 times observed, intact tropical closed canopy
1.077, DRC .820, Republic of Congo .833, and cropland 1.363. Productive rangeland remains high at
1.840, while boreal forest and arid low-fuel land remain low at .215 and .192. The next probe must
explain those state-defined contrasts and the four-month global peak error without geography or
region-specific coefficients. Optuna remains deferred because this is a real step but not a new
held-out ceiling near .8.


### Entry 101: true online fitting closes further residual-GLM expansion

The refreshed causal-memory HGB reaches .8006 on the fitted grid but only **.7525** under whole-
cell holdout from the official .729 incumbent. The training score is therefore not a deployable
result. A new 3.52-million-row diagnostic then fits the compact 20-term equation to every actual
site-month rather than a completed-period monthly climatology. Its fold coefficients are extremely
stable at .9955 minimum correlation, but held-out Overall rises only .0003 at strength .25 and
falls at stronger settings. This closes further coefficient stacking from the present residual
family and leaves Optuna unjustified; the next large step requires new causal state or validated
forcing rather than a harder fit to GFED5.

The active prediction remains official **.729** at `6a4587b`. Dead `REGION_PARAMS` and
`REGION_BOXES` constants were removed in `7c9e0a7`, so `model.py` no longer even carries an
inactive regional coefficient table. This cleanup is prediction-neutral.


### Entry 102: historical VPD route validates, but its online mechanism does not

Adding the four prepared candidate forcings raises the causal-memory HGB whole-cell OOF ceiling
from .7525 to **.7627**; an in-sample .8083 is rejected. Permutation importance identifies VPD
departure from a three-month local state as the only large new signal, with 12-24 month VPD
background secondary. NOAA 20CRv3 daily 2-m temperature and humidity were therefore tested as an
1850 bridge. The downloaded 1850 files are complete and finite, while a 2001 overlap derives VPD
with area-weighted correlation .959 against TerraClimate and monthly spatial correlations
.953-.973. The source route is credible, but the full forcing is not installed, so VPD remains
excluded from the coupled-ready model.

A reduced eight-term smooth equation distilled the VPD clue into rapid drying, antecedent fuel,
seasonal-climate, cropland-fragmentation, opportunity-saturation, and long-background terms. Its
climatological whole-cell OOF score was .7368 from .7290, but the decisive 3.52-million-row online
whole-cell fit scored only .7289 at its weakest strength and declined monotonically thereafter.
All formulations and reversions are retained in commits `15c7037` through `2f91f0c`. The
climatological gain was an averaging artifact, not deployable physics; no Optuna is justified.
The active model is restored and reverified at proxy and official **.729**.


### Entry 103: ecological capacity is the new coupled-ready official best

Exact Shapley over 13 components was stopped after 81 of 8,192 subsets because the exhaustive
run was disproportionate. Bounded leave-one-out checks show that spread, legacy, stubble, and
pasture are inactive at four-decimal precision, while curing contributes about .0043, lag .0003,
and soft-min is essential. Commit `629aa15` prunes the four inactive components with the proxy
unchanged at .7290. A stochastic rain-event dry-window family then scored .7288 at both tested
physical timescales and was rejected without tuning.

The successful formulation uses one smooth local-state fire-capacity equation everywhere. Cold
natural forest permits crown-fire amplification below the warm ignition threshold, antecedent
fine fuel provides a bounded arid recovery, and productive rangeland loses contiguous fuel to
grazing and fragmentation. Intermediate and isolated variants are preserved in `2c8c532` through
`f8a3685`; the selected balance is `9157bba`. Official Overall is a new best **.730** (bias .774,
RMSE .566, seasonal .835, spatial .907). Global area is 1.034 times observed, intact tropical
closed canopy 1.074, boreal forest .508 instead of .215, and productive rangeland 1.061 instead
of 1.840. The strong arid variant failed at .697 because the state was too spatially broad, so
the arid term remains deliberately weak.

The coupled audit passes. Halving every input after month 96 changes months 1-96 by normalized L1
0.000000000 and maximum absolute 0.000000000. Perturbing all frozen ED state fields together by
-50 percent changes area-weighted predictions 23.24 percent and total burned area to .8463 of
baseline; +50 percent changes predictions 28.48 percent and total area to 1.2362. The response is
sublinear in aggregate and finite, but materially dependent on coupled vegetation state as it
should be. No Optuna was run because the structural gain is only one official millipoint.
