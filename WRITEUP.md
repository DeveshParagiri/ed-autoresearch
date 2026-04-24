# ED Fire Module Replacement — Writeup

> **Scope:** entire writeup is about an OFFLINE evaluation. Formulas are trained and benchmarked against fixed TRENDY v14 ED-S3 NetCDFs + CRUJRA climate, not a coupled ED simulation. See "Coupling caveats" for what changes under live coupling.

This is the narrative for how Model C — a 3-mechanism, 12-parameter, closed-form fire module — reached rank #1 on the TRENDY v14 ILAMB burned-area benchmark (Overall Score **0.6703**), and how the final shipped artifact was produced.

## Baseline

ED v3's stock fire module (`EDv3` in the TRENDY v14 intercomparison) gets ILAMB Overall 0.4774 (native tier-2 aggregation from `scalar_database.csv`), near the bottom of the leaderboard. Goal: a closed-form replacement that

1. beats stock ED substantially
2. stays interpretable (every parameter has a physical meaning and a literature citation)
3. is a single global formula (no regional variants, no per-cell lookups)
4. runs at monthly resolution on standard CRUJRA + ED prognostic state

## Formula class

Model C is a product of bounded `[0, 1]` mechanism gates, raised to a global exponent:

```
fire(cell, month) = [onset(D̄) · suppress(D̄) · precip_floor(P_ann) ·
                     precip_dampen(P_month) · gpp_hump(GPP) · ign(T_air)]^fire_exp
```

- **Double sigmoid on D̄** (accumulated precipitation deficit): onset threshold + hyperarid suppression (Pausas & Keeley 2009; Krawchuk 2009)
- **Precipitation controls** (Knorr et al. 2014-style pyrogeography boundary): annual precipitation floor × monthly dampener
- **Monthly GPP hump** (van der Werf 2008): fire peaks at intermediate productivity
- **Monthly air-temperature ignition sigmoid** (Archibald 2010)
- **Global intensity exponent** (Bistinas 2014)

## Method summary

1. **Base formula class** — multiplicative product of sigmoids + hump, each tied to a physical mechanism.
2. **Iterative development (v2 → v8)** — added monthly-resolved drivers (monthly GPP, monthly air temp, monthly surface soil temp, per-cell GPP anomaly) one at a time; each change validated on real `ilamb-run`.
3. **Shapley decomposition on Model A v8** — 2⁸ = 256 subsets × 500 Optuna trials. Ranked mechanisms by marginal contribution to the training objective. Top-3 were `t_air_ign`, `precip`, `gpp_monthly` (52.8% of explained variance).
4. **Shapley-guided simplification** — built Model C by dropping the lower-5 mechanisms. Refit remaining 12 parameters with 2500 Optuna trials.
5. **Dbar canonicalization (this revision)** — original training used an internal `dbar` accumulator that had two bugs relative to ED's own `calcSiteDrynessIndex`: a fake "annual precip" (`month × 12`) plus an extra `pre_ann < 200` gate that hard-reset the accumulator anywhere rainfall exceeded ~17 mm/month. Fixed by adopting a canonical Thornthwaite + daylength accumulator (from Richard Owusu-Ansah's `prep_monthly_inputs.py`, byte-matched); re-ran the full 12-parameter fit on the canonical inputs.
6. **Final benchmark** — real `ilamb-run` with stock `ConfBurntArea` and stock ILAMB-distributed GFED4.1s. Model C Overall 0.6703, rank #1.

## Scores (real `ilamb-run`)

Pulled directly from `scalar_database.csv` (global row, ED-ModelC-final):

| Metric | Score |
|---|---:|
| Bias Score | 0.7137 |
| RMSE Score | 0.5128 |
| Seasonal Cycle Score | 0.8338 |
| Spatial Distribution Score | 0.7786 |
| **Overall Score** | **0.6703** |

Leaderboard comparison:

| Model | Overall |
|---|---:|
| **ED-Model C (ours)** | **0.6703** |
| CLASSIC | 0.6660 |
| CLM6.0 | 0.6606 |
| CLM-FATES | 0.6568 |
| ELM-FATES | 0.6568 |
| JULES-ES | 0.5905 |
| ELM | 0.5564 |
| VISIT-UT | 0.5545 |
| LPJmL | 0.5485 |
| LPJ-GUESS | 0.4792 |
| EDv3 (stock baseline) | 0.4774 |
| LPJ-EOSIM | 0.4635 |

## Why it wins

**Seasonal Cycle.** Three monthly-varying mechanisms with aligned phase (dry-season peak: low precip, high air temp, intermediate GPP at curing transition) produce one clean seasonal peak without phase smearing. Model C's Seasonal = 0.834, vs CLM6.0 0.758 and CLASSIC 0.782.

**Bias + Seasonal dominate under ILAMB's tier-2 weighting.** Overall = (2·Bias + 2·RMSE + Seasonal + Spatial) / 6. Model C loses on RMSE and Spatial to CLM6.0, but the +0.08 Seasonal edge plus decent Bias put it ahead.

## The dbar bug (and why it matters)

During initial iteration the offline driver pipeline computed `dbar_monthly.npy` via a Python function (`dbar_ed_hard_reset` in `src/modules/fire.py`) that differed from ED's C `calcSiteDrynessIndex`:

```python
# Our original (buggy):
pre_ann = pre_m * 12                                # fake annual precip from 1 month
is_dry = (pet_m / pre_m > 1.0) & (pre_ann < 200)     # extra gate ED doesn't have
```

The `pre_ann < 200` gate kills the accumulator anywhere rainfall exceeds ~17 mm/month. At 9.5°N, 22.5°E (Sahel), our original dbar was **0 across all 192 months**, while the correct accumulator produces values 100–2000 consistently. The formula's parameters jointly adapted to this zeroed-out input, which is why Model C still scored rank #1 on the buggy inputs (internal consistency) — but the resulting parameters are calibrated to a fake dbar distribution.

Fix: adopt Richard's `prep_monthly_inputs.py` as the canonical `dbar_monthly.npy` generator (Thornthwaite + daylength PET, `K=1`, continuous accumulator, reset at monthly precip ≥ 200 mm). Byte-match confirmed (md5 `d9d4ba583ee419b5abfc66d3f2aa7939` on raw array bytes). Retrain Model C's full 12 params against it. The retuned params are a substantially different calibration — `gpp_b` moved 100×, `D_low` went from 180 to 50, `gpp_d` from 24 to 0.56 — but the formula structure is unchanged.

**Under the corrected inputs and retuned params, Model C still wins (0.6703).** Component scores are nearly identical to the pre-bug-fix numbers (Bias 0.714 vs 0.721, Seasonal 0.834 vs 0.842, Spatial 0.779 vs 0.777), because the formula's structural expressiveness is what's doing the work, not the specific calibration.

## Reproducibility

Every pinned artifact is SHA256-hashed in `CHECKSUMS.txt`. Four-command reproduction:

```bash
python scripts/prep_monthly_inputs.py    # or download Drive bundle
python scripts/verify.py                 # pinned-hash check
python scripts/reproduce_modelC.py       # writes burntArea.nc
bash   scripts/run_ilamb.sh              # real ilamb-run
```

`scripts/dump_modelC_terms.py` writes a per-term NetCDF for coupled-ED debugging (inputs + 6 intermediate terms + final product at each cell/month).

## Coupling caveats

Everything above is OFFLINE. Training, Shapley, final benchmark — all against fixed TRENDY v14 ED-S3 NetCDFs. When the `.cc` is ported into `fire.cc` in a live coupled run:

- **Coupling-invariant inputs:** CRUJRA forcing (T_air, P_month, P_ann). These don't change between offline and coupled.
- **Partially coupling-sensitive:** D̄. Same CRUJRA raw inputs, but the accumulator formula itself must match — ED's `calcSiteDrynessIndex` is numerically different from our canonical Python version. Either port the canonical dbar into ED, or retune the 4 dbar params against ED's internal dbar. See README "ED integration" section.
- **Coupling-sensitive:** monthly GPP. ED prognostic, responds to fire feedback. Expect some drift.

Model C has the smallest coupling-sensitive surface area of any formula we tried — only GPP is truly prognostic. No AGB, LAI, canopy height, or carbon pools.

## What's not shipped in this revision

- Models A and B (8 and 5 mechanisms respectively). Their params were also fit against the buggy dbar; they have not been retuned. The `patches/fire_modelA.cc`, `fire_modelB.cc`, and `models/{A,B}/` directories have been removed for this revision.
- `models/shapley.json`. The Shapley decomposition was run against the Optuna objective (equal-weighted component scores), not ILAMB's native Overall; numerical φ values are Optuna-proxy deltas. The ranking of mechanisms is what drove Model C's construction. Not rerun against canonical dbar — future work.
- `t_deep` and `t_surf` arrays in the bundle. The current `prep_monthly_inputs.py` produces bit-identical values for `t_air`, `t_deep`, `t_surf` (known bug inherited from the upstream script). Model C doesn't use `t_deep` or `t_surf`, so the rank #1 result is unaffected. Would need fixing before shipping Models A or B.

## References

- ILAMB: Collier et al. 2018
- GFED4.1s: van der Werf et al. 2017
- TRENDY v14: Global Carbon Project 2025
- Pausas & Keeley 2009; Pausas & Ribeiro 2013
- Krawchuk et al. 2009
- van der Werf et al. 2008, 2010
- Archibald 2010, 2013
- Bistinas et al. 2014
- Lundberg & Lee 2017 (Shapley)
