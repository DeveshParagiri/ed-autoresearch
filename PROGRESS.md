# PROGRESS — ED Fire Optimization

Living log. Most recent on top. Update at end of every working session so Mac and Windows stay in sync via Drive.

Companion deck: `figures_and_tables.pptx` — single home for every table / schematic.

---

## 2026-06-09 — 1:1 / spatial-variance thread: diagnosed the 0.039 ceiling, prototyped the transform fix

George's bar: not satisfied until the model is good on the 1:1 per-cell scatter (it is a flat band
capping at 0.039 vs GFED5 0.104). tropfix2-k4 (canonical) fixed magnitude/false-positives but not this
ceiling. On the Mac (git blocked by Xcode license, no conda/ILAMB), so all work is uncommitted and
emulated-scored only; canonical files untouched. Commit + official-score from Windows.

- DIAGNOSED the ceiling (`scripts/diag_saturation.py`): two structural causes in the rate->fraction
  transform, neither param-tunable. (1) `fire_C` is a product of [0,1] sigmoids so rate is hard-capped at
  1.0 yr^-1 (max 0.997, zero cells hit FIRE_MAX=5). (2) `monthly_frac=(1-exp(-rate))/12` spreads annual
  fire evenly over 12 months -> monthly<=1/12=0.083 and seasonality flattened. Chain gives period-mean
  <=0.039; GFED5 0.104 is above even the 1/12 cap, so the model architecturally cannot reach peak cells.
- PROTOTYPED the fix (`scripts/proto_seasonal_transform.py`, tagged): use the physically-correct
  per-month disturbance fraction `1-exp(-rate/12)`. Lifts high-rate cells x1.5-x4, low-fire cells x1.05,
  so it raises the savanna core without re-inflating false positives. Ceiling 0.039 -> 0.057. Wrote
  `ilamb/MODELS_SEASONAL_PROTO/ED-ModelC-seasonal/burntArea.nc` and
  `NEW MAPS/proto_seasonal/per_cell_scatter_seasonal_proto.png`. Also made the canonical 1:1 figure
  `ED-ModelC-tropfix2-k4/figures/per_cell_scatter_k4.png` (k4 vs PRE-tropfix2, requested earlier).
- EMULATED ILAMB (`scripts/emulate_ilamb_ba.py`, 0.5deg, NOT official, trust deltas): prototype raised
  sigma (model/ref spatial std ratio) 0.747 -> 1.017 (near-perfect variance match), Spatial 0.7144 ->
  0.7782, Overall 0.5955 -> 0.6097, with Bias/RMSE/Seas flat. First thing to move sigma toward 1.
  CAVEAT: un-retuned, magnitude now ~1.47x GFED5; evidence the direction is right, not promotable as-is.
- STAGED official scoring: `scripts/score_proto_official.sh` (one command on the ed-fire machine).
- NEXT (Windows): commit; run official score on the prototype; then refit with the transform locked in
  (wire a `SEASONAL_TRANSFORM=1` flag into optimize_modelC_coupled.py + reproduce_modelC.py, re-run with
  MAG_BAND to pull total back to ~1.1x while keeping sigma~1); coupling-check the transform with Lei
  before any promotion (the /12 was there to match what coupled ED writes). See HANDOFF READ-THIS-FIRST.

## 2026-06-08 (later) — PROMOTED tropfix2-k4 to canonical (magnitude over-burn fixed)

Ran tropfix2 (2500 trials, 60 min): `PHYSICAL=1 MAG_BAND=1.12 FP_MIN=0.85 SAMPLER=nsga2
WARM=params.nsga2.json TAG=tropfix2 N_TRIALS=2500 TOPK=8`. FIX 2 dumped 8 Pareto candidates; scored all
8 + canonical with official ILAMB. Best official candidate = **k4** (trial 2465): BA Overall 0.6474,
Spatial 0.7620, magnitude 1.11x. (The earlier failed tropfix run dropped to 0.6416/0.7252 - tropfix2
held spatial because FIX 1 fixed the selection target and MAG_BAND was loosened 1.05->1.12.)

Richard decided to promote (his paper). Promotion executed and verified with official ILAMB:
- BA: 0.6485 -> **0.6473** (reproduce-regenerated shipped file; rank #3 held, #4 CLASSIC 0.6268).
- Magnitude: **1.26x -> 1.11x** GFED5 (881 vs 793 Mha/yr).
- Retuned combustion betas for the lower BA (internal fit 0.6537 -> 0.6653; beta_leaf 0.58 -> 0.92).
- fFire: 0.6465 -> **0.6534** (rank #5 -> **#4**); total 3.41 PgC/yr vs GFED5 3.40 (was 3.47). IMPROVED.

Net: corrected the over-burn the principled way (targeted tropical closed-canopy suppression, not a
global scalar) at a negligible -0.0012 BA cost, and emissions actually got better. Canonical files
swapped (params.json, betas.gfed5.json, burntArea.nc x2, fFire.nc); PRE-tropfix2 backups kept
(params.PRE-tropfix2.json, betas.PRE-tropfix2.gfed5.json, backups_PRE-tropfix2/). CLAUDE.md canonical
table + scores updated. New figures: NEW MAPS/tropfix2/{BA_k4_vs_gfed5,1_global_timeseries_k4}.png.

Gotchas: moved stray .nc (burntArea.{nsga2,tropfix,tropfix2}.nc, fFire (1..6).nc dups) out of the
scored folders to avoid ILAMB MonotonicityError; compute_emissions.py needs PYTHONIOENCODING=utf-8 on
Windows; its "GFED ref ~2.0 PgC/yr" note is stale (GFED5 is 3.40). TODO (optional): regenerate the
4-panel/seasonal paper figures (their scripts have hardcoded old score annotations to update).

## 2026-06-08 — Committed pending work, applied scorer fixes 1+2, launched tropfix2

Caught the docs up to git and cleared the backlog the 2026-06-08 handoff flagged. On Windows so git
works (the Mac Xcode-license block does not apply here).

- Committed the previously-uncommitted backlog as three clean commits: the tropfix CODE + 2026-06-07
  run (`39adac4`), the 2026-06-08 spatial-drop diagnosis (`2b843f7`), and CLAUDE/handoff state. Nothing
  canonical was overwritten in any of them.
- FIX 1 (`929a816`): optimizer internal Overall now uses official ILAMB weighting
  `(Bias + 2*RMSE + Seas + Spatial)/5` instead of the unweighted mean of the four components, so the
  selection target matches the official aggregation used for promotion.
- FIX 2 (`cb7e054`): optimizer now dumps the top-K Pareto candidates (env `TOPK`, default 5; NSGA-II
  only), each as its own params JSON + a scoreable `burntArea.nc` in its own model dir under
  `ilamb/MODELS_TOPK_{tag}/`, plus a manifest `models/C/topk.{tag}.json`. Because internal Overall and
  official Overall are on different grids/weightings, the internal-best is NOT necessarily the official
  winner; this lets us re-score every candidate with official ILAMB and promote the official best.
  Refactored NC encoding into `write_ba_nc()` and the params dict into `build_out()` (primary output
  unchanged). Smoke-tested end-to-end (5 trials, TOPK=2), artifacts verified then removed.
- Pushed all 5 commits to `origin/coupled-refit-gfed5` (`7f96cf6..cb7e054`).
- Launched the tropfix2 re-run in the background (handoff step 3 — cut magnitude WITHOUT flattening
  sigma): `PHYSICAL=1 MAG_BAND=1.12 FP_MIN=0.85 SAMPLER=nsga2 WARM=params.nsga2.json TAG=tropfix2
  N_TRIALS=2500 TOPK=8` -> `logs/opt_tropfix2.log`. Goal: total ~1.10-1.15x GFED5 while official Spatial
  holds near 0.769 and Overall >= 0.6482. Next: score the 8 top-K candidates with official ILAMB and
  promote the official winner only if Overall held/improved AND magnitude improved (NOT the internal best).

---

## 2026-06-03 — Global-scalar test and the error-distribution finding

Tested the "model is ~1.26x high, just multiply by 0.8" idea. Best scalar 0.792 (matches global mean),
0.779 (min global RMSE). It fixes the GLOBAL-SUM bias but NOT the ILAMB score. Two reasons, the second
is the real finding:
1. Spatial cancellation. Net signed bias +17.3 Mha hides +46.1 over / -28.8 under. Gross per-cell |bias|
   is 75.0 Mha (4.3x the net). ILAMB scores per cell, so the scalar dropping net to ~0 only moves gross
   75.0 -> 67.7 (~11%). Global-series correlation 0.405 is scale-invariant.
2. Dynamic-range / ceiling problem. Per-cell scatter shows model period-mean burned fraction caps at
   ~0.039 while GFED5 reaches ~0.10 — a horizontal band, not a 1:1 line. Over-prediction in dry/boreal
   low-fire cells, under-prediction in the African savanna core. The scalar helps the former and makes
   the latter worse, so ILAMB barely moves. Bias is "too flat," not uniform inflation; the fix is
   slope/saturation in a refit (tighter MAG_BAND, plus the false-positive objective), not a multiply.

Advisor asked about the error distribution; saved to auto-memory `error-distribution-scalar-finding.md`.
New figures: `NEW MAPS/Seasonal/{1_global_timeseries_scaled08,2_diff_scatter_scaled,3_map_scaled_ba}.png`.
Scaled outputs: `ilamb/MODELS_SCALED/ED-ModelC-{scaled08,scaled0792}/burntArea.nc`. Scorer:
`scripts/score_scaled_ba.sh`. ILAMB deltas are EMULATED — no ed-fire env / ILAMB on this machine
(system python3 has xarray+cartopy only); run the scorer on a real env for the official number. This
machine's git is blocked by an unsigned Xcode license, so commit from another machine if needed.

---

## 2026-05-15 — Native TRENDY v14 fFire comparison, final ranking, names locked

**Two important corrections rolled into this session:**

1. **The "2.0 PgC/yr GFED reference" Lei's docstring cited is the GFED4 number.** GFED5 actual global mean is **3.40 PgC/yr** (Chen et al., Nature Sci Data 2025, verbatim quote in `gfed5_emissions_nature.pdf`; matches direct integration of `ilamb_ref_official/.../GFED5/fFire.nc` at 3.38 PgC/yr). All earlier "magnitude penalty" framings were aimed at the wrong target. The honest variant is **ED-ModelC-Emissions** at 3.45 PgC/yr (within 1.5% of GFED5), not the HurttStrict at 2.06 PgC/yr that was tuned to GFED4.

2. **Initial fFire leaderboard scored every TRENDY peer using *our* GFED5-derived EF applied to their burned area, not their native fire emissions output.** Pulled native `<Model>_S3_fFire.nc` for nine models from `https://mdosullivan.github.io/GCB/` (TRENDY v14 GCB 2025): CLASSIC, CLM-FATES, CLM6, E3SM, ELM-FATES, JSBACH, SDGVM, VISIT, EDv3. Regridded each to 0.5° via `scripts/regrid_trendy_native_ffire.py` and replaced the EF-derived files in `ilamb/MODELS_LEADERBOARD_FFIRE_GFED5/<Model>/fFire.nc`. EF-derived backups kept at `<Model>/fFire.EF-derived.nc.bak`.

**Naming locked.** Dropped all "Hurtt" / "fFireTuned" labels. The keepers are:

- `ED-ModelC-GFED5` — Model C BA, criterion (a), tuned against GFED5 burned area
- `ED-ModelC-Emissions` — same BA + Hurtt 4-pool combustion-completeness fFire with tuned β values
- `ED-ModelC-EmpiricalEmit` — BA refit against GFED5 fFire + per-cell EF derived from GFED5

All dead variants (HurttConstrained, HurttStrict, mask{NN}, agb{NN}, x{NNN}, fFireTuned-Hurtt) moved to `ilamb/MODELS_LEADERBOARD_FFIRE_GFED5/_archive/`. Older ED-ModelA/B/C exploration variants also moved to `_archive/` on both BA and fFire sides.

Scripts renamed to match: `compute_emissions.py`, `tune_combustion_params.py`, `refit_modelC_empirical_emit.py`, `figure_emissions_landscape.py`, `maps_emissions_stack.py`, `regrid_trendy_native_ffire.py`. Param folders: `models/combustion-params/`, `models/C-empirical-emit/`.

**Tuned β values that go to Lei for coupled-ED implementation** (`models/combustion-params/betas.gfed5.json`):

| Constant | CLM5 default | Tuned |
|---|---:|---:|
| β_leaf | 0.90 | 0.141 |
| β_fine | 0.80 | 0.669 |
| β_coarse | 0.35 | 0.018 |
| β_litter | 0.80 | 0.996 |
| D_REF | 150 mm | 107 mm |

Pool partitions unchanged from Lei's `compute_ffire.py`: C_leaf=0.05·AGB, C_fine=0.10·AGB, C_coarse=0.85·AGB, C_litter=0.15·cSoil.

**FINAL native physics-vs-physics ranking on GFED5 fFire** (`ilamb_out_ffire_native/scores.csv`, TRENDY peers + our three kept variants):

| Rank | Model | Bias | RMSE | Seas | Spat | Overall | Global PgC/yr |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | CLM6 | 0.7583 | 0.5371 | 0.8374 | 0.7867 | **0.6913** | 3.57 |
| 2 | ELM-FATES | 0.7254 | 0.5375 | 0.8471 | 0.6910 | 0.6677 | 4.77 |
| 3 | **ED-ModelC-EmpiricalEmit** | 0.7111 | 0.5134 | 0.8161 | 0.7514 | **0.6611** | 3.48 |
| 4 | **ED-ModelC-Emissions** | 0.7067 | 0.5051 | 0.8097 | 0.7645 | **0.6582** | **3.45** |
| 5 | CLASSIC | 0.7291 | 0.5340 | 0.8325 | 0.6586 | 0.6576 | 3.05 |
| 6 | E3SM | 0.7327 | 0.4665 | 0.7955 | 0.8038 | 0.6530 | 4.00 |
| 7 | ED-ModelC-GFED5 (EF) | 0.7233 | 0.4938 | 0.7592 | 0.7529 | 0.6446 | 3.85 |
| 8 | CLM-FATES | 0.7159 | 0.5408 | 0.8215 | 0.4449 | 0.6128 | 2.38 |
| 9 | VISIT | 0.6557 | 0.5118 | 0.7995 | 0.5073 | 0.5972 | 2.10 |
| 10 | SDGVM | 0.7084 | 0.5029 | 0.4314 | 0.7000 | 0.5691 | 5.21 |
| 11 | JSBACH | 0.6861 | 0.5133 | 0.7911 | 0.2590 | 0.5526 | 1.43 |
| 12 | **EDv3 (TRENDY submission)** | 0.6483 | 0.4787 | 0.4820 | 0.1050 | **0.4385** | **0.46** |

GFED5 reference total: 3.40 PgC/yr.

**Headline for the paper (process-based emissions, controlled-variables comparison):**

> Replacing ED's legacy fire submodel with a Model-C burned-area formula plus a four-pool combustion-completeness emissions formulation, while holding ED's vegetation and climate backbone unchanged, improves UMD's TRENDY contribution from 0.4385 (EDv3) to 0.6582 on GFED5 fire carbon emissions. This places ED among the top three TRENDY single models (behind CLM6 at 0.6913 and ELM-FATES at 0.6677). Global magnitude improves from 0.46 to 3.45 PgC/yr against the GFED5 reference of 3.40 PgC/yr.

The +0.22 ILAMB-Overall gain is attributable entirely to the new fire physics: vegetation forcing (AGB, cSoil, GPP) and climate forcing (CRUJRA) are identical to EDv3.

**Component story for our model:**

- Spatial (0.7645) is 3rd best of all 12 models, behind only CLM6 and E3SM. Process-based formula gets carbon emissions in the right places.
- Seasonal (0.8097) is competitive but slightly behind CLM6/ELM-FATES.
- Bias and RMSE are mid-pack. Cell-level residuals could improve with herbaceous fuel addition (next paper iteration).

**Where the empirical-EF variant fits.** ED-ModelC-EmpiricalEmit (0.6611) beats ED-ModelC-Emissions (0.6582) by 0.003 on ILAMB but is methodologically weaker because it uses a GFED5-derived per-cell EF lookup. We carry it as a baseline / sensitivity check, not the submission.

**Caveat on coupling.** Both our variants currently run offline as post-processing on saved EDv3 vegetation and GPP fields. Lei's task is to implement the Hurtt formula with these tuned β values inside coupled ED. Per the Tier-2 hand-off, BA already reproduces bit-exact in coupled ED; fFire is the next step.

**Artifacts:**
- Figures: `NEW MAPS/EMISSIONS/FIG_landscape_BA_vs_fFire.png`, `FIG_burnedArea_stack_5models.png`, `FIG_fFire_stack_5models.png`, `FIG_ffire_stack_truth_vs_ED_variants.png`
- CSV: `NEW MAPS/EMISSIONS/scores_table.csv`
- Score databases: `ilamb_out_ba_renamed/scores.csv`, `ilamb_out_ffire_native/scores.csv`
- Tuned params: `models/combustion-params/betas.gfed5.json`
- References pulled to repo: `essd-15-5227-2023.pdf` (GFED5 BA paper), `gfed5_emissions_nature.pdf` (GFED5 emissions paper documenting 3.4 PgC/yr global mean)

---

## TL;DR (current state)

- **Paper framing pivoted.** No longer "we built a better fire model." New framing per Hurtt (2026-05-05):
  > *Advances in remote sensing alter the structural form of fire dynamics.*
- The contribution is showing that **changing the optimization criterion or the reference dataset (GFED4 vs GFED5) doesn't just shift parameters — it changes which input variables the model needs.**
- Two deliverables on different clocks:
  1. **GCB (May)** — one fire model into the Global Carbon Budget. Use the current GFED4 + every-fire-equal baseline (≈ Model C).
  2. **Paper (end of summer)** — full 6-row table (3 optimization criteria × 2 datasets), both calibrated and cross-evaluated.
- Lei's coupled-ED run with Model C: agreement looks good, **but only Africa was tested** and spinup used the old fire model (legacy contamination).

---

## The two axes of variation

**Optimization criterion** (3 choices)
- **Opt 1** — every fire equal *(current; overfits African grassland/cropland)*
- **Opt 2** — every continent equal *(N. America = Africa)*
- **Opt 3** — every fire type equal *(grassland = cropland = forest = shrubland)*

**Reference dataset** (2 choices)
- **GFED4** — what we use today; what GCB scores against
- **GFED5** — ~2× burned area, more small/cropland fires; community split on quality

→ 6 independently-optimized models (see table in `figures_and_tables.pptx`, slide 2).

## Expected structural changes (Hurtt's predictions)

| Setup | Inputs the optimizer is expected to keep |
|---|---|
| GFED4 + Opt 1 | Dryness, GPP *(current Model C)* |
| GFED4 + Opt 3 | Dryness, GPP, **Fuel** |
| GFED5 + Opt 3 | Dryness, GPP, Fuel, **Ignition** (humans/lightning), possibly **Land-use mask** |

If these expectations hold, the paper writes itself: better remote sensing → richer required model structure.

---

## Backlog

Code:
1. Generalize the offline pipeline so the loss function is pluggable (per-fire, per-continent, per-fire-type weighting).
2. Add fuel and ignition inputs to the candidate predictor set; let the optimizer drop unused ones.
3. Pull GFED5 alongside GFED4; add fire-type and continent masks for stratified scoring.
4. Cross-evaluation harness: calibrate on dataset A, score on dataset B.
5. Report selected variables per run, not just parameters.

Coordination:
- Confirm with Lei the coupled-run caveats (Africa-only, old-fire-model spinup).
- Settle optimizer API with Dev (function signature for `fire(params, drivers) → loss`).

---

## Working log

### 2026-05-08 — Mac — Model A magaware refit launched (going for #1)
- ILAMB leaderboard ran with all team variants in `ilamb/MODELS_LEADERBOARD/`. Result: ED-ModelC-Ours rank 7 at Overall 0.6446, ED-ModelA-final rank 4 at 0.6574 (slightly higher), CLASSIC rank 1 at 0.6665.
- Diagnosis. Model A has 8 mechanisms (27 params) vs Model C's 3 (12 params). The extra knobs let Model A fit GFED's spatial pattern and seasonal cycle better, while we beat A on Bias.
- Decision. Take Model A's full 8-mechanism architecture from `scripts/reproduce_v2.py` `fire_A` and rerun it through the magaware-annual pipeline that gave us ED-ModelC-Ours.
- Wrote `scripts/refit_modelA_magaware.py`. Uses canonical 1° CRUJRA + ED-static + TRENDY drivers (where Model A's full input set lives), magaware-annual loss `(1 - r) + 0.5 * |pred_mean - obs_mean| / obs_mean`, 8000 trials, train 2001-2010, test 2011-2016.
- Launched background refit `bhw4s4b84` with `caffeinate -dimsu` to prevent Mac throttling. 4.5h cap. Expected completion ~17:30.
- When done, will write 0.5° burntArea.nc to `ilamb/MODELS_LEADERBOARD/ED-ModelA-Ours/`, re-run ILAMB on the full leaderboard, and see if A-Ours can climb above CLASSIC (0.6665) for rank 1.
- **If A-Ours wins, the deployment path is.**
  1. Patch `ED_Source_Code/GlobalED/fire.cc` with Model A's 8-mechanism formula, using `patches/fire_modelC.cc` as the convention template.
  2. Build ED on Mac (half-day of dependency fighting against homebrew NetCDF/TBB/BerkeleyDB/libconfig++; Makefile paths and CXX=g++ to fix).
  3. Run a regional 50-year transient (e.g., US or African savanna) on top of an existing spin-up restart to skip the 1000-year cold start. Verify coupled burntArea matches offline Model A prediction.
  4. Hand off to Lei for the global production run on GEL. Lei recompiles ED with the patch and runs the standard spin-up + S3 pipeline, then exports the resulting burntArea.nc for GCB.
- **Three ED state fields will need to be added to Lei's NC export** for bit-exact offline calibration of any future Model A retune: `t_deep`, `t_surf`, `h_natr`. Coordinate with Lei when the time comes.

### 2026-05-13 evening — Fire carbon emissions scored, RANK 1 on GFED4
- Per Hurtt's "trial on a simple model" directive for carbon: applied S=0 per-cell EF approach from `score_fFire_simple.py`. Per-cell EF derived from `GFED4_fFire / GFED4_BA_per_sec`.
- Applied to all 22 leaderboard models (regridded to 0.5deg where needed). Wrote `ilamb/MODELS_LEADERBOARD_FFIRE/{model}/fFire.nc`.
- ILAMB scoring against GFED4 fFire reference:
  - **Rank 1: ED-ModelC-Ours at Overall 0.6380** (the magaware-annual GFED4-tuned variant)
  - Rank 2: ED-ModelC-Lei at 0.6279
  - Rank 10: CLASSIC at 0.5785
  - Rank 18: ED-ModelC-GFED5 at 0.5146 (low because GFED5-tuned BA over-predicts when × GFED4-derived EF)
- Three caveats logged.
  1. GFED5 emissions data not yet acquired (Zenodo 504-timeout on the file).
  2. Per-cell EF favors models tuned on GFED4 (the same product the EF comes from).
  3. Native fFire pipelines from TRENDY models (CLASSIC, CLM6) not used in this comparison — apples-to-apples requires using each model's own fFire if available.
- **Headline for GCB targeted at GFED4**: ED-ModelC-Ours is the single-model ship variant that wins on BOTH burned area (rank 7) AND fire emissions (rank 1). For GFED5 targeting, need GFED5 emissions to redo the EF derivation.

### 2026-05-13 — GFED5 retune, BEAT CLASSIC as a single model
- Per the meeting yesterday with Hurtt + Lei: switch GCB target from GFED4 to GFED5 (more novel, GCB community moving there).
- Downloaded GFED5 monthly burned area (Zenodo 7668424, 252 MB, 1997-2020 at 0.25 deg). Built ILAMB-compatible reference NC at 0.5 deg in `ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc` (240 months 2001-2020, % per month).
- Wrote `scripts/refit_modelC_gfed5.py` using the same ILAMB-weighted Taylor-aware loss that produced ED-ModelC-ILAMB.
- 10000-trial TPE refit, 4.2 hr wall, seeded from GFED4-tuned ED-ModelC-ILAMB best. Final params at `models/C-gfed5/params.gfed5.json`.
- **Result on actual ILAMB GFED5 benchmark.** ED-ModelC-GFED5 scored Overall 0.6371, RANK 5 of 20. **Beats CLASSIC (0.6268) by +0.010.**
- Component-wise gain over CLASSIC: Bias +0.023, Spatial +0.113. Lose on RMSE -0.036 and Seasonal -0.011. Net gain dominated by Spatial Distribution (which is what the Taylor-aware proxy was designed for).
- Same 12-param Model C structure. No new mechanisms (no lightning, no population). Just re-optimization against the new target.
- Implication for the paper. Same structural form transfers from GFED4 to GFED5 with reparameterization. The "structural-change" thesis (Hurtt's preferred narrative) may need to be qualified — at least at this 12-param Model C complexity, parameter change is sufficient.

### 2026-05-10 — Single-model ceiling confirmed at ~0.648
- Iteration 6 attempted ILAMB-in-the-loop scoring per trial. Killed at ~50 trials due to ~73s/trial wall pace (subprocess overhead). At that pace 2000 trials = 40 hours, infeasible.
- Iteration 7 ran ILAMB-faithful proxy (cell-wise Bias and RMSE matching ILAMB's formulas) for 20241 trials in 8h. Best params scored ILAMB Overall 0.6414, WORSE than ED-ModelC-ILAMB at 0.6482. Confirms the proxy-vs-ILAMB ranking mismatch is fundamental.
- **Conclusion.** Model C with current 12-param structure and Lei NC inputs has a real ILAMB ceiling at ~0.648 as a single model. ED-ModelC-ILAMB (iter 3) is that ceiling. To break past CLASSIC (0.667) as a single model, the architecture needs new mechanisms (lightning + population, HESFIRE-style) — not more refits of the existing structure.
- **Status.** Ensemble (rank 1, 0.6763) remains the team's #1 result but is multi-model. Best single-model coupled-consistent variant for GCB is ED-ModelC-ILAMB at 0.6482.

### 2026-05-09 — Iterative leaderboard climb to #1 via ensemble

- 5 iterations across the night to push past CLASSIC (Overall 0.6665).
- **Iteration 1** Magaware Model A. ILAMB Overall 0.609 (rank 9). Magnitude perfect, Spatial collapsed.
- **Iteration 2** Multi-obj NSGA-II Model A. 0.607. Same proxy/ILAMB Spatial mismatch.
- **Iteration 2b** Multi-obj NSGA-II Model C. 0.593. Same problem.
- **Iteration 3** ILAMB-weighted single-obj TPE on Model C with Taylor-aware Spatial proxy. Loss = `1 - (2*Bias+2*RMSE+Seasonal+Spatial)/6`. ED-ModelC-ILAMB at **0.6482, rank 7**. First gain over Model C-Ours (0.6446). Spatial Distribution recovered from 0.762 to 0.776.
- **Iteration 4** Same approach on Model A. 0.6344. 27 params still too flexible for proxy-only optimization.
- **Iteration 5** Round 2 TPE on Model C with iteration 3 best as seed, 12688 trials. Best trial = seed itself. Proxy ceiling at ~0.654. Maps to ILAMB ceiling at ~0.65 for Model C.
- **Pivot to ensemble.** Wrote `scripts/build_ensemble_variants.py`. Built four no-refit variants by averaging existing burntArea NCs.
- **Result.** ED-Ensemble-Weighted (score-weighted mean of C-ILAMB, A-final, B-final) and ED-Ensemble-CAB (simple mean of same three) both score **Overall 0.6763, rank 1** on the ILAMB GFED4.1s benchmark. Beats CLASSIC (0.6665) by +0.010.
  - Wins on Seasonal Cycle (0.839 vs 0.782, +0.057) and RMSE (0.510 vs 0.507, +0.003).
  - Loses slightly on Bias (0.731 vs 0.739, -0.008) and Spatial (0.791 vs 0.798, -0.007).
  - Net Seasonal lead exceeds the sum of CLASSIC's leads, so weighted Overall favors us.
- **Caveat for GCB.** Ensemble is a multi-model average. If GCB requires single-parameterization, the best single-model variant is **ED-ModelC-ILAMB at 0.6482** (coupled-consistent on Lei's NC). Need Hurtt+Lei sign-off on whether ensemble-of-team-variants is acceptable as the team's GCB submission.

### 2026-05-08 evening — Model A magaware result and lesson

- Model A magaware-annual finished cleanly (8000 trials, 58 min CPU). Train r 0.463, test r 0.432, train rel_bias 0.18% (essentially perfect magnitude).
- Generated 0.5 deg burntArea.nc via uncoarsen, placed in `MODELS_LEADERBOARD/ED-ModelA-Ours/`, re-ran ILAMB.
- **ED-ModelA-Ours scored Overall 0.6090, rank 9 of 15 — WORSE than ED-ModelA-final at 0.6574.** Spatial Distribution Score collapsed from 0.7834 to 0.5680. Bias and RMSE essentially tied.
- **Lesson.** Magaware loss is not strictly better. For Model C (12 params, 3 mechs) it gave a small gain over `1 - r`. For Model A (27 params, 8 mechs) the extra knobs let the optimizer satisfy magnitude at the cost of spatial pattern. Single-objective magaware does not work as a one-size-fits-all loss.
- Current team-best variant on the leaderboard remains **ED-ModelA-final at rank 4 (0.6574)**, with its original tuning.
- Three honest options for next move.
  1. **Ship ED-ModelA-final for GCB.** It's our current best by ILAMB. Verify it's coupled-consistent first (i.e., that its params were tuned on inputs that match coupled ED's prognostic state).
  2. **Multi-objective Model A retune.** Replace the magaware loss with a Pareto-style optimizer that balances Bias + RMSE + Seasonal + Spatial directly. Avoids the magnitude-vs-pattern trade.
  3. **Stay with ED-ModelC-Ours for GCB** (rank 7, 0.6446). It's the magaware-retuned variant on Lei's NC inputs, so it's already coupled-consistent. Rank 7 is honest and shippable, but lower than what Model A could give.
- Need Hurtt and Lei sign-off before deciding between (1) and (3).
- Map display bug fix from this morning. `scripts/maps_ilamb_leaderboard.py` had a unit-mismatch error (treated GFED reference's `units = '%'` as if it were `units = '1'`, overscaling truth panel by 100x). ILAMB scoring was unaffected, but my display layer was wrong. Fixed and saved a memory note (`feedback_check_units_before_plotting.md`) so I unit-check before plotting.

### 2026-05-08 — Mac — Model C-fuel variant launched
- Lei sent `EDv3_S3_cVeg.nc` (ED v3.0 cVeg, 1700-2024 annual, 0.5 deg, kg/m^2). Multiplied by 0.8 to get AGB and built `global_baseline_modelCfuel_inputs_1997-2016.nc` by appending an `AGB` variable to Lei's hand-off NC (annual broadcast to monthly).
- Wrote `scripts/refit_modelCfuel_magaware.py`. Adds a multiplicative fuel factor `sigmoid(AGB; fuel_k, fuel_low)` to the existing Model C predictor, mirroring CTEM's three-factor structure. Two new free params (`fuel_k`, `fuel_low`); other 12 params remain free with magaware-annual loss.
- Output paths kept separate from Model C so comparisons stay clean. `models/C-fuel/params.lei-magaware-annual.json` and summary, plus future maps under `NEW MAPS/Cfuel/`.
- Launched background refit, ID `bb4kjrv8r`, 6000 trials, 3.5h cap, magaware-annual mode. Expected completion ~05:30.
- Hypothesis to test in maps. The fuel sigmoid should suppress predicted fire in low-AGB regions where Model C currently paints false positives (western N. America, southern S. America, central Asia). If r improves AND rel_bias stays ~10%, this confirms Hurtt's structural-change prediction.
- **V1 result (throttled).** Mac App Nap heavily throttled the run. Only 985 of 6000 trials completed in 3.5h wall (CPU usage ~16%). Best train r = 0.4635, test r = 0.4447, rel_bias 2.5%/13.0%. Fuel sigmoid converged to nearly-flat (fuel_k = 0.023) so the fuel factor was effectively turned off. C-fuel v1 lost 0.02 r vs Model C globally and in every regional split.
- **V2 launched with `caffeinate -dimsu`** to disable App Nap. ID `bqfkj1qj3`, output `models/C-fuel-v2/`, seed 7, 4h cap, expected completion ~14:20.
- **V1 maps saved to `NEW MAPS/Cfuel/20-25.png`.** Truth, Model C, C-fuel, bias maps, C-fuel minus C difference. Visual story: C-fuel-v1 looks essentially identical to Model C with slightly suppressed magnitudes everywhere, no targeted false-positive suppression yet.
- **V2 result (clean, 6000 trials).** Train r 0.470, test r 0.448, rel_bias 2.2%/10.7%, in 152 min. Best params have `fuel_k = 0.014` and `fuel_low = 0.56`, which makes the fuel sigmoid nearly flat — the optimizer effectively turned the fuel knob off.
- **Region head-to-head (C vs C-fuel-v2 on land mask).** Global r 0.483 vs 0.461. Africa r 0.538 vs 0.507. Non-Africa r 0.193 vs 0.190.
- **Conclusion.** Adding AGB as a multiplicative fuel gate on top of Model C does not improve the fit. The pattern ceiling at r ~ 0.48 holds because GPP times area_frac already carries the spatial fuel-availability signal. To break this ceiling we need information Model C does not already encode, candidates are anthropogenic ignition pressure (HESFIRE-style), lightning frequency (LIS/OTD), or a different fuel functional form (additive, piecewise, or interacting with moisture).
- **Maps regenerated** at `NEW MAPS/Cfuel/20-25.png` with V2 params. Visual story is unchanged from v1, C-fuel looks like a slightly faded Model C, no targeted false-positive suppression in western N. America, southern S. America, or central Asia.

### Next steps when Richard returns
1. **Ship Model C (magaware-annual) for GCB**, not C-fuel. C-fuel does not improve over Model C and adds two parameters with no benefit.
2. **Try an anthropogenic-ignition variant next** (Model C-ign). Use HYDE population density as a multiplicative ignition-pressure factor. This is the term Le Page (2015) shows dominates fire regime in temperate and tropical biomes, and it adds information Model C does not already encode through GPP.
3. **Or try a lightning variant** using LIS/OTD climatology as Pi. Cheaper to acquire than HYDE-based fits and tests the natural-ignition leg of the three-factor structure.
4. **Or drop the bolt-on approach** and rebuild Model C-3F as a true CTEM-style three-factor (Pf = Pb · Pm · Pi) where biomass, moisture, and ignition each have their own dedicated factor rather than being mixed into a single product. This would be more invasive but might break the ceiling.

### 2026-05-07 — Mac — Lei sent global refit JSON, pipeline reproduction probe, magnitude-aware refits
- Lei replied with `analysis/modelC_refit_global.json` (copied to `from_lei/`). His note: "this refit has not been run through ED. Only the Africa refit has been."
- Lei's reported numbers: best_loss = 0.881, pearson_r = 0.6299, gfed_mean_pct = 3.08, pred_mean_pct = 1.69. Sahel and Congo pred_pct are NaN even in his global fit.
- Inference: Lei's loss is NOT 1-r (would have been ~0.37). Likely magnitude-aware (MSE, NMSE, or weighted r + magnitude penalty).
- Plugged Lei's `best_params` into our `predict_modelC_lei.py`. Result: r = 0.291 monthly, far from his 0.63. So gap is partly code-side, not just loss.
- **Time-aggregation probe.** Re-scoring same predictions on annual sums:
  - r = 0.469 (GFED-active mask), 0.483 (land-only mask), 0.529 (16-yr time-mean per cell).
  - obs_mean reaches 3.74% with land-only mask, close to Lei's 3.08%.
  - pred_mean stays 0.62-0.73% vs Lei's 1.69%, 2.7× too low regardless of mask.
- **Annual-mean drivers probe** (feeding fire_C with annual means): r = 0.51, similar story. Time-aggregation choice does not close the full gap.
- Wrote `scripts/refit_modelC_magaware.py` with magnitude-aware loss `loss = (1 - r) + 0.5 * |pred_mean - obs_mean| / (obs_mean + eps)`, supporting both monthly and annual scoring modes. Ran both as parallel background Optuna runs, 6000 trials each, ~3.3 hr each.
- **Magaware-monthly result.** Train r = 0.339, test r = 0.332, magnitude bias 1.13x (vs 15x in prior 1-r refit). Saved `models/C/params.lei-magaware-monthly.json`.
- **Magaware-annual result.** Train r = 0.481, test r = 0.474, magnitude bias 1.11x. Saved `models/C/params.lei-magaware-annual.json`. **Best Model C we have on Lei's NC.**
- Diagnosis. Magnitude-aware loss worked as designed. Pattern ceiling for our pipeline at annual scale is r ~ 0.48 regardless of loss. Remaining ~0.15 gap to Lei's 0.63 is structural (mask boundary, fire_max_rate cap, formula variant, or driver preprocessing in Lei's setup).
- **Africa-vs-rest probe (resolves the r=0.63 question).** Scored Lei's params and our magaware-annual params on the same regional masks. Africa box lat -35..38, lon -20..52.
  - Lei params, Africa land: r = 0.644, pred = 1.13%, obs = 10.67% (matches his reported 0.63, but magnitude is 9x under).
  - Lei params, non-Africa land: r = 0.232, pred = 0.45%, obs = 1.49%.
  - Our magaware-annual, Africa land: r = 0.538, pred = 10.09%, obs = 10.67% (magnitude essentially nailed).
  - Our magaware-annual, non-Africa land: r = 0.193, pred = 2.70%, obs = 1.49%.
- **Resolution of the r gap.** Lei's reported r = 0.63 is what Africa already gives. The signal outside Africa is structurally thin in this Model C formulation. Lei wins r in Africa (0.64 vs 0.54). We win magnitude in Africa (10.1% vs 1.13% against 10.67% truth). Different optimization targets explain the rest of the gap, his 1-r style optimizer rewards pattern at any magnitude, our magaware loss enforces faithful magnitude.
- **Status.** Two viable GCB candidates depending on what GCB rewards. If GCB scores against burned-area magnitude, ship `params.lei-magaware-annual.json`. If GCB scores pattern-only on Africa, Lei's params win there. The honest framing for either is "Model C captures Africa well, struggles elsewhere," consistent with the Sahel and Congo NaN flags in Lei's stats.
- **Spatial maps generated** (`NEW MAPS/10..15`). Truth, Lei, Ours, plus bias maps and per-cell r-difference. Visual story is clear, Lei is pattern-faithful but magnitude-starved (Africa ~3% pred vs 10%+ obs), ours is magnitude-faithful but paints spurious fire in western N. America, Mexico, southern South America, parts of Europe and Central Asia. Neither captures GFED's sharp on/off concentration.

### Next steps when Richard returns to ED work
1. **Ship for GCB May submission.** Adopt `models/C/params.lei-magaware-annual.json` as the GCB Model C. Regenerate `ilamb/MODELS/ED-ModelC-final/burntArea.nc` via `scripts/reproduce_modelC.py` (point it at the new params or copy them into `models/C/params.json`). Run `scripts/verify.py` and `scripts/run_ilamb.sh`. Tag the commit.
2. **Lock the three-panel comparison figure into `figures_and_tables.pptx`.** Truth, Lei, Ours side by side is the strongest visual evidence for the paper's structural-change thesis (no params for current C can be both pattern- and magnitude-faithful). Add as a new slide before the 6-row evaluation table.
3. **Start the next predictor variant with a fuel input** (Hurtt's prediction for GFED4 + Opt 3, see Backlog item 2). Call it `Model C-fuel` or `Model D` to keep the comparison clean. The over-prediction zones in `12_ours_pred_annualmean_pct.png` (western N. America, southern S. America, central Asia) are exactly where a fuel input is expected to suppress false positives.
4. **Do not chase Lei's r = 0.63 globally further.** The Africa probe showed it is structurally an Africa number. Sunk-cost.

### 2026-05-06 — Mac — Lei hand-off wired in, first refit, diagnostic maps

### 2026-05-06 — Mac — Lei hand-off wired in, first refit, diagnostic maps
- Read Lei's `handoff_modelC_inputs.md` and verified `global_baseline_modelC_inputs_1997-2016.nc` (240 months, 0.5°, 11 vars, ~80% NaN over ocean).
- Wrote `scripts/predict_modelC_lei.py` — clean predictor using Model C per-landuse, area-weighted, ED saturation transform.
- Confirmed the dryness scale mismatch: current canonical-dbar params give Pearson r = **0.014** on Lei's data (essentially random) — exactly as expected from README, since ED's internal D_bar is 3 orders of magnitude bigger than canonical Thornthwaite dbar.
- Wrote `scripts/refit_modelC_lei.py` (4-param dryness-only refit, README option 2). Saturated at r ≈ 0.25 — bottleneck is the other 8 frozen params.
- Wrote `scripts/refit_modelC_full.py` — full 12-param refit on Lei's inputs. Train 2001-2010, test 2011-2016. Loss = 1 − r (Pearson, cos-lat-weighted, on cells with any GFED activity in train).
- 8000-trial run completed in 3h 9m. Result:
  - Train r = **0.370** (loss = 0.630)
  - Test  r = **0.373**  (generalizes cleanly)
  - Lei's reference target: r ≈ 0.63 — we're 0.26 short.
  - **Key finding:** pred_mean = 2.4e-4 vs obs_mean = 3.6e-3 — the optimizer crushed magnitude by 15× to chase pattern. Our `1 − r` loss is scale-invariant; that's the blame.
  - Best params saved to `models/C/params.lei-full-refit.json`.
- Wrote `scripts/maps_modelC_lei.py` — diagnostic maps in `NEW MAPS/`:
  - `01–03` train (obs / pred / bias),  `04–06` hold-out,  `07` per-cell Pearson r on hold-out.
- Drafted email to Lei in Gmail (rowusuan@terpmail.umd.edu) asking for `analysis/modelC_refit_global.json` so we can verify our pipeline against his reported r ≈ 0.63 before chasing further improvements.
- **Status:** waiting on Lei. Optional next steps (no waste either way): (a) magnitude-aware re-refit, (b) inspect maps and write up regional residual notes.

### 2026-05-06 — Mac (earlier) — environment + new direction setup
- Conda env `ed-fire` created from `environment.yml`. `verify.py`: 24/24 inputs present, 23 hash-OK, 1 cosmetic mismatch on regenerated `out_terms/modelC_terms.nc`.
- Read transcript of 2026-05-05 meeting with Hurtt (`NEW DIRECTION - MEETING.gdoc`).
- Created `figures_and_tables.pptx` with the 6-row evaluation table, two-axes schematic, train/test split, vocabulary cards, Model C anatomy, 2-family view, GPP-vs-biomass.
- Reset this file to reflect the new direction.
