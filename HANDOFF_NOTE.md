# HANDOFF NOTE — ED fire submodule (Model C)

Last updated: 2026-06-09. Read `CLAUDE.md` first for environment, file locations, and conventions.
This note is the "where are we, what's next" narrative.

## >>> READ THIS FIRST (2026-06-09) — the 1:1 / spatial-variance thread (active work) <<<

CONTEXT: George said he will not be satisfied until the model is good on the **1:1 plot** (the per-cell
scatter of model vs GFED5 burned fraction). Right now that scatter is a flat horizontal band that caps
at ~0.039 while GFED5 reaches 0.104 — the "too flat" / dynamic-range ceiling. tropfix2-k4 (canonical,
still shipped, NOT changed this session) fixed magnitude + false positives but did NOTHING for this
ceiling (by design — it only ever suppresses). So this is the next problem.

DONE THIS SESSION (all on the Mac, so NOTHING is committed — Xcode-license blocks git here; commit from
Windows). Canonical files were NOT touched. New scripts + a tagged prototype only.

1. DIAGNOSED where the 0.039 ceiling comes from (`scripts/diag_saturation.py`). It is TWO structural
   facts in the rate->fraction transform, neither tunable by the 14 params:
   - Cause 1 (smaller): `fire_C` is a PRODUCT of [0,1] sigmoids ^fire_exp, so the fire rate is hard-capped
     at 1.0 yr^-1 (measured max 0.997; ZERO cells ever hit FIRE_MAX=5, so that cap is irrelevant).
   - Cause 2 (bigger): the ED transform `monthly_frac = (1 - exp(-rate))/12` spreads one annual fraction
     evenly across 12 months, so monthly frac can never exceed 1/12 = 0.083 and seasonality is flattened.
     Chain: rate<=1 -> annual_frac<=0.63 -> /12 -> monthly<=0.053 -> period-mean<=0.039. GFED5 reaches
     0.104, which is ABOVE even the 1/12 hard cap, so the model ARCHITECTURALLY cannot match peak cells.

2. PROTOTYPED the fix (`scripts/proto_seasonal_transform.py`, tagged, canonical untouched): replace the
   transform with the physically-correct per-month disturbance fraction `1 - exp(-rate/12)`. This lifts
   high-rate cells x1.5 (rate~1) to x4 (rate 5) while barely touching low-fire cells (x1.05), so it
   raises the savanna core WITHOUT re-inflating false positives. Wrote a scoreable nc to
   `ilamb/MODELS_SEASONAL_PROTO/ED-ModelC-seasonal/burntArea.nc` and the 1:1 figure to
   `NEW MAPS/proto_seasonal/per_cell_scatter_seasonal_proto.png`. Per-cell period-mean ceiling moved
   0.039 -> 0.057 (GFED5 0.104) — about halfway; the rest needs Cause 1 (let rate exceed 1).

3. EMULATED-scored it (`scripts/emulate_ilamb_ba.py`; this Mac has NO conda/ILAMB, system python only,
   so this is NOT official — it mirrors the optimizer's Collier score_BA on the 0.5deg grid; trust the
   DELTAS, not the absolute, which sits on the internal ~0.60 scale not the official ~0.65 scale):
   - canonical k4:  Overall 0.5955  Spatial 0.7144  sigma 0.747
   - PROTOTYPE:     Overall 0.6097  Spatial 0.7782  sigma 1.017
   The transform fix raised sigma (model/ref spatial std ratio) 0.747 -> ~1.0 = a near-perfect variance
   match, lifting Spatial +0.064 and Overall +0.014 with Bias/RMSE/Seas all flat. This is the FIRST thing
   that moves sigma toward 1 — exactly the "keep amplitude while cutting the total" lesson from the failed
   2026-06-07 tropfix run. CAVEAT: prototype is UN-RETUNED and magnitude is now ~1.47x GFED5 (the
   transform extracts more frac per unit rate). Emulated Bias barely moved, but official Bias (regrids,
   region-weights) may treat the inflation less kindly. So the prototype is EVIDENCE the direction is
   right, NOT promotable as-is.

NEXT STEPS ON WINDOWS (in order):
0. Commit this session's uncommitted work. Confirm `git status` first. Per .gitignore, ONLY the 5 new
   scripts commit (`scripts/{diag_saturation,proto_seasonal_transform,emulate_ilamb_ba,
   per_cell_scatter_k4}.py`, `scripts/score_proto_official.sh`) plus the HANDOFF/PROGRESS edits. The
   prototype nc (*.nc), the NEW MAPS/ figure, and the ED-ModelC-tropfix2-k4/ figure are all gitignored
   (Drive-only, regenerable by re-running the scripts) — that is fine and intended.
1. RUN OFFICIAL ILAMB on the prototype to confirm the emulated +Spatial holds:
   `bash scripts/score_proto_official.sh`  (activates ed-fire, deletes AppleDouble, runs ilamb-run on
   ilamb/MODELS_SEASONAL_PROTO, prints global Overall; output -> ilamb_out_proto_seasonal/). Compare
   Spatial + Overall to canonical k4 (re-score canonical the same way if needed for an apples-to-apples
   official delta). If official Spatial also rises, the transform direction is confirmed.
2. REFIT with the corrected transform LOCKED IN. The transform lives in `scripts/optimize_modelC_coupled.py`
   `ed_transform()` (line ~142) and `scripts/reproduce_modelC.py` `main()` (line ~201). Suggest wiring a
   `SEASONAL_TRANSFORM=1` env flag into BOTH so canonical behavior is unchanged when off (Claude offered
   to do this; not done yet). Then re-run the optimizer with the magnitude band to pull total back to
   ~1.1x GFED5 WHILE keeping sigma~1, e.g. start from:
   `SEASONAL_TRANSFORM=1 PHYSICAL=1 MAG_BAND=1.12 FP_MIN=0.85 SAMPLER=nsga2 WARM=params.tropfix2.k4.json
    TAG=seasonal N_TRIALS=2500 TOPK=8 python scripts/optimize_modelC_coupled.py`
   Goal: official Overall >= 0.6473 AND Spatial UP AND magnitude ~1.1x. Score the TOPK with official
   ILAMB (find ... -name '._*' -delete first), pick the OFFICIAL winner.
3. Cause 1 (optional deeper lever): if sigma still short of 1 after the transform refit, let the fire
   RATE exceed 1.0 (an additive/exponential fuel term instead of a pure product of [0,1] sigmoids) so
   peak savanna cells can drive annual_frac->1. Bigger model-structure change; only if step 2 is not enough.
4. COUPLING CHECK before promoting any transform change: the `/12` even-spread was put in to MATCH what
   coupled ED writes to TRENDY-format burntArea (see reproduce_modelC.py docstring). Changing it means
   Lei's ED would need to emit the same sub-annual fire timing. Confirm with Lei (lma6@umd.edu) that the
   coupled run can reproduce `1-exp(-rate/12)` before this becomes canonical.

Acceptance for a promotable result: official BA Overall held/improved vs 0.6473, Spatial UP (sigma toward
1, the 1:1 band climbing the diagonal), magnitude ~1.0-1.15x GFED5, AND a coupling-consistent transform.

The 2026-06-08 block below is STILL the canonical state (tropfix2-k4 shipped, unchanged this session).

## >>> READ THIS FIRST (2026-06-08, updated later same day) — tropfix2-k4 PROMOTED to canonical <<<

DONE THIS SESSION: the magnitude over-burn fix is finished and SHIPPED. tropfix2 candidate **k4**
(trial 2465) is now the canonical model. Richard made the call to promote (his paper; George guides).

What the promotion achieved (all official ILAMB, verified):
- Burned area: 0.6485 -> **0.6473** (rank #3 held; #4 CLASSIC is far back at 0.6268).
- Magnitude: **1.26x -> 1.11x** GFED5 (the over-burn roughly halved; the whole point).
- Fire emissions: 0.6465 -> **0.6534** (rank #5 -> **#4**) after retuning betas for the lower BA;
  total 3.41 PgC/yr vs GFED5 3.40. Emissions IMPROVED, not just held.

What changed on disk (canonical table in CLAUDE.md updated to match):
- `models/C/params.json` <- k4 (14 params, adds trop_agb_crit/trop_k_veg). Source of truth tagged copy:
  `models/C/params.tropfix2.k4.json`.
- `ilamb/MODELS/ED-ModelC-final/burntArea.nc` and `.../MODELS_LEADERBOARD/ED-ModelC-Hybrid/burntArea.nc`
  regenerated from k4 (kept in sync manually; reproduce only writes the first).
- `models/combustion-params/betas.gfed5.json` <- retuned for k4 BA. fFire regenerated.
- Backups of the PRE-tropfix2 canonical: `models/C/params.PRE-tropfix2.json`,
  `models/combustion-params/betas.PRE-tropfix2.gfed5.json`, and `backups_PRE-tropfix2/` (the old .nc).
  `params.nsga2.json` is unchanged and still equals the old canonical params.
- SELF-CONTAINED DELIVERABLE BUNDLE: `ED-ModelC-tropfix2-k4/` (model files + corrected-annotation
  figures + README + scores/SCORES.md). This is the folder to show George or zip/send. It is gitignored
  (Drive deliverable, has the .nc + pngs); regenerate figures with `scripts/bundle_k4_figures.py`.

GOTCHAS handled this session (note for next time): both scored model folders had stray .nc that would
trigger ILAMB MonotonicityError (burntArea.{nsga2,tropfix,tropfix2}.nc and fFire (1..6).nc dups) -
moved to backups_PRE-tropfix2/strays/. compute_emissions.py crashes on a unicode print under Windows
cp1252 - run it with `PYTHONIOENCODING=utf-8`. The "GFED ref ~2.0 PgC/yr" note in that script is STALE
(GFED5 here is actually 3.40 PgC/yr).

NEXT (optional, not blocking):
1. The bundle figures (`ED-ModelC-tropfix2-k4/figures/`) are DONE with CORRECT new-score annotations.
   But the OLDER paper figures in `NEW MAPS/Hybrid_GFED5/` and `NEW MAPS/Seasonal/` still come from
   scripts with HARDCODED old annotations (maps_hybrid_ba_ffire.py titles say ILAMB=0.6482 / rank #12 /
   mean 6.27%; maps_seasonal*.py). If you want those refreshed, update their hardcoded strings to the
   new numbers (BA 0.6473 #3, fFire 0.6534 #4) before re-running, or just rely on the bundle figures.
2. Send Lei / George the updated bundle if desired (the old Lei draft references the pre-tropfix2 model).
3. Paper writeup: the tropfix2 story (principled tropical-canopy suppression, not a scalar; magnitude
   1.26x->1.11x at ~zero ILAMB cost; emissions improved) is a clean methods + results narrative.

The 2026-06-03 section below is STALE. Everything above supersedes the old 5-step plan (which is DONE).

### What the 2026-06-07 tropfix run produced (NOT promoted, canonical untouched — correct)

- Code (step 1) is fully wired: `fire_C` in `reproduce_modelC.py:86-92` applies a gated tropical
  closed-canopy AGB suppression; the optimizer imports it and searches `trop_agb_crit`/`trop_k_veg`.
- Run output: `models/C/params.tropfix.json`, `ilamb/MODELS/ED-ModelC-final/burntArea.tropfix.nc`,
  scored in `ilamb_out_tropfix/`. Recipe used: `PHYSICAL=1 MAG_BAND=1.05 FP_MIN=0.85 SAMPLER=nsga2`.
- Result: MAGNITUDE GOAL HIT. Annual total 1001 -> 818 Mha/yr = 1.05x GFED5 (was 1.29x). False
  positives largely cleared (fp_score 0.956). BUT official ILAMB Burned Area DROPPED 0.6485 -> 0.6416,
  below the 0.6482 promotion bar. So it was correctly NOT promoted.

### Diagnosis of WHY (done 2026-06-08, this is the key finding)

The whole official drop is the SPATIAL DISTRIBUTION SCORE (0.7691 -> 0.7252, -0.044). Every other
component held or improved (RMSE even got better, 0.4754 -> 0.4827, because magnitude is closer). And
it is a VARIANCE effect, not correlation: spatial correlation rho actually rose (0.54 -> 0.56), but the
model/ref std ratio sigma fell 0.77 -> 0.69. The suppression made an already-too-flat map flatter
(GFED5 burn-cell std 0.0111, canonical 0.0089, tropfix 0.0079). This is the SAME ceiling /
dynamic-range problem as [[error-distribution-scalar-finding]]. Killing tropical fire removed amplitude
the model could not spare.

LESSON: cutting the total cannot come from suppression ALONE — it flattens the field and the Taylor
spatial term punishes it. The next run must pair the tropical cut with KEEPING (or lifting) amplitude
in the cells that genuinely burn, so sigma stays near 1.

Two real scorer-fidelity bugs also surfaced (independent of the science):
1. The optimizer's internal Overall is an UNWEIGHTED mean of 4 components
   (`optimize_modelC_coupled.py:216`). Official ILAMB weights RMSE double: `(Bias + 2*RMSE + Seasonal +
   Spatial)/5` (verified: reproduces official Overall to 4 decimals for both models). FIX 1 below.
2. The internal per-component values are offset from official (internal RMSE 0.38 vs official 0.48,
   internal Bias 0.68 vs 0.70) because internal scores on the 1deg grid with burn-mask weighting while
   official scores on 0.5deg with ILAMB regridding. So internal Overall (0.6501) and official (0.6485)
   are DIFFERENT SCALES. Never compare the internal number to the official 0.6482 bar — re-score
   candidates with official ILAMB before any promotion decision (FIX 2 below).

### BEFORE YOU START (pre-flight checklist — this drive moves between Mac and Windows)

This drive is exFAT and moves between machines. On whatever machine you are on:
1. `source $(conda info --base)/etc/profile.d/conda.sh && conda activate ed-fire` and confirm
   `python -c "import optuna, xarray; print('ok')"` AND `which ilamb-run`. If the env is missing,
   recreate it: `conda env create -f environment.yml` (the yml is on the drive; the env is not).
2. Confirm git works: `git status`. (On the office Mac git is blocked by an unaccepted Xcode license,
   `sudo xcodebuild -license`; on Windows git works fine.) See PENDING COMMITS below.
3. If on WINDOWS: run the ILAMB scoring from a bash shell (WSL or Git Bash), not PowerShell — the
   CLAUDE.md scoring recipe uses `$PWD`/`ilamb-run`/`rm -rf`/`cp`. The Python scripts are path-portable.
4. exFAT GOTCHA: macOS leaves AppleDouble `._*` files all over the drive. ILAMB globs `*.nc` in a model
   folder and `._burntArea.nc` matches, so it will try to read an AppleDouble file as NetCDF and crash
   the model-dir merge. In ANY model folder you score, run `find . -name '._*' -delete` first. (Same
   failure family as the MonotonicityError gotcha in CLAUDE.md.)
5. Confirm the big data files are present (they ARE on the drive as of 2026-06-08): the two
   `global_baseline_*.nc` dumps, `ilamb_ref_official/DATA/{burntArea,fFire}/GFED5/*.nc`,
   `data/crujra/*_monthly.npy` (6 files), `ilamb/MODELS_LEADERBOARD*`.

### PENDING COMMITS — DONE 2026-06-08 (on Windows, git works)

All committed as separate, clearly-messaged commits and PUSHED to origin. Nothing canonical overwritten.
- The 2026-06-03 scalar/error-distribution session: already committed earlier as `e65149f`.
- The 2026-06-07 tropfix run (params.tropfix.json, logs; the .nc files are gitignored): `39adac4`.
- The 2026-06-08 diagnosis: `2b843f7`.
(Left untracked on purpose: the thesis-docs folder, a `betas (1).gfed5.json` download dup, and
`prep_monthly_inputs.LOCAL-BACKUP.py`. The thesis folder's `.gitignore` entry has literal quotes so it
never matches — fix the quoting if you want it actually ignored.)

### NEXT STEPS (in order)

1. ~~APPLY FIX 1~~ DONE (`929a816`): optimizer Overall now uses `(bias + 2*rmse_s + seas + spatial)/5`.
2. ~~APPLY FIX 2~~ DONE (`cb7e054`): optimizer dumps the top-K Pareto candidates (env `TOPK`, default 5;
   NSGA-II only), each as its own `models/C/params.{tag}.k{rank}.json` + a scoreable `burntArea.nc` in
   `ilamb/MODELS_TOPK_{tag}/ED-ModelC-{tag}-k{rank}/`, plus manifest `models/C/topk.{tag}.json`.
3. ~~RE-RUN to cut magnitude WITHOUT flattening sigma~~ LAUNCHED (running in background ->
   `logs/opt_tropfix2.log`): `PHYSICAL=1 MAG_BAND=1.12 FP_MIN=0.85 SAMPLER=nsga2 WARM=params.nsga2.json
   TAG=tropfix2 N_TRIALS=2500 TOPK=8`. Goal: total ~1.10-1.15x GFED5 while official Spatial holds near
   canonical's 0.7691 and official Overall >= 0.6482. If spatial still sags, the deeper lever is RAISING
   the African-savanna core toward GFED5's dynamic range (the ceiling problem), not more suppression.
4. TODO (do when the run finishes): SCORE the 8 top-K candidates with official ILAMB. They are already
   laid out as model dirs, so:
   `find ilamb/MODELS_TOPK_tropfix2 -name '._*' -delete` then score with
   `--model_root "$PWD/ilamb/MODELS_TOPK_tropfix2"` (CLAUDE.md recipe). Read `models/C/topk.tropfix2.json`
   for the candidate list. Check BOTH (a) annual Mha/yr is ~1.0-1.15x GFED5 and (b) official Overall >=
   0.6482 AND Spatial held. Pick the OFFICIAL winner, not the internal best (different grid/weighting).
5. TODO: Only promote if official Overall held/improved AND magnitude improved. Back up canonical first
   (CLAUDE.md table). Then regenerate emissions + figures, update HANDOFF/PROGRESS, commit.

Acceptance: magnitude ~1.0-1.15x GFED5 (down from 1.29x) AND official ILAMB Overall >= 0.6482 with the
Spatial Distribution Score not below ~0.76. The tropfix run already proved magnitude is achievable; the
open problem is doing it without losing spatial variance.

## >>> END READ-THIS-FIRST <<<

## This session (2026-06-03) — global-scalar / error-distribution exploration

Explored "the model is ~1.26x high everywhere, what if we just multiply by 0.8?" Answer: it fixes the
GLOBAL-SUM bias but not the ILAMB score, and the reason is the real finding.
- Best scalar: 0.792 matches the global mean (model 83.4 vs GFED5 66.1 Mha/month), 0.779 min global RMSE.
- Spatial cancellation: net signed bias +17.3 Mha = +46.1 over cancelling -28.8 under. Gross per-cell
  |bias| is 75.0 Mha (~4.3x the net). ILAMB scores per cell so it feels the 75. A 0.8x scalar drives net
  to ~0 but only drops gross 75.0 -> 67.7 (~11%). Correlation/phase (global r=0.405) is scale-invariant.
- The deeper cause is a **dynamic-range / ceiling problem** (seen in the per-cell scatter): model
  period-mean burned fraction caps at ~0.039 while GFED5 reaches ~0.10. Points form a horizontal band,
  not a 1:1 line. So errors split: over-prediction in low-fire dry/boreal cells (scalar helps) vs
  under-prediction in the high-fire African savanna core (scalar makes WORSE). For 1666 cells with
  GFED5>0.04, model 0.024 vs GFED5 0.054; scaled drops to 0.019. Net bias vanishes, ILAMB barely moves.
- Takeaway: bias is not uniform amplitude inflation, it is "too flat." Fix is slope/saturation in the
  refit (tighter MAG_BAND, currently 1.3 vs effective 1.26; plus the false-positive objective), not a
  post-hoc multiply. A global scalar is also metric-tuning a reviewer would flag.
- IMPORTANT: ILAMB score deltas above are EMULATED. This machine has no conda / `ed-fire` env / ILAMB
  (system python3 has xarray+cartopy only). `scripts/score_scaled_ba.sh` is ready to produce the OFFICIAL
  number once on a machine with the env. Also note: this machine's `git` is blocked by an unsigned Xcode
  license (`sudo xcodebuild -license`), so this session may need to be committed from another machine.
- Advisor asked about the error distribution; saved to auto-memory `error-distribution-scalar-finding.md`.

Artifacts this session (all new, none overwrite canonical):
- `NEW MAPS/Seasonal/1_global_timeseries_scaled08.png` — GFED5 vs original vs 0.792x overlay.
- `NEW MAPS/Seasonal/2_diff_scatter_scaled.png` — per-cell scatter + difference histogram (shows ceiling).
- `NEW MAPS/Seasonal/3_map_scaled_ba.png` — 4-panel map GFED5/original/scaled/(scaled-GFED5).
- `ilamb/MODELS_SCALED/ED-ModelC-scaled08/` and `-scaled0792/` — scaled `burntArea.nc` (not canonical).
- `scripts/score_scaled_ba.sh` — one-command official ILAMB scoring of canonical vs both scaled variants.

## >>> NEXT TASK TO RUN AT HOME (Claude: execute this) <<<

Goal from the professor: the model burns too much in total (1.26x GFED5) and that is wrong even where we
do not match GFED5 spatially. Bring the total down the PRINCIPLED way, not with a post-hoc scalar.

Why this is tractable (diagnosis already done 2026-06-03, annual basis):
- Model 1001 vs GFED5 793 Mha/yr -> excess +208 Mha/yr (1.26x).
- **76% of the excess (157 Mha/yr) is FALSE POSITIVES**: fire in cells where GFED5 is essentially zero
  (annual burned fraction < 0.5%). Mostly productive, seasonally-dry CLOSED-CANOPY TROPICAL FOREST
  (Amazon, Congo, SE Asia/Indonesia), plus scattered boreal/temperate.
- In cells that genuinely burn, model is ~balanced on net (over +394 cancels under -343 -> only +51).
- **Removing the false positives alone takes the total from 1001 -> 844 Mha/yr = 1.06x GFED5.** Magnitude
  essentially solved, WITHOUT touching the African savanna core (which is already too LOW, so a global
  scalar would wrongly cut it). See auto-memory [[error-distribution-scalar-finding]].

Do this (steps):
0. First commit the pending 2026-06-03 work (see auto-memory [[pending-commit-scalar-session]]). Then
   `conda activate ed-fire`; confirm git works (this needs a machine where Xcode license is accepted).
1. Add a TROPICAL CLOSED-CANOPY fire-suppression term to `fire_C` (in the Model C core; the suppression
   hook already exists and activates only when its params are present — the earlier GLOBAL veg term did
   NOT help, so make this one TROPICAL + canopy-specific, not global). Form: suppress burned fraction
   where AGB is high and the cell is closed-canopy tropical forest, e.g. multiply BA by
   `1 / (1 + (AGB/agb_crit)**k_veg)` gated to |lat|<~23.5 and high-AGB, with `agb_crit`,`k_veg` as new
   optimizable params. AGB field: `global_baseline_modelCfuel_inputs_1997-2016.nc` (already a driver for
   emissions). Keep it OFF by default (only active when the new params are in `params.json`).
2. Re-run the optimizer with a TIGHTER magnitude band and stronger false-positive weight so the total is
   forced down while the ILAMB spatial score is protected:
   `PHYSICAL=1 MAG_BAND=1.05 FP_MIN=0.85 W_FP=0.30 SAMPLER=nsga2 WARM=params.nsga2.json TAG=tropfix \
    N_TRIALS=2500 python scripts/optimize_modelC_coupled.py`
   (Canonical came from MAG_BAND=1.3 FP_MIN=0.80 SAMPLER=nsga2 WARM=params.nsga2.json. Start from there
   and tighten; do not overwrite canonical files — use the TAG.)
3. Regenerate BA from the new params (`scripts/reproduce_modelC.py`), score with official ILAMB
   (CLAUDE.md "How to score"), and check BOTH: (a) annual total Mha/yr is now ~1.0-1.1x GFED5, and (b)
   ILAMB Overall did not drop below the canonical 0.6482 (ideally it rises as false positives clear).
4. Regenerate the diagnostic figures (`scripts/diagnose_false_positives.py`, `scripts/maps_hybrid_ba_ffire.py`,
   the seasonal maps). Compare false-positive Mha before/after.
5. ONLY promote to canonical if total improved AND ILAMB held/improved. Back up canonical first (see the
   CLAUDE.md "DO NOT overwrite without backing up" table). Update HANDOFF_NOTE/PROGRESS and commit.

Acceptance: annual total ~1.0-1.1x GFED5 (down from 1.26x), false-positive Mha cut by most of the 157,
ILAMB Overall >= 0.6482. If ILAMB drops, the suppression is too aggressive — loosen `agb_crit`/band.

Fallback if the code change is too involved in one sitting: do step 2 alone (refit with tighter
MAG_BAND/W_FP, no new term) as a quick first cut, then add the tropical term in a second pass.

## >>> END NEXT TASK <<<

## TL;DR

The Model C burned-area submodule has been refit to be **coupled-consistent with Lei's ED GPP** and
tuned against **GFED5**. It is the canonical shipped model. A bundle was prepared for Lei. Seasonal
visualizations were built for Hurtt. The main open scientific problem is **tropical-forest false fire**
(we burn the Amazon/Congo interior where GFED5 is near zero).

## Where we are

### The model (canonical, shipped)
- Burned area: ILAMB Overall **0.6482, rank #3 of 10 external TRENDY** (behind CLM6 0.6562, ELM-FATES 0.6502).
- Fire emissions: ILAMB Overall **0.6465, rank #5 of 10**.
- Files: see CLAUDE.md "canonical model" table. `models/C/params.json` is the live parameter set.

### How it was produced (in order, all done)
1. Hybrid driver loader: CRUJRA climate + Lei's coupled GPP (closes Lei's correctness ask).
2. Fixed the optimizer scorer to be ILAMB-faithful Collier-2018 (centralized RMSE, crms normalizer,
   mass weighting). Earlier scorer was gaming the metric.
3. Constrained `fire_exp >= 1` so the suppression cascade behaves physically (values < 1 smear fire globally).
4. Added a physical objective: `0.55*ILAMB + 0.25*false_positive_score + 0.20*hotspot_score`, magnitude band.
5. NSGA-II multi-objective (ILAMB Overall vs false-positive rate) gave the best Pareto point -> canonical.
6. Retuned combustion betas for the new BA -> `betas.gfed5.json`.
7. A vegetation-AGB suppression term was tried; it did NOT help (0.6472 vs 0.6482), so it is OFF in the
   canonical model. The hook exists in `fire_C` (activates only if `k_veg`/`agb_crit` are in params).

### Figures built (in `NEW MAPS/`)
- `Hybrid_GFED5/BA_four_panel.png`, `fFire_four_panel.png`, bias stacks — model vs GFED5/CLM6/ELM-FATES.
- `Seasonal/1..4` — BA seasonal: global timeseries, regional cycles, peak-month map, Hovmoller.
- `Seasonal/5..8` — same four for fFire.
- `Seasonal/9_false_positive_map.png` — where we over-predict.

### Diagnosis of the over-prediction (important for the paper)
Area-weighted, Model C total is **1.26x GFED5** (1001 vs 793 Mha/yr) — close, better than CLM6/ELM-FATES
on magnitude. The error splits into:
- **False positives: 77 Mha (8%)** — fire where GFED5 is near-zero. Concentrated in **Amazonia (28 Mha)**,
  then SE Asia/Indonesia, E. North America, Central Africa forest, Europe.
- **Intensity: 1.16x** in cells that genuinely burn.
The single biggest fixable error is **tropical closed-canopy forest false fire** (Amazon, Congo). Our
climate+GPP formula sees productivity + seasonal dryness and lights forests that don't burn in reality.
This is exactly what CLM6's separate "tropical deforestation fire" type and ELM-FATES canopy structure
handle and we don't.

## What needs to be done (priority order)

1. **Send Lei the bundle** if not already sent. A draft email to Lei only (`lma6@umd.edu`, no CC) is in
   Gmail Drafts. The bundle is preserved at `NEW MAPS/Seasonal/modelC_for_lei_v2/` (the `/tmp` copy was
   cleared). Re-zip that folder and attach. Contents are the canonical model + maps + a README. NOTE the
   draft says "version that uses your full dump (climate too)" is available on request — we have NOT
   built that full-dump version; if Lei asks, re-run the optimizer with the climate drivers also read
   from the dump (revert `load_coupled_drivers` to pull D_bar/T_air/P_* from the dump).

2. **Tropical-forest false-fire fix (most promising scientific lever).** Add a suppression that kills
   fire in high-AGB closed-canopy cells, applied specifically to the tropics (not the global veg term we
   already tried). Target the 28 Mha Amazon error directly. Re-run NSGA-II, re-score, regenerate maps.
   This is the next real improvement, not more global tuning.

3. **Hurtt's seasonal-visualization request is DONE** (the `Seasonal/` figures). If he wants more, the
   natural additions are: per-region Taylor diagrams, or an interannual-variability (detrended anomaly)
   timeseries.

4. **Paper / writeup.** Framing points already established: we are the only model tuned to GFED5; Model C
   is coupled-consistent with ED's own GPP so it transfers into the coupled run; CLM6/Li2013 uses a
   four-fire-type structure (non-peat, agricultural, tropical-deforestation, peat) plus explicit
   ignition + population suppression that Model C lacks — that structural gap is the honest explanation
   for our residual over-prediction.

## Reproduce from scratch on the new machine

```bash
source $(conda info --base)/etc/profile.d/conda.sh && conda activate ed-fire
cd <repo>
# 1. regenerate BA from canonical params
python scripts/reproduce_modelC.py
# 2. emissions
python scripts/compute_emissions.py --model ED-ModelC-Hybrid \
  --betas-json models/combustion-params/betas.gfed5.json --out-suffix ""
# 3. score (see CLAUDE.md "How to score with official ILAMB")
```

## Data the new machine MUST have (large, mostly git-ignored)

These are not in git (`.gitignore` excludes `data/`). If the new computer syncs the same Google Drive
folder they come along automatically; otherwise copy them:
- `data/crujra/*_monthly.npy` (CRUJRA climate drivers)
- `global_baseline_modelC_inputs_1997-2016.nc` (Lei's coupled GPP dump, ~420 MB, repo root)
- `global_baseline_modelCfuel_inputs_1997-2016.nc` (AGB for emissions, repo root)
- `ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc` and `.../fFire/GFED5/fFire.nc` (references)
- `ilamb/MODELS_LEADERBOARD/` and `ilamb/MODELS_LEADERBOARD_FFIRE_GFED5/` (comparator TRENDY models)

## Housekeeping done this session

Cleaned ~4 GB of dead intermediate output: sync duplicates, ~53 stale `ilamb_out_*` dirs (kept only
`ilamb_out_nsga2_lb` and `ilamb_out_ffire_nsga2`), ~43 old logs, ~24 dead `params.*.json` and ~14 dead
`burntArea.*.nc` variants. Not yet touched (unsure, from earlier project phases): `models_full/`,
`out_terms/`, `stageB_models/`, `out_ffire/`, and the older `ED-ModelC-{GFED5,GFED5cont,GFED5type,
l02s7,Emissions,EmpiricalEmit}` leaderboard slots. Check references before deleting those.
