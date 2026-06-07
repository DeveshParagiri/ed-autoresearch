# HANDOFF NOTE — ED fire submodule (Model C)

Last updated: 2026-06-03. Read `CLAUDE.md` first for environment, file locations, and conventions.
This note is the "where are we, what's next" narrative.

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
