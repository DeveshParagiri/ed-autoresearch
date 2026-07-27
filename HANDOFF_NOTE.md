# HANDOFF NOTE — ED fire submodule (Model C)

Last updated: 2026-07-27. Read `CLAUDE.md` first for environment, file locations, and conventions.
This note is the "where are we, what's next" narrative.

## >>> READ THIS FIRST (2026-07-27) — MODEL F SHIPPED TO LEI; F vs G DECIDED <<<

>>> FULL TECHNICAL RECORD: `GDP_HUMAN_TERM_FINDINGS.md` (Steps 1-9). All committed + pushed to
origin/coupled-refit-gfed5. The 2026-07-24 block below has the George meeting directions. <<<

### The two candidate models (renamed to follow C, D, E)
- **Model F = the regional-GDP model** (was "A"/"regional-GDP"/"ED-ModelC-gdpreg"). Model E on ED's
  own dump drivers + a smooth per-region GDP human-suppression term. **RECOMMENDED / promoted to Lei.**
  Official ILAMB vs GFED5: **burned area 0.679, fFire emissions 0.667** (best of our models on BOTH
  TRENDY fire rows). Base params `models/C/params.coupledE_gdp.json`; regional gamma
  `data_human/gdp_regional_gamma.json`; BA `ilamb/MODELS_GDP_REGIONAL/ED-ModelC-gdpreg/burntArea.nc`.
- **Model G = F + grass-curing term** (was "B"). Fixes temperate-grassland BURNED AREA (Kazakh steppe
  6->10, Australia 22->39) but scores LOWER (BA 0.654, fFire 0.656) and adds ~no carbon there (grass =
  low biomass). Curing is `CURING=1` in `optimize_modelC_coupled.py`; base `params.coupledE_cure.json`.

### THE DECISION (scientific): promote **Model F** to Lei, NOT G
Reasons: (1) the coupled run is a CARBON budget -> emissions matter, and F leads emissions; G's grassland
burned area emits almost no carbon. (2) F wins BOTH scored TRENDY rows (Burned Area Extended + Fire
Emissions Extended) -- Richard confirmed both are scored in the official ILAMB scorecard, and both use
the AGGREGATE metric that F wins. (3) F is simpler/more robust to couple (no extra unvalidated pathway).
G only wins if the coupling needs grassland burned-area EXTENT itself (not carbon) -- unlikely for TRENDY.
Curing stays a PAPER result (it's the cleanest demo of the metric thesis), not a coupled component.

### Lei email — DRAFTED, ready to send (in Gmail thread "Model E code question")
Draft recommends Model F, notes the George-meeting provenance of the GDP term, points Lei at branch
coupled-refit-gfed5 + the spec, gives the 2 implementation checks (live GPP + dryness scale; lat orient).
BEFORE SENDING: attach 3 files (repo root) -- `COUPLING_SPEC_for_Lei.md`, `modelF_vs_G_burned_area.png`,
`modelF_vs_G_emissions.png`. (Gmail API can't attach; big NetCDFs stay on the branch.) Gotcha: Gmail
auto-links filenames ending .nc/.cc/.md (country TLDs) -- the draft was reworded to drop those extensions.

### IMMEDIATE NEXT (pick up here)
1. Confirm the Lei email went out (attach the 3 figures/spec first). Coupling deliverable is otherwise DONE.
2. Advisor figure for the next weekly George meeting: 4-panel "human factor" (score ladder C->D->E->F,
   fire-vs-GDP, biome-gamma map, regional fidelity bars). Designed, deferred, not built.
3. Paper: rebuild the outline TOPIC-SENTENCE-first (George's writing ask). Then the schematic diagram.
4. Future physics (separate item the fFire work surfaced): the Australia/boreal EMISSIONS gap is a
   BIOMASS/fuel limitation, not a burned-area one -- G fixed Australia's area but not its carbon.

### Scripts added this session (all committed)
fire_vs_{gdp_country,gdp_partial,pop_partial,landuse_partial}.py; add_gdp_{term,regional}.py;
make_ed_gdp_netcdfs.py + check_ed_gdp_netcdfs.py (the Lei NetCDFs, validated to 0.6783);
diag_steppe_terms.py; test_steppe_gamma.py; proto_curing.py; assemble_regional_cure.py; sweep_regcure.py;
make_george_pair.py; make_george_figure.py + make_george_ffire_figure.py (F-vs-G figures).

## >>> READ THIS FIRST (2026-07-24) — ADVISOR MEETING (George, 07/23) + HUMAN/GDP TERM DONE <<<

>>> FULL TECHNICAL RECORD IS IN `GDP_HUMAN_TERM_FINDINGS.md` (repo root). Meeting notes/transcript:
`paper_gmd/meeting with advisor on 07_23_2026.pdf`. <<<

### The meeting (main directions George gave)
1. **Paper thesis** = "how much can sophisticated OPTIMIZATION improve a global fire model?" The headline
   metric is the improvement of the UNOPTIMIZED expert baseline vs the optimized model (before/after ILAMB).
2. **Biome/PFT-specific, not continent-specific.** Understand the PHYSICAL reason regional models work
   (= our structural indistinguishability finding). Then generalize to PFT-specific / seasonal. This is the
   same as keying params to VEGETATION STATE (solves cerrado-vs-Africa AND Lei's prognostic-migration issue).
3. **Human term — the explicit assignment:** "go pull GDP + population + fire frequency for every country,
   build the fire-vs-GDP plot", fit a functional form, bolt on MULTIPLICATIVELY. Make it socioeconomic, NOT
   latitude-dependent. => DONE this session (see below).
4. **More physics = next steps:** wind, topography (fires burn uphill), landscape fragmentation, humans.
5. **Coupling tiers:** only forward-runnable + long-time-series forcings go in the global carbon budget
   (land use, population, GDP qualify; roads / remote sensing do NOT — paper-only exploration). Remote
   sensing's real value = vegetation STRUCTURE, not biomass.
6. **Writing:** rebuild the outline as TOPIC-SENTENCE-only (one point per paragraph); then bullets, then prose.
7. **Schematic diagram** must be fixed (box-type consistency, ED vs non-ED drivers, optimization loop +
   implementation/coupling loop, stack of candidate models) and "make its way to the ED model".
8. **Logistics:** meet WEEKLY for the next 3 weeks. George sending his doc + a commented parallel version.

### What was BUILT this session (2026-07-24) — the GDP human-suppression term
- `scripts/fire_vs_gdp_country.py` -> raw plot: fire falls with wealth, slope -0.92/decade, r -0.55 (164 countries).
- `scripts/fire_vs_gdp_partial.py` -> CLIMATE-CONTROLLED: partial wealth slope -0.70/decade, r -0.47, p 4e-10.
  Wealth keeps 76% of the raw slope after removing climate => the socioeconomic signal is REAL, not a
  savanna-climate artifact. (This is the answer to George's obvious first objection.)
- `scripts/add_gdp_term.py` -> bolted a GDP multiplier onto the single-global dump model (params.coupledE.k2),
  magnitude pinned, scored on official ILAMB: **Overall 0.6547 -> 0.6603 (+0.0056), ALL in the Spatial score
  (0.805 -> 0.821), Seasonal untouched.** Model prefers a MILD term (gamma ~0.15-0.3); over-suppresses past 0.3.
- `scripts/fig_gdp_term.py` -> `gdp_term_figure.png` (4-panel advisor figure: driver, multiplier, dBA map, score-vs-gamma).
- Data: World Bank GDP/pop CSVs (JSON API blocked here, CSV download endpoint works), Natural Earth 50m polygons.
  Gridded GDP driver cached at `data_human/gdp_pcap_grid_1deg.npy` (forward-runnable => satisfies Lei's coupling rule).

### WHERE IT ENDED (all committed + pushed to origin/coupled-refit-gfed5)
- GDP term evolved: bolt-on (0.6603) -> JOINT refit (0.6695, retired the fire_exp=7 hack, gamma=1.36 but
  over-burned Asia) -> REGIONAL/biome gamma (**0.6783, BEST MODEL**, coupling-ready, regionally faithful,
  no seams). `scripts/add_gdp_regional.py`, gamma in `data_human/gdp_regional_gamma.json`,
  BA in `ilamb/MODELS_GDP_REGIONAL/ED-ModelC-gdpreg/`. Beats paper E (0.6646) and CLM6 (0.6562) on dump climate.
- HUMAN-FACTOR SURVEY COMPLETE: GDP = only forcing with independent skill (in model). Population tested,
  no signal (p=0.93). Land use tested, real but redundant with GDP + the land-use-weighted GPP (p=0.13).
  Roads/remote-sensing = no long time series (paper-only). All in `GDP_HUMAN_TERM_FINDINGS.md` Steps 1-7.
- Optimizer now has `GDP_TERM=1` (fits gdp_gamma jointly). `SEASONAL_TRANSFORM=1` is REQUIRED to reproduce
  the coupledE family (the bolt-on/assemble use 1-exp(-rate/12)).

### IMMEDIATE NEXT (pick up here)
- (a) Refresh the advisor figure for the weekly George meeting: biome-gamma map (hot Africa, cold Asia) +
  the C->D->E->GDP->regional score ladder. Highest value (meeting is the near deadline).
- (b) fFire/combustion retune on the new 0.6783 BA, then package the regional model as the coupling handoff to Lei.
- (c) Rebuild the paper outline topic-sentence-first (George's writing ask).
- Boreal is the one weak region (0.52x): fuel-limited, GDP suppresses wealthy boreal -- needs a base fuel fix, not a human term.
- Lei thread below still open (reply drafted, not sent). The regional model IS the coupling deliverable (dump climate, no seams, forward-runnable).

## >>> READ THIS FIRST (2026-07-23) — LEI COUPLING THREAD: coupling-ready model BUILT, reply DRAFTED not sent <<<

>>> FULL TECHNICAL RECORD IS IN `COUPLED_REFIT_FINDINGS.md` (repo root). Read that + this block. <<<

### The situation
Lei (lma6@umd.edu, thread "Model E code question") asked for TWO things before Model E goes into
coupled ED for the GCB/TRENDY run (due Aug 31):
1. Use ED's own `D_bar` from the dump, so ED has ONE dryness definition, not two.
2. Drop the per-continent regional blocking (blocky seams propagate into coupled veg/carbon).

### What was built (nothing promoted to canonical; paper Model E untouched)
- `DUMP_CLIMATE=1` mode in `scripts/optimize_modelC_coupled.py` (default OFF = canonical unchanged):
  climate from the dump instead of CRUJRA, widened dryness ranges (ED D_bar hits ~4.8e6 vs CRUJRA ~7e4).
  Also added env `FIRE_EXP_LO/HI` to bound the fire_exp concentrator.
- SINGLE-GLOBAL dump fits (`params.coupledE*.json`, `params.coupledE_fx*.json`): score WELL on ILAMB
  (0.6523-0.6532) but are REGIONALLY BROKEN — boreal 2 Mha vs GFED5 50, S.America 196 vs 65. The good
  global total is compensating error. Constraining fire_exp did NOT fix it.
- DIAGNOSIS (`scripts/diag_coupledE.py`): (a) fire_exp~7 crushes marginal (boreal) cells and
  over-concentrates savanna; (b) STRUCTURAL — cerrado and African savanna have near-identical drivers
  (base_product 0.56 vs 0.53) but 6x different GFED5 fire, so ONE global form cannot separate them.
  => a single global equation is OUT for a carbon budget.
- SOLUTION (Option 3): keep per-continent params but BLEND THEM SMOOTHLY (Gaussian on log params,
  SIGMA=4 deg) so there are no hard seams. Proven on CRUJRA first (0.6649 -> 0.6641, ~free).
- PRODUCTION BUILD: 7 continental fits on dump climate (`params.coupledE_{af,bor,sam,sea,eur,nam,aus}.json`)
  + global-dump fallback, smooth-blended by `scripts/assemble_smooth_coupledE.py`.
  Output: `ilamb/MODELS_SMOOTH_COUPLED/ED-ModelC-smooth/burntArea.nc`.
  **ILAMB Overall 0.6426, every region 0.88-1.38x GFED5, global 1.05x, boreal 63 vs 50.**

### The key scientific finding (paper's thesis, confirmed on ED's drivers)
ILAMB Overall ranks the regionally BROKEN single-global model (0.6532) ABOVE the regionally faithful
smooth model (0.6426). The aggregate metric does NOT reward regional fidelity and can prefer a
physically worse model. For a carbon budget the regional fluxes are what matter.

### Known limitation we raised OURSELVES (do not drop this)
The smooth model's params are still keyed to GEOGRAPHY (lat/lon boxes), so they do NOT migrate when
the vegetation does. Over a long TRENDY run with land-use change + prognostic vegetation, a cell that
converts forest->pasture keeps its old region's params. Smoothing fixes the artifacts, not this.
REAL FIX = key params to VEGETATION STATE (PFT fractions / tree cover / AGB) instead of lat/lon. That
also solves the cerrado-vs-Africa discrimination. Precedent exists: the tropical suppression term
already keys off AGB.

### >>> IMMEDIATE NEXT STEP <<<
A reply to Lei is DRAFTED IN GMAIL (thread "Model E code question", reply to Lei's 2026-07-22 22:26
message) but NOT SENT. Before sending:
- ATTACH `coupling_ready_maps.png` manually (Gmail API cannot attach; the draft text references it).
- DELETE the older stale unsent draft on that thread if still there.
The draft asks Lei TWO questions that gate all further work:
  Q1. Is the objection the HARD SEAMS, or ANY spatial parameter variation at all?
      (seams -> smooth model is the answer; any variation -> we must choose between a regionally
       wrong model and a regionally right one, needs a conversation)
  Q2. Can the fire submodule see VEGETATION STATE at runtime inside ED (PFT fractions / tree cover /
      AGB per patch)? If yes, the vegetation-keyed build is tractable and is the right next model.
Files to hand Lei when asked: the 7 param JSONs + `assemble_smooth_coupledE.py` + the BA nc.
Recommendation was to commit those to Devesh's repo rather than email loose files (NOT done, needs
Richard's approval — outward action).

### ENVIRONMENT GOTCHAS LEARNED 2026-07-23 (important)
- THIS Mac DOES have the full stack in `ed-fire`: optuna 4.8, cmaes, ilamb-run, cartopy, scipy,
  python-pptx. CLAUDE.md's "home Mac = system python only, no conda/ILAMB" is STALE for this machine.
- T7 drive throws recurring EPERM ("Operation not permitted") on file I/O and silently ROLLED BACK
  .md edits once. FIX = eject + replug the drive. ALWAYS `sync` after writing, and re-verify edits.
- Long background jobs get KILLED after ~20 min. Run continental refits ONE region per job.
- ILAMB `build_dir` MUST be on local APFS (scratchpad), never the exFAT T7, or the harvest step
  crashes on regenerated `._*` AppleDouble files. Clean `._*` from model dirs before every run.

### ALSO DONE 2026-07-23 (unrelated to Lei)
- CPA Section II.C schematic BUILT -> `cpa/framework_goals.png` (generator `cpa/make_framework.py`),
  caption written into `cpa/CPA_OwusuAnsah.md`. **CPA Section II is now COMPLETE** pending Richard's
  read. Next CPA targets: Section VI (annotated bib) or Section VII (dissertation idea paper).
  See `cpa/STATUS.md`.
- Advisor talking-aid deck for the C/D/E ladder -> `ModelCDE_advisor_deck.pptx`
  (generator `scripts/build_ladder_deck.py`), 11 slides, speaker notes carry the plain-language script.

## >>> READ THIS FIRST (2026-07-20) — FULL DRAFT + ALL FIGURES DONE; MODEL D PR SENT TO DEVESH <<<

The GMD paper ("Development and Optimization of a Global Fire Model using Autoresearch AI") has a
COMPLETE draft (Abstract + Intro + Methods + Results + Discussion + Conclusions) and a COMPLETE figure
set, all on the clean single-lever Model E. ALL paper work lives in `paper_gmd/` (GITIGNORED, drive-only,
NOT on GitHub) — `git log` on the coupled-refit-gfed5 branch DOES show it (paper_gmd/ files are force-added
there), but it never goes to the remote.

>>> TO RESUME: read `paper_gmd/STATUS.md` (exact resume point) then `paper_gmd/DRAFT.md` (all prose). <<<

### What is DONE (this stretch, through 2026-07-20)
- FULL DRAFT complete and verified against the clean E (0.6646).
- FIGURES complete + captioned + cited in the prose:
  - Fig 1 autoresearch schematic (2.2) — rebuilt honest + compact near-square layout (figures/make_schematic.py).
  - Fig 2 burned-area maps (3.1), Fig 3 per-cell scatter (3.2/3.3).
  - Fig 4 fire-emissions maps (3.5) — PROMOTED from supplement to a MAIN figure; GFED5 vs E, stacked,
    verified totals 3.40 / 3.15 PgC/yr (figures/make_fig_ffire_maps.py).
  - Fig S1 difference maps (cited 3.1 P3); Fig S2 regional seasonal cycles (cited 4.3, figures/make_fig_seasonal.py).
  - CUT: the threshold-sensitivity figure (robustness point carried by one sentence in 3.3;
    threshold_sensitivity.py still prints the grounding table).
- References consolidated in `paper_gmd/references/CITED_PAPERS/` + MANIFEST.md (21 cited PDFs). Zotero
  workflow: drag the folder in -> "Retrieve Metadata for PDF"; add optuna/NSGA-II by DOI; Copernicus style.

### MODEL D shared with Devesh (OUTWARD ACTION — on GitHub now)
- Devesh cleaned up `origin/main` (github.com/DeveshParagiri/ed-autoresearch) into a `models/paper/`
  layout (C.json, D.json, E/). His `D.json` was a WRONG placeholder (amp-enabled, 15 params, objective
  "ILAMB Overall"); his own note flagged it as a stub awaiting the colleague file.
- The real Model D = `models/C/params.paperD.k1.json` (12 params, C form, spatial objective, trial 1497,
  spatial_taylor 0.7495, official ILAMB 0.6411). PR #1 opened replacing his D.json with a corrected
  version (objective field fixed off the hardcoded-label bug):
  https://github.com/DeveshParagiri/ed-autoresearch/pull/1  (branch modelD-paper-params).
  AWAITING Devesh's review/merge. Our coupled-refit-gfed5 branch was NOT pushed; paper_gmd never left the drive.
- E needs NOTHING — Devesh already has all E region params in `models/paper/E/`.
- GH AUTH GOTCHA: two accounts are logged in; the DEFAULT active one (RichardOwusuAnsah-Apps) has NO
  write access (403). Use `gh auth switch --user RichardOwusu-Ansah` before pushing to this repo.

### NEXT / OPEN (all decisions, nothing blocking)
- Real model names (ED-stock/C/D/E placeholders) — advisor decision; baked into Abstract too, so a rename
  is a find-replace across DRAFT.md.
- CRUJRA: no public v3.5 exists; Harris et al. 2020 + Kobayashi et al. 2015 (the 2 MISSING refs) are
  blocked on the data preparer (Lei / fire_autoresearch). Confirm version before citing.
- Sync the finished DRAFT.md sections into the Google Doc (the live paper the group reads).
- Devesh's email: he set up a `paper/` dir that live-renders `paper.md` (his in-repo paper, distinct from
  our drive-only paper_gmd/) — worth looking at with Claude Code. Zoom Monday to discuss.
- Verified-citation standard is UNCHANGED (full-text PDF, exact quote+page). Do not drop it.

The 2026-06-12 and earlier blocks below describe the pre-paper MODEL-DEVELOPMENT phase (still valid as
background, but the paper phase above supersedes them for "what to do next").

## >>> READ THIS FIRST (2026-06-12, later) — seasonal-aware combustion does NOT help (stopped early) <<<

Tried the SEAS_W lever on the COMBUSTION objective to recover the fFire seasonal gap (0.794 vs
canonical 0.825). FINDING: it does not work, and the reason is structural, not tuning. Combustion is
fFire = BA x (beta-weighted fuel) x dryness-gate; the betas are 4 per-pool scalars + D_REF that only
rescale per-cell MAGNITUDE. The month-to-month SHAPE of fFire is inherited almost entirely from the BA
seasonal cycle, which this step does not touch. Evidence from the SEAS_W=0.4 fits (vs the SEAS_W=0
fits): Africa seas 0.680->0.684, S.America 0.769->0.780, N.America 0.542->0.542 (no change), Boreal
0.710->0.710 (no change). The seasonal score barely moves and overall does not improve. So the emissions
seasonal gap lives UPSTREAM in the burned-area seasonality, not in combustion. The real lever would be
improving the continental-BA seasonal cycle (the BA-side SEAS_W work), which trades against the BA
spatial win - a separate, bigger effort, not pursued now.

STOPPED the run early (Richard's call) at 6/7 regions to continue on a different machine. The per-cont
emissions result STANDS at official fFire 0.6490 (the block below) - the seasonal-aware run would not
have beaten it (best case ~0.650, still under canonical 0.6534; keep-best assembler cannot regress it).

STATE ON DISK: 6/7 seasonal betas in models/combustion-params-continental-seas/betas.{Africa,SAmerica,
NAmerica,Boreal,SEAsia,Australia}.json (Europe NOT fitted - run was killed mid-Europe). To RESUME if
ever wanted: `REGION=Europe SEAS_W=0.4 python scripts/tune_combustion_continental.py --out
"$PWD/models/combustion-params-continental-seas"`, then `CAND=seas python scripts/
assemble_combustion_continental.py` writes ED-ModelC-continental-percont-seas/fFire.nc to officially
confirm the (expected null) seasonal effect. But the conclusion above already settles it - this is
optional confirmation only. The PRODUCTION emissions model is still ED-ModelC-continental-percont (0.6490).

## >>> READ THIS FIRST (2026-06-12) — per-continent COMBUSTION closes the emissions regression <<<

CHASED the BA-vs-emissions tradeoff (next-step #2). The continental BA win had cost ~0.02 on emissions
(0.6534 -> 0.6334) because the single global beta set no longer mapped the savanna-shifted BA onto
GFED5's fFire pattern. FIX: fit combustion betas (beta_{leaf,fine,coarse,litter} + D_REF) SEPARATELY per
continent, same REGION machinery + keep-best-per-region as the BA work. Canonical untouched, not promoted.

RESULT (official ILAMB, single run, GFED5 fFire):
  | fFire model                          | Bias | RMSE  | Seas  | Spatial | Overall |
  | canonical k4 (ED-ModelC-Hybrid)      |0.6914|0.5112 |0.8249 | 0.7282  | 0.6534  |
  | continental + GLOBAL betas (-Hurtt)  |0.6913|0.5056 |0.7783 | 0.6861  | 0.6334  |
  | continental + PER-CONTINENT betas    |0.6919|0.5088 |0.7942 |**0.7412**| **0.6490** |
Emissions 0.6334 -> **0.6490 (+0.0156)**: spatial 0.686 -> 0.741 (now ABOVE canonical's 0.728), regression
vs canonical shrunk from -0.020 to **-0.0044**. Magnitude 3.21 PgC/yr (GFED5 3.40). All 7 regions adopted
(each beat global betas on its own box by +0.05..+0.13 overall). Remaining shortfall is the SEASONAL cycle
(0.794 vs 0.825) - the same seasonal-vs-spatial frontier as BA, not a magnitude/pattern problem.
SO THE TRADEOFF IS LARGELY RESOLVED: continental model is now BA 0.6723 (big win) AND emissions ~matched
(0.6490 vs 0.6534), making it defensible to promote on BOTH axes.

WHAT'S ON DISK (gitignored .nc, committed params/scripts):
- `ilamb/MODELS_LEADERBOARD_FFIRE_GFED5/ED-ModelC-continental-percont/fFire.nc` - the new stitched fFire.
- `models/combustion-params-continental/betas.{Africa,SAmerica,NAmerica,Boreal,SEAsia,Australia,Europe}.json`
- `scripts/tune_combustion_continental.py` (REGION-restricted beta tuner),
  `scripts/assemble_combustion_continental.py` (keep-best stitch + scoreable fFire).
REGENERATE: `for R in Africa S.America N.America Boreal SEAsia Australia Europe; do REGION=$R python
scripts/tune_combustion_continental.py; done` then `python scripts/assemble_combustion_continental.py`.
SCORE: ILAMB build_dir MUST be on local APFS (e.g. /tmp), NOT the exFAT drive - macOS regenerates ._*
AppleDouble files in the output dir mid-run and ILAMB's harvest step crashes reading them as NetCDF.

NEXT (unchanged priority): #1 send Lei the email (LEI_EMAIL_DRAFT.md) - still gates promotion. The
emissions tradeoff (#2) is now ANSWERED. Optional: chase the fFire seasonal cycle (0.794->~0.825) with a
seasonal-aware combustion objective (analogue of SEAS_W on BA).

## >>> SESSION-END SUMMARY (2026-06-11) — read this, then the dated blocks below for detail <<<

WHERE WE ARE. Built a NEW best model this session: the **continental Model C** (per-continent parameters
+ a fuel-driven savanna term). It is NOT promoted; canonical on disk is still tropfix2-k4, and the
lab/Lei-shipped model is still PRE-tropfix2. The continental model lives at
`ilamb/MODELS_CONTINENTAL/ED-ModelC-continental/burntArea.nc` (gitignored; regenerate with
`SEASONAL_TRANSFORM=1 python scripts/assemble_continental.py`).

THE RESULT (official ILAMB, single-model runs):
- Burned area: **0.6723** (was canonical 0.6473; above CLM6 0.6562), magnitude **1.03x** GFED5, per-cell
  1:1 slope **0.34 -> 0.65** (r 0.49 -> 0.70). This is the big win on George's 1:1 bar.
- Fire emissions: **0.6334** (a ~0.02 REGRESSION vs canonical k4 0.6534), magnitude 3.42 vs 3.40 PgC/yr.
  Tradeoff: emissions = BA x fuel, so moving fire to low-biomass savanna + cutting the Amazon over-burn
  removes carbon GFED5 puts in higher-fuel forest fires. BA win, emissions secondary cost.
- BOTH held-out validations PASS: years (fit 2001-2012, test 2013-2016) r drop -0.05; cells (blocked
  10deg-tile CV) r drop -0.015. The result is genuine structure, not overfitting.

HOW IT WAS BUILT (the arc, all flag-gated, canonical-safe):
1. B (spatial scorer, score_spatial.py) showed the 1:1 bar is an r problem (slope=r*sigma, r~0.5 capped).
2. A (fire_amp / then the physical FUEL_AMP fuel-scaled amplitude) let the rate exceed 1.
3. Per-continent fits (REGION env) + the Africa FUEL form (diag_africa_residual.py found Model C had GPP
   backwards in savanna) broke the r wall: Africa r 0.47 -> 0.66. assemble_continental.py stitches them.
4. SEAS_W blended seasonal into the per-continent objective to recover the seasonal cycle.

ED COUPLING CHECK (read Lei's ED_Source_Code/GlobalED): our approach is fundamentally CONSISTENT - the
fuel fix IS ED's native fire (fuel*fp1*dryness^10), the transform matches ED's 1-exp disturbance, rate>1
is allowed (cap fire_max_disturbance_rate=0.2 needs raising), per-region branching exists. 4 concrete ED
changes documented for Lei (see the 2026-06-11 detail block). The 2x coupled over-burn Lei saw = GPP/
biomass feedback -> recalibrate in coupled run, structure transfers.

FIGURES: NEW MAPS/continental_model/ (BA map, fFire map, global timeseries, regional seasonal, 1:1
scatter; gitignored, regenerate with `python scripts/figures_continental_model.py`). Continental betas at
models/combustion-params-continental/betas.gfed5.json.

WHAT'S NEXT (in priority order):
1. **Send Lei the email** (draft saved in `LEI_EMAIL_DRAFT.md`) to get the coupled-side answers we need
   (monthly fire timing / PATCH_FREQ, the fire_max cap, per-site region tagging, his coupled refit). This
   gates promotion.
2. **Decide the BA-vs-emissions tradeoff**: accept the BA win as primary (George's bar) with emissions
   secondary, OR chase the emissions pattern - the combustion step uses one global beta set; a
   per-continent / fuel-aware combustion (same idea that fixed BA) could lift the 0.686 emissions spatial
   score. Optional follow-up.
3. **Promotion decision** (Richard's call) - gated on the Lei coupling check. If promoting, back up
   canonical first (CLAUDE.md table), swap params/BA/betas/fFire, update the canonical table + scores.
4. Optional deeper BA: N.America + Australia still on the global model (their fits regressed); the seasonal
   frontier in Africa/Amazon (can't gain seasonal without losing spatial with the current form).

RESUMING ON A NEW MACHINE (Mac mini): `git pull` gets all code + params + docs. The .nc OUTPUTS and big
data are gitignored - you need the driver dumps (global_baseline_*.nc), GFED5 refs, CRUJRA npy, and the
ed-fire/edfire conda env present (Drive sync usually brings the data; recreate the env from
environment.yml). NOTE per CLAUDE.md: a Mac with system-python-only has NO optuna/ILAMB - you can
diagnose/emulate but must run the optimizer + official ILAMB on a machine with the env (Windows/edfire).
Regenerate the continental model from committed params: assemble_continental.py -> compute_emissions.py
(continental betas) -> figures_continental_model.py.

## >>> READ THIS FIRST (2026-06-11) — ED-source coupling consistency CHECKED + held-out validation <<<

Read the ED source (`ED_Source_Code/GlobalED`, Lei's code) to check whether our offline work can become
coupled-canonical. VERDICT: fundamentally consistent, in places MORE consistent than our legacy form.
Key files: fire.cc, disturbance.cc (line 33: disturbance_rate[1]=fire()), patch.cc:718/762 (area burned),
mortality.cc, edmodels.h, ED_params.defaults.cfg.
- ED prognostic fire (fire.cc update_fuel:216): `ignition_rate = fuel * fp1 * (dryness/30000)^10`. Fire
  scales LINEARLY with fuel -> our Africa fuel-amplitude fix IS ED's native mechanism (we rediscovered it).
- Burned area (patch.cc:718,762): `area*(1 - exp(-rate*deltat*PATCH_FREQ))`. ED uses the EXPONENTIAL
  disturbance = our SEASONAL_TRANSFORM (1-exp), not the legacy /12. In ED monthly mode (PATCH_FREQ=1,
  which Lei is moving toward per edmodels.h:134) it is 1-exp(-rate/12) per month -> exactly our transform.
- Rate>1: ED's rate is NOT a product of [0,1] sigmoids; it is capped by `fire_max_disturbance_rate`
  (default 0.2 in ED_params.defaults.cfg). So rate>1 is allowed structurally; the cap just needs RAISING
  to reach GFED savanna levels (one config value).
- Per-continent: ED already branches fire-suppression by region (AFRICA/SOUTH_AMERICA/EUROPE/... in
  fire.cc) and treefall by climate_zone (disturbance.cc:36). Per-continent fire params fit the
  architecture; a GLOBAL run needs per-SITE region/zone tagging (currently keys off one data->region).
- All our drivers (dryness=D_bar, precip, temp, fuel/biomass, GPP) are ED state variables.
4 CONCRETE ED CHANGES for Lei to adopt this: (1) put Model C's driver response (or the fuel amplitude +
per-continent calibration) into update_fuel; (2) raise fire_max_disturbance_rate above 0.2; (3) run
monthly patch dynamics PATCH_FREQ=1; (4) add per-site region/zone tagging for global runs. The 2x coupled
over-burn Lei saw is the GPP/biomass FEEDBACK (fire->veg->fuel->fire), so params need RECALIBRATION in the
coupled run, but the STRUCTURE/PHYSICS transfer.

HELD-OUT VALIDATION (is the result genuine or overfit?):
- Held-out YEARS (fit 2001-2012, score unseen 2013-2016; FIT_Y0/FIT_YF; validate_holdout.py): active-fire
  r TRAIN 0.695 -> TEST 0.645 (drop only -0.05), slope 0.643->0.609. PASS - generalizes across years.
- Held-out CELLS (blocked 10deg-tile spatial CV; CELL_HOLDOUT; validate_holdout_cells.py): active-fire
  r TRAIN-tiles 0.700 -> TEST-tiles (unseen cells) 0.685 (drop only -0.015), taylor identical. PASS - the
  per-continent params generalize to unseen cells, did NOT memorize the map. This is the STRONGER overfit
  test and it passes cleanly. (Reproduce: `ASSEMBLY=cell python scripts/assemble_continental.py` then
  `python scripts/validate_holdout_cells.py`.)
BOTH validations pass (years drop -0.05, cells drop -0.015) -> the per-continent + fuel result is genuine
structure, not overfitting, on both the time and space axes. The science is defensible.
assemble_continental.py now takes ASSEMBLY=best|ho|cell (writes to MODELS_CONTINENTAL{,_HO,_CELL}).

## >>> READ THIS FIRST (2026-06-10, latest) — C DELIVERED: per-continent + fuel form break the pattern wall <<<

Both tasks done: (1) all 7 continents fitted, (2) the Africa FORM change. The pattern wall (r~0.5) is
broken. Canonical untouched, nothing promoted. The best model is the assembled continental in
`ilamb/MODELS_CONTINENTAL/ED-ModelC-continental/burntArea.nc` (regenerate: `SEASONAL_TRANSFORM=1 python
scripts/assemble_continental.py`).

THE BREAKTHROUGH - Africa fuel form (`scripts/diag_africa_residual.py` -> `fire_C` FUEL_AMP):
The Africa residual (GFED - Model C) is dominated by GPP/fuel: GFED fire RISES with GPP in savanna
(fuel-limited) but Model C was ANTI-correlated (-0.29) because the global GPP hump suppresses productive
cells. Added a fuel-scaled amplitude `rate *= 1 + fuel_k*GPP/(GPP+fuel_half)` (fuel capacity = period-mean
GPP). Africa r 0.469 -> **0.664** (~the 0.676 driver ceiling). This is ALSO the physical (fuel-selective)
form of the rate>1 lever, replacing the crude constant fire_amp. fuel_k=4.71.

THE RESULT - best assembled continental (fuel-Africa + Boreal + S.America + SEAsia + Europe; N.America &
Australia kept global because their fits regressed r - assembly keeps-best-per-region):
  per-cell 1:1 (active-fire): r 0.505 -> **0.695**, sigma 0.93, slope 0.496 -> **0.647**, taylor 0.84.
  official ILAMB: Bias 0.7514 (best), RMSE 0.4753, Seas **0.7450**, Spatial **0.8756** (was 0.7617
  canonical!), Overall 0.6645.
George's 1:1 band has climbed across the whole arc: canonical slope 0.34 -> spatial-k1 0.50 -> continental
**0.65** (~2/3 up the diagonal); r 0.49 -> 0.70. Figure: NEW MAPS/proto_seasonal/per_cell_scatter_continental.png.

SEASONAL-AWARE RE-FIT DONE (SEAS_W=0.35 blends a region seasonal score into the SPATIAL_OBJ objective).
Re-fit Boreal/SEAsia/Europe (Africa+Amazon picker reverted to their warm = spatial-only, see below).
Best model now = `ilamb/MODELS_CONTINENTAL/ED-ModelC-continental/burntArea.nc` (assemble_continental.py
uses africafuel + borealseas + samerica + seasiaseas + europeseas). Official ILAMB:
  Bias 0.7479  RMSE 0.4822  Seas **0.7748** (was 0.745)  Spatial **0.8747**  Overall **0.6723**.
So Overall 0.6645 -> **0.6723** (best of ALL versions; canonical 0.6473, CLM6 0.6562) with Spatial held
at 0.875 and magnitude 1.03x. The seasonal-aware blend works.

STILL slightly short of full seasonal recovery (Seas 0.775 vs canonical 0.823) because the candidate
PICKER ranked Africa+Amazon by pure spatial_taylor, so their seasonal-blend candidates were not chosen
(picker reverted to the warm = spatial-only fit). FIXED in optimize_modelC_coupled.py (rank by the blend
when SEAS_W>0; manifest now records spatial_seas). NEXT: re-run JUST Africa (FUEL_AMP) + S.America
seasonal fits with the fixed picker, re-assemble -> should pull Seas toward 0.82 and Overall past 0.68.

WHAT EACH CONTINENT NEEDED (the paper's per-continent story):
- Africa: FORM change (fuel term) -> r 0.47->0.66. The pattern fix.
- Boreal: magnitude (was burning 4% of GFED) -> fixed by regional params.
- S.America/Amazon: magnitude (over-burned 2x) -> fixed (near its driver ceiling on r, can't do more).
- SEAsia, Europe: regional params helped r a lot (Europe -0.15 -> +0.15).
- N.America, Australia: regional fits regressed; kept global (try fuel form / seasonal-aware objective).

REMAINING (in priority order): (1) seasonal-aware per-continent objective + re-fit (recovers Overall);
(2) try the fuel form on the other savanna continents (Australia, N.America) which the constant-amp fits
failed; (3) smooth the continent seams toward the single "unified model"; (4) FUEL form is the physical
fire_amp, so the fuel-selective upgrade is effectively done; (5) Lei coupling check still gates promotion.

## >>> READ THIS FIRST (2026-06-10, later) — workstream C STARTED: drivers are good, per-continent fits running <<<

Started the per-continent work (C). Two findings + machinery + a running campaign. Canonical untouched.

SCOUTING (committed cd17818, `scripts/diag_continent_headroom.py`): for each continent, compared Model C's
current spatial r (spatial-k1) to the best a free regression of GFED5 on the EXISTING drivers+squares can
reach (held-out). Result decides HOW to build C:
  region          modelC_r  driver_ceiling  verdict
  Africa            0.513      0.676         FORM headroom (build structure)
  S.America         0.387      0.474         near driver ceiling (Amazon = magnitude/suppression, not pattern)
  N.America         0.390      0.599         FORM headroom
  Boreal Eurasia    0.202      0.542         FORM headroom (big)
  Trop/SE Asia      0.096      0.465         FORM headroom (big)
  Australia         0.415      0.765         FORM headroom (big)
  Europe           -0.215      0.457         FORM headroom (model is ANTI-correlated!)
=> The DRIVERS are good enough (they support r 0.46-0.77 everywhere); the GLOBAL formula is the
bottleneck. Per-continent tuning has large headroom in 6 of 7 regions with the SAME drivers (no new data
needed). Only the Amazon is near its driver ceiling. This validates C strongly.

MACHINERY (committed f919c61): `optimize_modelC_coupled.py` now takes `REGION=Africa|S.America|N.America|
Boreal|SEAsia|Australia|Europe` to restrict the spatial objective + magnitude band + FP masks to one
continent. REGION="" => global (bit-identical). `scripts/assemble_continental.py` stitches per-continent
params into ONE global prediction (each cell uses its continent's params; fallback = spatial-k1 elsewhere)
-> the meeting's "one unified model". Piecewise/hard borders for now (smooth the seams later).

RESULT (Africa, Boreal, S.America fitted; logs/opt_continents.log). The per-continent approach WORKS,
and it moved r for the first time. Assembled via `scripts/assemble_continental.py` ->
`ilamb/MODELS_CONTINENTAL/ED-ModelC-continental/burntArea.nc`. Official ILAMB:
  | model                       | Bias  | RMSE  | Seas  | Spatial | Overall |
  | canonical k4 (lab-shipped)  | 0.6972| 0.4771| 0.8234| 0.7617  | 0.6473  |
  | spatial-k1 (A+B global)     | 0.6902| 0.4980| 0.8177| 0.7985  | 0.6605  |
  | CONTINENTAL (2 regions)     | 0.7096| 0.5114| 0.8203| 0.8064  | **0.6718** |
EVERY component improved; Overall 0.6718 = +0.0245 over the lab-shipped model and clear of CLM6's 0.6562
(would be #1 on BA; positioning, not "we beat them"). Global spatial r rose 0.505 -> **0.546** (the
FIRST real movement on r), taylor 0.752 -> 0.765, RMSE-on-burning-cells better.

WHAT FIXED WHAT (per-region, the key science):
- Boreal: was burning ~4% of GFED (sigma 0.09, magx 0.04) -> after its own fit magx 0.81, sigma 0.77.
  Catastrophic under-burn FIXED by a regional param set.
- S.America/Amazon: over-burned 2x (magx 2.03, sigma 3.30) -> fit brought it to magx 1.01, sigma 0.78.
  Over-burn FIXED.
- Africa: its regional fit UNDERPERFORMED the global spatial-k1 (r 0.469 -> 0.389), so the assembly KEEPS
  spatial-k1 for Africa (see assemble_continental.py REGION_PARAMS). LESSON: Africa's gap is PATTERN (r),
  and re-tuning the current FORM regionally does not raise r -> pattern-limited regions need a FORM change
  (a new term), not just regional params. Magnitude-broken regions (Boreal/Amazon) are fixed by params.

So C splits the problem cleanly: (1) magnitude-broken continents -> regional param fits work great (done
for Boreal+Amazon); (2) pattern-limited continents (Africa + the near-zero-r ones) -> need form changes.

NEXT:
1. Fit the 4 remaining continents (N.America, SEAsia, Australia, Europe) - all high-headroom, mostly
   magnitude/pattern fixes; re-assemble. The assembly already keeps-best-per-region (drop any fit that
   does not beat the global on its region, as done for Africa).
2. For Africa (and other pattern-limited regions): try a FORM change (new term) - the headroom diagnostic
   says the drivers support r~0.68 in Africa vs the 0.47-0.51 the current form reaches.
3. Smooth the continent seams toward a single "unified model" (currently hard borders).
4. FUEL-SELECTIVE fire_amp upgrade (physical form) and the Lei coupling check still gate ANY promotion.
The continental .nc is gitignored (regenerate with assemble_continental.py); params + logs are committed.

## >>> READ THIS FIRST (2026-06-10) — B scorer built, the 1:1 bar is an r-problem, A+B refit running <<<

Started workstream B (new goodness-of-fit) + A (fire-physics). Both flag-gated, canonical untouched.
The B scorer immediately produced the most important finding of the whole 1:1 thread.

WHAT WAS BUILT (committed df4bc20, bfd1b4d):
- `scripts/score_spatial.py` — workstream-B spatial-pattern scorer on the cells that ACTUALLY BURN
  (GFED5>0). Metrics per cell: r (spatial correlation), sigma (std ratio), slope = r*sigma (the 1:1
  line slope George wants -> 1), magx, Taylor skill (single scalar for the optimizer). `spatial_metrics()`
  is importable. Includes a per-continent breakdown.
- `scripts/diag_rate_amp.py` — sweep of the new `fire_amp` lever (A).
- `reproduce_modelC.py` fire_C — optional `fire_amp` multiplier lets the annual fire rate exceed 1.0
  (savanna multi-burn). Active only when `fire_amp` is in params; canonical bit-identical when absent.
- `optimize_modelC_coupled.py` — `RATE_AMP=1` adds fire_amp to the search; `SPATIAL_OBJ=1` makes the
  first NSGA-II objective the spatial Taylor skill (B) instead of ILAMB Overall. Both flag-gated.

THE KEY FINDING — George's 1:1 bar is an r (correlation) problem, NOT just amplitude:
- The per-cell 1:1 slope is ~0.35 (George wants 1.0). slope = r * sigma, and r is only ~0.48-0.50 on
  active-fire cells. So even with a PERFECT sigma=1, the slope caps at ~0.5. Amplitude fixes (seasonal
  transform, fire_amp) raise sigma (0.70->0.80 already) but CANNOT reach the diagonal until r improves.
- The `fire_amp` sweep confirms A breaks the structural rate<=1 cap: the per-cell ceiling reaches GFED5's
  0.104 at fire_amp~2, BUT a uniform lift overshoots sigma (1.7) and magnitude (2.1x). So A needs a
  co-fitted reshape, and even then is bounded by r.

THE REGIONAL BREAKDOWN — why r is low, and the case for workstream C (run `python scripts/score_spatial.py`):
The single global formula does OPPOSITE, wrong things per continent (active-fire cells, tropfix2-k4):
  - Africa (savanna core): r~0.5 (right pattern) but UNDER-burns (magx 0.67, sigma 0.5) -> A is the fix here.
  - S.America (Amazon):     r~0.3, OVER-burns 3x (magx 2.7, sigma 2.5) -> needs SUPPRESSION; A makes it worse.
  - Boreal Eurasia:         r~0.36, UNDER-burns 4x (magx 0.26, sigma 0.33) -> needs its own boreal regime.
  - Trop/SE Asia, Australia: r~0.1-0.2 (near-zero pattern skill).
  - Europe:                  r~0.00 (no skill at all).
The regions need OPPOSITE fixes, so ONE global amplitude/criterion cannot win. This is the quantitative
case for continent-specific fire models (workstream C). seasonal-k1 even HURT boreal/SE-Asia r
(0.36->0.22, 0.20->0.11) because dry-season concentration misfires in monsoonal/boreal regimes.

A+B REFIT DONE (logs/opt_spatial.log, 65 min). WINNER = **spatial-k1** (trial 2305, fire_amp=5.45,
fire_exp=1.52). Official ILAMB single-model run (ilamb_out_topk_spatial/), apples-to-apples:
  | model        | Spatial | Overall | magnitude |
  | canonical k4 | 0.7617  | 0.6473  | 1.11x |
  | seasonal-k1  | 0.7797  | 0.6495  | 1.11x |
  | spatial-k1   | 0.7985  | 0.6605  | 0.94x |
So official Overall 0.6473 -> **0.6605 (+0.0132)** and Spatial 0.7617 -> **0.7985 (+0.037)** -- the best
official BA score yet, ABOVE CLM6's 0.6562 (it would be leaderboard #1 on burned area; frame as
positioning, not "we beat them"). On George's 1:1 plot (B scorer, active-fire cells): the band climbed
from slope 0.344 (canonical) -> 0.385 (seasonal) -> **0.496** (spatial-k1), sigma 0.70 -> **0.98**
(solved), per-cell ceiling 0.039 -> **0.095** (now reaching GFED5's 0.104). Figure:
`NEW MAPS/proto_seasonal/per_cell_scatter_spatial_k1.png` (3-panel progression). Candidates dumped to
`ilamb/MODELS_TOPK_spatial/`, params `models/C/params.spatial.k{1..6}.json`, manifest `topk.spatial.json`.

THE WALL CONFIRMED: r went 0.49 -> 0.505 only. All the slope gain came from sigma. slope=r*sigma so the
band PLATEAUS at slope ~ r ~ 0.5. George's slope=1 is unreachable without raising the correlation r ->
workstream C (continent-specific structure). The A+B refit banked the dynamic-range half of the problem
and set a record official score; C is the only lever left for the correlation half.

CAVEATS before any promotion (NOT promoted; canonical untouched):
- fire_amp maxed at 5.45 (top of its 1..6 range): physically 5 fires/yr is too high; the optimizer is
  just using the scalar to force sigma->1. The PRINCIPLED version is the FUEL-SELECTIVE fire_amp (step 3
  of the agreed plan) — amplitude tied to grass-fuel curing so only genuine savanna multi-burns, not a
  global scalar (which CLAUDE.md flags as metric-tuning). Do that before promoting spatial-k1's approach.
- RMSE dipped 0.477 -> 0.498 (the higher amplitude enlarges per-cell errors in the over-burning regions,
  e.g. Amazon) but the Spatial gain dominates Overall. A region-aware amplitude (C) would avoid this.
- Coupling check with Lei still gates promotion (the transform + now the rate>1 must be reproducible in
  coupled ED).

NEXT (the fork for Richard): the path to George's bar runs through workstream C (continent-specific
structure to raise r), not more global tuning. Two concrete sub-steps already scoped:
  1. Upgrade fire_amp to the FUEL-SELECTIVE form (step 3) and re-fit — makes spatial-k1's gain defensible
     and likely recovers the RMSE dip.
  2. Begin C: per-continent fire structure to raise r (Africa amplitude, Amazon suppression, boreal
     regime). The regional breakdown (run `python scripts/score_spatial.py`) is the blueprint. Confirm
     scope with George (near-term vs eventual) — see the open question in the meeting block below.

## >>> NEXT TO DO — Fire Meeting outcomes (2026-06-09) — Richard will action these later <<<

These came out of the June-09 fire meeting (George + group). They RE-PRIORITIZE the work. None are
started; the SEASONAL_TRANSFORM refit (block below) is a partial down-payment on A/B but is NOT the
finish line. Canonical tropfix2-k4 is still shipped and unchanged. NOT promoting seasonal-k1 yet.

GEORGE'S HARD BAR (restated, firmer): the model is NOT acceptable to him until we are **EQUAL on the
1:1 line** of the per-cell scatter (model vs GFED5 burned fraction). Not "climbing toward" the diagonal
— ON it. We are currently at sigma ~0.87, per-cell ceiling 0.050 vs GFED5 0.104. So this is the gate
on everything.

The four workstreams:

A. FIRE-PHYSICS — let the fire RATE exceed 1.0 (meeting agenda item "fire-physics"; our long-standing
   "Cause 1"). Today `fire_C` is a product of [0,1] sigmoids so the rate hard-caps at 1.0 yr^-1, which
   (even with the corrected transform) caps the per-cell band at ~0.05. To reach GFED5's 0.104 the peak
   savanna cells need annual_frac -> ~1, i.e. rate must be allowed > 1 via an additive/exponential fuel
   term instead of a pure product. This is the direct mechanical lever for George's 1:1 bar. Structural
   change to the model core; keep it behind a flag like SEASONAL_TRANSFORM so canonical is unchanged.

B. NEW OPTIMIZATION / GOODNESS-OF-FIT CRITERIA — the paper's intellectual core, and the reason the band
   is flat. We optimized ILAMB Overall, which is dominated by bias/RMSE and barely rewards spatial
   pattern ("ILAMB does not produce a good global total"; "we are missing a very strong spatial
   pattern"). Build and COMPARE alternative objectives:
     (a) ILAMB Overall (current baseline)
     (b) global mean / average ANNUAL burned area  ("average annual burned area should be the focus")
     (c) PIXEL-LEVEL SPATIAL CORRELATION on the annual-mean map (the one that should pull the band onto
         the diagonal)
   Restrict the fit/scoring to GRID CELLS THAT ACTUALLY BURN (GFED5 > 0) — meeting page-2 item 3.
   Design an objective that explicitly WEIGHTS the spatial pattern. Also wanted: a "structural"
   goodness-of-fit criterion (page-2 item 2) alongside the statistical ones. Paper framing: "different
   optimization criteria give different results, and that is interesting precisely because ILAMB is the
   community standard." This is a scoring/criteria module + a comparison figure/table; touches the
   optimizer's objective, not necessarily the model core. OPEN QUESTION for Richard: does the new
   criterion REPLACE ILAMB as our optimization target, or sit ALONGSIDE it as a paper comparison?

C. CONTINENT-SPECIFIC MODELS + A UNIFIED MODEL (page-2 item 1) — "different fires need different models
   for different continents, and then one unified model that explains them." Separate fire formulations
   per continent (African savanna vs Amazon deforestation vs boreal vs ...), then unify. Bigger
   structural research direction (echoes CLM6's multi-fire-type design that Model C lacks); longer
   horizon. OPEN QUESTION: near-term task or eventual? (Confirm scope with George before building.)

D. LEI'S COUPLED RUN (agenda item 3) — Lei reported the coupled global total burn is **2x too high**,
   and he refitted the equation (all parameters) in the coupled run. Sub-tasks: (a) test with Africa,
   (b) reconcile the 2x global over-burn. This is the coupling-consistency thread and is partly Lei's
   job, but it directly gates whether OUR offline refits transfer. It is ALSO the same coupling check
   that blocks promoting any transform change (the legacy /12 even-spread matched coupled-ED output;
   see the SEASONAL_TRANSFORM block below). Coordinate with Lei (lma6@umd.edu).

SUGGESTED ORDER (Claude's read, not George's ruling): A and B are the same problem from two ends — B's
spatial-correlation objective is what should DRIVE the fit onto the 1:1 line, and A is the mechanism
that lets peak cells get there. Natural to build B's scoring/criteria module first (low risk, no
canonical change, produces the paper's comparison figure), then add A's rate>1 term and refit against
the new criterion. C is longer-horizon; D is coordinated with Lei. CONFIRM priority with Richard/George
before starting — see the OPEN QUESTIONS under B and C.

## >>> READ THIS FIRST (2026-06-09, Windows session) — SEASONAL_TRANSFORM refit DONE, k1 is the official winner (NOT yet promoted) <<<

CONTEXT: George said he will not be satisfied until the model is good on the **1:1 plot** (the per-cell
scatter of model vs GFED5 burned fraction). The scatter was a flat band capping at ~0.039 while GFED5
reaches 0.104 — the "too flat" / dynamic-range ceiling. The Mac session (below) diagnosed it and
prototyped the fix; THIS Windows session wired the fix behind a flag and ran the full refit.

WHAT THIS WINDOWS SESSION DID (committed; canonical files NOT touched):
1. Committed the Mac work (commit 000b57f). Pushed.
2. OFFICIAL-scored the Mac prototype vs canonical k4 (apples-to-apples single-model run): the corrected
   transform 1-exp(-rate/12) raised official Spatial 0.7617 -> 0.8135 and Overall 0.6473 -> 0.6596 with
   NO component dropped. Direction CONFIRMED (the emulated +Spatial held under official ILAMB). Caveat as
   flagged: prototype was un-retuned at ~1.47x GFED5, so not promotable — the refit fixes that.
3. WIRED a `SEASONAL_TRANSFORM=1` env flag into BOTH `optimize_modelC_coupled.py` (`ed_transform`) and
   `reproduce_modelC.py` (`main`). OFF by default => canonical k4 behavior is bit-unchanged (commit 5c3fc46).
4. RAN the refit (52.7 min): `SEASONAL_TRANSFORM=1 PHYSICAL=1 MAG_BAND=1.12 FP_MIN=0.85 SAMPLER=nsga2
   WARM=params.tropfix2.k4.json TAG=seasonal N_TRIALS=2500 TOPK=8`. 6 Pareto candidates dumped to
   `ilamb/MODELS_TOPK_seasonal/`, params to `models/C/params.seasonal.k{1..6}.json`, manifest
   `models/C/topk.seasonal.json`. (params.seasonal.json == k1, trial 2438.)
5. OFFICIAL-scored all 6 + canonical k4 in ONE single-model run (`ilamb_out_topk_seasonal/`):

   | model            | Bias  | RMSE  | Seas  | Spatial | Overall | mag  |
   |------------------|-------|-------|-------|---------|---------|------|
   | canonical k4     | 0.6972| 0.4771| 0.8234| 0.7617  | 0.6473  | 1.11x|
   | **seasonal-k1**  | 0.6917| 0.4796| 0.8169| **0.7797** | **0.6495** | **1.11x** |
   | seasonal-k2      | 0.6946| 0.4843| 0.8123| 0.7645  | 0.6480  | 0.95x|
   | seasonal-k3/k4   | 0.6909| 0.4731| 0.7868| 0.7758  | 0.6399  | 0.97x|
   | seasonal-k5      | 0.6906| 0.4706| 0.7903| 0.7709  | 0.6386  | 0.95x|
   | seasonal-k6      | 0.6883| 0.4701| 0.7902| 0.7581  | 0.6353  | 0.91x|

   OFFICIAL WINNER = **seasonal-k1 (trial 2438)**: Overall 0.6473 -> **0.6495 (+0.0022)**, Spatial
   0.7617 -> **0.7797 (+0.018)**, magnitude **1.11x = same as canonical**. Strictly better than canonical
   on the metrics that matter, at equal magnitude. (The refit gives back some of the prototype's +0.052
   Spatial because MAG_BAND pulls the 1.47x inflation back to 1.11x — expected tradeoff; k1 is the best
   balance. The lower-mag k2-k6 score lower because Seasonal drops ~0.03 and they go slightly UNDER GFED5.)
6. 1:1 FIGURE (`scripts/per_cell_scatter_seasonal_k1.py` -> `NEW MAPS/proto_seasonal/
   per_cell_scatter_seasonal_k1.png`): per-cell period-mean ceiling lifted **0.0393 -> 0.0503**,
   sigma_ratio (model/GFED5 burnable-cell std) **0.768 -> 0.865**, mean diff ~0 (magnitude unchanged).
   Band is climbing the diagonal. Still short of GFED5's 0.104 max because Cause 1 (rate<=1 cap) remains.

NEXT STEPS (in order — promotion is NOT done; needs Richard's call + a coupling check):
1. COUPLING CHECK WITH LEI (lma6@umd.edu) BEFORE promoting. The legacy `/12` even-spread existed to MATCH
   what coupled ED writes to TRENDY-format burntArea (see reproduce_modelC.py docstring). The corrected
   `1-exp(-rate/12)` is per-month disturbance fraction — Lei's coupled ED must be able to emit the same
   sub-annual fire timing. CONFIRM this is reproducible in the coupled run before k1 becomes canonical.
2. IF coupled-consistent AND Richard approves: PROMOTE k1. Back up canonical first (CLAUDE.md table):
   cp models/C/params.json -> params.PRE-seasonal.json, cp betas + the two burntArea.nc + fFire.nc to a
   backups_PRE-seasonal/ dir. Then `SEASONAL_TRANSFORM=1 python scripts/reproduce_modelC.py` with k1's
   params copied to params.json (regenerates burntArea.nc; ALSO copy to MODELS_LEADERBOARD/ED-ModelC-Hybrid/
   — reproduce only writes the MODELS/ED-ModelC-final one). RETUNE combustion betas for the k1 BA
   (`scripts/tune_combustion_params.py`), regenerate fFire, re-score emissions. Update CLAUDE.md canonical
   table + scores, HANDOFF/PROGRESS, commit. NOTE: emissions betas were retuned for k4's BA; k1's BA is
   spatially redistributed (more dry-season concentration) so betas SHOULD be re-tuned, not reused.
3. OPTIONAL deeper lever (Cause 1) if George wants the band to reach GFED5's 0.104: let the fire RATE
   exceed 1.0 (additive/exponential fuel term instead of a pure product of [0,1] sigmoids). Bigger
   model-structure change; the transform fix alone moved sigma 0.77 -> 0.87, not all the way to 1.

The Mac-session diagnosis that motivated all this is preserved below as background.

CONTEXT (Mac session, 2026-06-09 — the diagnosis + prototype that this Windows session built on):
Canonical files were NOT touched. New scripts + a tagged prototype only.

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

[The original Mac-session "NEXT STEPS ON WINDOWS" list — commit, official-score, wire the flag, refit,
score TOPK — is DONE as of this Windows session; see the top block. Steps that remain (Lei coupling
check, optional Cause-1 lever) are restated in the top block's NEXT STEPS.]

Acceptance for a promotable result (MET by seasonal-k1 except the coupling check): official BA Overall
held/improved vs 0.6473 (0.6495 ✓), Spatial UP / sigma toward 1 (0.7617->0.7797 ✓), magnitude ~1.0-1.15x
GFED5 (1.11x ✓), AND a coupling-consistent transform (NOT yet confirmed — needs Lei).

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
