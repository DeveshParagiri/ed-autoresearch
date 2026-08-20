# HANDOFF NOTE — ED fire submodule, the GMD paper, and the CPA

Last updated **2026-08-19, Windows session**. Read `CLAUDE.md` first, then this file top to bottom.
Branch `coupled-refit-gfed5`. **All commits are LOCAL. DO NOT PUSH** unless the session says otherwise.

## HOW TO ORIENT IN FIVE MINUTES

1. This file, all of it.
2. `git log -20 --oneline`.
3. `cpa/STATUS.md` for the portfolio and `cpa/CPA_Presentation_notes.md` for the committee talk.
   **`cpa/` AND `paper_gmd/` ARE NOT IN GIT.** Both are gitignored, lines 70 and 73. They exist only
   on the Drive, so `git log` will never show that work. Do not force-add them. History was rewritten
   once with git-filter-repo to remove them after a `git add -f` put them in.
4. `paper_gmd/TOPIC_SENTENCES_v3.md` is the paper live outline.
5. `paper_gmd/references/VERIFIED_CITATIONS.md` before writing any sentence with a citation in it.

---

## 2026-08-19, WINDOWS SESSION. COUPLING CONSISTENCY, THE FUEL TERM, AND THE HUMAN TERM

Three threads ran. Which version can actually go into Lei coupled ED, making the fuel term dynamic,
and finding a human driver that survives George rule that it must reach 1850.

### 1. WHICH VERSION IS COUPLING CONSISTENT

**`coupledE_fx` / `coupledFW` is the coupling legal version, official ILAMB 0.6532.** Model I is not,
because a per continent parameter set cannot be carried into a global coupled run without seams. That
is the same objection that ruled out Model F earlier, and Model I inherits it.

**Model F beats the default coupled scheme on 20 of 20 ILAMB components**, so the direction is right.
But **both under burn badly, 168 and 177 Mha against 793 observed.** Scored 2001 to 2016 against
both GFED4 and GFED5 at Lei request. `scripts/prep_coupled_for_ilamb.py` does the slicing.

**A trap that silently drops a model from ILAMB.** The time and time_bounds encodings must share an
epoch. With different epochs ILAMB builds 193 edges for 192 months and drops the model without an
error. Both are now written with `days since {y0}-01-01` and calendar noleap.

**I was wrong about the fire cap and Lei corrected me.** I attributed the under burn to a low
`FIRE_MAX`, from `ED_params.defaults.cfg` on the Drive. That file does not match the run, which
already used 5.0. `scripts/diag_fire_cap.py` shows the cap is not the binding term, 5.0 and 1.0 both
give 793 Mha offline. Conceded in the email draft. **Do not diagnose a coupled run from a config file
on the Drive.**

### 2. THE FUEL TERM IS NOW DYNAMIC

It was a whole record mean, which is a constant in a coupled run and therefore not a fuel term at all.
It is now a **causal trailing mean over the previous `FUEL_WINDOW` months**, `trailing_mean()` in
`scripts/reproduce_modelC.py`, default window 60. Causal matters, a centred window would let the model
see the future.

### 3. THE HUMAN TERM. GDP IS OUT, POPULATION IS IN

**GDP is disqualified.** It does not reach 1850, and George rule for the coupled runs is that a driver
must. This is worth 793 against 317 Mha, so it is the largest single missing piece and losing it hurts.

**Land use can act as the human term** and is already in the dump, `LANDUSE_TERM=1` in the optimizer.
If it were used for attribution it would take a new model letter, since it changes one attribute.

**Population density is the replacement.** HYDE, from Lei, 0.5 degree annual 1700 to 2025, gridded not
national, already used by TRENDY. `scripts/make_population_driver.py` regrids it. Three transformations
each of which can ruin the field silently, latitude runs north to south in the source, 0.5 to 1 degree
must be an **area weighted mean because density is intensive**, and annual to monthly is a repeat, not
an interpolation, because population has no seasonal cycle.

**The July population test was wrong, and it is worth knowing why.** It used NATIONAL AVERAGE density
and found nothing, p equals 0.93. The population fire relationship is **humped**, so both a linear
correlation (r = +0.139) and a national average destroy it. The gridded hump fits at a peak of
9.77 capita per km2, against 10 to 35 in the literature.

`POP_TERM=1` adds `pop_amp`, `pop_peak`, `pop_sig`. Peak is expressed as a DENSITY rather than as a
position on the log axis, because every parameter is sampled log uniform and a log axis position can
be negative.

### THE POPULATION RESULT, AND WHY IT IS NOT PROMOTABLE AS IT STANDS

Unconstrained, **official ILAMB 0.6711 against a 0.6554 control**, a real gain. **But burned area went
to 1179 Mha against 793 observed.** Richard flagged this correctly. It is the paper own thesis
repeating itself, a higher aggregate score on a physically worse model, which is exactly the Model G
against Model I result.

**So the run was repeated with `MAG_BAND=1.3`, and it finished at 2500 trials.** Internal 0.6126
against the warm start 0.6094, **875 Mha, 1.10x observed**, `models/C/params.coupledPOPmag.json`.

**THE ANSWER, FROM A CONTROLLED PAIR. `ilamb_out_pair/`. POPULATION SURVIVES AND IS WORTH HAVING.**

Two runs, identical config, one flag apart, `logs/opt_pairPOP.log` and `logs/opt_pairCTL.log`.

    DUMP_CLIMATE=1 SEASONAL_TRANSFORM=1 FUEL_WINDOW=60 MAG_BAND=1.3
    SAMPLER=tpe SEED=42 N_TRIALS=2500 WARM=params.coupledFW.json   POP_TERM=1 | 0

The warm starts agree, 0.6002 and 0.5989, which is the check that the pair is actually controlled.
Every earlier attempt at this comparison failed that check.

| | Official Overall | Bias | RMSE | Seasonal | Spatial | Mha/yr |
|---|---|---|---|---|---|---|
| control, `coupledFW` | 0.6532 | 0.7102 | 0.4655 | 0.8009 | 0.8240 | 921 |
| pairCTL, refit, no population | 0.6489 | 0.6934 | 0.4945 | 0.8203 | 0.7420 | 866 |
| **pairPOP, refit, with population** | **0.6652** | 0.7031 | 0.5096 | 0.8258 | 0.7777 | **722** |

**Population is worth +0.0163 official against its own control**, and it also beats the standing
0.6532. Magnitude goes 866 to 722 against 793 observed, so 1.09x over becomes 0.91x under, both
inside the band.

**The fitted hump is the part that matters more than the score.** `pop_amp` 3.77, `pop_peak` 23.95
capita per km2, `pop_sig` 0.96. The optimizer could place that peak anywhere from 0.05 to 1000 and
could set the amplitude to effectively zero, which is exactly what it did in the earlier run. It
instead chose a strong hump centred at 24 people per km2. **The independent binning of GFED5 puts
the observed peak between 10 and 35.** That is corroboration from a direction the objective knew
nothing about, and it is the argument to give Lei and to put in the paper.

**I WROTE THE OPPOSITE CONCLUSION TWICE BEFORE THIS AND BOTH WERE WRONG.** First from a partial
optuna log at trial 1432, then from the `coupledPOPmag` run which fitted `pop_amp` at 0.0027 and
looked like the term being selected against. The visible difference is that this pair carries
`SEASONAL_TRANSFORM=1` and that run did not, but that cannot be confirmed because the run is not
reproducible. **Which is the reason it must not be the basis of a claim.** Use the pair.

### A TRAP THAT VOIDED A WHOLE RUN. `DUMP_CLIMATE` IS NOT VISIBLE IN THE ENV VAR LIST

**A coupling-mode run and a CRUJRA run are not comparable and never belong in the same table.**
`DUMP_CLIMATE=1` takes D_bar, T_air and P from Lei ED dump instead of CRUJRA. It costs offline GFED5
skill BY DESIGN, because ED D_bar is a derived diagnostic rather than an observation. That is why the
paper Model E used CRUJRA and why the coupling series does not.

I refitted a control without it, got 0.6381, and nearly reported that as the cost of removing the
population term. It was the cost of changing the climate driver. **The tell is the FIRST LINE of the
log**, either `[setup] COUPLING-READY: climate ... all from global_baseline...` or `[setup] CRUJRA
climate from data/crujra/`. **Check that line before comparing any two runs.** The warm start score
disagreeing between two runs off the same WARM file is the symptom.

**A second and worse finding from the same check. The historical `coupledPOPmag` run cannot be
reproduced.** Matching drivers, seasonal transform, fuel window, trop mask, sampler and seed still
gives a warm start of 0.5989 against its recorded 0.6094. Something about how it was launched is not
recoverable from its log. **So its 0.6601 is not a baseline anyone can defend.** The optimizer records
the fitted parameters but NOT the environment it was run under, which is the actual gap. Worth fixing
by dumping the full env into the params json at write time.

Because of that the population question is being settled by a FRESH CONTROLLED PAIR, `logs/opt_pairPOP.log`
and `logs/opt_pairCTL.log`, identical config, one flag apart, both then scored by official ILAMB.

    DUMP_CLIMATE=1 SEASONAL_TRANSFORM=1 FUEL_WINDOW=60 MAG_BAND=1.3
    SAMPLER=tpe SEED=42 N_TRIALS=2500 WARM=params.coupledFW.json   POP_TERM=1 | 0

**Also, background runs still get killed, exactly as the older note in this file warns.** The wrapper
around that pair was killed while both python children survived, so the completion signal was lost but
the runs were fine. Check `ps` before assuming a killed job means a dead run.

### NEW FILES THIS SESSION

| File | What |
|---|---|
| `scripts/make_population_driver.py` | HYDE to the model grid, `data_human/pop_density_1deg_2001_2016.npy` |
| `scripts/build_pop_candidates.py` | rebuild burned area for the five recovered candidates |
| `scripts/prep_coupled_for_ilamb.py` | slice Lei 8 GB runs into something ILAMB will read |
| `scripts/diag_fire_cap.py` | the cap test that disproved my own diagnosis |
| `scripts/reproduce_modelC.py` | `trailing_mean()` added, `gpp_fuel` with a back compatible fallback |
| `scripts/optimize_modelC_coupled.py` | `FUEL_WINDOW`, `LANDUSE_TERM`, `POP_TERM` |

### WHAT IS LEFT ON THIS THREAD

1. **Let the MAG_BAND run finish**, but the conclusion is already visible. If it ends with trial 0
   still best, the honest report to Lei is that population does not survive a magnitude constraint.
2. **Send the Lei email.** Gmail draft `r7041293207578388834`, subject "Model F coupled run, one
   question before the 24th". Delete the duplicate draft on the BA reconstruction thread. Unlink
   `gdp_pcap.nc` and `gdp_gamma.nc`, which Gmail turned into URLs.
3. **PFT dump from Lei**, earliest two weeks from 08-19. Still blocks Model J.

---

## 2026-08-13, WINDOWS SESSION. THE PAPER REBUILT TO GEORGE'S 08/06 SKELETON

**Read `paper_gmd/TOPIC_SENTENCES_v3.md` first. It supersedes `TOPIC_SENTENCES_COMPLETE.md`,**
which is now the working file holding evidence and bracketed notes. The clean document for George is
`paper_gmd/Topic_Sentence_Outline_v3.docx`, in his own bullet arrangement, rebuilt by
`python paper_gmd/build_outline_docx.py`.

### What George asked for on 08/06, and what was done

His written skeleton and the meeting transcript are in `paper_gmd/`. Methods became one overview plus
four steps in the order the work was done. ED is introduced for the first time. The offline design and
its consequence are stated in Methods. The version table moved out of Methods into Results.
Socioeconomics moved from Methods to Results. Results became Model performance, with Globally and
Regionally as subsections, then Attribution, then the coupled model.

**A correction worth remembering.** I first kept a grid-cell subsection on the strength of one
transcript line. George rejected it twice, in speech, "fire intensity or robustness with held data,
these have got to go in somewhere else, these are too small a point", and by leaving it out of what he
wrote. **When the transcript and the written skeleton disagree, the skeleton wins.** The per-cell
numbers now live in the body of Sect. 3.1.1 and the scatter is supplementary.

### Runs completed this session, all official ILAMB, all reproducible

| Output | What |
|---|---|
| `paper_gmd/scoring/ba_regions/` | burned area, global plus the seven fitting regions, eleven versions |
| `paper_gmd/scoring/ba_gfed14/` | burned area, the fourteen standard GFED regions |
| `paper_gmd/models_ffire_paper/` | fFire computed for all eight paper versions, `scripts/run_ffire_all_versions.sh` |
| `paper_gmd/scoring/ffire_regions/` | fFire scored over global, the seven and the fourteen |

**The headline result of the session.** Model G wins the global score and is the best version in one
region of seven. Model I is best in four of seven, first on the regional mean, and third globally. On
the fourteen GFED regions the same inversion appears, I best in six and G in three. The compensating
error argument is now a systematic measurement rather than one basin.

**Regional emissions are the other new finding.** Model G's global emissions land within four percent
of GFED5 while its regional emissions are wrong by up to a factor of six, Africa over and everything
else under. Correcting burned area did not correct emissions, which is Discussion P6.

**Do not reuse 3.15 Pg C.** That figure came from an earlier combustion calibration. Model E gives 2.82
with the current betas and Model G gives 3.24.

### Documents and how to regenerate them

- `python scripts/build_tables.py` writes `paper_gmd/TABLES.md`, Table 1 global, Tables 2 to 8 one per
  region with identical rows and columns, Table 9 the version attributes, authored inside the script,
  Table 10 a stub. Emissions appear beside burned area in every table.
- `python paper_gmd/build_figs_tables_v2.py` writes `paper_gmd/Figures_and_Tables_v2.docx`, every
  figure and table in the order the outline calls for it. **Version 2 supersedes
  `Figures_and_Tables.docx`.**
- `python paper_gmd/build_outline_docx.py` writes the outline docx.
- Figures. `make_fig_maps_all.py`, `make_fig_ffire_all.py`, `make_fig_regional_matrix.py`,
  `make_fig_attribution.py`, `make_fig_scatter.py`, all run with the BASE python, not edfire.

**George's one-page rule is built into the figure scripts.** Whatever the item, the whole of it sits on
one page. The map scripts compute their own grid at three columns, so a tenth version will not spill.

### The three things that are not done, and who they need

1. **Model J, the plant functional type version.** Committed, reserved in Table 9, and blocked on a
   productivity split by plant functional type from Lei. The current dump splits by land use only.
   This is George's own mechanism, see his fine-fuel diagram.
2. **The coupled run, Table 10 and Sect. 3.3.** Needs Lei. **The version to reintegrate is not chosen**,
   and the paper's own argument makes it awkward, since the aggregate score picks Model G while Model I
   is the physically faithful one.
3. **Held-out validation now appears nowhere.** George cut it from Results and it is not what he meant
   by testing. The work exists, Model E lost 0.050 and 0.015 of its correlation. A referee will ask
   about overfitting.

Ten open decisions are listed at the end of `TOPIC_SENTENCES_v3.md` and on the last page of the outline
docx.

### LATER THE SAME DAY, 2026-08-13. THE FIGURE AND TABLE PASS. RESUME THE PAPER HERE

Richard reviewed the built figures and tables and gave two rules. Both are now in the code, not in a
note, so they hold for anything built later.

**RULE ONE. Draw every figure at final print size.** Both figures he rejected were legible on screen
and unreadable in Word for the same reason. They were laid out on a wide canvas with small type, so
inserting them at column width scaled the type down to about 6 pt. The schematic was 13 inches with
10 pt text and the matrix was 11 inches with 8 pt cells. Both are now drawn at the size they are
inserted at, so nothing is rescaled. **Check any new figure by asking what point size the labels end
up at after the insert, not by looking at the PNG.**

- `autoresearch_schematic.png` is 8.2 by 3.2 inches, insert at 7.5.
- `regional_matrix_fit7.png` is 7.3 by 3.65 inches, insert at 7.5 or less.

**RULE TWO. No caption text on a figure.** The title and the footnote came off the matrix figure
because Richard writes captions as text below the figure and the figure was duplicating them. The same
rule already removed the Model F caveat from both map figures. Anything a caption would say does not
belong on the image.

**Two content changes that came out of that pass.**

- **The schematic's version stack read C to F and now reads C to I.** It was drawn before Model G, H
  and I existed. The front card is the newest version, not a claim about which one is reported.
- **The asterisk on Model F is gone everywhere,** figures and tables. It was explained by a footnote
  that is now removed, so it pointed at nothing. The caveat itself still matters and belongs in prose.
  Model F was fitted on the coupled model's own climate with its global total pinned to the observed
  value, so its bias score reflects a constraint we imposed rather than skill.

**EVERY PERFORMANCE TABLE NOW COMES OUT OF ONE FUNCTION.** `format_table()` in
`scripts/build_tables.py` is called for the global table and each of the seven regions, and
`format_matrix()` shares its precision rules. They used to share a shape by inspection, which is how
the columns drifted apart. Do not hand-format a table. Add a column once and it appears in all eight.

Richard's formatting spec, now implemented and not to be relitigated.

| | Rule |
|---|---|
| Columns | Version, BA (Mha yr-1), Bias, RMSE, Seasonal, Spatial, BA score, F1, Emissions (Pg C yr-1), Emis. score |
| Precision | burned area integer, emissions two decimals, every ILAMB score and F1 three decimals |
| Observed | stated once in the caption, never repeated as a column |
| Best | bold, per column |
| Composites | **BA score and Emis. score. They must never both read Overall.** This was the worst of it |

Two judgment calls inside that spec, both defensible and both already in the captions. **Best means
largest for the scores and F1 but closest to observed for burned area and emissions,** since bolding
the largest burned area would bold the worst model. **Bold is decided on the printed string, not the
float,** so two versions that round to the same value are both bold or neither. South America has such
a tie, G and I both at 76 Mha.

**The Word builder now reads its captions out of `TABLES.md`,** so the wording and the numbers in the
document come from the same place the table does and cannot drift from it.

**A new cross-region matrix TABLE exists** in `TABLES.md` and in the docx, versions down and regions
across, the tabular form of Figure 4. **It has no number**, because the outline fixes Tables 2 to 8 as
regional, 9 as attributes and 10 as coupled, and because it carries numbers identical to Figure 4.
Printing both prints the same content twice. Undecided which one the paper uses.

**Two smaller things noticed and not acted on.** F1 is nearly flat across versions within a region,
0.897 to 0.914 across all of Africa, so it may be earning its column only in Table 1. And the observed
global burned area is 793 Mha yr-1 with observed emissions 3.40 Pg C yr-1, which is what every caption
now states.

**What to ask Lei in person about Model J,** since this came up and the answer decides the cost. One
sentence is enough, "can I get PFT-resolved output on the global baseline dump, same 0.5 degree monthly
1997 to 2016". The files are `global_baseline_modelC_inputs_1997-2016.nc` and
`global_baseline_modelCfuel_inputs_1997-2016.nc`. What is needed is GPP, above-ground biomass and area
fraction split by plant functional type instead of by land use, plus the PFT index table. **The question
that decides everything is whether ED carries PFT as a real output axis or aggregates it on write.** If
the per-cohort PFT state never reaches the region files, Model J costs a full ED rerun rather than a
repackaging.

## IF YOU ARE ON THE MAC

Per `CLAUDE.md` the Mac has **no conda and no ILAMB**, system python only. So on the Mac:
- **Do the writing.** The paper outline, the CPA, the prose. That is where the work is anyway.
- **Do NOT attempt** optimizer runs, assemblies, or ILAMB scoring. Nothing needs re-running.
- Figures may work if matplotlib is installed, but every figure the paper needs already exists.
- The numbers in this file are final and were produced on Windows. Quote them, do not recompute.

---

# PART 1. THE MODELS

## The scores. Eleven versions, one official ILAMB run, `paper_gmd/scoring/ba_withI/`

| Model | Bias | RMSE | Seasonal | Spatial | Overall | Mha/yr | x obs | F1 | Congo | Amazon |
|---|---|---|---|---|---|---|---|---|---|---|
| ED-stock | 0.5437 | 0.4652 | 0.4298 | 0.2085 | 0.4225 | 2500 | 3.15 | 0.456 | - | - |
| C | 0.6977 | 0.4754 | 0.8246 | 0.7691 | 0.6485 | 1001 | 1.26 | 0.768 | 3.7x | 21.3x |
| D | 0.6951 | 0.4662 | 0.7914 | 0.7864 | 0.6411 | 1219 | 1.54 | 0.775 | - | - |
| E | 0.7514 | 0.4753 | 0.7455 | **0.8756** | 0.6646 | 816 | 1.03 | 0.766 | 2.4x | **0.04x** |
| F | 0.7531 | 0.5120 | 0.7745 | 0.8400 | 0.6783 | 785 | 0.99 | - | - | - |
| G (5 reg) | 0.7430 | 0.4885 | 0.8419 | 0.8471 | 0.6818 | 1002 | 1.26 | - | - | - |
| G6 | 0.7473 | 0.4913 | 0.8437 | 0.8503 | 0.6848 | 943 | 1.19 | - | - | - |
| **G7** | 0.7481 | 0.4936 | **0.8455** | 0.8502 | **0.6862** | 946 | 1.19 | **0.813** | **10.1x** | 3.5x |
| H | 0.7258 | 0.5201 | 0.8353 | 0.8081 | 0.6819 | 830 | 1.05 | 0.804 | 1.9x | 4.0x |
| **I** | **0.7552** | 0.4798 | 0.8200 | 0.8478 | 0.6765 | **794** | **1.00** | - | **0.68x** | **0.74x** |
| Ibest | 0.7561 | 0.4793 | 0.8220 | 0.8464 | 0.6766 | 806 | 1.02 | - | 0.68x | 0.74x |

Congo and Amazon columns are mean annual burned fraction on closed-canopy cells (AGB > 10 kg C m-2)
as a multiple of GFED5. GFED5 observes 3.49 %/yr in the Congo box and 0.62 %/yr in the Amazon box.
GFED5 global total is 792.9 Mha/yr.

## THE HEADLINE FINDING

**The best-scoring model is not the best model.**

Model G7 scores highest at 0.6862 and burns Congo closed-canopy rainforest at ten times the observed
rate. Model I scores 0.6765, fifth, and is the only version that reproduces the observed global total
(794 against 793) AND gets both tropical forests right. The score gap is 0.010. The Congo error differs
by a factor of fifteen.

An aggregate benchmark computed over a land surface that is mostly free of fire ranked a physically
implausible model above a physically sound one. This is the paper's most valuable result and it is now
Discussion P4 and Conclusions P3. **Richard found it by looking at a map, not by reading a score.**

## What each version IS

- **ED-stock** ED's native fire scheme, never fitted by us. The floor. George has RULED OUT the name
  "ED-stock" and has not supplied a replacement. Current substitute in prose is "the original
  formulation", which is his own phrase from the AGU abstract.
- **A, B** historical, GFED4.1s target, buggy early driver pipeline, code removed from the repo. They
  CANNOT appear in a GFED5 table. 8 and 5 mechanism gates respectively.
- **C** `models/C/params.nsga2.json`. 12 params, S_overall objective, global. The start of the ladder.
- **D** `models/C/params.paperD.k1.json`. Identical 12-param form to C, objective changed to
  spatial-Taylor with SEAS_W=0 on the annual mean map. The ONLY clean single-lever step.
- **E** the "clean" continental assembly, 5 regions. 16 params, adds fuel headroom (`fuel_k`,
  `fuel_half`) and the biomass gate (`trop_agb_crit`, `trop_k_veg`) and switches the seasonal transform
  ON. Changes TWO attributes from D, form and spatial. Best spatial score and best precision.
- **F** `params.coupledE_gdp.json`. Global base plus a regionally varying GDP coefficient, fitted on
  ED-dump climate with the global total PINNED to GFED5 by construction. NOT comparable to the rest.
  Report as development history, not as a rung.
- **G** C's form and C's objective, fitted per continent. G7 is all seven regions and is the version to
  quote. `params.G_{Africa,Boreal,SAmerica,SEAsia,Europe,NAmerica,Australia}.json`.
- **H** `params.H.json`. C plus a GDP-per-capita term, on CRUJRA, `gdp_gamma` fitted JOINTLY at 1.81,
  nothing pinned. 13 params. The comparable replacement for F.
- **I** `params.Gtrop_*.json`. G's recipe with `TROP_MASK=1`, 14 params, adding ONLY the biomass gate.
  Its own letter because one attribute changed. The unbuilt "G + GDP" proposal moved off I to O.

## Regional detail worth keeping

Per-region internal scores, Model C on that region, then G, then I:

| Region | C | G | I | I - G |
|---|---|---|---|---|
| Africa | 0.6048 | 0.6343 | 0.6369 | +0.003 |
| Boreal | 0.4796 | 0.5974 | 0.5970 | -0.000 |
| S.America | 0.3827 | 0.6245 | 0.6376 | +0.013 |
| SEAsia | 0.5704 | 0.5975 | 0.5987 | +0.001 |
| Europe | 0.5024 | 0.5729 | 0.5557 | **-0.017** |
| N.America | 0.4012 | 0.5905 | 0.5845 | -0.006 |
| Australia | 0.5053 | 0.5132 | 0.5453 | **+0.032** |

The biomass gate is switched on only within 23.5 degrees of the equator, so it is INERT in Europe and
Boreal. Where inert it slightly HURTS, because two useless parameters enlarge the search space for the
same 1500 trials. Australia gains most (+0.032) because northern Australia is tropical woodland, and
Australia was the one region regional fitting alone did nothing for.

Keep-best-per-region must be judged on the ASSEMBLED global score, not the regional one. Australia's own
fit looked poor yet G7 beats G6, so Australia is kept.

## How to reproduce anything

```
# optimizer, env-var driven, see CLAUDE.md for the full flag list
PHYSICAL=1 MAG_BAND=1.3 FP_MIN=0.80 SAMPLER=nsga2 WARM=params.nsga2.json TROP_MASK=0 \
  SEASONAL_TRANSFORM=0 REGION=Africa N_TRIALS=1500 TAG=... python scripts/optimize_modelC_coupled.py

# assembly presets in scripts/assemble_continental.py: best, ho, cell, clean, G, G6, G7, I, Ibest
SEASONAL_TRANSFORM=0 ASSEMBLY=I ASSEMBLE_FALLBACK=params.nsga2.json python scripts/assemble_continental.py

# scoring, from the base env, and use nohup or it gets killed
export ILAMB_ROOT="$PWD/ilamb_ref_official"
nohup ilamb-run --config OUT/ilamb.cfg --model_root "$PWD/paper_gmd/models" \
  --regions global --build_dir OUT > log 2>&1 &
```

Driver scripts already written: `scripts/run_modelG.sh` (takes REGIONS), `run_modelH.sh`,
`run_africa_trop.sh`, `run_Gtrop_rest.sh`, `finish_G7_H.sh`.

## BUG FIXED 2026-08-05, commit 8f5d0f0. Affects any past regional run

`score_BA` ignored `REGION_MASK` entirely. Its weights came from the global land mask, so `REGION=X`
with the default S_overall objective was a GLOBAL fit with only a regional false-positive penalty. The
tell was `[warm] Overall=0.5987` printed identically for Africa, Boreal and S.America. Fixed with
SCORE_MASK aliases defined after REGION_MASK, bit-identical when REGION is unset (asserted at import).
**Model E is unaffected**, its fits used SPATIAL_OBJ=1 which was always region-aware, and that is
probably WHY that objective was chosen.

## TWO ENVIRONMENT TRAPS THAT COST HOURS

1. **matplotlib cannot render in the `edfire` env on Windows.** Every `savefig` dies with Windows fatal
   exception 0xc06d007f, in every format, reproducible in three lines. Use `base`
   (`C:/Users/owusu/miniforge3/python.exe`) for all figures. `edfire` is still the optimizer env.
   ILAMB also lives in `base`.
2. **Background ILAMB runs were killed twice.** Launch detached with `nohup ... &`.

---

# PART 2. THE PAPER

## The live document

`paper_gmd/TOPIC_SENTENCES_COMPLETE.md`. This is what goes to George. 39 paragraphs, Introduction
through Code and Data Availability, built on Richard's own v2 structure. `TOPIC_SENTENCE_OUTLINE.md` is
the older evidence-bearing file and is ONE REVISION BEHIND. Do not edit both; COMPLETE is authoritative.

## FIVE RULES a topic sentence must pass. Richard's, learned the hard way. Honor them.

1. Read the topic sentences in sequence with nothing else. They must form a coherent argument alone.
2. No pronoun or demonstrative pointing at something stated only in body text. "This difficulty" is
   allowed in Introduction P3 because P2's topic sentence names the difficulty. "That shortfall" was
   rejected because no topic sentence had stated a performance gap.
3. Main verb inside the first fifteen words. Assert. Do not open with a suspended subject clause.
4. One idea per topic sentence. Importance and failure joined by "yet" is TWO ideas.
5. Everything else in the section is EVIDENCE for that one idea, not a paragraph of its own. A section
   with five paragraphs where one claim would do is wrong.

Also: register follows **Ma et al. 2022 (ED v3.0)**, which George co-authored. Introduction topic
sentences carry NO numbers; Results topic sentences DO lead with the measured value against its
benchmark. Name the versions (C, D, E, G, H, I), never "the optimized model".

## What changed in the paper's argument, and it changed twice

The paper began as "functional form versus goodness-of-fit criterion, and the form is the binding
constraint". That is WITHDRAWN. Model G improves on C by +0.038 with NO change to the equation, more
than the form-and-spatial bundle bought. Then the Congo result added the second reversal, that the
version winning the benchmark is physically wrong. The paper now says regional fitting bought most of
the score, and the form change bought the physics.

## Verified citations. READ THIS BEFORE WRITING ANY CITED SENTENCE

`paper_gmd/references/VERIFIED_CITATIONS.md` has every claim with its quote and page. Two traps:

- **Li et al. 2024 must NOT be cited for a wide burned-area spread.** Its own finding is that "most
  CMIP6 models simulate the present-day global burned area and fire carbon emissions within the range
  of satellite-based products". It supports the 0.28 to 0.70 spatial correlation and the failure to
  reproduce the observed two-decade decline, nothing more.
- **GEDI IS NOT GLOBAL.** Dubayah 2020 p.1 gives coverage "between 51.6 N and S latitude", which
  EXCLUDES the boreal. Any claim of global canopy-height coverage is false. This is a design constraint
  on CPA Chapter 2, not a wording fix.

Still unverified: Cardoso 2003 (PDF held but text would not extract; it is George's own paper and the
closest precedent for Chapter 3).

## Figures, all in `paper_gmd/figures/`

**All of these were rebuilt for the paper's eight versions on 2026-08-13. Model I IS in them now.**
Read the figure and table pass in the 08-13 block above before regenerating any of them, because both
rules there are enforced in the scripts and easy to undo by accident.

- `ba_all_versions.png` every version, each panel titled with its Mha/yr. Three columns, one page.
- `ffire_all_versions.png` the same panels for fire carbon emissions.
- `regional_matrix_fit7.png` the score matrix, regions down and versions across. No title and no
  footnote on the figure, both are caption material. `_gfed14` is the fourteen-region version, scored
  and held back.
- `attribution.png` the score change contributed by each attribute, against Model C.
- `autoresearch_schematic.png` Figure 1, the loop. Drawn at print size, insert at 7.5 inches.
- `ba_diff_all_versions.png` the same against GFED5.
- `ba_single/` one map per version.
- `_archive/` five figures built for the version set that ended at E. Superseded, kept not deleted.
- `finefuel_pft.png` **George's own fine-fuel sketch reproduced from the model's drivers.** Three
  panels. Fine fuel has his two humps, grass peaking near 1375 mm and forest later. Total productivity,
  which is what the submodel actually uses as its fuel proxy, does NOT turn over at the wet end.
  Observed burning tracks fine fuel, not productivity. In Congo cells with AGB>10 and rain>1500 mm,
  productivity is 36.2 against a global land median of 3.9 while fine fuel is 6.9 against 3.7. This
  figure explains the Congo failure and justifies the biomass gate.
- Generators: `make_fig_maps_all.py`, `make_fig_finefuel_pft.py`. Run from `base`, not `edfire`.

## The version table

`cpa/Fire_Model_Version_Table_CORRECTED.docx`, generated by `cpa/build_version_table_corrected.py`.
NEVER edit the docx, edit the script and regenerate. Nine columns to Richard's layout, rows
alphabetical, plain unshaded headers. Table 1 has a **Vegetation dependence** column that explains the
Congo result at a glance, since every version with a blank there over-burns the rainforest.
STILL NEEDED: Model I's scores in Table 2, and its spatial cell changed from "Africa fitted, rest
pending" to "Continental (7 regions)".

## OPEN WITH GEORGE

1. **The baseline's name.** He has ruled out "ED-stock" and given no replacement.
2. **Whether the criterion search stays a headline contribution.** His submitted abstract describes a
   TWO-tier search (forms, parameter values). The work searched THREE, including the goodness-of-fit
   criterion, which is what Model D is. This decides how much of Results 3.2 survives.
3. **Whether the version table regains a functional-form column.** Without it the table says A equals B
   and says the E step was spatial alone. Both false.
4. **His submitted abstract says 0.42 to 0.66.** It predates Model G and understates the result, which
   is 0.69. Tell him before he presents.
5. **PFT.** The drivers carry NO grass/tree split, only land-use tiles and AGB, so the PFT column cannot
   be filled without a new dump from Lei with PFT-resolved productivity and biomass. What exists is
   vegetation-STATE dependence keyed on biomass, which is NOT the same thing and must not be presented
   as PFT. The honest line is that the question behind the column now has direct evidence, worth a
   factor of ten in the Congo, and that closing it properly is his and Lei's decision.
6. **Which version ships to the coupled runs and the carbon budget.** The best-scoring and the
   best-behaved are different models. Magnitude and the rainforest argue for I. The leaderboard argues
   for G.

---

# PART 3. THE CPA

**`cpa/` IS NOT IN GIT.** It lives only on the Drive. `cpa/STATUS.md` is the detailed resume point and
has been updated alongside this file. The working document is `cpa/CPA_OwusuAnsah.md`.

## Section status, measured 2026-08-06

| Section | State | Words |
|---|---|---|
| I. Curriculum Vitae | EMPTY | needs Richard's CV file |
| II. Goal Description | WRITTEN | 2933 |
| III. Coursework | PARTIAL | 268 plus `cpa/coursework_tables.docx` (all 66 courses) |
| IV. Research Experiences | WRITTEN | 3049 |
| V. Professional Experiences | EMPTY | needs Richard's input on what to include |
| VI. Analytical and Integrative Thinking | EMPTY | raw material EXISTS, see below |
| VII. Initial Dissertation Planning | WRITTEN 2026-08-06 | 2398 |
| VIII. Documentation | EMPTY | appendix, mechanical |

## What happened on the CPA today

**Section VII was drafted in full and is being reviewed subsection by subsection with Richard.**
Do not rewrite it wholesale. He is walking A through G in order.

- **A. Title** unchanged from the pre-proposal, "Advances in Remote Sensing and Modeling for Fire in the
  Carbon Cycle". Flagged as describing a field rather than a contribution. Not yet decided.
- **B. Introduction (real-world problem) APPROVED.** Five paragraphs, 605 words, twelve citations, all
  verified. Richard rejected two earlier drafts. The lessons: the real-world problem is NOT the modeling
  gap, the modeling gap is the obstacle; paragraph one must contain NO models at all; every factual
  claim needs a citation; and the section must return to the real-world consequence at the end.
- **C. Research Questions PRESENTED, NOT YET RESOLVED.** Chapter 1's question was rewritten to match
  George's submitted abstract, "Can an automated, artificial-intelligence-assisted search, referred to
  here as Autoresearch AI, develop a global fire model that predicts observed burning substantially
  better than its original formulation?" FOUR ISSUES RAISED AND STILL OPEN: the overarching question is
  a topic not a question and is inherited from the pre-proposal; Chapter 1 carries the unverified
  calibration claim; Chapter 2's paragraph ends flatly on "and the Ecosystem Demography model does";
  Chapter 3 has no citation and no numbers because the six-to-nine-month skill figure from the
  pre-proposal has no source. **RESUME HERE.**
- **D, E, F, G written but NOT yet reviewed with him.**

**Model G is the endpoint used throughout Section VII, at Richard's explicit instruction.** Therefore
score 0.42 to 0.69, F1 0.46 to 0.81, recall 0.32 to 0.83, spatial 0.21 to 0.85, and burned area falling
to 1.19 times observed. The "within one percent" figure belongs to Model I and appears only in the
Congo paragraph, as what the vegetation correction buys. Do not mix the two.

## Section VI, the next substantial piece

The raw material already exists at
`THE THESIS JOURNEY -REAL ONE/Annotated_Bibliography_Owusu-Ansah_May2026.md`. Eight papers in two
themes, the ED lineage and other DGVMs' fire schemes, each with a stated gap, plus a closing synthesis.
Richard's own guideline note in Section VI says it is "essentially this section already".

Three things to fix before moving it in:
1. **Its closing synthesis names a gap the dissertation no longer fills.** It defines the missing piece
   as a richer mechanistic fire module combining three-factor probability, behaviour physics and
   anthropogenic drivers with ED's cohort structure. Chapter 1 as executed is an automated search over
   an existing formulation.
2. **But one sentence in it has become the most important thing in the portfolio.** "None of these
   approaches operates on a resolved size and age distribution of live vegetation." Written in May,
   before any of this. In August the top-scoring version burned the Congo tenfold precisely because it
   could not distinguish rainforest from savanna. That gap statement was right and now has evidence.
3. **Two literatures are missing**, and papers for both are already verified in the repository. Nothing
   on observations and benchmarking (GFED5, ILAMB, FireMIP, CMIP6), which is the ground Chapter 1 stands
   on. Nothing on machine learning in Earth-system modeling (Reichstein 2019, Zhu 2022, Son 2024,
   Grundner 2025, Boardman 2025), which is what makes Autoresearch AI a contribution rather than a tool.

## Richard's standing CPA preferences. `cpa/STATUS.md` section 3 has the full list. The critical ones

- House style, Hurtt's: **no em-dashes, no en-dashes, no semicolons, no colons in body prose.**
- Formal academic register. Casual phrasing has been rejected repeatedly.
- **Never copy phrasing or structure from the sample portfolios.** Format only.
- Goal headers are noun phrases, not "I want to" sentences.
- He dislikes the word **"program"** as in research program.
- An academic goal must be a RECOGNIZED, TRANSFERABLE field. Benchmarking and ILAMB and Optuna are
  METHODS shown under a real field, never goals.
- Section II research GOALS are durable field-level contributions. Section IV is the concrete projects.
  Keep them from overlapping.

---

# PART 4. NEXT ACTIONS, in priority order

**Rewritten 2026-08-13 at the end of the figure and table pass. Items 4, 5, 6 and 8 of the old list
are DONE and have been removed. Richard moved back to the CPA at this point, so the paper resumes at
item 4 below.**

1. **CPA Section VIII, Documentation.** The last unwritten section, 29 words of stub. Mechanical, and
   everything it must list already exists.
2. **CPA Section VII is probably over the five-page cap** at 3156 words. The identified next cut is
   Section B or the second research-literature paragraph, about 225 words.
3. **CPA, the burned-area figure for VII.** Richard's own idea, the original formulation beside the
   best version. Not started, and cheap now that both maps exist.
4. **Paper, decide figure or table for the cross-region matrix.** They carry identical numbers and the
   table has no number until this is settled. My read is the figure goes in the main text and the table
   goes to the supplement.
5. **Paper, ask Lei for the PFT dump.** Blocks Model J. The one sentence to say and the question that
   decides the cost are in the 08-13 block above.
6. **Paper, the coupled run, Table 10 and Sect. 3.3.** Needs Lei, and needs the reported version chosen.
7. **Paper, held-out validation appears nowhere.** A referee will ask about overfitting. The work
   exists, Model E lost 0.050.
8. **Five items still need George,** listed under OPEN WITH GEORGE above. The two that block finishing
   are the dissertation title and the committee names, both for the CPA.
9. **Optional runs, NOT on the Mac.** Model N (G refitted on spatial-Taylor, about 2 h) would let the
   spatial and form levers be attributed with the statistic held fixed. Model O (G plus the GDP term,
   about 3 h) would test whether the regional and human gains are additive, though as currently defined
   it would inherit G's rainforest fault, so the better version is I plus GDP.

---

# ARCHIVE OF EARLIER SESSIONS BELOW

## >>> READ THIS FIRST — THE BEST-SCORING MODEL IS NOT THE BEST MODEL <<<

Eleven versions, one official ILAMB run against GFED5 (`paper_gmd/scoring/ba_withI/`).

| Model | Bias | RMSE | Seasonal | Spatial | Overall | Mha/yr | Congo (AGB>10) |
|---|---|---|---|---|---|---|---|
| ED-stock | 0.5437 | 0.4652 | 0.4298 | 0.2085 | 0.4225 | 2500 | - |
| C | 0.6977 | 0.4754 | 0.8246 | 0.7691 | 0.6485 | 1001 | 3.7x |
| D | 0.6951 | 0.4662 | 0.7914 | 0.7864 | 0.6411 | 1219 | - |
| E | 0.7514 | 0.4753 | 0.7455 | **0.8756** | 0.6646 | 816 | 2.4x (Amazon 0.04x) |
| F | 0.7531 | 0.5120 | 0.7745 | 0.8400 | 0.6783 | 785 | - |
| G (7 regions) | 0.7481 | 0.4936 | **0.8455** | 0.8502 | **0.6862** | 946 | **10.1x** |
| H (C + GDP) | 0.7258 | 0.5201 | 0.8353 | 0.8081 | 0.6819 | 830 | 1.9x |
| **I (G + gate)** | **0.7552** | 0.4798 | 0.8200 | 0.8478 | 0.6765 | **794** | **0.68x** |

**Model G scores highest and is physically wrong.** It burns Congo closed-canopy forest at 35 %/yr
against 3.5 observed.

**Model I is physically right and scores fifth.** 794 Mha against an observed 793, the best bias score
of any version, and the ONLY version that gets both the Congo (0.68x) and the Amazon (0.74x) right.
Model E fixes the Congo but extinguishes the Amazon at 0.04x; Model G over-burns both.

**That contrast is the paper's most valuable result.** An aggregate score computed over a mostly
fire-free land surface ranked a physically wrong model above a physically sensible one. The score gap
is 0.010; the Congo error differs by a factor of fifteen. Richard found it by looking at the map.

### Model I, what it is
Model G's recipe with `TROP_MASK=1`, 14 parameters, adding only `trop_agb_crit` and `trop_k_veg`. Not
Model E's fuel terms, not the seasonal transform. One attribute changed from G, so it takes its own
letter under George's rule. The unbuilt G+GDP proposal moves off I to O.

Regional fits gain where the gate can act (inside 23.5 deg of the equator): S.America +0.013,
Australia +0.032, Africa +0.003, SEAsia +0.001. They lose slightly where it is inert (Europe -0.017,
N.America -0.006) because two inert parameters enlarge the search space for the same 1500 trials. A
keep-best variant (`Ibest`) was built and scores the same, 0.6766.

### The live question the paper must answer
**Which version ships?** The best-scoring and the best-behaved are different models. For the coupled
runs and the carbon budget the magnitude and the rainforest matter, which argues for I. For the
leaderboard, G. The paper should say which and why.

### Writing state
`paper_gmd/TOPIC_SENTENCES_COMPLETE.md` is the document for George. 39 paragraphs, Introduction through
Code and Data Availability, built on Richard's v2 structure. Five topic-sentence rules recorded in it:
read down and the argument must hold alone; no demonstrative pointing at body text; main verb inside
fifteen words; one idea per sentence; everything else is evidence, not a paragraph.

TWO SENTENCES WERE ACTIVELY WRONG until 2026-08-06 and are fixed. Results 3.3 P3 and Conclusions P3
both claimed the vegetation term corrected the Congo "at no cost in overall score". It costs 0.010.

STILL TO DO in the writing: Results 3.1 and 3.2 still present Model G as the headline without the
caveat that it is physically wrong. `TOPIC_SENTENCE_OUTLINE.md` (the evidence file) is one revision
behind `TOPIC_SENTENCES_COMPLETE.md`.

### Table state
`cpa/Fire_Model_Version_Table_CORRECTED.docx`, nine columns, rows alphabetical, plain headers, and a
new Vegetation dependence column that explains the Congo result at a glance. NEEDS: Model I's scores in
Table 2, and its spatial cell changed from "Africa fitted, rest pending" to "Continental (7 regions)".

### Figures
`paper_gmd/figures/` holds `ba_all_versions.png` (every version, each titled with its Mha/yr),
`ba_diff_all_versions.png`, `ba_single/` with one map per version, and `finefuel_pft.png`, George's
fine-fuel sketch reproduced from the model's own drivers. Model I is not yet in the map figures.

### Two environment gotchas, both cost time today
- matplotlib CANNOT render in the `edfire` env on this machine. Every `savefig` dies with Windows fatal
  exception 0xc06d007f. Use `base` (`C:/Users/owusu/miniforge3/python.exe`) for figures.
- Background ILAMB runs were killed twice. Launch with `nohup ... &` so they survive.

### Bug fixed 2026-08-05, affects any past regional run
`score_BA` ignored `REGION_MASK`, so `REGION=X` with the default S_overall objective was a GLOBAL fit
with a regional false-positive penalty. Commit 8f5d0f0. Bit-identical when REGION is unset, asserted at
import. Model E unaffected, its fits used SPATIAL_OBJ=1 which was always region-aware.

### Open with George
1. The baseline's name. He has ruled out "ED-stock" without giving a replacement.
2. Whether the criterion search stays a headline contribution. His abstract describes two tiers; the
   paper argues three.
3. Whether the version table regains a functional-form column.
4. His submitted abstract says 0.42 to 0.66. It predates Model G and understates the result.
5. PFT. The drivers carry no grass/tree split, so the column cannot be filled without a new dump from
   Lei. What exists is vegetation-STATE dependence keyed on biomass, which is not the same thing.

---

## >>> READ THIS FIRST — THE BEST MODEL IS NOW G, AND THE PAPER'S CONCLUSION HAS CHANGED <<<

### The scores that matter (one official ILAMB run, `paper_gmd/scoring/ba_final/`)

| Model | Bias | RMSE | Seasonal | Spatial | Overall | Mha/yr | F1 |
|---|---|---|---|---|---|---|---|
| ED-stock | 0.5437 | 0.4652 | 0.4298 | 0.2085 | 0.4225 | 2500 | 0.456 |
| C | 0.6977 | 0.4754 | 0.8246 | 0.7691 | 0.6485 | 1001 | 0.768 |
| D | 0.6951 | 0.4662 | 0.7914 | 0.7864 | 0.6411 | 1219 | 0.775 |
| **G (7 regions)** | 0.7481 | 0.4936 | **0.8455** | 0.8502 | **0.6862** | 946 | **0.813** |
| E | 0.7514 | 0.4753 | 0.7455 | **0.8756** | 0.6646 | **816** | 0.766 |
| **H (C + GDP)** | 0.7258 | 0.5201 | 0.8353 | 0.8081 | **0.6819** | 830 | 0.804 |
| F | 0.7531 | 0.5120 | 0.7745 | 0.8400 | 0.6783 | 785 | - |

### The five things that changed

1. **Model G is the best version, 0.6862.** It makes NO change to the equation. Model C's original
   12 parameters, fitted separately for seven regions. Headline is now ED-stock 0.42 to 0.69, and the
   F1 pair is 0.46 to 0.81 with recall rising 0.318 to 0.826.
2. **The drafted conclusion is withdrawn.** It said the functional form rather than the scoring
   criterion is the binding constraint. C to G is +0.0377 with no form change at all, against D to E's
   +0.0235. Not supported. Withdraw, do not soften.
3. **Model E is no longer the best version.** It ranks fifth. What it still owns is the best spatial
   score (0.8756), the best precision (0.854), and the closest magnitude of the reanalysis versions.
4. **Model H rescues the human factor.** C + GDP on the SAME drivers as everything else, gdp_gamma
   fitted jointly at 1.81, nothing pinned. 0.6819, above Model F's 0.6783. The submitted abstract's
   socioeconomic claim is now supportable by a comparable version. Report F as development history.
5. **Three levers, three jobs.** Regional fitting buys pattern and seasonality, barely moves magnitude
   (1001 to 946). The form change buys magnitude (1219 to 816), costs seasonality. The human term buys
   magnitude best of all (1001 to 830), costs RMSE.

### Tell George before he presents
The submitted AGU abstract says the score rose "from 0.42 to 0.66". That was written before Model G
existed and is now an understatement. The paper carries 0.69.

### The one run worth doing next
**Model I, Model G plus the GDP term.** G and H each buy about +0.035 over C by DIFFERENT routes, G
through spatial pattern and H through magnitude. If they are close to additive, I would be the best
model in the paper by a clear margin. Seven continental runs with `GDP_TERM=1`, about 3 h.
Richard's instruction was to run this only if H understated the human effect. It did not, so it was
left for him to decide.

### Bug fixed this session, affects any past regional run
`score_BA` in `optimize_modelC_coupled.py` ignored `REGION_MASK` entirely, so `REGION=X` with the
default S_overall objective was a GLOBAL fit with a regional false-positive penalty. Fixed in commit
8f5d0f0, bit-identical when REGION is unset (asserted at import). Model E is unaffected, its fits used
SPATIAL_OBJ=1 which was always region-aware, and that is likely WHY that objective was chosen.

### Environment gotcha, Windows
matplotlib in the `edfire` env CANNOT render. Every `fig.savefig` dies with Windows fatal exception
0xc06d007f in all formats. Use the `base` env for figures, `C:/Users/owusu/miniforge3/python.exe`,
which is also where ILAMB lives. `edfire` remains the env for the optimizer.

### Where the writing is
`paper_gmd/TOPIC_SENTENCE_OUTLINE.md` is complete, Introduction through Conclusions, all updated to
these scores. Five writing rules are recorded in it, all from Richard's pushback:
1. No numbers in Introduction topic sentences; numbers in Results ones. Register from Ma et al. 2022.
2. A topic sentence states the paragraph's claim, not everything the paragraph covers, about 30 words.
3. A paragraph needs 5 or 6 sentences of material or it is not a paragraph.
4. Results reports, Discussion argues.
5. Name the versions, never "the optimized model".

### Figures
`paper_gmd/figures/ba_all_versions.png` (ten panels, every version, each titled with its Mha/yr),
`ba_diff_all_versions.png`, and `ba_single/` with one map per version. Regenerate with
`base` python on `paper_gmd/figures/make_fig_maps_all.py`; it picks up new versions automatically.

### Still open with George
- He does not want "ED-stock" but has given no replacement. Best candidate from his own abstract is
  "its original formulation".
- The abstract describes a two-tier search (forms and parameters); the paper argues three, including
  the goodness-of-fit criterion. This decides how much of Results 3.2 survives.
- Whether the version table regains a Functional form column. Without it the table says A equals B and
  says D to E was spatial alone, both false.

---

## >>> READ THIS FIRST (2026-08-06) — MODEL G CHANGES THE PAPER'S CONCLUSION <<<

Two things happened this session that the next session must not miss. A bug in the optimizer's scorer,
and a new model version that overturns the drafted paper's central claim.

### 1. BUG FIXED in `optimize_modelC_coupled.py` — `score_BA` ignored REGION (commit 8f5d0f0)
`score_BA`, which computes the S_overall objective, weighted bias and RMSE by `mass_w`, seasonal by
`mass_w_burn`, and the spatial term by `land_mask` and `w2_burn`. All four are built from the GLOBAL
land mask and none consulted `REGION_MASK`. Only `spatial_taylor` (SPATIAL_OBJ=1) and `physical_score`
were region-aware.
- CONSEQUENCE: `REGION=X` with the default S_overall objective was a GLOBAL fit with a regional
  false-positive penalty. NOT a per-continent fit. The tell was `[warm] Overall=0.5987` printed
  identically for Africa, Boreal and S.America.
- This is also WHY Model E's continental fits used SPATIAL_OBJ=1. It was the only region-aware
  objective the code had. Model E is unaffected and its 0.6646 stands.
- Fix adds SCORE_MASK / SCORE_MASS_W / SCORE_MASS_W_BURN / SCORE_W2_BURN after REGION_MASK. Bit-identical
  to the old globals when REGION and CELL_HOLDOUT are unset, asserted at import. Verified: global warm
  start reproduces 0.5987 exactly.
- Any past REGION run on S_overall is suspect. Three invalid G fits were discarded; their log is
  `logs/opt_modelG.INVALID-global-scoreBA.log`.

### 2. MODEL G RAN, AND IT IS THE BEST VERSION IN THE PAPER (commit a74fa54)
G = Model C's 12-parameter form and C's S_overall objective, fitted PER CONTINENT over the same five
regions as E's clean assembly. Only the spatial parameterization differs from C. Built to isolate the
column that D to E confounds. Params `models/C/params.G_*.json`, `ASSEMBLY=G` in
`assemble_continental.py`, scored in `paper_gmd/scoring/ba_withG/`.

Official ILAMB, ALL versions in one run against GFED5:

| Model | Bias | RMSE | Seasonal | Spatial | Overall | Mha/yr |
|---|---|---|---|---|---|---|
| ED-stock | 0.5437 | 0.4652 | 0.4298 | 0.2085 | 0.4225 | 2500 |
| C | 0.6977 | 0.4754 | 0.8246 | 0.7691 | 0.6485 | 1001 |
| D | 0.6951 | 0.4662 | 0.7914 | 0.7864 | 0.6411 | 1219 |
| **G** | 0.7430 | 0.4885 | **0.8419** | 0.8471 | **0.6818** | 1002 |
| E-clean | 0.7514 | 0.4753 | 0.7455 | **0.8756** | 0.6646 | 816 |

**WHAT THIS OVERTURNS.** The drafted paper concludes "the functional form, not the scoring criterion,
is the binding constraint on burned-area skill". NOT SUPPORTED. C to G is +0.0333 on the same statistic,
D to E is +0.0235. Spatial parameterization alone bought more than the form-and-spatial bundle. That
sentence must be WITHDRAWN, not softened.

**WHAT THE TWO LEVERS ACTUALLY DID.** Regional fitting buys pattern and keeps seasonality (Spatial
0.7691 to 0.8471, Seasonal 0.8419) but does NOT fix magnitude (1002 Mha, essentially C's 1001, against
793 observed). The form change buys magnitude (1219 to 816) and pushes Spatial to 0.8756, but costs
seasonality (0.7914 to 0.7455). Different jobs, which is a better finding than one lever winning.

**CAVEAT, MUST BE STATED.** G and E also differ in OBJECTIVE (S_overall vs spatial-Taylor). ILAMB
reports the composite, so G was fitted close to the metric being reported and E was not. Part of G's
margin is that alignment.

**THE ONE RUN THAT WOULD CLOSE IT.** Model G refitted on spatial-Taylor (SPATIAL_OBJ=1 SEAS_W=0),
matching D and E exactly, about 2 h via `scripts/run_modelG.sh` with those env vars. Then D to Gspatial
isolates spatial and Gspatial to E isolates form, statistic held fixed across all three.

### 3. WRITING: topic sentences done for Introduction and Results
`paper_gmd/TOPIC_SENTENCE_OUTLINE.md` is the live writing document. Introduction rebuilt to 7 paragraphs
(was 5 drafted), Results to 3.1 through 3.6. Discussion and Conclusion NOT yet done and the Conclusion
must be rewritten around the G result.

FIVE WRITING RULES were established this session, all recorded in that file, all from Richard's
pushback. Apply them to Discussion and Conclusion:
1. Introduction topic sentences carry NO numbers. Results topic sentences DO. Register is set by
   Ma et al. 2022 (`references/CITED_PAPERS/Ma2022_ED-v3.pdf`), which George co-authored.
2. A topic sentence states the paragraph's CLAIM, not everything the paragraph covers. Ma's run ~30
   words. Ours were 47 to 55 before this was caught.
3. One paragraph needs 5 or 6 sentences of material behind it. If a claim cannot sustain that it is a
   sentence inside another paragraph. 3.1 went 5 -> 3 on this.
4. RESULTS REPORTS, DISCUSSION ARGUES. Ma names discrepancies without explaining them. A trial drafting
   of 3.1 was full of "therefore" and "this explains"; those lines are parked in the outline for reuse
   in the Discussion.
5. NAME THE VERSIONS (ED-stock, C, D, E, F, G), never "the optimized model" or "the unoptimized module".
   Descriptive labels assert the conclusion in the label.

### 4. AGU abstract is FINAL and trimmed
George returned it 08-05 with his own closing sentence. His text was 2130 chars, 130 OVER the AGU limit.
Trimmed to 1938 with 62 to spare, in `paper_gmd/AGU_2026_abstract_v2.md`, which holds both his untrimmed
return and the final, with every cut itemized. The human-factor clause is what came out.

### 5. OPEN WITH GEORGE
- He does NOT want the baseline called "ED-stock" (said 08-06) but has not given a replacement. Working
  name kept. Best candidate from his own abstract is "its original formulation".
- The abstract describes a search over structural forms and parameter values, TWO tiers. The paper
  argues THREE, including the goodness-of-fit criterion. Either the abstract gains a clause or the
  paper demotes the criterion from a headline contribution. This decides how much of 3.2 survives.
- Whether Model F is a paper row. It is on dump climate, magnitude-pinned by construction, and not
  GCB-viable. Model H would make it comparable.
- Whether the version table gets a Functional form column back. Without it the table says A equals B
  and says D to E was spatial alone, both false.

## >>> READ THIS FIRST (2026-08-04) — AGU ABSTRACT IS THE LIVE DELIVERABLE <<<

The active work has moved OFF coupling and ONTO writing. Two live threads, the AGU abstract (deadline
~August, so it is the near one) and the CPA. The coupling deliverable is parked exactly where the
07-30 block below left it.

### AGU 2026 abstract — v2 written to George's comments, ONE open question
George returned the abstract with tracked changes and five comments (`paper_gmd/AGU Abstract_GH
Comments.docx`). Revised version is `paper_gmd/AGU_2026_abstract_v2.md`, 278 words / 1873 chars, under
the 2000-char limit. That file also carries the comment-by-comment resolution table, the numbers
accounting, the F1 method note, and the score verification, so it is the single place to resume.
- His retitle stands: "Development of a New Global Fire Model for Carbon Cycle Science Using
  Autoresearch AI". The deliverable is now framed as A NEW MODEL, not as an optimization study.
- FRAMING FLIPPED POSITIVE (his "masked" comment, Richard read it as "stop trashing our own model").
  Every result leads as a gain. The "more than three times the observed value" clause is GONE.
- STILL OPEN, the only blocker: George's comment said "the version of our model used was masked" and
  the positive reframe does not mention masking at all. ASK HIM whether he wants masking stated in the
  abstract. Recommendation is to keep it out of 250 words and carry it in the paper.
- Submit to the CARBON MONITORING SYSTEM session George co-chairs. AGU 2026 San Francisco, ~Dec 10-14.

### The search-scale numbers (his biggest comment) — DO NOT let his placeholder through
He inserted "thousands of alternative model structures with millions of parameter combinations". NEITHER
IS SUPPORTABLE. The three tiers are different things and the abstract now names them separately:
- STRUCTURES = distinct equations (the gated optional terms in `fire_C`: AGB veg suppression, tropical
  closed-canopy, fuel amplitude, scalar amplitude, seasonal transform, GDP multiplier, curing, plus the
  historic 8-gate A and 5-gate B). Order of TEN actually run.
- CONFIGURATIONS = structure x spatial scope x objective x saved param set. ~168 (189 JSON in models/C
  minus 21 topk manifests). Many are k1..k8 siblings of ONE run, or the same structure fit per continent,
  which is a SPATIAL change and not a form change.
- PARAMETER COMBINATIONS = optuna trials. 62,600 logged.
The early Model A/B autoresearch phase is real but UNCOUNTABLE. Its winning models, `shapley.json` and
`MASTER_RESULTS.md` survive in git history but no trial-by-trial logs were ever committed. Folding it in
would also overstate the fire search because that phase tuned SIX ED modules, not fire alone.
Reported floor: "more than 160 configurations, more than sixty thousand parameter combinations".

### F1 added (his "F1 scores?" comment) — `scripts/compute_fire_f1.py`
Fire presence per grid cell on the 2001-2016 annual climatology, 0.5 deg, threshold 0.1%/yr (the same
threshold as the paper's active-fire mask), GFED5 as truth, 50,790 valid cells.
**ED-stock 0.452 -> Model E 0.764.** Holds at every threshold, 0.01% gives 0.486 -> 0.802 and 1% gives
0.425 -> 0.682, so it is not an artifact of the cutoff. Precision/recall 0.783/0.318 vs 0.850/0.694.
THE TALKING POINT: the baseline failure is RECALL, not precision. ED-stock misses two thirds of the cells
GFED5 says burn while burning 3x too much area in the ones it does find, which is how a model
over-predicts total burned area and under-detects fire extent at the same time. Monthly cell-month F1 is
0.256 -> 0.442, held in reserve and not in the abstract.
UNITS GOTCHA handled in the script: GFED5 burntArea is PERCENT, model output is FRACTION.

### ED-stock baseline CONFIRMED (the 0.42 and 0.21 were unverified until now)
ED-stock is NOT in the clean `paper_gmd/scoring/ba_clean` run (that holds only C, D, E-clean,
E-seasonal), so the floor numbers were inherited rather than verified. Ran ED-stock + E-clean together
through official ILAMB against GFED5 in one apples-to-apples run:
  ED-stock  Bias 0.5437  RMSE 0.4652  Seas 0.4298  Spatial 0.2085  Overall 0.4225
  E-clean   Bias 0.7514  RMSE 0.4753  Seas 0.7455  Spatial 0.8756  Overall 0.6646
Reproduces the recorded table exactly, so the abstract's 0.42 -> 0.66 and 0.21 -> 0.88 are both verified.
Evidence at `paper_gmd/scoring/ba_stockcheck/scalar_database.csv`.
**GOTCHA WORTH REMEMBERING:** ILAMB's `scalar_database.csv` `Source` column reads `c` instead of `GFED5`
in some runs. It is a COLUMN QUIRK, not a different reference dataset. That is what made the older
`scoring/ba` run look unverified, and it cost an investigation. It was GFED5 all along.

### CPA — Section IV COMPLETE (drive-only, gitignored, `cpa/STATUS.md` to resume)
IV.A research narrative (undergrad mining study, MS Boston flood + DC food deserts, doctoral fire work as
centerpiece written to the NEW thesis framing, Great Basin, biomass-uncertainty review, NASA Central
Africa), IV.B four competencies needing preparation, IV.C Chapters 2 and 3, IV.D outputs list. Exported to
`cpa/CPA_OwusuAnsah.docx`. Sections II and III were already complete. Sections I, V, VI, VII, VIII still
skeleton. Next CPA target is VII (dissertation idea paper) or VI (annotated bib reformat).

### Also worth knowing
- 2026-08-01 (never written up until now): thesis reframed to "how much can optimization improve a global
  fire model"; repo audit of George's version table -> `paper_gmd/version_table_audit.md` + corrected
  table `cpa/Fire_Model_Version_Table_CORRECTED.docx`; AGU abstract v1. All detail in `paper_gmd/STATUS.md`.
- SESSION-CONTEXT LESSON: a `/compact` late on 08-04 lost a further abstract refinement that had never
  been written to a file. Everything of value now lives in `paper_gmd/AGU_2026_abstract_v2.md`. Write
  deliverables to FILES as they are agreed, do not leave them in the conversation.

## >>> READ THIS FIRST (2026-07-30) — GEORGE MEETING: E & F ARE OUT FOR COUPLING (1850 rule) <<<

### THE PIVOT (today's George meeting) -- this changes the coupling deliverable
- NEW HARD CONSTRAINT: every input to the coupled run must extend back to ~1850 (historical carbon-budget
  run). GDP does NOT go back that far -> GDP is DISALLOWED for coupling. This REVERSES last week's
  "forward-runnable is enough" rule that had let GDP in. Richard flagged the reversal himself.
- Therefore **Model F (GDP) AND Model E (per-continent seams) are BOTH OUT for the coupled run.** F had
  already been "shipped to Lei" -> it is now SUPERSEDED; Lei must be told the coupling deliverable changed.
- NEW coupling target (George's words): a model BETTER than C and D, NOT per-continent (no seams), NO GDP.
- **WE ALREADY HAVE IT: the single-global ED-dump refit = the `coupledE` family** (`models/C/params.coupledE*.k*.json`).
  ~0.652-0.655 ILAMB BA, above C (0.6473) and D (0.6411). Single global (no seams), no GDP, runs on ED state
  (biomass, GPP) + climate -- all extend to 1850. Built in the July coupling work, passed over THEN only
  because E/F scored higher (now disallowed). See the 2026-07-23 coupling block below for the coupledE build.
- HONEST CAVEAT to carry: this single-global model is the "regionally broken" one (good global TOTAL via
  compensating regional errors -- boreal under-burns, S.America over-burns). It is the CONSTRAINED best
  under no-seams + no-GDP + back-to-1850. E / F / the GDP story become PAPER / OFFLINE results, not coupled.
- PRINCIPLED long-term fix George himself pointed at (07/23): key params to VEGETATION STATE (AGB / tree
  cover / PFT), not lat/lon or GDP -> removes seams, needs no GDP, extends to 1850, migrates with veg. NOT
  built. This is the right next model IF single-global regional errors matter for the carbon budget.

### IMMEDIATE NEXT (coupling), pick up here
1. Confirm the exact best `coupledE` variant + its ILAMB score (several k-files; nail down the pick).
2. Retune combustion betas (fFire) on that BA so emissions are consistent (scripts/tune_combustion_params.py).
3. Package + tell Lei that F is superseded; the coupled deliverable is now the single-global no-GDP model.

### SCHEMATIC (paper Fig 1) -- REBUILT to George's actual hand-drawn sketch (grounded in the transcript)
Read the 07/23 transcript this session (`paper_gmd/meeting with advisor on 07_23_2026.pdf`, extracted with
pypdf) instead of guessing. Final `paper_gmd/figures/make_schematic.py` -> autoresearch_schematic.{png,pdf}:
- PROPER FLOWCHART SHAPES, shape = type, NO legend/key (George: "you can't have different kinds of things
  in the same kind of boxes"): Inputs = PARALLELOGRAM (data; split ED biomass/GPP vs non-ED climate/GDP,
  GDP-only human input); fire model versions = STACKED rectangles (C/D/E/F, F promoted); Burned area +
  carbon = PARALLELOGRAM (a model PRODUCT); Score / compare to benchmarks = RECTANGLE (an ACTIVITY).
  [BA=product, Score=activity is George's exact wording -- I had them backwards before.]
- TWO loops, FUNCTIONAL labels only (NO person names -- it is a publication figure): (1) auto-research loop
  (score iterates back to the model, top); (2) implementation loop (couple the chosen model back into ED,
  bottom, green) -- points at the ED PART of the inputs ONLY (biomass, GPP). REMOVED: the GCB/TRENDY end
  box, the "converged?" diamond, the native-fire box (all were my inventions, not George's).
- Still pending George's confirm. make_schematic.py is the source of truth; png/pdf regenerable.

### CPA (Comprehensive Portfolio) -- Richard pivoted to this at end of session
`cpa/` (Drive-only, gitignored). Section II COMPLETE (II.A original goals verbatim; II.B current goals =
4 academic / 3 research / 4 professional; II.C 3x3 goals-framework schematic `cpa/framework_goals.png`,
rows Academic/Research/Professional x cols Past/Present/Future, overarching-goal + scholar-practitioner
banners). Sections I/III/IV/V/VI/VII/VIII still skeleton. Next drafts: VI (reformat the annotated bib) or
VII (dissertation idea paper from paper_gmd + preproposal Ch2/3). Needs from Richard: CV, transcript, PAC
member names. Read `cpa/STATUS.md` to resume. HOUSE STYLE strict (no em/en-dash, no semicolon/colon in
prose, formal register, noun-phrase goal headers, avoid the word "program", never borrow sample phrasing).

### ALSO this session
- ADVISOR DECK for the GDP human-factor work built + verified (the 2026-07-29 block below has the detail).
  NOTE: the GDP story is now a PAPER result, not coupled (see the pivot above).
- ILAMB scoring learned from Collier2018 (`paper_gmd/references/CITED_PAPERS/Collier2018_ILAMB.pdf`):
  Overall = (Sbias + 2*Srmse + Sphase + Siav + Sdist)/6, RMSE double-weighted; IAV omitted for fire, so
  OUR Overall = (Bias + 2*RMSE + Seasonal + Spatial)/5. Collier 2.3 themselves warn "a higher score does
  not necessarily reflect a more process-oriented model" = our metric-thesis in the benchmark authors' words.
- COMMIT still NOT run this session (a classifier outage earlier + Richard deferred). Staged/ready: the deck
  scripts + docs; the schematic generator lives in paper_gmd (Drive-only). When committing: LOCAL only, NO push.

## >>> READ THIS FIRST (2026-07-29) — GDP->Model F ADVISOR DECK BUILT (weekly George meeting) <<<

Lei email confirmed DELIVERED (Richard). Coupling deliverable stays DONE. This session built the
advisor slide deck for George's human-factor / GDP assignment. Four-beat story, GDP only (no C/D/E
backstory -- George knows it): plot it -> prove it is real -> prove it is the right factor -> build it
in biome-specifically -> it wins.

DECK: `GDP_ModelF_advisor_deck.pptx` (repo root; *.pptx is gitignored -> carried on Drive, regenerate
from the two scripts). 6 slides, 16:9. Speaker notes carry the spoken script + backup numbers, with the
two honesty asterisks baked in where they belong: (1) the biome-gamma map is CALIBRATED to GFED5, not
derived from first principles; (2) boreal 0.52x + Australia 0.37x run low = a base FUEL/biomass limit,
not the human term. Flow: S1 title -> S2 beat1 (signal real) -> S3 beat2 (right factor) -> S4 beat3
(biome-specific gamma map) -> S5 beat4 (regional payoff) -> S6 bottom line.

SCRIPTS (committed, re-runnable in edfire; python-pptx pip-installed into edfire this session):
- `scripts/make_advisor_gdp_deck_figs.py` -> figs_gdp_advisor/ (4 PNGs, gitignored dir). All from cached
  data, no model re-run. Beat 1 names exemplars (DR Congo/Zambia/S.Sudan/Angola vs USA/Germany/Norway)
  on BOTH the raw and climate-removed panels -- the same countries hold position after climate is removed.
  Beat 2 reproduces the CANONICAL nested-model F-test from the cached country CSVs (the original
  fire_vs_{pop,landuse}_partial.py have hardcoded Mac paths but wrote the CSVs, which carry every column).
- `scripts/build_gdp_advisor_deck.py` -> assembles the pptx.

NUMBERS VERIFIED vs GDP_HUMAN_TERM_FINDINGS.md: raw -0.92/decade r-0.55 (164 countries); partial
-0.70/decade r-0.47 p2e-10; population F=0.1 p=0.93; land use F=1.9 p=0.13; regional global 0.98x,
Africa 1.13x, boreal 0.52x, Australia 0.37x. Controlled GDP-only score jump (if George asks): 0.6547 -> 0.6603.

### SCHEMATIC -- our v2 is WRONG; George gave a hand-drawn target. REBUILD to his sketch (Mac mini)
Our v2 (`paper_gmd/figures/make_schematic.py`, color-coded rounded boxes + two shaded loop panels) does
NOT match George. He handed a HAND-DRAWN schematic (photo shown 07-29). THE ACTUAL TARGET:

- **Flowchart SHAPES encode TYPE, not color** (his explicit point; the two loose shapes he drew = a key).
  Use real flowchart geometry: PARALLELOGRAM = data / input-output (drivers, GFED5, scores);
  RECTANGLE = process (Fire Model, compute burned area + carbon); DIAMOND = decision (converged?);
  STADIUM/rounded = start-end terminator; STACKED rectangles = the model-version set. Drop the color legend.
- **Single flat pipeline + a feedback loop** (NOT our two big panels):
  Inputs -> Fire Model -> Burned area + Carbon -> Scores / compare to benchmarks -> "backward revision"
  feedback loop back into Fire Model (+ an outer return loop). Simpler and flatter than ours.
- **Inputs box split ED vs non-ED**: ED = Biomass, GPP;  non-ED = Climate, **GDP ONLY** (Richard: drop
  POP and Landuse from the figure even though George's sketch listed them -- stats found them redundant).
- **"Burned area + Carbon" is ONE output box** (not our split predict-BA vs coupling-carbon).
- **Scores box = "compare to benchmarks"** (the TRENDY leaderboard framing), not just "Score vs GFED5".
- **Coupled run**: the return/coupling path George verbally called "Lei" is the COUPLED RUN -- label it
  "Coupled run", NOT "Lei" (that was him narrating the box).
- **Model versions labelled C, D, E, F** (our naming), shown as the stacked-rectangles set.
- The optimization loop is its own explicit element feeding the model stack.
RICHARD may still probe George for more. Treat the current make_schematic.py as scrap; rebuild from the
sketch photo. NOTE the figure content vs DRAFT.md (still Model-E-only) -> fold-in-F decision still open.

### COMMIT STATUS -- NOT committed this session (deferred by Richard + a classifier outage)
Staged/ready but NOT run: the deck scripts (make_advisor_gdp_deck_figs.py, build_gdp_advisor_deck.py),
.gitignore, HANDOFF_NOTE.md, PROGRESS.md. The schematic generator lives in paper_gmd (drive-only, force-added
on this branch). When committing: local only, DO NOT push (Richard's instruction). Deck pptx + figs_gdp_advisor
PNGs are gitignored (regenerable).

NEXT (unchanged from 07-27, minus the now-done Lei email): (a) run the deferred local commit (no push).
(b) SCHEMATIC: get George's actual preferred layout, rework make_schematic.py (home/Mac mini). (c) paper
outline topic-sentence-first (George's writing ask) -- this is where the fold-in-F-into-the-paper decision
gets made. (d) the Australia/boreal EMISSIONS gap is a biomass/fuel fix in the base model, not a human-term fix.

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
