# CLAUDE.md — ED fire submodule (Model C) project

Persistent context for Claude Code. Read this first, then `HANDOFF_NOTE.md` for current state.

## What this project is

Model C is a burned-area + fire-emissions submodule for the ED vegetation model (part of the
UMD/Hurtt group's ED-coupled TRENDY work). It predicts monthly burned-area fraction from climate +
GPP drivers, is scored against GFED5 via official ILAMB, and is compared against the other TRENDY
v14 land models. A separate combustion step turns burned area into fire carbon emissions (fFire).

**Team (one UMD group, not competitors):** George Hurtt (PI), Richard Owusu-Ansah (this repo / fire
submodule), Lei Ma (runs coupled ED, `lma6@umd.edu`), Devesh (`deveshp@terpmail.umd.edu`).

**Target reference is GFED5** (not GFED4). We are the only model in the leaderboard actually tuned to
GFED5; CLM6 etc. were tuned to older GFED3/GFED4 and only evaluated against GFED5.

## Environment

```bash
source $(conda info --base)/etc/profile.d/conda.sh && conda activate ed-fire
```
All Python (xarray, optuna, ILAMB, cartopy, h5py, cmaes) lives in the `ed-fire` conda env.
`Date.now`-style nondeterminism is irrelevant here; runs are seeded.

## The canonical (shipped) model — DO NOT overwrite without backing up

| File | What |
|---|---|
| `models/C/params.json` | Final 12-parameter set (NSGA-II refit). |
| `ilamb/MODELS/ED-ModelC-final/burntArea.nc` | Canonical offline BA output (0.5deg monthly 2001-2016). |
| `models/combustion-params/betas.gfed5.json` | Retuned combustion betas for fFire. |
| `ilamb/MODELS_LEADERBOARD_FFIRE_GFED5/ED-ModelC-Hybrid/fFire.nc` | Canonical fFire output. |
| `models/C/params.nsga2.json` | Tagged copy of the canonical params (source of truth). |
| `models/C/params.PRE-coupled.json` | Pre-refit backup (the original benchmark). |

Current scores vs **10 external TRENDY models** (excluding our own old variants):
- Burned area: ILAMB Overall **0.6482, rank #3** (CLM6 0.6562 #1, ELM-FATES 0.6502 #2).
- Fire emissions: ILAMB Overall **0.6465, rank #5**.

## Drivers (hybrid — this is the key design decision)

- **GPP**: from `global_baseline_modelC_inputs_1997-2016.nc` (Lei's coupled-ED dump), area-weighted
  sum of `GPP_month_{ntrl,scnd,past}`. This makes params transfer into Lei's coupled run.
- **Climate** (D_bar, T_air, P_ann, P_month): from CRUJRA `data/crujra/*_monthly.npy`. Using ED's
  internal climate from the dump instead drops the GFED5 score ~0.03, because ED's D_bar is a derived
  diagnostic; CRUJRA is the upstream observation ED itself ingests.
- **AGB** (for emissions + optional veg suppression): `global_baseline_modelCfuel_inputs_1997-2016.nc`.
- Model C computes on a 1deg grid (drivers coarsened), output expanded to 0.5deg for ILAMB.

## Key scripts

| Script | Purpose |
|---|---|
| `scripts/reproduce_modelC.py` | Regenerate `burntArea.nc` from `params.json` + drivers. |
| `scripts/optimize_modelC_coupled.py` | The optimizer. Env-var driven (see below). |
| `scripts/compute_emissions.py` | BA + biomass + betas -> fFire. `--betas-json ... --out-suffix ""`. |
| `scripts/tune_combustion_params.py` | Retune combustion betas against GFED5 fFire. |
| `scripts/maps_hybrid_ba_ffire.py` | 4-panel BA + fFire maps vs GFED5/CLM6/ELM-FATES. |
| `scripts/maps_seasonal.py` / `_ffire.py` | Seasonal figures (timeseries, regional cycles, peak-month, Hovmoller). |
| `scripts/diagnose_false_positives.py` | Quantify + map where we over-predict vs GFED5. |

### optimize_modelC_coupled.py env vars
`N_TRIALS` (default 2500), `SAMPLER` (`tpe`|`cmaes`|`nsga2`), `SEED`, `WARM` (path under models/C/),
`TAG` (output suffix), `MAG_BAND` (hard magnitude band, e.g. 1.3 = within 1.3x GFED5 mean),
`MAG_PENALTY` (soft log-ratio penalty), `PHYSICAL=1` (enable false-positive + hotspot objective),
`W_ILAMB`/`W_FP`/`W_HOT` (physical objective weights), `QUIET_THRESH`, `FP_MIN` (NSGA-II Pareto filter).
The canonical model came from: `PHYSICAL=1 MAG_BAND=1.3 FP_MIN=0.80 SAMPLER=nsga2 WARM=params.nsga2.json`.

## How to score with official ILAMB

```bash
export ILAMB_ROOT=ilamb_ref_official
OUT="$PWD/ilamb_out_run"; rm -rf "$OUT"; mkdir -p "$OUT"
cp ilamb/burntArea_gfed5.cfg "$OUT/ilamb.cfg"   # or ffire_gfed5.cfg for emissions
ilamb-run --config "$OUT/ilamb.cfg" --model_root "$PWD/ilamb/MODELS_LEADERBOARD" \
  --regions global --build_dir "$OUT"
```
Read scores from `$OUT/scalar_database.csv` (filter Region==global, ScalarName=='Overall Score').
Gotcha: ILAMB needs an `ilamb.cfg` copy *inside* the build dir. Remove stale `burntArea.*.nc` from a
model folder before running or ILAMB raises `MonotonicityError` (it tries to merge all .nc in the dir).

## Conventions / preferences

- **Writing**: no en-dashes, em-dashes, semicolons, or colons in body prose; avoid heavy quoting.
- **Paper framing is positioning, never "we beat them".** It's one UMD team; the leaderboard shows
  where ED sits among TRENDY models, not a competition.
- **Check units before plotting** (a `%` vs `1` mismatch once caused a 100x display error).
- When excluding our own old variants from a leaderboard, skip: `ED-ModelC-{l02s7,GFED5,GFED5cont,
  GFED5type,Emissions,EmpiricalEmit}`, `ED-ModelA-final`, `ED-ModelB-final`, `ED-Ensemble-*`, `_archive`.
- Area-weighted total (Mha) is the physical magnitude; the unweighted mean-of-percent over-counts small
  high-latitude cells (it reads ~6% vs GFED 1%, but area-weighted we're only 1.26x).

## Other docs

- `HANDOFF_NOTE.md` — current state + next steps (read after this).
- `HYBRID_REFIT_SUMMARY.md` — detailed final results + every optimization round tried.
- `PROGRESS.md` — living log (most recent on top).
- `handoff_modelC_inputs.md` — schema of Lei's coupled dump.
- `README.md`, `WRITEUP.md` — older project overview / methods.
