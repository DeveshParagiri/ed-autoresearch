# Historical ED-Fire models

This directory preserves and replays the model work that predates the clean ED-Fire research line. It is evidence of what was tried, not a source of admissible candidates. Optimizing parameters against the GFED5 score is valid under the current research goal. The old generators cross a different boundary: they read benchmark-derived fire masks while constructing outputs, and some recompute a normalization directly from the target field.

`registry.toml` is the concise ledger for the thirteen final variants that can be discussed or replayed as named models: legacy A, B, and C, then GFED5 C, D, E, F, G, G6, G7, H, I, and Ibest. It records each variant's source, parameters, mechanism count, archived result, replay result, and caveat. `parameter-inventory.csv` checksums the 303 retained branch-tip JSON files. `parameters/abc-gfed4.1s/` holds the final A, B, and legacy C parameters, while `parameters/coupled-gfed5/` holds the 300 final sets, regional fits, ranked candidates, checkpoints, and failed or superseded directions recovered from the coupled branch tip.

That branch-tip view is not the complete history. Files such as `models/A/params.json`, `models/B/params.json`, `models/C/params.json`, `models/paper/D.json`, and `patches/fire_modelC.cc` were replaced in place. `commit-artifacts/` exposes every committed version from every commit reachable from the three remote branches. Its 406 files cover 381 original paths and 373 unique Git blobs across 115 commits. `manifest.csv` maps each original path and Git blob to its materialized file, checksum, model family, stage, first and last appearance, and branch reachability. `model-attempts.csv` is the JSON-only working ledger, including parameter sets, optimizer checkpoints, ranked-candidate records, assemblies, negative directions, combustion fits, and human-driver variants. `coverage.json` pins the branch tips and records the exact coverage totals.

The exposed history includes all committed files under `models/`, `configs/`, `data_human/`, `patches/`, and `HPC_AFRICA_HANDOFF/reference/`. It therefore retains the early A and B generations, every C overwrite, paper C through E, the coupled refits, tropical and seasonal variants, regional and held-out fits, combustion variants, GDP, population, land-use and curing work, Models G through I, and the coupled handoff. An optimizer trial is not renamed as a separate scientific model; committed checkpoint and top-k files preserve those trials at the level the repository actually recorded them.

`upstream/ed-autoresearch.bundle` is the complete source repository, including every script, log, figure, commit, and code version on `main`, `coupled-refit-gfed5`, and `modelD-paper-params`. The visible commit-artifact ledger does not duplicate all of that source. `upstream/abc-gfed4.1s-bffdba5/` is a convenient readable snapshot of the commit that fixed the final A, B, and C parameter files. Generated NetCDFs, ILAMB products, metrics, and figures go under `reproduced/`; Git ignores that directory because retained replay outputs are regenerated from the pinned inputs and code here.

Run the history audit with:

```bash
python scripts/sync_other_model_history.py
```

Use `--write` only when deliberately rebuilding the materialized files from the pinned bundle. The command refuses to delete unexpected files, reconstructs a temporary bare repository, verifies all three branch tips, verifies every Git blob, and compares every generated file and ledger byte for byte.

## Replay

Install the historical numerical dependencies and run the complete replay with:

```bash
uv sync --extra historical
uv run --extra historical python scripts/reproduce_other_models.py --models all --evaluate
```

The `historical` extra pins the numerical package versions used by the verified replay. The script verifies protected benchmark hashes before and after execution, reconstructs every model, runs the pinned ILAMB evaluator, writes `reproduced/metrics.csv`, and records the runtime, script hash, output hashes, per-metric deltas, and tolerances in `reproduced/verification.json`. It deliberately labels every output `admissible_for_current_research=false`.

Models B and legacy C reproduce from the committed GFED4.1s equations and parameters. Model A is only a partial reconstruction: its exact generated NetCDF and frozen LAI preprocessing artifact were never committed, so the replay uses period-mean LAI from the official ED simulation. The coupled GFED5 Models C, D, E, F, G, G6, G7, H, I, and Ibest reproduce from the recovered branch inputs and parameters. Model H has a documented platform-sensitive phase-tie drift: all nonseasonal scores reproduce, while tiny floating-point differences move the seasonal score from 0.835265 to 0.836376 and Overall from 0.681881 to 0.682103 on the current macOS environment.

Ibest is an exploratory regional hybrid that selected between the G and Gtrop parameterizations using their regional GFED5 objective values. That selection is ordinary development optimization. Ibest remains historical because its assembled output inherits the GFED4.1s fire-presence mask and does not descend from the controlled stock line. Its name does not mean it has the highest global score in the recovered ladder.

Git history contains no generated model-output NetCDF, NumPy archive, pickle, or equivalent result field. The only tracked NetCDF and NumPy files in the audited history are the two coupled GDP inputs and the one-degree GDP grid. Deleted outputs that were never committed cannot be recovered from Git. The final named variants are reproducible because their generators, parameters, inputs, and reported scores were recoverable; earlier commit artifacts remain provenance unless a separate replay is added.

To inspect the original repository without a network dependency, clone the bundle and detach at the coupled branch tip:

```bash
git clone model/other-models/upstream/ed-autoresearch.bundle /tmp/ed-autoresearch-history
git -C /tmp/ed-autoresearch-history switch --detach 11ee71418e597e977a4d49f6fda166e20c098e9f
```

## Figures

The coupled branch contains 33 figure-oriented scripts. Thirty-one write figures, twenty-nine name particular NetCDF layouts, and twenty-nine set their own map or axis limits. One also depends on an absolute `paper_gmd` path. They document the sequence of analyses, but they do not define one comparable plotting interface and remain available in the Git bundle as provenance.

After a verified replay, run:

```bash
uv run --extra historical python scripts/figure_other_models.py
```

This writes the two model-ladder overviews and the six-image SciencePlots suite for Model Ibest under `reproduced/figures/Ibest/`. The 1800 × 1200 absolute-field image and 1800 × 800 difference image have separate colorbars directly below the maps they describe; the other four images are 1800 × 1200. Pass `--models all` to render the same score summary, maps, seasonal cycles, spatial distributions, and benchmark-sensitivity figure for all thirteen archived models. The script evaluates each selected output against both GFED products, reuses matching ILAMB results, and keeps the active v2 contract’s scales, regions, ordering, filenames, renderer, and bundled fonts. These files remain descriptive historical comparisons. New proposed models receive their official figures from `scripts/evaluate_burned_area_v2.py`; the historical plotting script cannot write candidate evidence.

## Scientific boundary

The A, B, and legacy C workflow masks cells where GFED4.1s reports fire and recomputes a scale factor from the GFED4.1s mean. Coupled Models C, D, F, and H use a GFED5 fire-presence mask to define generation or fitting support. Models E, G, G6, G7, I, and Ibest inherit a GFED4.1s fire-presence output mask. Model F also pins global burned area directly to GFED5. These target-fed construction steps keep the ladder historical. Ordinary use of GFED5 metrics for parameter optimization remains valid.
