# ED-Fire models A-I

This directory contains the model work that this project is extending. Models A, B, C, D, E, F, G, G6, G7, H, I, and Ibest are not external or miscellaneous models; they are the existing ED-Fire development record. Their equations, parameters, source commits, optimizer artifacts, reconstructed outputs, scores, and known failures are kept together here.

`registry.toml` is the concise record for the thirteen retained variants: the GFED4.1s versions of A, B, and C, followed by the GFED5 refit of C and Models D through Ibest. It records the source, parameter set, mechanism count, reported result, replayed result, and caveat for each variant. `parameter-inventory.csv` checksums all 303 retained parameter JSON files. `parameters/abc-gfed4.1s/` contains the final A, B, and C sets. `parameters/coupled-gfed5/` contains the final sets, regional fits, ranked candidates, checkpoints, and failed or superseded directions from the coupled work.

`commit-artifacts/` preserves all 406 committed versions of model-bearing files across the 115 commits reachable from the three recovered remote branches. Its manifest maps each original path and Git blob to the materialized file, checksum, model family, stage, first and last appearance, and branch reachability. `model-attempts.csv` records the committed parameter sets, optimizer checkpoints, ranked candidates, assemblies, combustion fits, and human-driver variants. `coverage.json` pins the audited branch tips and coverage counts.

`upstream/ed-autoresearch.bundle` is the complete source repository for those branches. It retains source code, scripts, logs, figures, and commits without depending on GitHub. `upstream/abc-gfed4.1s-bffdba5/` is a readable snapshot of the commit that fixed the retained A, B, and C parameter files. `stock-ed/` is separate because it is the official ED v3.0 publisher release, not one of the A-I development variants.

Generated NetCDFs, ILAMB products, metrics, and figures go under `reproduced/`. Git ignores that directory because these products are regenerated from the pinned code, parameters, and project data. The source record and parameter evidence remain tracked.

## Verify the source record

Run:

```bash
python3 scripts/sync_models.py
```

The command reconstructs a temporary bare repository from the bundle, verifies all three branch tips and every Git blob, and compares every materialized file and ledger byte for byte. `--write` is only for deliberately rebuilding the tracked materialization from the bundle; it refuses to delete unexpected files.

## Reproduce models A-I

Install the replay and plotting dependencies, then reproduce all thirteen variants and their recorded ILAMB values:

```bash
uv sync --extra historical
uv run --extra historical python scripts/reproduce_models.py --models all --evaluate
```

The replay verifies the GFED files before and after execution, regenerates every model, writes `reproduced/metrics.csv`, and records the runtime, output hashes, score deltas, tolerances, and per-model status in `reproduced/verification.json`.

Models B and the GFED4.1s version of C reproduce from their committed equations and parameters. Model A is partial because its exact generated NetCDF and frozen LAI preprocessing artifact were never committed; its replay uses period-mean LAI from the retained ED simulation. Models C through Ibest reproduce from the recovered coupled inputs and parameters. Model H has a known platform-sensitive seasonal phase tie, but the selected pinned run reproduced its archived Seasonal score of 0.835 and Overall score of 0.682. Model F is reproducible but not comparable as an unconstrained candidate because its global magnitude was pinned to GFED5.

No generated model-output NetCDF, NumPy archive, pickle, or equivalent result field was committed in the recovered Git history. The tracked numerical files are the coupled GDP inputs and the one-degree GDP grid. The retained variants are reproducible because their generators, parameters, inputs, and reported scores survived; an earlier output that was never committed cannot be recovered from Git.

## Generate the comparable figure suite

After replay, run:

```bash
uv run --extra historical python scripts/figure_models.py --models all
```

The script writes two A-I overview figures and the same six deterministic SciencePlots figures for every retained model. Each model receives the score summary, benchmark and model maps, difference maps, seasonal cycles, cell-level spatial distributions, and benchmark-sensitivity figure used by the active v2 presentation. The renderer uses the repository’s Basier Square fonts, fixed dimensions, fixed scales, fixed region order, and no randomness. Display annotations use three decimal places; machine-readable metrics retain their full precision.

The durable record is the completed `experiment.models-a-i-reproduction` experiment, not the ignored `reproduced/` preview directory. Its [selected reproduction](../research/experiments/experiment.models-a-i-reproduction/runs/run.20260819T113431Z.96777d94/run.json) passed all thirteen replay checks and retained 78 per-model figures plus two overviews. Run it again only as a new recorded attempt:

```bash
uv run --extra historical python scripts/run_experiment.py experiment.models-a-i-reproduction
```

The existing A-I outputs are valid evidence about what this project tried and how those calculations behave. They are not automatically valid candidates under the active clean-candidate contract. The A, B, and GFED4.1s C workflow uses GFED4.1s fire presence and target normalization during output construction. Coupled C, D, F, and H use GFED5 fire presence in generation or fitting. E, G, G6, G7, I, and Ibest inherit a GFED4.1s fire-presence output mask. A new experiment may extend any of their mechanisms or parameter sets, but it must regenerate its candidate from declared non-benchmark inputs and let only the trusted evaluator read GFED fields.

To inspect the recovered repository directly:

```bash
git clone model/upstream/ed-autoresearch.bundle /tmp/ed-autoresearch-history
git -C /tmp/ed-autoresearch-history switch --detach 11ee71418e597e977a4d49f6fda166e20c098e9f
```
