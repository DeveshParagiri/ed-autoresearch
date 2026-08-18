# ED-Fire autoresearch

This repository is the working record for research on the mechanistic fire model used by Ecosystem Demography. It has a recorded, reproducible burned-area baseline: a pinned TRENDY v14 EDv3 S3 output is staged as the candidate and evaluated by ILAMB 2.7.3 against GFED5 and GFED4.1s over 2001–2016. Two matched runs at the same code revision reproduced the candidate, complete metric vector, ILAMB score products, and five canonical figures byte for byte. The original ED source revision and producing command are not present, so this is an evaluation floor rather than source-level reproduction of ED. The measured result and decision live in `research/experiments/experiment.stock-baseline/experiment.md`.

The structure is intentionally small. One document defines the science, one records the current handoff, one reusable skill tells an agent how to operate, and the remaining files either identify data, lock an evaluation, execute an experiment, or preserve its evidence. There is no `ar` CLI, separate memory directory, archive, data cache, or UI.

## What each top-level document owns

`README.md` is the operator’s map. It explains the tree, setup, commands, and the path from an experiment declaration to a recorded result.

`research.md` is the ED-Fire scientific charter. It defines the research target, allowed changes, evidence standard, benchmark boundary, leakage rules, branch policy, Optuna policy, and decisions that require human approval. It should change only when the project’s scientific rules change.

`memory.md` is the short handoff for the next research session. It names the current frontier, material blockers, the researcher’s priority, and the next action. It is deliberately one file. It does not replace experiment records, run evidence, data manifests, contracts, or Git.

`skills/autoresearch/SKILL.md` is the reusable agent procedure. It explains how to recover state, declare and execute an experiment, interpret evidence, and update memory. It contains no ED-Fire-specific scientific claims.

## File tree

```text
ed-fire/
├── .gitignore                        local data links, generated state, and run work exclusions
├── README.md                         operator guide and tree
├── research.md                       ED-Fire scientific charter
├── memory.md                         current frontier and next action
├── pyproject.toml                    Python package and optional Optuna dependency
├── uv.lock                           locked Python environment
├── data/
│   ├── README.md                     data-layer rules
│   ├── catalog.toml                  stable dataset IDs, roles, paths, and checks
│   ├── sources.toml                  provenance, acquisition, coverage, and limits
│   ├── inputs/                       model-facing climate, ecosystem, human, and candidate inputs
│   ├── benchmarks/
│   │   ├── observations/             GFED evaluation products
│   │   ├── configs/                  tracked ILAMB configurations
│   │   ├── comparison-models/        native ED and external model outputs
│   │   └── source/                   source products used to build or audit references
│   ├── reference/                    pinned conversion code and source notes
│   └── derived/                      created when a recorded run yields a reusable product
├── evals/contracts/
│   └── burned-area-eval-v1.json      locked burned-area evaluation contract
├── research/
│   ├── experiments/
│   │   └── <experiment-id>/
│   │       ├── experiment.md         question, change, plan, result, and decision
│   │       └── runs/<run-id>/        immutable evidence from one terminal attempt
│   └── autoresearch/                 generic scaffolder for other research projects
├── scripts/
│   ├── check_workspace.py            validate data, contracts, lineage, and runs
│   ├── run_experiment.py             execute one declared experiment and record it
│   ├── stage_stock_baseline.py       stage the pinned native ED baseline
│   ├── evaluate_burned_area.py       trusted ILAMB evaluation and canonical figures
│   ├── install_data.py                plan or create manifest-declared data links
│   ├── install_all_data.sh            orchestrate public retrieval, builds, links, and validation
│   ├── download_public_data.py        retrieve public GFED and ILAMB source products
│   ├── download_luh2_gcb2026.py      retrieve and verify the large LUH2 release
│   ├── build_gfed5_burned_area.py    construct the conservative GFED5 ILAMB reference
│   └── build_gfed5_fire_emissions.py construct the conservative GFED5 emissions reference
├── skills/autoresearch/
│   ├── SKILL.md                       reusable agent procedure
│   └── agents/openai.yaml             Codex display metadata and default invocation
├── src/autoresearch/
│   ├── __init__.py                    package boundary
│   ├── catalog.py                     ED-Fire catalog and integrity validation
│   ├── data_install.py                safe source-root linking and recovery guidance
│   ├── experiments.py                 experiment parsing, lineage, and event validation
│   ├── workspace.py                   one combined workspace status report
│   ├── runner.py                      protected execution and evidence recording
│   └── optuna_records.py              portable Optuna study and trial export
└── tests/
    ├── test_catalog.py                catalog integrity cases
    ├── test_data_install.py           safe linking and provenance cases
    ├── test_experiments.py            experiment-DAG and search declaration cases
    └── test_optuna_records.py         portable Optuna record cases
```

Only experiments form the research DAG. A dataset is an input to an experiment, a benchmark is protected evidence, an Optuna trial is part of one run, and a metric or figure is an experiment result. None of those becomes a second kind of graph node.

The tree includes semantic data and run directories that are created only when used. Empty directories are not retained with placeholder files.

The run directory contains `run.json`, the frozen `contract.json`, `metrics.json`, `artifacts.json`, `events.jsonl`, canonical `figures/`, durable `artifacts/`, command and evaluator `logs/`, and disposable `work/`. Interpretation is written back to the experiment record, so the runner does not create a second notes file. Git ignores only `work/` and `tmp/`; the evidence files are meant to be retained.

## Why there is no cache directory

Large scientific assets have a canonical home outside this repository and appear here through stable symlinks. A second project cache would create another ambiguous copy of the same data. Procurement scripts download to controlled temporary locations, validate what they obtained, and place or link the final asset at its manifest path. `data/derived/` is for a reusable scientific product with provenance; transient run products stay in that run’s `work/` directory.

## Data state

`data/catalog.toml` currently declares 38 assets, and all 38 resolve. The required stock run consumes only the pinned native ED burned-area output, GFED5 burned area, GFED4.1s burned area, and the two tracked ILAMB configurations. GFED5 fire emissions and the wider climate, ecosystem, coupled-state, human, LUH2, TRENDY, and candidate-driver inventory are installed for later experiments but do not silently enter this baseline.

`data/sources.toml` states where each asset came from, how it was produced or linked, its coverage and units, its known limitations, and how to recover it. A manifest ID, not a filename guessed by an agent, identifies data consumed by an experiment.

To inspect the intended links without changing them, run:

```bash
uv run python scripts/install_data.py plan
```

To link an authorized local data root and validate it, run:

```bash
bash scripts/install_all_data.sh
```

To also fetch the public products allowed by the manifests, run:

```bash
bash scripts/install_all_data.sh --fetch-public
```

The installer does not replace a regular file or a symlink pointing somewhere else. Restricted, lab-derived, coupled, and TRENDY assets require an authorized seed root; the scripts do not substitute a superficially similar public file.

## How one experiment runs

An experiment begins as `research/experiments/<experiment-id>/experiment.md`. Its front matter fixes its parents, manifest input IDs, evaluation contract, scientific kind, execution mode, tool, adapter, and exact argument vector before computation begins.

`scripts/run_experiment.py` first validates the workspace and requires a clean, committed code state. It creates a unique run directory, snapshots the Git revision, dirty diff, environment, declared input identities, and active contract, then invokes the declared adapter. The adapter may write only inside that run’s work area.

The trusted evaluator then removes any candidate-authored official metrics or canonical figures, verifies every protected hash, evaluates the candidate, creates the fixed figure suite, and verifies the protected files again. The recorder validates required metrics and artifacts before marking the run complete. Failed and invalid attempts remain separate run IDs instead of being overwritten.

For the stock experiment, `stage_stock_baseline.py` verifies and stages the pinned native ED file as `work/model/Candidate/burntArea.nc`. `evaluate_burned_area.py` runs full ILAMB evaluations against both GFED products, preserves their scalar databases and logs, writes the metric vector, copies the candidate output into durable artifacts, and generates the same five 1800 × 1200 figures for every candidate under this contract.

## What `burned-area-eval-v1.json` does

`evals/contracts/burned-area-eval-v1.json` is the machine-enforced burned-area evaluation contract. It pins the two benchmark files, two ILAMB configurations, native ED baseline, trusted evaluator and its hash, candidate-output interface, ILAMB version, period, regions, metric IDs, aggregation rule, five canonical figures, and the fact that promotion is disabled. The runner reads this file before every experiment and snapshots it into the run.

The contract is not a node, a research diary, or a second copy of `research.md`. `research.md` says what evidence is scientifically acceptable; the contract makes one approved evaluation executable. If a benchmark, period, mask, region, metric, baseline, evaluator, aggregation rule, or figure specification changes, the old contract remains attached to its runs and a new version is created. That is why the filename names the evaluation family rather than the whole project.

## Commands

Install the base environment and validate the workspace with:

```bash
uv sync
uv run python scripts/check_workspace.py
```

Run the declared stock baseline with:

```bash
uv run python scripts/run_experiment.py experiment.stock-baseline
```

Install Optuna only when an approved experiment declares a parameter search:

```bash
uv sync --extra optuna
```

Optuna is a tool inside one scientific experiment, not the research controller. Its objectives, directions, parameter space, sampler, pruner, seed, budget, and selection rule are fixed before the study begins. Every completed, pruned, and failed trial is retained. The selected parameters must produce the ordinary candidate output and pass through the same trusted evaluator; a later non-adaptive experiment confirms the selection.

## Script names

Executable scripts use lowercase `verb_object.py`. `check_` is read-only, `install_` creates declared local state, `download_` fetches upstream data, `build_` performs a deterministic transformation, `stage_` adapts an existing candidate, `run_` executes a declared workflow, and `evaluate_` produces trusted evidence. The names describe one responsibility; this repository does not add `utils.py`, `helpers.py`, numbered scratch scripts, or another CLI layer.

## Generic scaffolder

`research/autoresearch/` creates the same minimal structure for another mechanistic model, simulation engine, program, or hybrid research problem. It supports `mechanistic`, `simulation`, and `hybrid` execution modes because those are project-level possibilities. ED-Fire itself uses the mechanistic mode. The generated workspace starts with a draft contract and proposed baseline and cannot run until its owner supplies real data, an adapter, a trusted evaluator, canonical outputs, approval, and a Git baseline.
