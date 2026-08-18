# __PROJECT_NAME__ autoresearch

__PROJECT_DESCRIPTION__

This repository uses one experiment graph, one data boundary, and one evaluation boundary. It does not create separate nodes for datasets, benchmarks, metrics, figures, findings, or optimizer trials. It also does not use an installed autoresearch CLI, a memory directory, placeholder tracking files, or an archive of superseded protocols.

## Document responsibilities

`README.md` explains the repository and its commands. `research.md` defines project-specific science, evidence, constraints, and approval boundaries. `memory.md` is one short handoff containing the active frontier, material blockers, researcher priority, and next action. `skills/autoresearch/SKILL.md` defines the reusable agent procedure.

The durable record has higher authority than `memory.md`. Data identity belongs in `data/catalog.toml` and `data/sources.toml`. Evaluation identity belongs in `evals/contracts/`. Questions and decisions belong in experiment records. Measured evidence belongs in run directories. Git identifies the code that produced a run.

## File tree

```text
__PROJECT_SLUG__/
├── .gitignore                             local data links and disposable run work exclusions
├── README.md                              repository map and commands
├── research.md                            project scientific charter
├── memory.md                              current frontier and next action
├── pyproject.toml                         base package and optional Optuna dependency
├── data/
│   ├── catalog.toml                       stable dataset IDs, roles, paths, and checks
│   └── sources.toml                       provenance, acquisition, coverage, and limits
├── evals/contracts/
│   └── baseline-v1.json                   draft machine evaluation contract
├── research/experiments/
│   └── experiment.baseline/
│       └── experiment.md                  baseline question, execution, result, and decision
├── scripts/
│   ├── check_workspace.py                 read-only validation entry point
│   ├── install_data.py                    manifest-driven link installer
│   └── run_experiment.py                  controlled experiment entry point
├── skills/autoresearch/
│   ├── SKILL.md                           reusable agent operating procedure
│   └── agents/openai.yaml                 Codex display metadata and default invocation
├── src/autoresearch/
│   ├── __init__.py                        package boundary
│   ├── validation.py                      manifests, contracts, lineage, and run validation
│   ├── data_install.py                    safe project-path linking
│   ├── runner.py                          protected execution and evidence recording
│   └── optuna_records.py                  portable Optuna study and trial export
└── tests/test_workspace.py                end-to-end kernel tests
```

The installer creates `data/inputs/`, `data/benchmarks/`, `data/reference/`, or `data/derived/` only when a manifest entry uses that category. The runner creates `research/experiments/<experiment-id>/runs/<run-id>/` only when an experiment executes. Empty directories are not kept alive with `.gitkeep` files.

`data/inputs/` is for values consumed by the research target. `data/benchmarks/` is for protected observations, fixed baselines, and evaluation configuration. `data/reference/` is for pinned external code or source notes needed to understand or reproduce an asset. `data/derived/` is for reusable scientific products with recorded provenance. Temporary output belongs in a run’s `work/` directory, not in a project cache.

## What the evaluation contract does

`evals/contracts/baseline-v1.json` is executable policy, not a note and not a graph node. Before activation, replace its placeholders and give the file a name of the form `<evaluation-family>-v<integer>.json`. The contract pins the trusted evaluator, protected benchmark files, development and promotion boundaries, required metrics, comparison baselines, aggregation rule, and canonical figures. A run snapshots the contract and rejects protected-file drift. A material change to ranking or presentation requires a new version rather than editing the comparison family in place.

## Experiments and runs

An experiment is the only graph node. Its `parents` field creates lineage. Its scientific `kind` states what is being tested. Its execution `mode` is `mechanistic`, `simulation`, or `hybrid`. Its execution `tool` names the implementation route, normally `direct` or `optuna`. The mode, kind, and tool are separate because a simulator can be run directly, a mechanistic model can be tuned with Optuna, and a hybrid experiment can invoke several project tools without changing the graph schema.

Every experiment keeps its question, isolated change, prediction, result, decision, and revisit condition in one `experiment.md`. One terminal attempt creates one run ID. Retries and failures are new runs rather than edits to old evidence.

A run contains `run.json` for identity, status, revision, inputs, commands, and failure stage; `contract.json` for the exact evaluation snapshot; `metrics.json` for trusted scalar evidence; `artifacts.json` for artifact paths and checksums; `events.jsonl` for the append-only execution timeline; `logs/` for stdout, stderr, environment, Git status, and diffs; `figures/` for canonical figures; `artifacts/` for durable non-figure outputs; and `work/` for disposable candidate state. Interpretation is written back to `experiment.md`, so the runner does not create a second notes file.

## Public commands

Install the base project and validate it with:

```bash
uv sync
uv run python scripts/check_workspace.py
```

Link data declared by the manifests with:

```bash
uv run python scripts/install_data.py plan
uv run python scripts/install_data.py install
```

After the baseline adapter and contract are complete, active, reviewed, and committed, run:

```bash
uv run python scripts/run_experiment.py experiment.baseline
```

There is no `ar` command. These three script entry points are the complete public harness surface.

## Script naming

Executable scripts use lowercase `verb_object.py`. `check_` is read-only validation, `install_` creates manifest-declared local state, `download_` retrieves an upstream asset, `build_` deterministically transforms data, `stage_` adapts an existing candidate, `run_` executes a declared workflow, and `evaluate_` produces trusted evidence. A version or product suffix is added only when it disambiguates the object. Names such as `utils.py`, `helpers.py`, `main.py`, `pipeline.py`, and numbered scratch scripts are not part of the public surface.

Project-specific adapters, simulators, preprocessing, evaluators, and plotting code may live in `scripts/` while small and may move into a project package when reused. A new top-level file must own a responsibility that cannot fit an existing authoritative file. A new tracker is not added when an experiment record, run artifact, manifest, contract, or `memory.md` already owns the information.

## Optuna

Optuna is optional and is installed with `uv sync --extra optuna`. One study belongs to one scientific experiment. Its objective metrics and directions, parameter-space implementation, sampler, pruner, seed, budget, and selection rule are declared before execution. `optuna_records.py` exports the complete study, all trial states, and selected parameters into the run’s artifacts. The selected candidate then passes through the same trusted evaluator as a direct candidate and is confirmed later without adaptive search.

The generated project is structurally valid but not scientifically runnable. It becomes runnable only after real data, a target adapter, a trusted evaluator, canonical outputs, an active approved contract, and a Git baseline exist.
