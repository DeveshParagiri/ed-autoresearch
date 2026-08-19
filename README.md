# ED-Fire autoresearch

ED-Fire autoresearch is a reproducible research project for improving the fire process in the Ecosystem Demography model. The current objective is to find mechanisms and parameters that make ED reproduce GFED5 monthly burned area as closely as possible under a fixed 2001–2016 ILAMB evaluation. GFED4.1s is reported alongside GFED5 as a benchmark-sensitivity check.

The autoresearch agent works through one declared experiment at a time. Each experiment states its question and controlled change, runs the relevant model or reproduction, evaluates the output with the same locked contract, saves the metrics and figures, and records what the result means for the next experiment. The agent may write new analysis and execution tools, but it cannot alter the benchmarks, evaluator, comparison period, or figure definitions during an experiment.

This repository contains the official ED v3.0 source and inputs, the source and parameters for Models A–I, dataset installation scripts, evaluation contracts, and recorded experiments. Read [`research.md`](research.md) for the scientific rules, [`memory.md`](memory.md) for the current state and next task, and [`research/experiments/`](research/experiments/) for results.

## What each top-level document owns

`README.md` is the operator’s map. It explains the tree, setup, commands, and the path from an experiment declaration to a recorded result.

`research.md` is the ED-Fire scientific charter. It defines the research target, allowed changes, evidence standard, benchmark boundary, leakage rules, branch policy, Optuna policy, and decisions that require human approval. It should change only when the project’s scientific rules change.

`memory.md` is the short handoff for the next research session. It names the current frontier, material blockers, the researcher’s priority, and the next action. It is deliberately one file. It does not replace experiment records, run evidence, data manifests, contracts, or Git.

`research/meetings/README.md` defines the private intake for transcripts, Granola exports, and meeting notes. The source and structured private record stay under the ignored `research/meetings/private/` path. Only their approved research consequences enter public project files.

`skills/autoresearch/SKILL.md` is the reusable agent procedure. It explains how to recover state, declare and execute an experiment, interpret evidence, and update memory. It contains no ED-Fire-specific scientific claims.

## File tree

```text
ed-fire/
├── .gitignore                        large project-local data, generated state, and run work exclusions
├── README.md                         operator guide and tree
├── research.md                       ED-Fire scientific charter
├── memory.md                         current frontier and next action
├── pyproject.toml                    Python package and optional Optuna and replay dependencies
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
│   ├── reference/                    pinned conversion code and the official ED release archives
│   └── derived/                      created when a recorded run yields a reusable product
├── model/
│   ├── README.md                     model A-I record, replay procedure, and scientific limits
│   ├── registry.toml                 A-I definitions, provenance, results, and caveats
│   ├── parameter-inventory.csv       checksums for every retained parameter artifact
│   ├── parameters/                   final sets, trial sets, checkpoints, and top-k records
│   ├── commit-artifacts/              every model-bearing file version from every remote commit
│   ├── upstream/                     complete old Git history plus a readable A-C snapshot
│   ├── reproduced/                   ignored regenerated outputs, ILAMB evidence, and figures
│   ├── stock-ed/
│   │   ├── UPSTREAM.toml             publisher identity, hashes, license, and provenance limit
│   │   ├── Makefile, *.cc, *.h       exact files from the official ED v3.0 source archive
│   │   ├── external_apps/source_code/ official dependency source tarballs, ignored by Git
│   │   └── EDv3_inputs               link to the canonical official input bundle under data/
├── evals/contracts/
│   ├── burned-area-eval-v1.json      preserved contract for completed stock runs
│   └── burned-area-eval-v2.json      active contract for future candidate runs
├── research/
│   ├── meetings/
│   │   ├── README.md                   private meeting intake and propagation rules
│   │   └── private/                    ignored source material and structured records, created when used
│   ├── framings/                       optional durable questions used by named branches
│   │   └── <framing-id>.md
│   └── experiments/
│       └── <experiment-id>/
│           ├── experiment.md         research record plus curated figure and result links
│           ├── figures/              current presentation derived from the selected run
│           └── runs/<run-id>/        immutable evidence from one terminal attempt
├── scripts/
│   ├── check_workspace.py            validate data, contracts, experiment graph, and runs
│   ├── run_experiment.py             execute one declared experiment and record it
│   ├── stage_stock_baseline.py       stage the pinned native ED baseline
│   ├── stage_models_a_i.py            regenerate the thirteen retained model variants
│   ├── evaluate_burned_area.py       preserved v1 evaluator used by completed stock runs
│   ├── evaluate_burned_area_v2.py    active ILAMB evaluator for future candidate runs
│   ├── evaluate_models_a_i.py         verify A-I replay evidence and retain all figure suites
│   ├── reproduce_models.py           reconstruct and verify models A-I
│   ├── burned_area_figures.py        shared SciencePlots layout for the six canonical images
│   ├── render_experiment_figures.py  refresh current figures from selected-run evidence
│   ├── render_models_overview.py     readable two-column A-I maps from selected-run evidence
│   ├── figure_models.py              fixed A-I overviews and per-model figure suites
│   ├── sync_models.py                expose and verify every commit-derived model artifact
│   ├── install_all_data.sh            orchestrate public retrieval, deterministic builds, and validation
│   ├── install_stock_ed.py            verify and install every official ED v3.0 release asset
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
│   ├── experiments.py                 experiment parsing, parent edges, and event validation
│   ├── workspace.py                   one combined workspace status report
│   ├── runner.py                      protected execution and evidence recording
│   └── optuna_records.py              portable Optuna study and trial export
└── tests/
    ├── test_catalog.py                catalog, provenance, and canonical-path integrity cases
    ├── test_experiments.py            experiment-DAG and search declaration cases
    ├── test_optuna_records.py         portable Optuna record cases
    └── test_models.py                 model registry, parameter, and bundle integrity cases
```

Only experiments form the research DAG. A framing appears as a labeled container only when an `experiment.md` explicitly names it; an experiment with no `framing` field stays at the graph root. Framings do not add parent edges. A dataset is an input to an experiment, a benchmark is protected evidence, an Optuna trial is part of one run, and a metric or figure is an experiment result. None of those becomes another node type.

Meetings do not appear in the graph. Their raw source and structured records stay local and ignored; approved consequences are written into the experiment, framing, charter, memory, or evaluation contract that owns the decision.

When a framing is needed, `research/framings/<framing-id>.md` uses `schema: autoresearch-framing/v1` and records `id`, `title`, `status`, and `created_at` in front matter, followed by Question, Scope, Current position, and Revisit when sections. The experiment opts in with `framing: <framing-id>`; no field or framing file is created when it does not apply.

The tree includes semantic data and run directories that are created only when used. Empty directories are not retained with placeholder files.

The run directory contains `run.json`, the frozen `contract.json`, `metrics.json`, `artifacts.json`, `events.jsonl`, canonical `figures/`, durable `artifacts/`, command and evaluator `logs/`, and disposable `work/`. Interpretation is written back to the experiment record, so the runner does not create a second notes file. A completed `experiment.md` names its selected run and contains human-labeled Markdown references to the figures, results, and outputs chosen for the research reader. The experiment may keep a current presentation under its own `figures/` directory, but those images must be regenerated from the selected run; metrics and scientific outputs still link to that immutable run. When a graph node is selected, the viewer displays the experiment file's front matter and renders its Markdown body and references. It does not construct another narrative or guess which files matter. Git ignores only `work/` and `tmp/`; the evidence files are meant to be retained.

## Why there is no cache directory

Large scientific assets live inside this project at the paths declared in `data/catalog.toml`; they are ignored by Git, not stored elsewhere. A second cache would create an ambiguous copy. Procurement scripts use partial files only while downloading, verify publisher sizes and digests, then place the validated asset at its canonical path. The official ED archives are retained under `data/reference/ed-v3-release/archives/` because they are provenance evidence, not a cache. Older workspaces may symlink to this project, never the reverse. `data/derived/` is for a reusable scientific product with provenance; transient run products stay in that run’s `work/` directory.

## Data state

`data/catalog.toml` declares 42 assets. They include the immutable official ED source, run deck, and release evidence in addition to the climate, ecosystem, coupled-state, human, LUH2, GFED, TRENDY, ILAMB, and candidate-driver inventory. The exact 1-degree historical GDP grid is retained only for replaying Models F and H. The completed stock evaluation still consumes only the pinned native ED burned-area output, GFED5 burned area, GFED4.1s burned area, and the two tracked ILAMB configurations. Installed data do not silently enter an experiment; its declaration must name every consumed catalog ID.

`data/sources.toml` states where each asset came from, how it was produced or linked, its coverage and units, its known limitations, and how to recover it. A manifest ID, not a filename guessed by an agent, identifies data consumed by an experiment.

To validate the data already owned by this project, run:

```bash
bash scripts/install_all_data.sh
```

To also fetch the public products allowed by the manifests, run:

```bash
bash scripts/install_all_data.sh --fetch-public
```

With `--fetch-public`, the script retrieves and verifies the official ED release, GFED4.1s, GFED5 burned area, GFED5.1 emissions, LUH2-GCB2026, the ILAMB GFED4.1s reference, and the pinned ILAMB-Data repository at their project paths. Restricted, lab-derived, coupled, and TRENDY assets are retained locally but are not replaced by superficially similar public files.

## How one experiment runs

An experiment begins as `research/experiments/<experiment-id>/experiment.md`. Its front matter fixes its experiment parents, manifest input IDs, evaluation contract, scientific kind, execution mode, tool, adapter, and exact argument vector before computation begins. It may name one existing framing when that context is useful; otherwise the `framing` field is omitted. `selected_run` is added only after evidence has been interpreted.

`scripts/run_experiment.py` first validates the workspace and requires a clean, committed code state. It creates a unique run directory, snapshots the Git revision, dirty diff, environment, declared input identities, and active contract, then invokes the declared adapter. The adapter may write only inside that run’s work area.

The trusted evaluator then removes any candidate-authored official metrics or canonical figures, verifies every protected hash, evaluates the candidate, creates the fixed figure suite, and verifies the protected files again. The recorder validates required metrics and artifacts before marking the run complete. Failed and invalid attempts remain separate run IDs instead of being overwritten.

After interpretation, the agent writes the result, interpretation, and decision into `experiment.md`, sets `selected_run`, and authors an `Evidence` section with relative Markdown embeds for selected figures and links to selected-run metrics and scientific outputs. A later presentation update may regenerate experiment-level figures from the selected run without changing that run. The fixed document sequence is Question, Rationale, Change, Prediction, Plan, Result, Evidence, Interpretation, Decision, and Revisit when. The document must stand on its own: it explains why the experiment exists or is running, what it tests, what happened when results exist, and what follows from the evidence. Raw logs, disposable work files, and unselected artifacts remain available in the run record but do not enter the research reader automatically.

For the completed stock experiment, `stage_stock_baseline.py` verified and staged the pinned native ED file as `work/model/Candidate/burntArea.nc`, and the v1 evaluator produced its recorded evidence. `render_experiment_figures.py` can present that selected run with the current six-image renderer while leaving the v1 evidence intact.

The completed A-I reproduction is also a first-class experiment and graph node. `stage_models_a_i.py` regenerated all thirteen variants and their archived replay metrics inside the run, and `evaluate_models_a_i.py` verified the output identities, protected files, 78 canonical figures, thirteen suite manifests, and two overview figures. Its `experiment.md` selects the completed run, summarizes the scientific limitations, embeds the score overview and a readable two-column map presentation, and links all six figures for every model. `render_models_overview.py` can refresh that tall map sheet from the selected run without changing immutable run evidence.

Future candidate experiments use `evaluate_burned_area_v2.py`. It runs the same ILAMB evaluations against both GFED products, preserves their scalar databases and logs, writes the metric vector, copies the candidate output into durable artifacts, and delegates the six images to the protected SciencePlots renderer.

## Burned-area contracts

`evals/contracts/burned-area-eval-v1.json` remains intact for the completed stock runs. `evals/contracts/burned-area-eval-v2.json` is the active machine-enforced contract for future candidates. It preserves the v1 benchmarks, ILAMB metrics, period, regions, ranking rule, and candidate interface while versioning the figure presentation. It also pins the trusted evaluator, shared renderer, and bundled Basier Square font files by hash. The runner reads the contract named by an experiment and snapshots it into the run.

The contract is not a node, a research diary, or a second copy of `research.md`. `research.md` says what evidence is scientifically acceptable; the contract makes one approved evaluation executable. If a benchmark, period, mask, region, metric, baseline, evaluator, aggregation rule, or figure specification changes, the old contract remains attached to its runs and a new version is created. That is why the filename names the evaluation family rather than the whole project.

The v2 figure scales are fixed before future candidate evaluation. The 1800 × 1200 absolute-field figure places the GFED4.1s and GFED5 benchmarks above stock ED and the candidate. Its shared 0–80 percent-per-year colorbar sits directly below those four maps. The separate 1800 × 800 difference figure places candidate-minus-GFED4.1s beside candidate-minus-GFED5, with its shared ±60 percent-per-year colorbar directly below the pair. Cell-density panels use a fixed 0–80 percent-per-year window and report the fraction of valid pairs shown. Seasonal panels use region-specific limits fixed across every candidate so low-fire regions remain readable. Score differences use ±0.15, regional differences use ±25 percent per year, and global period means use 0–1 percent per month. End colors include values outside map limits. These presentation limits do not alter ILAMB masks, intersections, scores, or the GFED5 objective.

Rendering is deterministic for the locked software environment: the backend is noninteractive, each figure’s dimensions are fixed by the contract, the SciencePlots style and repository fonts are hash-pinned, plot ranges and ordering are contract-owned, and no renderer uses randomness. Each run records figure checksums, so a matched rerun can be compared byte for byte.

The full-period GFED5 Overall score is the visible optimization target. Direct iteration and Optuna may query that trusted score repeatedly. GFED4.1s remains visible as a sensitivity diagnostic. Benchmark protection fixes the comparison files and evaluator while the optimizer receives the declared objective.

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

Regenerate and record Models A-I with their complete standardized figure suites with:

```bash
uv run --extra historical python scripts/run_experiment.py experiment.models-a-i-reproduction
```

Refresh the stock experiment's current figure presentation from its selected run with:

```bash
uv run --extra historical python scripts/render_experiment_figures.py experiment.stock-baseline --label "Staged ED-stock"
```

Install Optuna only when an approved experiment declares a parameter search:

```bash
uv sync --extra optuna
```

Optuna is a tool inside one scientific experiment, not the research controller. Its objectives, directions, parameter space, sampler, pruner, seed, budget, and selection rule are fixed before the study begins. Every completed, pruned, and failed trial is retained. Under the active contract, a study may maximize the trusted GFED5 Overall score over 2001–2016. The selected parameters must produce the ordinary candidate output and pass through the same evaluator and six-image suite. A later non-adaptive run confirms reproducibility on the same benchmark.

## Models A-I

`model/` contains the ED-Fire work this project is extending. Its registry names the thirteen retained A-I variants, its parameter inventory covers all 303 retained parameter files, and its source bundle preserves every source file, script, log, figure, and commit on the three recovered remote branches. The materialized commit record exposes 406 versions of model-bearing files across 115 commits, including overwritten and deleted A, B, C, combustion, paper, coupled, human-driver, G, H, and I artifacts. Run `python3 scripts/sync_models.py` to prove that the tracked record still matches the bundle.

The [selected A-I reproduction](research/experiments/experiment.models-a-i-reproduction/runs/run.20260819T113431Z.96777d94/run.json) is the durable comparison record. It regenerated all thirteen variants, passed all thirteen replay checks, and retained 78 canonical figures plus two overviews. Run `uv sync --extra historical`, then `uv run --extra historical python scripts/reproduce_models.py --models all --evaluate` only for a disposable direct replay under `model/reproduced/`. `uv run --extra historical python scripts/figure_models.py --models all` directly refreshes the ignored model-level previews. The declared experiment command above is the route for a new recorded attempt. Displayed statistics use three decimal places while the JSON and CSV evidence retain full precision.

These models are our prior work, not a miscellaneous comparison collection. Their mechanisms and parameters may be extended in a new experiment. Their existing outputs are not automatically valid under the active clean-candidate contract because the original generators used benchmark-derived masks or target normalization. The exact coverage, replay result, and per-model limitations live in `model/README.md`.

## Script names

Executable scripts use lowercase `verb_object.py`. `check_` is read-only, `install_` creates declared local state, `download_` fetches upstream data, `build_` performs a deterministic transformation, `stage_` adapts an existing candidate, `run_` executes a declared workflow, `evaluate_` produces trusted evidence, and `render_` derives a current presentation from recorded evidence. The names describe one responsibility; this repository does not add `utils.py`, `helpers.py`, numbered scratch scripts, or another CLI layer.
