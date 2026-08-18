# Generic autoresearch scaffolder

This directory creates a small repository-local research workspace for a mechanistic model, simulation engine, hybrid system, program, or other target that can be changed and evaluated. The generated project has one experiment DAG, one current-memory file, manifest-addressed data, versioned evaluation contracts, protected execution, and ordinary project scripts. It contains no ED-Fire assumptions.

The template is an exact allowlist of 20 files. `scaffold.py check` fails when a required file is missing or an unexpected file appears. This prevents the template from accumulating placeholder trackers, duplicate protocols, dormant UI code, and generic helper files with no runtime owner.

## Scaffolder tree

```text
research/autoresearch/
├── README.md                              this design and usage guide
├── scaffold.py                            validate, render, and atomically copy the template
├── template/                              the exact generated project
│   ├── .gitignore                         local data links and disposable run work exclusions
│   ├── README.md
│   ├── research.md
│   ├── memory.md
│   ├── pyproject.toml
│   ├── data/
│   │   ├── catalog.toml
│   │   └── sources.toml
│   ├── evals/contracts/baseline-eval-v1.json
│   ├── research/experiments/experiment.baseline/experiment.md
│   ├── scripts/
│   │   ├── check_workspace.py
│   │   ├── install_data.py
│   │   └── run_experiment.py
│   ├── skills/autoresearch/
│   │   ├── SKILL.md
│   │   └── agents/openai.yaml
│   ├── src/autoresearch/
│   │   ├── __init__.py
│   │   ├── validation.py
│   │   ├── data_install.py
│   │   ├── runner.py
│   │   └── optuna_records.py
│   └── tests/test_workspace.py
└── tests/
    ├── test_scaffold.py                  generator and runtime integration tests
    └── test_skill_quality.py             skill structure and metadata tests
```

`scaffold.py` has one job: substitute the project name, slug, description, execution mode, data-root environment variable, and creation time, then copy the allowlisted template through a temporary staging directory. It refuses an existing destination and has no overwrite mode.

The template README explains every generated file, the run artifacts, node semantics, contract role, data categories, and script naming convention. The source package provides deterministic plumbing behind the three public scripts and exposes no second user interface. `validation.py` reads and checks the repository state. `data_install.py` creates only manifest-declared links. `runner.py` enforces the candidate/evaluator boundary and records terminal evidence. `optuna_records.py` gives optional parameter studies one portable output format. No separate graph module is needed because lineage already lives in experiment front matter and validation already traverses it.

## Why this shape

The published evidence does not prove one universal filesystem layout, so the structure here is an engineering inference from repeated findings about agent interfaces and long-horizon work.

[SWE-agent, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html) found that interface design changes agent performance and that a custom interface improved repository navigation, editing, and test execution. The scaffold therefore exposes three named scripts instead of an `ar` command with many subcommands.

[PaperBench, ICML 2025](https://proceedings.mlr.press/v267/starace25a.html) reports that paginated file access, context management, and repeated “take the next step” interaction helped with long replication tasks and reduced premature termination. The scaffold therefore keeps authoritative context in a few predictable root files and makes one experiment the unit of progress.

[MLAgentBench, ICML 2024](https://proceedings.mlr.press/v235/huang24y.html) identifies long-term planning and hallucination as persistent weaknesses in autonomous experimentation. The scaffold therefore puts predictions, results, decisions, failures, and revisit conditions in durable experiment records rather than relying on chat history.

[CORE-Bench, TMLR 2025](https://tmlr.infinite-conf.org/paper_pages/BsMMc4MEGS.html) treats computational reproducibility as a prerequisite for research agents, while [PaperBench](https://proceedings.mlr.press/v267/starace25a.html) evaluates executed artifacts against explicit criteria. The scaffold therefore separates candidate execution from a trusted evaluation contract and retains exact commands, environment, revision, metrics, figures, and checksums.

[Nemotron-CORTEXA, ICML 2025](https://proceedings.mlr.press/v267/sohrabizadeh25a.html) shows that precise localization of relevant repository context improves agent efficiency. The scaffold therefore uses conventional names, one owner per fact, and no archive or placeholder file that competes for attention.

[Structurally Aligned Subtask-Level Memory, ICML 2026](https://openreview.net/forum?id=2CoRS45Ucj) finds that whole-episode memory can retrieve the wrong experience when the current functional step differs. The scaffold keeps `memory.md` short and current while preserving granular experiment declarations, event timelines, command logs, and run evidence under the experiment that produced them. It does not create a separate memory filesystem.

## Markdown rules

No peer-reviewed result establishes one correct Markdown layout. The rules here are engineering inferences from findings about how models retrieve and use written context. [Lost in the Middle, TACL 2024](https://aclanthology.org/2024.tacl-1.9/) shows that relevant information can become less usable when buried inside a long context. [DocPrompting, ICLR 2023](https://iclr.cc/virtual/2023/poster/11358) shows that retrieving relevant documentation improves code generation for unfamiliar interfaces. PaperBench and Nemotron-CORTEXA provide the repository-agent evidence described above.

The README, research charter, memory file, and skill therefore state their responsibility at the start. The active handoff appears at the start of `memory.md`; an experiment keeps its decision and revisit condition at the end of a fixed question, change, prediction, plan, result, decision, and revisit sequence. Stable, descriptive headings mirror the scientific workflow. A fact has one authoritative owner and other files link to it instead of paraphrasing it. Paths, manifest IDs, contract IDs, commands, units, and evidence references are exact. `memory.md` stays short; experiments, runs, contracts, manifests, and Git retain durable history. Root documents do not accumulate chronological change logs. Tables are reserved for repeated exact mappings, while scientific judgment stays in prose.

## Node and contract model

Only experiments are nodes. A parent relation means that the child extends, repairs, replicates, ablates, combines, or reopens an earlier experiment. Runs are executions of a node. Datasets are declared inputs. Benchmarks are protected evaluation evidence. Metrics, figures, findings, and optimizer trials are outputs attached to a run or interpreted in the experiment. This makes a future DAG viewer straightforward without forcing unrelated repository files into a graph.

An evaluation contract is active machine policy. It defines what the trusted evaluator is, which files are protected, what evidence is required, how comparisons are aggregated, which baselines appear, and how canonical figures are rendered. The contract is versioned because changing a benchmark, mask, period, metric, aggregation rule, baseline, evaluator, or figure presentation changes the comparison family. It is not another research document and not a tracking file.

## Create a project

Validate the bundled template with:

```bash
python3 research/autoresearch/scaffold.py check
```

Preview a mechanistic project with:

```bash
python3 research/autoresearch/scaffold.py create /path/to/new-project \
  --name "Project Name" \
  --description "What this project is trying to understand." \
  --mode mechanistic \
  --dry-run
```

Create it by removing `--dry-run`. Use `--mode simulation` when the research target is a simulation engine or scenario system. Use `--mode hybrid` only when one experiment genuinely combines a process model and a simulation layer. The mode can vary by later experiment; the creation flag only sets the baseline correctly.

## Extension rule

The scaffold is extensible through content and ordinary project code, not through permanent empty layers. New datasets extend the two manifests. New evaluation families add a versioned contract. New research directions add experiment directories. New target adapters and evaluators add scripts with concrete names. New reusable behavior enters the project package only after at least one real workflow needs it.

Executable scripts use lowercase `verb_object.py`. The supported verbs are semantic: `check_` reads and validates, `install_` creates declared local state, `download_` fetches an upstream source, `build_` deterministically transforms data, `stage_` adapts a candidate, `run_` executes a declared workflow, and `evaluate_` produces trusted evidence. Vague names such as `utils.py`, `helpers.py`, `main.py`, `pipeline.py`, and `misc.py` are rejected as design choices even when Python would accept them.

The generated baseline and contract are deliberately incomplete. Structural validation can pass before scientific execution is possible. A real run requires declared data, an adapter, a trusted evaluator, canonical outputs, an active approved contract, and a Git baseline.
