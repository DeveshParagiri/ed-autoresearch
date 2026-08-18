---
name: autoresearch
description: Conduct persistent, experiment-driven research from repository state. Use when Codex must recover a research frontier, design or execute mechanistic or simulation experiments, use Optuna, protect evaluation integrity, interpret evidence, or preserve durable decisions.
---

# Autoresearch operating procedure

## Module 1: Recover state and choose the next question

Run `uv run python scripts/check_workspace.py`. Read `README.md` for the repository interface, `research.md` for project rules, `memory.md` for the current handoff, and the active experiment with its ancestors and recent runs. Open manifests, contracts, logs, metrics, and artifacts only when the current decision needs them.

Treat experiment records, runs, contracts, manifests, artifacts, Git, and explicit researcher decisions as authority. Treat `memory.md` as a short pointer. Repair stale memory from the durable record and state unresolved uncertainty instead of relying on chat recollection.

Choose one next experiment from an active leaf, unexplained anomaly, missing control, required replication, satisfied revisit condition, or scientifically justified combination. Prefer the least costly experiment that can change the decision. Do not reject a scientific direction merely because one implementation, simulator execution, or optimization attempt failed.

Create exactly one experiment node before material execution. Record its parents, scientific kind, execution mode, manifest input IDs, evaluation contract, question, isolated change, prediction, expected evidence, known failure modes, cost, stopping rule, and revisit condition. Use multiple parents only for a real combination or dependency. Datasets, benchmarks, findings, metrics, figures, runs, and optimizer trials are attached evidence, not graph nodes.

Use `mechanistic` mode for equations, processes, or structural model work; `simulation` for a simulation engine or scenario system; and `hybrid` only when both are part of the same experiment. Keep scientific kind, execution mode, and execution tool separate.

## Module 2: Build and execute a controlled experiment

Resolve each consumed dataset through its manifest ID and inspect its provenance, coverage, units, transformations, limitations, and integrity checks. Do not substitute a source, release, period, mask, grid, or preprocessing route without a new declaration.

Use ordinary project code for adapters, simulators, preprocessing, diagnostics, evaluation, plotting, and optimization. Add a file only when it owns a responsibility that no existing file owns. Name executable scripts `verb_object.py` and keep candidate writes inside the run directory.

Treat an active evaluation contract as immutable machine policy. Candidate code produces target output. Trusted evaluation code produces official metrics and canonical figures. Candidate code, simulator callbacks, optimizers, and diagnostic plotters may not change protected evidence, evaluator code, periods, masks, metrics, aggregation, baselines, or canonical figure rules.

For direct execution, invoke the declared adapter once. For Optuna, declare objective metrics and directions, parameter-space implementation, sampler, pruner, seed, budget, and selection rule before the first trial. Keep every trial inside one experiment and preserve completed, pruned, and failed trials. Allow search code to see only declared development objectives; never expose sealed promotion evidence.

Export `optuna-study.json`, `optuna-trials.jsonl`, and `selected-parameters.json`. Make the selected parameters produce the ordinary candidate output, then pass that output through the same trusted evaluator used for direct execution. Require a later non-adaptive confirmation experiment before promotion.

Execute through `uv run python scripts/run_experiment.py <experiment-id>`. Keep retries, replications, repairs, interruptions, and invalid attempts as separate run IDs. Preserve the exact command, revision and diff, environment, inputs, contract snapshot, events, stdout, stderr, metrics, canonical figures, artifacts, timings, and failure stage.

## Module 3: Interpret evidence and preserve the result

Compare the result with the written prediction and inspect the complete evidence vector. Report regressions, local disagreement, instability, benchmark sensitivity, and transfer failure instead of promoting from one aggregate score.

Classify a negative result as data failure, target or simulator execution failure, implementation failure, numerical failure, evaluation failure, optimization failure, insufficient or noisy evidence, benchmark sensitivity, or evidence against the proposed explanation. Match causal claims to controls and require ablation for multi-factor attribution.

Choose one recorded continuation: replicate, repair, deepen, ablate, attribute, combine, investigate an anomaly, confirm selected parameters, request expensive validation, pause, close, or reopen under a satisfied trigger. A paused branch keeps its evidence and `Revisit when` condition.

Write the result, interpretation, decision, and revisit condition into `experiment.md`. Preserve run evidence unchanged. Update the single root `memory.md` with the active frontier, material blockers, researcher priority, and next action, then run `uv run python scripts/check_workspace.py` again.

Do not create another state file, memory tree, protocol collection, finding node, or tracking layer. Keep project science in `research.md`, repository operation in `README.md`, data identity in manifests, evaluation identity in contracts, execution evidence in runs, and reusable agent behavior in this skill.
