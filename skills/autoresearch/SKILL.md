---
name: autoresearch
description: Run persistent, experiment-driven research from repository state. Use when Codex must recover a research frontier, execute mechanistic or simulation experiments, use Optuna, protect evaluation integrity, incorporate private meeting feedback, interpret evidence, or preserve decisions.
---

# Autoresearch operating procedure

## Module 1: Recover state and choose the next question

Run `uv run python scripts/check_workspace.py`. Read `README.md` for the repository interface, `research.md` for project rules, `memory.md` for the current handoff, and the active experiment with its ancestors and recent runs. If the experiment names a framing, read that framing too. Open manifests, contracts, logs, metrics, and artifacts only when the current decision needs them.

Treat experiment records, runs, contracts, manifests, artifacts, Git, and explicit researcher decisions as authority. Treat `memory.md` as a short pointer. Repair stale memory from the durable record and state unresolved uncertainty instead of relying on chat recollection.

When the researcher supplies a transcript, Granola export, or meeting notes, follow `research/meetings/README.md`. Preserve the source unchanged under the ignored private meeting path and write one structured private record that separates explicit feedback, decisions, actions, open questions, and later interpretation. Never commit or expose that private material. Apply confirmed consequences to the experiment, framing, charter, memory, or evaluation contract that owns them; meetings do not become graph nodes, and a public experiment must explain its rationale without relying on the private record.

Choose one next experiment from an active leaf, unexplained anomaly, missing control, required replication, satisfied revisit condition, or scientifically justified combination. Prefer the least costly experiment that can change the decision. Do not reject a scientific direction merely because one implementation, simulator execution, or optimization attempt failed.

Use a framing only when a durable question genuinely groups multiple experiments or needs its own reopening boundary. In that case, create or reuse the framing and add its ID to the experiment front matter. Otherwise omit `framing`; do not manufacture one for schema completeness. Framings appear as graph containers, while experiment parent links form the DAG.

Create exactly one experiment node before material execution. Record its parents, scientific kind, execution mode, manifest input IDs, evaluation contract, question, rationale for choosing it now, isolated change, prediction, expected evidence, known failure modes, cost, stopping rule, and revisit condition. Include the optional framing ID only when one applies. Omit `selected_run` until evidence has been interpreted. Use multiple parents only for a real combination or dependency. Datasets, benchmarks, findings, metrics, figures, runs, and optimizer trials are attached evidence, not graph nodes.

Use `mechanistic` mode for equations, processes, or structural model work; `simulation` for a simulation engine or scenario system; and `hybrid` only when both are part of the same experiment. Keep scientific kind, execution mode, and execution tool separate.

## Module 2: Build and execute a controlled experiment

Resolve each consumed dataset through its manifest ID and inspect its provenance, coverage, units, transformations, limitations, and integrity checks. Do not substitute a source, release, period, mask, grid, or preprocessing route without a new declaration.

Before changing a target implementation, verify the source release or revision, build configuration, and runtime inputs. A publisher source archive is usable code, but it is not proof that the same code and configuration produced a separate baseline artifact.

Use ordinary project code for adapters, simulators, preprocessing, diagnostics, evaluation, plotting, and optimization. Add a file only when it owns a responsibility that no existing file owns. Name executable scripts `verb_object.py` and keep candidate writes inside the run directory.

Treat an active evaluation contract as immutable machine policy. Candidate code produces target output. Trusted evaluation code produces official metrics and canonical figures. Candidate code, simulator callbacks, optimizers, and diagnostic plotters may not change protected evidence, evaluator code, periods, masks, metrics, aggregation, baselines, or canonical figure rules.

Read the contract to determine how benchmark information may be used. A visible development benchmark may be queried repeatedly by direct iteration or an optimizer; score-directed calibration is often the research goal. Candidate-generation code must not read protected target fields, masks, residuals, or summary statistics to construct its output unless the contract explicitly declares them as model inputs. Search code may use metrics returned by the trusted evaluator. Do not invent a holdout requirement or describe a confirmation replay as independent validation unless the project charter defines an independent evaluation.

For direct execution, invoke the declared adapter once. For Optuna, declare objective metrics and directions, parameter-space implementation, sampler, pruner, seed, budget, and selection rule before the first trial. Keep every trial inside one experiment and preserve completed, pruned, and failed trials. Allow search code to see declared development objectives. If the contract defines separate sealed evidence, keep it outside the search path.

Export `optuna-study.json`, `optuna-trials.jsonl`, and `selected-parameters.json`. Make the selected parameters produce the ordinary candidate output, then pass that output through the same trusted evaluator used for direct execution. Run a later non-adaptive confirmation when project rules require it and state whether it proves reproducibility on the same evaluation or performance on independent evidence.

Execute through `uv run python scripts/run_experiment.py <experiment-id>`. Keep retries, replications, repairs, interruptions, and invalid attempts as separate run IDs. Preserve the exact command, revision and diff, environment, inputs, contract snapshot, events, stdout, stderr, metrics, canonical figures, artifacts, timings, and failure stage.

## Module 3: Interpret evidence and preserve the result

Rank candidates by the declared objective and compare the result with the written prediction. Inspect the complete evidence vector and report regressions, local disagreement, instability, benchmark sensitivity, and transfer failure without replacing the declared ranking rule after seeing results.

Classify a negative result as data failure, target or simulator execution failure, implementation failure, numerical failure, evaluation failure, optimization failure, insufficient or noisy evidence, benchmark sensitivity, or evidence against the proposed explanation. Match causal claims to controls and require ablation for multi-factor attribution.

Choose one recorded continuation: replicate, repair, deepen, ablate, attribute, combine, investigate an anomaly, confirm selected parameters, request expensive validation, pause, close, or reopen under a satisfied trigger. A paused branch keeps its evidence and `Revisit when` condition.

Write the result, evidence, interpretation, decision, and revisit condition into `experiment.md`. Select the run that supports the decision, record its ID in `selected_run`, and author the `Evidence` section as the experiment's public research record. Embed the selected canonical figures with descriptive alt text and link the selected-run metric record and scientific outputs with human labels. Use relative paths under `runs/<selected-run>/`. The Rationale must explain why the experiment was chosen, and the Interpretation must explain why the evidence supports the decision. The document must contain everything a reader needs to understand why the experiment ran or is running, what it tested, what happened when results exist, and why the recorded decision follows. Do not make the viewer infer important files, and do not surface `work/`, temporary files, raw logs, events, hashes, or unselected artifacts as research evidence.

Preserve run evidence unchanged. If the experiment names a framing, update that framing's current position only when the branch-level understanding changes. Update the single root `memory.md` with the active frontier, material blockers, researcher priority, and next action, then run `uv run python scripts/check_workspace.py` again.

Do not create another state file, memory tree, protocol collection, finding node, or tracking layer. Keep project science in `research.md`, branch questions in framings, repository operation in `README.md`, data identity in manifests, evaluation identity in contracts, execution evidence in runs, curated experiment evidence in `experiment.md`, and reusable agent behavior in this skill.
