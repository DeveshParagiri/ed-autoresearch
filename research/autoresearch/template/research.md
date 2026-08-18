# __PROJECT_NAME__ research charter

This file defines the project-specific scientific target, evidence, constraints, branch policy, and approval boundaries. Setup belongs in `README.md`, the current handoff belongs in `memory.md`, data identity belongs in the manifests, executable evaluation policy belongs in contracts, and reusable agent procedure belongs in `skills/autoresearch/SKILL.md`.

## Objective

__PROJECT_DESCRIPTION__

State what the project is trying to understand or improve and what would count as a credible result. Do not reduce the objective to one score unless the problem genuinely has one sufficient outcome.

## Research target

Describe the model, simulation engine, program, process, or system being researched. State which parts may change and which interfaces must remain stable.

Experiments may use `mechanistic` execution for process or structural work, `simulation` execution for simulator and scenario studies, or `hybrid` execution when both matter. The scientific experiment kind and the execution mode are separate.

## Inputs

Define the minimum baseline inputs, optional candidate inputs, provenance standard, coverage requirements, transformations, and rules for introducing new data. Reference assets by manifest ID.

## Evaluation

Define the evidence vector, development evaluation, sealed promotion evaluation, comparison baselines, canonical figures, uncertainty, and regressions that must remain visible. Evaluation contracts and trusted code own machine-enforced details.

## Search and optimization

Define which scientific directions may be explored and which costs constrain them. Optimization tools such as Optuna operate inside one fixed experiment. Trials are not research branches. Search code may use only declared development objectives and must preserve the complete trial history and selected parameters.

## Constraints and oversight

State hard scientific constraints, leakage risks, protected evidence, cost tiers, and decisions requiring human approval. A changed evaluation rule creates a new comparison family.

## Memory and branch policy

Experiments and runs are durable memory. Each experiment states its parents, prediction, result, decision, and conditions for revisiting it. A failed execution is not automatically evidence against the research idea.

## Research sequence

Define which controls must precede model changes, how work progresses from cheap diagnostics to expensive validation, and which prerequisites unlock new inputs, optimization, or coupled execution. Completion state and the immediate next action belong in experiment records and `memory.md`.

## Open questions

Record the unresolved questions that should guide the next experiments.
