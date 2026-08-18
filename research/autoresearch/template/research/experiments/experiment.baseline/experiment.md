---
schema: autoresearch-experiment/v1
id: experiment.baseline
title: __BASELINE_TITLE_YAML__
kind: baseline
status: proposed
created_at: __CREATED_AT__
parents: []
inputs: []
contract: evals/contracts/baseline-v1.json
execution:
  mode: __EXECUTION_MODE__
  tool: direct
  adapter: null
  argv: []
search: null
---

# Question

What command and output contract establish a clean, reproducible baseline for this project?

# Change

Introduce no research change. Connect the existing target to a run-owned output directory through one adapter.

# Prediction

The first execution will either establish the baseline or identify a precise missing interface, dependency, or evaluation requirement.

# Plan

Define the expected evidence, likely execution or interpretation failures, cost tier, and stopping rule before activating this experiment.

# Result

No run has been recorded.

# Decision

Keep the experiment proposed until the adapter and evaluation contract are complete.

# Revisit when

Run when the target revision, adapter, inputs, trusted evaluator, canonical outputs, and Git baseline have been reviewed.
