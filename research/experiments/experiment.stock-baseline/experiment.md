---
schema: autoresearch-experiment/v1
id: experiment.stock-baseline
title: Establish the stock ED fire baseline
kind: baseline
status: proposed
created_at: 2026-08-18T00:00:00Z
parents: []
inputs:
  - benchmark.ed-stock.burned-area
  - benchmark.gfed5.ilamb-burned-area
  - benchmark.gfed4.ilamb-burned-area
  - benchmark.config.gfed5-burned-area
  - benchmark.config.gfed4-burned-area
contract: evals/contracts/burned-area-v1.json
execution:
  mode: mechanistic
  tool: direct
  adapter: scripts/stage_stock_baseline.py
  argv: ["{python}", "scripts/stage_stock_baseline.py"]
search: null
---

# Question

Can the pinned native ED burned-area output produce a reproducible comparison against GFED5 and GFED4.1s without using Model A through E assets?

# Change

This experiment introduces no scientific change. It stages the pinned TRENDY v14 EDv3 S3 native output and passes it to the locked ILAMB evaluator. The native ED source revision and producing command are not present, so this is an evaluation baseline rather than source-level reproduction.

# Prediction

The run should reproduce the same GFED5 score vector already associated with the pinned file, add a matched GFED4.1s score vector, and generate the five contract figures. It must not be interpreted as a candidate-model improvement or as proof that native ED can be rebuilt here.

# Result

No run has been recorded. The adapter, evaluator, data, and active contract are present.

# Decision

Run this experiment, inspect the complete evidence, then repeat it once before opening mechanistic changes. Do not introduce Optuna until a direct candidate and its objective are reproducible.

# Revisit when

Reopen the setup if the pinned stock output, benchmark product, ILAMB version, period, region boxes, score definition, or figure contract changes.
