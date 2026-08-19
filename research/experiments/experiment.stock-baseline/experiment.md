---
schema: autoresearch-experiment/v2
id: experiment.stock-baseline
title: Establish the stock ED fire baseline
kind: baseline
status: completed
created_at: 2026-08-18T00:00:00Z
parents: []
inputs:
  - benchmark.ed-stock.burned-area
  - benchmark.gfed5.ilamb-burned-area
  - benchmark.gfed4.ilamb-burned-area
  - benchmark.config.gfed5-burned-area
  - benchmark.config.gfed4-burned-area
contract: evals/contracts/burned-area-eval-v1.json
execution:
  mode: mechanistic
  tool: direct
  adapter: scripts/stage_stock_baseline.py
  argv: ["{python}", "scripts/stage_stock_baseline.py"]
search: null
selected_run: run.20260818T171220Z.a7122045
---

# Question

Can the pinned native ED burned-area output produce a reproducible comparison against GFED5 and GFED4.1s without using Model A through E assets?

# Rationale

This control was required before proposing another fire mechanism because the project needed one trusted evaluation floor, one fixed candidate interface, and evidence that the same committed state produces the same metrics and figures. Without it, a later score change could not be separated from evaluator, staging, or presentation drift.

# Change

This experiment introduces no scientific change. It stages the pinned TRENDY v14 EDv3 S3 native output and passes it to the locked ILAMB evaluator. The native ED source revision and producing command are not present, so this is an evaluation baseline rather than source-level reproduction.

# Prediction

The run should reproduce the same GFED5 score vector already associated with the pinned file, add a matched GFED4.1s score vector, and generate the five contract figures. It must not be interpreted as a candidate-model improvement or as proof that native ED can be rebuilt here.

# Plan

Expected evidence is the staged candidate hash, both complete ILAMB score vectors, the five canonical figures, command logs, environment identity, and unchanged protected-file hashes. Likely failures are a missing source link, an incompatible ILAMB environment, candidate-interface drift, or evaluator output that violates the contract. One run costs roughly two minutes on this machine. Stop after one complete run and one matched repeat at the same revision; invalidate any attempt that changes project or protected files.

# Result

Three terminal attempts completed. The [initial attempt](runs/run.20260818T170639Z.ed6f7759/run.json) established the result. A bookkeeping repair then removed a duplicated evaluator lock. The [first matched repeat](runs/run.20260818T171022Z.bc08c8d2/run.json) and [selected repeat](runs/run.20260818T171220Z.a7122045/run.json) reproduced the result from the same code state. Both commands exited successfully, protected hashes passed before and after evaluation, the worktree remained unchanged, and no failure was recorded.

The repeats produced byte-identical metrics, staged candidates, ILAMB scalar databases, score files, and all five canonical figures. Their run records retain the exact checksums. Each figure is 1800 × 1200 and was visually checked for complete panels, readable labels, consistent scales, and obvious rendering failures. The evaluation records differ only where they record each attempt's unique absolute working paths.

Because the candidate is the pinned stock file by construction, the candidate and ED-stock evidence vectors are identical. This is the intended control rather than a model improvement.

| Metric | GFED5 | GFED4.1s |
| --- | ---: | ---: |
| Benchmark-period mean burned area (%) | 0.487 | 0.327 |
| Model-period mean burned area (%) | 0.107 | 0.119 |
| Bias (%) | -0.381 | -0.209 |
| Bias score | 0.622 | 0.681 |
| RMSE (%) | 0.870 | 0.643 |
| RMSE score | 0.466 | 0.489 |
| Seasonal-cycle score | 0.467 | 0.439 |
| Spatial-distribution score | 0.166 | 0.290 |
| Overall score | 0.437 | 0.478 |
| Phase shift (months) | 2.784 | 3.225 |

The overall score changes by -0.040 when the benchmark changes from GFED4.1s to GFED5. The stock field underestimates the global mean under both products, and spatial distribution is the weakest scalar component, especially against GFED5. Those are research targets for a future mechanism; they are not reasons to alter the locked evaluator.

# Evidence

The [selected repeat](runs/run.20260818T171220Z.a7122045/run.json) supplies the evidence below. Its run directory retains the complete provenance record.

## Figures

These six figures were regenerated from the selected run's unchanged metrics and model artifact with the current v2 SciencePlots renderer. The selected run still retains the five byte-matched v1 figures produced during execution. Run `uv run --extra historical python scripts/render_experiment_figures.py experiment.stock-baseline --label "Staged ED-stock"` to refresh this presentation.

![GFED5 and GFED4.1s score summary](figures/01-score-summary.png)

The staged candidate and ED-stock bars are identical by construction. The lower panel gives the benchmark and model period means on each ILAMB comparison intersection.

![Mean burned-area fields](figures/02a-mean-burned-area.png)

The two benchmark fields sit above ED-stock and the staged copy used as the candidate. The bottom maps are identical, and all four panels use the same fixed scale.

![Stock ED burned-area differences from GFED4.1s and GFED5](figures/02b-burned-area-differences.png)

These maps show stock ED minus each benchmark. Blue marks less burning than the benchmark and red marks more burning, with one fixed diverging scale.

![Global and regional seasonal cycles](figures/03-seasonal-cycles.png)

The ED-stock and staged-candidate curves overlap exactly. Fixed regional scales keep the same comparison shape available for later candidates.

![Cell-level spatial distributions](figures/04-spatial-distribution.png)

The upper and lower rows repeat because the candidate is ED-stock. Each panel compares annual burned area on benchmark fire cells and reports correlation, median bias, and the fraction shown inside the fixed window.

![Benchmark and regional sensitivity](figures/05-benchmark-sensitivity.png)

The top panels match because both refer to the same stock field. The lower-right panel is zero everywhere because the staged candidate introduces no change from native ED.

## Results and outputs

[Complete metric vector](runs/run.20260818T171220Z.a7122045/metrics.json)

[Evaluation record](runs/run.20260818T171220Z.a7122045/artifacts/evaluation.json)

[GFED5 ILAMB scores](runs/run.20260818T171220Z.a7122045/artifacts/ilamb/gfed5/scores.csv)

[GFED4.1s ILAMB scores](runs/run.20260818T171220Z.a7122045/artifacts/ilamb/gfed4_1s/scores.csv)

[Evaluated stock burned-area field](runs/run.20260818T171220Z.a7122045/artifacts/model-output/burntArea.nc)

# Interpretation

The matched repeats show that the pinned stock field can be staged and evaluated deterministically under the preserved v1 contract. They establish a reliable comparison floor and expose the stock field's largest measured weaknesses, but they do not establish source-level reproducibility because the producing ED revision, build, input selection, and command remain unknown.

# Decision

Close the baseline as completed. The evaluation floor and canonical presentation are reproducible, but the experiment does not supply an editable ED fire implementation. The next model experiment must begin from the relevant version-controlled ED fire source and configuration or from a clean, reviewable mechanistic implementation that emits the same candidate interface. It should isolate one process and predict its regional and seasonal effect before execution. Do not introduce Optuna until that direct candidate and its development objective reproduce.

# Revisit when

Reopen the setup if the pinned stock output, benchmark product, ILAMB version, period, region boxes, score definition, or figure contract changes.
