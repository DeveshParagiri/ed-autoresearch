# ED-Fire autoresearch

## Goal

Improve `model.py` as a mechanistic model of monthly burned area for fire. Your hard goal is to maximize official GFED5 ILAMB Overall with a soft requirement of ensuring performance gain is across regions and not concentrated in one core region. Directly edit only model.py, do not modify `inputs/`, `results.tsv`, GFED5, or the `ar` cli implementation.

Coupled-ready candidates must be pointwise across grid cells because ED sites run independently: do not use neighbour or other cross-cell operations. Use only predictors marked coupled-valid in `inputs/README.md`.

Official Overall is necessary but not sufficient evidence of a better model. Before promoting a candidate, inspect regional breadth, spatial maps, seasonal cycles, global burned area, and ecological plausibility across high- and low-fire land-cover regimes. Reject gains that depend on implausible regional compensation, such as severe false burning in closed-canopy forest, even when the global scalar improves; diagnose such failures by general biome or observable land state rather than adding a geographic mask or region-specific correction.

Distinct regional ecology may use distinct mechanisms, but they must be smooth mathematical functions of local observable state with globally shared coefficients. Do not dispatch on region labels or coordinate boxes, use region-specific coefficient tables, or implement geographic `if`/`else` branches.

## Model interface

`INPUTS` lists the exact variables loaded from `inputs/*.nc`. `PARAMS` holds the current coefficients. `SEARCH_SPACE` lists the coefficients the Optuna (hyper param optim tool) may tune. `COMPONENTS` can name at most 15 physical terms for "shapley-style" ablation.

`predict(data, params, components)` returns the fraction of each grid cell burned in each month from January 2001 through December 2016. Its shape is `(192, 180, 360)`, and every value must be finite and between zero and one. When `components` is `None`, use every declared component; otherwise use exactly the requested subset. The runtime checks this contract.

## Research tools

You are encouraged to use these tools in any way to triage or go off of for coming up with new research directions, subsequent experiments and anything which helps improve the model.

`uv run ar list` shows available tools and `uv run ar COMMAND --help` their arguments.

`uv run ar optuna` tunes `SEARCH_SPACE` against GFED5. It attempts at most 500 trials by default and stops after 50 completed trials without a new three-decimal improvement. It prints the winning coefficients but does not edit `model.py` or `results.tsv`.

`uv run ar evaluate --description TEXT` runs official ILAMB once, reports global and 14 regional metrics, appends one measurement and the current Git commit to `results.tsv`, updates the single external `progress.png`, and creates temporary comparison and seasonal-cycle figures. It returns measurements only and leaves `model.py` unchanged. It refuses to evaluate an uncommitted `model.py` so every recorded experiment is recoverable.

`uv run ar ablate` evaluates every subset of `COMPONENTS` with fixed `PARAMS`. It reports exact global and regional Shapley contributions plus leave-one-out effects.

`uv run ar figures` recreates the temporary GFED5/model/difference map and observed/model seasonal-cycle figure without recording another experiment.

## Experiment loop

Repeat indefinitely:

1. Read `results.tsv` and inspect the current model. The highest recorded three-decimal Overall is the "objective best" so far, but the current model may be exploring a different line. You are free to try any other model directions from first-principles thinking.
2. Use global metrics, regional, maps, seasonal cycles, ablations to identify physical weaknesses.
3. Form hypotheses based on thinking about fire from first principles in relation to the inputs available, along with your inputs on ablations, figures and past evals to modify `model.py`.
4. Run Optuna and copy its winning coefficients into `PARAMS` .
5. Commit `model.py` with a concise message, then run evaluation once with a concise, concrete description of the hypothesis.
6. Inspect all returned evidence, decide whether to continue, revise, combine, or abandon the line, and begin the next experiment. To restore any recorded one-file model, run `git restore --source COMMIT -- model.py` from this directory.

Every distinct `model.py` formulation that reaches a proxy score is an experiment and must be committed, including intermediate and non-improving results. Restore the objective-best formulation in a later explicit commit rather than erasing a rejected experiment from history.

## Scientific judgment

A lower Overall score can still justify another experiment when it improves a relevant region or metric, corrects a spatial or seasonal failure, or exposes a useful component through ablation, or might look like a promising research direction regardless. Do not accept or reject an input after one arbitrary transformation. Test distinct, physically plausible formulations as separate experiments exhaustively.

Prefer changes that explain observed fire behavior. Do not add complexity without diagnostic evidence. Failed commands are not experiments: repair the failure and rerun. Continue until interrupted.

## Approach

Soft guidance, not hard rules:

- Lean toward genuinely *families* rather than long hyperparameter sweeps — don't get stuck tuning one recipe.
- **A better method than the baseline exists**, so "no improvement found" / "baseline is optimal" is never a valid place to stop — being stuck means try a different family, not that you're done.
- Roughly every ~8 ideas explored, do a pruning round: try dropping each component you've stacked on and keep only what still earns its place, so complexity and dead weight don't accumulate.

## Scratchpad

Keep and organize your durable memory in `scratchpad/` — you may be compacted or restarted, so read it before continuing. Organize it however you like, with one mandatory file: `scratchpad/thread.md`, a running log where you record every decision and its outcome. Everything else is up to you.
