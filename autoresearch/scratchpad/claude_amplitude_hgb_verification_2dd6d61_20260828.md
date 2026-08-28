# Verification of Claude amplitude and HGB ceiling claims at `2dd6d61`

This audit is read-only with respect to the canonical model and evaluator ledger. The current `autoresearch/model.py` blob is `0d05b1c75489fbdde6a1996aa993ed1e67657c71`, exactly the blob at `2dd6d61`. The exact incumbent proxy is Overall `0.720105466`, bias `0.757826197`, RMSE `0.548783038`, seasonal `0.860630117`, and spatial `0.884504940`.

## Evaluator semantics

For each cell, `fast_ilamb.py` first forms the sixteen-year monthly climatology, then centers it with the unweighted arithmetic mean

\[
\bar C=\frac1{12}\sum_{m=1}^{12} C_m,\qquad A_m=C_m-\bar C.
\]

Only after centering does it apply month-day weights inside the squared anomaly error,

\[
E=\sqrt{\frac{\sum_m d_m(A_m-R_m)^2}{\sum_m d_m}},\qquad
S_{\rm RMSE}=\left\langle e^{-E/\sigma_R}\right\rangle_{a\bar R}.
\]

The spatial average is weighted by cell area times the reference mean. Seasonal skill uses the cellwise climatological argmax. Bias and spatial distribution use the time-weighted mean map. Consequently, a clean anomaly intervention can hold bias and spatial skill exactly fixed while changing RMSE and seasonal skill.

The older `structural_headroom_121c83c.py` amplitude diagnostic is not this intervention. It first divides each cycle by a month-length-weighted mean, then matches the unweighted RMS of that dimensionless normalized shape. It therefore matches a relative coefficient of variation, not the absolute centered anomaly that the evaluator scores. `shape_probe.py` also describes a month-length-weighted centering operation as “centered rmse as scored,” but the evaluator centers with an unweighted mean. Its decomposition is therefore not exact evaluator semantics.

## Correct anomaly decomposition

Let `A_M` and `A_R` be the correctly centered model and reference anomaly vectors, and let `sigma_M` and `sigma_R` be their day-weighted RMS amplitudes. The pure amplitude-only diagnostic is

\[
A_{\rm amp}=\frac{\sigma_R}{\sigma_M}A_M,
\]

which retains the model anomaly direction and phase. The pure shape-only diagnostic is

\[
A_{\rm shape}=\frac{\sigma_M}{\sigma_R}A_R,
\]

which retains model amplitude while replacing the anomaly direction, including phase and decay-limb geometry. In both cases the incumbent evaluator mean map is retained.

The unconstrained mathematical amplitude-only result is Overall `0.738695951` and RMSE `0.595259308`, with bias, seasonal, and spatial unchanged apart from floating-point phase ties. It is not a valid burned-fraction candidate because the requested amplitude makes some monthly values negative. The cellwise `[0,1]`-feasible projection, which caps only the anomaly scale and still holds the incumbent mean map, scores Overall `0.734091888` and RMSE `0.583749151`. The desired amplitude scale is capped over `72.3289%` of reference fire weight, so even this result does not mean that all observed amplitudes are physically reachable at the incumbent mean.

The pure unconstrained shape-only result is Overall `0.771301543`, RMSE `0.628573891`, and seasonal `0.957028796`, with incumbent bias and spatial scores. Its valid bound-projected version scores Overall `0.772079372` and RMSE `0.630518464`, but `18.0676%` of reference weight is amplitude-capped; the slight score increase is therefore not a pure shape-only comparison. The unconstrained number is the cleaner orthogonal decomposition, while the projected number is the valid evaluator counterfactual.

These results reverse the old conclusion that perfect amplitude buys only about `0.0014` Overall. Under the evaluator’s actual absolute-anomaly definition, amplitude-only headroom is about `0.0186` for the mathematical oracle and `0.0140` under the valid mean-preserving projection. Shape direction remains the larger problem, but amplitude is material.

## Log amplitude ratios

Two commonly reported global ratios are not interchangeable. The ratio of reference-weighted mean amplitudes is `0.681177946`, while the reference-weighted geometric mean of the cellwise ratio is `0.429851281`. The latter is `exp(E_w[log(sigma_M/sigma_R)])` and is the appropriate summary of stratified log ratios. Reporting only the former substantially hides the typical cellwise deficit.

| Equal-reference-fire quintile by observed annual fraction | Weighted mean log ratio | Geometric model/reference amplitude | Weight with model amplitude below reference |
| --- | ---: | ---: | ---: |
| 1, lowest annual fire | -0.970063263 | 0.379059057 | 58.59% |
| 2 | -1.351314775 | 0.258899642 | 70.38% |
| 3 | -0.892903889 | 0.409464984 | 80.57% |
| 4 | -0.442393951 | 0.642496474 | 81.59% |
| 5, highest annual fire | -0.564867320 | 0.568435562 | 90.87% |

The amplitude deficit is therefore broad rather than confined to negligible-fire cells. It is strongest in geometric terms in the second quintile, and even the highest-fire quintile is under-amplitude in more than ninety percent of its reference weight.

## Stacked whole-cell HGB claim

The unchanged `clean_feature_oof_ceiling_b049b4d.py` was replayed against the current canonical model. Its exact incumbent reproduction passed. Its best current-model depth-three score was `0.724813329` at the separate annual-map `0.25` and normalized-cycle `1.0` stack. It does not reproduce `0.733371`.

The stronger depth-four protocol from `deep_reverse_ml_121c83c.py` was then reproduced at `2dd6d61` with 4,452 cells, 854,784 site-month rows, 48 point-local current or prefix-causal features, and four disjoint 15-degree whole-cell folds. Annual OOF R-squared was `0.628855434`; normalized-cycle OOF R-squared was `0.304728898`. The best fixed factorized stack in the declared grid was annual `0.5`, cycle `1.0`, scoring Overall `0.733194948`, bias `0.775491531`, RMSE `0.566252178`, seasonal `0.867656722`, and spatial `0.890322133`. This is `0.000176052` below the claimed `0.733371`.

A different direct-hazard combination of the same two OOF heads scored `0.736972054` at annual `0.5`, cycle `1.0`. That is not a reproduction of the claimed factorized ceiling; it is evidence that the reported “ceiling” depends materially on the stacking equation and held-data blend selection. No existing result or fixed configuration inspected here produced exactly `0.733371`.

## Leakage and weighting audit

The individual HGB row predictions are genuinely out of fold by whole cell, and coordinates, regions, and cell identifiers are absent from the feature matrices. That part of the claim passes.

The score-level ceiling is nevertheless optimistic. `clean_feature_oof_ceiling_b049b4d.py` selects the top 85 percent of reference-weighted GFED fire before splitting and applies learned corrections only inside that observation-derived mask. The deeper protocol selects the union of cells covering 90 percent of observed fire and 90 percent of incumbent excess relative to GFED, then likewise corrects only that target-derived mask. Membership in the corrected population therefore contains benchmark information even though it is not an HGB feature.

Both scripts compare multiple blend strengths and stacking laws on the same OOF predictions and report the maximum evaluator score. OOF protects row predictions from in-fold fitting, but it does not make hyperparameter selection on those held targets an independent test. The selected maximum is optimistic unless a second outer spatial split fixes the stack and blend.

The normalized-cycle target is a twelve-month allocation normalized by an unweighted monthly sum. The official RMSE instead scores absolute centered anomalies, normalizes each cell by reference temporal standard deviation, applies day weights after centering, and spatially weights the resulting exponential score. The HGB training loss and weights are therefore useful diagnostics but not an exact surrogate for Overall.

`seasonal_shape_ceiling.py` is not an OOF ceiling at all: it fits and predicts the same land cells, uses calendar harmonics, and uses full-record per-cell means, standard deviations, and quantiles. `online_residual_glm.py` is also not a clean current reproduction path because it attempts to read prepared `annual_precipitation` as a feature even though the current model no longer declares that input; historically that prepared field was a completed-year future leak.

## Conclusion

Claude’s amplitude correction is substantively right only when “amplitude” means the absolute centered anomaly used by the evaluator. Under that definition, amplitude is materially wrong and the earlier `+0.0014` ceiling is a normalization artifact. The exact `0.733371` stacked-HGB claim is not reproducible from the inspected current scripts. A nearby depth-four factorized score of `0.733194948` and a different direct-hazard score of `0.736972054` are reproducible, but both inherit target-derived population selection and OOF blend selection. They should be described as optimistic conditional diagnostics, not leakage-free current-input ceilings.

The reproducer is `claude_amplitude_hgb_verification_2dd6d61_20260828.py`. It does not edit canonical model, results, progress, research, input, country-audit, or thread artifacts.
