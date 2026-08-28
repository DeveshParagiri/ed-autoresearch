# Waveform-direction reverse-ML diagnostic at `2dd6d61`

## Result

The current inputs contain a modest, spatially held signal for correcting monthly waveform direction in the incumbent-defined high-fire footprint. The coordinate-free depth-four HGB explains 0.260038 of the incumbent-weighted direction residual and 0.163772 of the evaluator-weighted residual. Evaluator-weighted OOF R2 is positive in every whole-cell fold at 0.040151, 0.276751, 0.201899, and 0.149205. This is evidence of real residual structure, not a canonical model or a rigorous ceiling.

The dominant error is a broadening problem. At the incumbent peak month, the evaluator-weighted observed-minus-model unit-direction correction averages -0.436510, while onset and recession average +0.034487 and +0.046449. The model is therefore too concentrated at its own peak and needs fire direction moved into the shoulders. Recession is the most predictable limb, with evaluator-weighted OOF R2 0.185208, versus 0.072623 for onset and 0.066855 for peak. Each limb nevertheless has a negative held-fold R2 in fold 0, so none of the learned surfaces is globally transferable as written.

## Audit geometry and leakage boundary

The incumbent is loaded from commit `2dd6d61`; its blob `0d05b1c75489fbdde6a1996aa993ed1e67657c71` matches the working `autoresearch/model.py`. The high-fire population is the 2,364 cells covering 85.0048% of incumbent area-weighted fire mass. It covers 72.4216% of reference fire mass, but GFED is not used to select cells. Whole cells are assigned to four 15-degree spatial blocks with 640, 570, 545, and 609 cells.

The label is the observed-minus-incumbent difference between unit monthly anomaly directions. Each 12-month cycle is centered with the evaluator's unweighted monthly mean and divided by its day-weighted RMS anomaly amplitude. GFED therefore enters only the supervised label and the separate evaluator-weighted diagnostics. Training weights use incumbent fire mass and calendar-month duration. Features are the 48 current-input, incumbent-state, and point-local prefix-causal fields from the earlier clean feature builder. They contain no coordinate, region, cell identifier, calendar harmonic, benchmark-derived value, or future summary. Coordinates assign folds only. The incumbent peak and trough use a completed incumbent climatology only to label onset, peak, and recession after OOF prediction; those labels are never learner features or proposed runtime state.

Five incumbent-selected cells have effectively zero observed anomaly amplitude. They remain in the target-free population as false-positive model cells, but receive zero or negligible evaluator weight; no GFED-dependent exclusion is made.

## Held evidence by limb

| Weighting | Limb | Baseline direction RMSE | OOF RMSE | OOF R2 | Fold R2 |
|---|---:|---:|---:|---:|---:|
| Incumbent | Onset | 0.663613 | 0.592821 | 0.193100 | 0.150447, 0.256640, 0.253060, 0.133323 |
| Incumbent | Peak | 1.187819 | 0.943329 | 0.058578 | -0.074793, 0.171280, 0.037905, 0.017070 |
| Incumbent | Recession | 0.715915 | 0.620031 | 0.206951 | 0.119655, 0.319176, 0.169255, 0.159753 |
| Evaluator | Onset | 0.619332 | 0.595494 | 0.072623 | -0.044837, 0.178494, 0.244012, 0.016291 |
| Evaluator | Peak | 0.945527 | 0.810215 | 0.066855 | -0.066389, 0.186743, 0.058570, 0.009084 |
| Evaluator | Recession | 0.583943 | 0.525431 | 0.185208 | -0.026494, 0.331970, 0.159401, 0.191356 |

A diagnostic-only climatological renormalization preserves the incumbent anomaly amplitude and applies the held OOF direction correction. At strengths 0.25, 0.50, and 1.00 it reduces evaluator-weighted unit-direction RMSE by 0.015235, 0.028202, and 0.038723. Every whole-cell fold improves at every strength. At strength 0.50 the fold gains are 0.012023, 0.050640, 0.035817, and 0.020500, with onset, peak, and recession gains of 0.023931, 0.020192, and 0.034115. This operation averages the OOF correction into a completed climatology and renormalizes it, so it is a shape-headroom diagnostic rather than a deployable causal mechanism.

## Fold-stable interactions

Two hundred eight adjacent tree-feature pairs occur in every fold, but only a small subset also has the same held partial-dependence interaction sign across every fold and limb. The strongest interpretable pair is current lightning by rangeland fraction, positive in all twelve fold-limb contrasts. Its interaction ranges from +0.02549 to +0.12674 on onset, +0.02440 to +0.14480 at peak, and +0.02687 to +0.13990 on recession. Trailing lightning by GPP curing is negative in all twelve contrasts, ranging from -0.05663 to -0.01078 on onset, -0.12134 to -0.01598 at peak, and -0.01989 to -0.00255 on recession. Lightning pulse by current temperature is positive in all twelve contrasts. Current temperature by three-month warming is negative in all twelve contrasts. Current lightning by twelve-month temperature is also positive in all twelve contrasts, but it is less mechanistically distinct from the active rare-ignition family.

The sign pattern is coherent: rangeland fire needs more lightning-linked temporal opportunity, but coincident persistent lightning and strong curing should saturate rather than compound; likewise a fresh lightning pulse benefits from absolute warmth, while absolute warmth and rapid warming behave as substitute thermal indicators rather than independent multiplicative vetoes. These statements describe stable response geometry only. HGB split thresholds and fitted response magnitudes are not copied.

## Compact mechanism families

The first family is a saturating rangeland ignition-curing allocator. Let `L12_t = EMA12(L)_t`, `I_t = L12_t / (L12_t + 0.02)`, `c_t = max((GPP3_t - GPP_t) / (GPP3_t + GPP_t + 0.2), 0)`, `C_t = c_t / (c_t + 0.05)`, and let `R_t` be current LUH2 rangeland fraction. Define

`S^R_t = R_t [I_t + C_t - I_t C_t]`

and probe replacement of the existing conjunctive surface timing score by

`h'_{surf,t} = h_{surf,t} exp{kappa [S^R_t - EMA12(S^R)_t]}`.

The probabilistic-union term has positive rangeland-by-lightning geometry and negative lightning-by-curing interaction by construction. It reallocates an existing surface hazard and uses incumbent physical scales rather than HGB thresholds. It is globally shared and prefix causal. Because the causal EMA subtraction is not exactly annual-mass neutral, any test must use a small predeclared bracket and audit area as well as waveform. This should be a replacement probe, not an additive source, because the active rare-ignition and surface-bank paths already contain these ingredients.

The second family replaces the triple-conjunctive thermal part of rare natural onset with a smooth shoulder-broadening union. Let `A_t = sigmoid((T_t - 5) / 3)`, `W_t = sigmoid((T_t - T3_t - 0.5) / 1.5)`, `p_t = max((L_t - L3_t) / (L_t + L3_t + 0.002), 0)`, and `P_t = sigmoid((p_t - 0.05) / 0.1)`. Define

`S^P_t = P_t [A_t + W_t - A_t W_t]`.

In a bounded probe, `S^P_t` should replace, not supplement, the active `lightning_arrival * thermal_window * heat_onset` timing factor while leaving its fuel, dryness, natural-share, and opportunity-gap terms unchanged. The equation preserves the positive lightning-pulse-by-temperature interaction and makes absolute warmth and rapid warming substitutes, matching their negative held interaction. It is globally shared, pointwise, and prefix causal, and its constants are existing physical response scales rather than learner fits.

No third equation is justified. The remaining stable lightning-temperature pairs either duplicate the active rare-ignition and arrival-order code or reproduce the thermal-coherence capacity family already falsified in Entry 219.

## Relation to Entries 213–235

This diagnostic does not reopen the failed thermal-curing multiplier from Entry 213: that family multiplied absolute warmth, warming, and uncured fuel, whereas the present held sign calls for saturation or a probabilistic union. It does not reopen Entry 217's opportunity-relative by temperature brake or Entry 219's trailing-lightning by temperature-variability capacity law; the present signal is within-season direction and is strongest on recession, not annual climate capacity. It does not reconstruct litter quantity or depletion, so it is distinct from Entries 221 and 228–234. It also does not infer precipitation intermittency as in Entry 224 or reuse the dryness-temperature EMA window rejected in Entry 235.

There is still substantial overlap with the active rare-ignition and arrival-order ingredients. The novel claim is therefore narrow: their current conjunctive interaction geometry may be too peak-concentrating, and a saturating replacement could broaden the shoulders. Fold 0's negative limb-specific R2 is a hard caution. The two equations merit only fixed low-strength scratch replacement probes with the existing whole-cell annual, allocation, raw-cycle, prefix, and ecology gates; the evidence does not justify canonical coefficients, Optuna, or a learned surface.

## Reproduction

Run `uv run python autoresearch/scratchpad/waveform_direction_reverse_ml_2dd6d61_20260828.py` from the repository root. The script and this report are uniquely named scratch artifacts. No canonical model, results, progress, research, input documentation, thread log, git index, or commit was changed by this task.
