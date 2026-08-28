# Structural headroom at the weak dead-fuel incumbent

This audit is pinned to the `autoresearch/model.py` blob `b82c285259f35f0f942ddc8a78663d8d14dd36b1`, shared by `121c83c` and the execution HEAD `abb2bbff5b4b90879c5646a3479f020a4dfa0c7b`. The exact proxy incumbent is 0.719892388 and the official rounded score is 0.720. GFED5 is used only after prediction for diagnostic counterfactuals and held-block losses. No observation, observation-derived rank, coordinate, region, or future value enters a candidate equation.

## Metric headroom

The repeated observed GFED5 climatology scores 0.873649519. This is an empirical evaluator oracle, not a mathematical maximum. Its 0.153757131 gap above the incumbent decomposes exactly as follows under the evaluator's 1:2:1:1 metric weights.

| Metric | Incumbent | Empirical oracle | Overall contribution | Share of oracle gap |
| --- | ---: | ---: | ---: | ---: |
| Bias | 0.757875412 | 0.893361132 | 0.027097144 | 17.62% |
| RMSE | 0.548437882 | 0.766118927 | 0.087072418 | 56.63% |
| Seasonal cycle | 0.860504847 | 0.958937180 | 0.019686466 | 12.80% |
| Spatial distribution | 0.884205917 | 0.983711428 | 0.019901102 | 12.94% |

The nominal gap to a score of one is 0.280107612, but 0.126350481 of that remains even for the repeated-observation climatology because of evaluator construction and grid aggregation. The empirical 0.87365 value should therefore not be called a theoretical ceiling.

## Which structural object is wrong

The counterfactuals below change only the named object and retain the other incumbent factor. All observation-derived interventions are diagnostics, not candidate constructions.

| Intervention | Overall | Delta |
| --- | ---: | ---: |
| Global magnitude scaling with incumbent local map and cycle | 0.715036319 | -0.004856069 |
| Observed local rank with incumbent magnitude distribution and cycle | 0.777732409 | +0.057840021 |
| Observed marginal magnitude distribution with incumbent local rank and cycle | 0.718951236 | -0.000941152 |
| Full observed annual map with incumbent cycle | 0.784867514 | +0.064975126 |
| Observed peak month with incumbent map, amplitude, and waveform | 0.741560018 | +0.021667630 |
| Observed amplitude with incumbent map, phase, and waveform | 0.721317010 | +0.001424622 |
| Observed phase and amplitude with incumbent map and residual waveform | 0.743687929 | +0.023795541 |
| Full observed normalized cycle with incumbent annual map | 0.778080587 | +0.058188199 |

The annual-map error is overwhelmingly a local-ordering problem. Correcting rank while preserving the incumbent magnitude distribution recovers 89.02% of the full annual-map oracle gain, whereas a global rescale and a marginal-distribution correction both lose score. The seasonal error is not mainly amplitude. Perfect amplitude buys only 0.001425; phase buys 0.021668; and waveform shape beyond corrected phase plus amplitude still buys 0.034393, or 59.11% of the full cycle gain. Full annual-map and full-cycle corrections are similar in isolation, and correcting both together has another 0.030594 of superadditive gain because the RMSE and bias penalties couple them.

The honest current-input ceiling is much smaller than either oracle. The immediately preceding four-fold, 15-degree, whole-cell depth-two diagnostic used only valid point-local current or prefix-causal inputs and no incumbent-fire feature. It moved 0.719756369 to 0.722807209, a gain of 0.003050840. The restored weak dead-fuel state lifts the mechanistic base to 0.719892388 but does not add a new exogenous observable, so the defensible remaining current-input range is still roughly 0.003 to 0.005, not 0.154 and nowhere near a path to 0.8.

## One new mechanism family and fixed-bracket result

The tested new family is compound fire-weather opportunity duration. A bounded monthly opportunity multiplies lightning chance by dry, rain-free, warm combustion. Causal trailing sums of the opportunity and its square define an effective number of ignition-ready months, `N_eff = sum(q)^2 / sum(q^2)`. The candidate increases natural surface and woody event footprint where annual opportunity has high coverage but is concentrated into few effective months. This is distinct from the incumbent's marginal lightning variability, combustion-temperature alignment, one-month arrival order, fuel banks, and realized-fire recurrence because it represents the temporal topology of the joint exogenous opportunity distribution rather than one marginal moment or stored hazard.

On 4,463 whole cells covering 92.51% of observed fire and 90.42% of incumbent excess fire, fixed positive strengths 0.05, 0.10, 0.20, and 0.40 improve annual log error in all four spatial blocks. The smallest annual gains are positive in every bracket and the aggregate cycle trade remains within the preregistered five-percent allowance. A future-half input perturbation changes the first 96 months of the state by exactly zero.

The exact full-grid result falsifies the apparent held-block promise. Overall deltas are -0.000070006, -0.000223551, -0.000683994, and -0.002326167 as strength rises. The smallest form slightly improves RMSE and spatial skill but loses more bias than it earns; the loss grows monotonically. No bracket should enter the canonical model, official ledger, tuning loop, or progress artifacts.

## Decision

The plateau is not a scalar-calibration or amplitude problem. It is the combination of wrong cellwise annual ordering and missing within-cell waveform geometry, with RMSE carrying 56.63% of the empirical-oracle gap. Current valid monthly inputs expose only about three to five thousandths of held-out recoverable Overall, and the one genuinely distinct compound-duration family fails the exact grid despite a clean held-block signal. Another broad architecture search over the same observables is unlikely to produce a structural step. The next credible route is new clean state, especially pre-fire ground litter or small-cohort fuel, patch or fuel-weighted moisture, separable snow or SWE, and management timing; these must come from a provenance-clean coupled export rather than GFED-derived runtime fields.

The reproducibility scripts are `structural_headroom_121c83c.py` and `compound_opportunity_duration_121c83c.py`. They edit no canonical, official, result, progress, input-README, or country-audit artifact.
