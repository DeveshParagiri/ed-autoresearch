# Strong annual cell-ranking audit at `121c83c`

This scratch-only audit pins the canonical model source to `121c83c`, blob
`b82c285259f35f0f942ddc8a78663d8d14dd36b1`, and reproduces its exact proxy
score of `0.719892388`. The clean learner inputs are the current local monthly
precipitation, air temperature, dryness, LUH2 cropland, pasture, rangeland,
primary, secondary and urban signals, and lightning climatology. The learner
does not receive the incumbent prediction, completed-year precipitation,
frozen GFED-descended ED state, unresolved GPP, modern-only weather fields,
population, coordinates, regions, cell identity, calendar labels, future
summaries, targets, or learned runtime state. The observations define only the
diagnostic residual target and objective weights. Coordinates define four
disjoint whole-cell folds and never enter a predictor.

The fixed input-derived land mask contains 18,316 cells. The folds contain
4,954, 4,573, 4,349 and 4,440 whole cells. The feature matrix has 47 current or
point-local causal predictors and 3,516,672 land-month rows. Every fourth month,
879,168 rows in total, is used for fitting to keep the three forest comparisons
tractable, but each learner emits a prediction for every held land-month. No
target-selected correction carrier exists: every land cell is corrected only
by the learner that did not see any cell from its fold.

The exact local files used were `climate.nc` SHA-256
`792997db1c25909e2dc6535483f7009da65d5cfb4534d9f06f86802203e38802`,
`luh2.nc` `367d8d5061fc0dbc4d75a0496eadcb892cad3157407c21c639b3d7ec310abbbd`,
`lightning.nc` `b3cb4dae055ad0b331f5052fd5f59c9aa4169b093e212387110b79d00a881af4`,
the canonical base-only `ed.nc`
`fcc246bc20975d97ceca8998e6efbd27a3c792d73bc85a8d2a2eecb88683f558`,
and `gfed5.nc` `46594753a3f111e0ddc5526370708b0648f7f7b67458b42153dc04b7d035051b`.

## Honest OOF ceiling

The tested diagnostics are a 63-leaf, 280-stage deeper histogram-gradient
boosting ensemble, 96 ExtraTrees of depth 18, and 72 random-forest trees of
depth 16. Each annual log-residual target is repeated across its cell's monthly
prefix states, but whole-cell folding prevents any target from appearing in
both the training and held sides. Corrections are fixed-strength multiplicative
hazard corrections built entirely from unseen predictions.

| Learner | Fold OOF residual R2 | Joint OOF R2 | Best strength | Best Overall | Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| Deeper HGB | 0.5142, 0.4122, 0.5789, 0.3379 | 0.462536 | 0.25 | **0.723734682** | **+0.003842294** |
| ExtraTrees | 0.4783, 0.3871, 0.5522, 0.4012 | 0.455227 | 0.25 | 0.723106053 | +0.003213665 |
| Random forest | 0.4445, 0.3666, 0.5362, 0.3143 | 0.416109 | 0.25 | 0.723207650 | +0.003315262 |

The best honest strict-clean ceiling among these substantially stronger
transparent diagnostics is therefore `0.723734682`, or `+0.003842294` above
the incumbent. This is a tested model-class ceiling, not a mathematical bound.
Its components are bias `0.764513694`, RMSE `0.551687081`, seasonal cycle
`0.860275786`, and spatial distribution `0.890509767`.

At the winning deeper-HGB strength, held annual-log loss improves in all four
folds by `+0.118934822`, `+0.086539048`, `+0.087348106` and `+0.054649732`.
Normalized-allocation gains are `-0.000141536`, `-0.000177647`,
`-0.000258259` and `+0.000322047`; raw-cycle gains are `-0.000083343`,
`+0.000323835`, `-0.000308053` and `-0.000457358`. The learner therefore
confirms annual ordering headroom but does not pass a scientific all-metric
held gate. The earlier depth-four result of `0.725454849` is not the honest
strict-clean comparison because it used an incumbent-derived opportunity
feature and applied correction only on a target-selected 4,463-cell carrier.

## Stable clean feature and interaction patterns

Three main effects keep the same sign in all twelve fold-models across the
three learner families. Higher LUH2 primary fraction predicts positive annual
residual, trailing 12- and 24-month lightning predict positive residual, and
crop-plus-urban fragmentation pressure predicts negative residual. A fresh
lightning departure is also positive in all twelve models, although its random-
forest magnitude is small. These are residual associations, not promoted laws.

Two interactions have an exactly stable sign in every fold of every learner
family. LUH2 primary fraction by twelve-month temperature variability is
negative: its contrasts are `[-0.174, -0.314, -0.489, -0.362]` for deeper HGB,
`[-0.420, -0.228, -0.275, -0.236]` for ExtraTrees, and
`[-0.083, -0.125, -0.139, -0.308]` for random forest. Primary landscapes are
underburned only where the trailing thermal season is coherent; thermal
variability removes that positive primary-capacity residual. Fragmentation
pressure by rangeland is also negative in every fold and family, but that is
not novel because the incumbent already contains a fragmented managed-
recurrence brake.

The clean LUH2-primary by thermal-coherence interaction is the strongest novel
consensus pattern. It is more defensible than the prior lightning by thermal-
variability pattern because it survives all three stronger learner families
without using the incumbent opportunity or GFED-descended natural-vegetation
state. A secondary pasture by thermal-variability pattern is negative in all
eight deeper-HGB and random-forest fold-models, but it is not claimed as a
three-family consensus.

## Smooth-law translation and held result

The consensus interaction was translated into one global smooth capacity with
no learned threshold or coefficient. Let

`sigma_t = sqrt(EMA12(T_t^2) - EMA12(T_t)^2)`,

`C_t = 4 C / (4 C + sigma_t)`, and

`M_t(k) = 1 + k * primary_t * C_t`.

The twelve-month state is the physical annual horizon and 4 C is the fixed
thermal-variability saturation already used in the prior physical audit. The
candidate multiplies incumbent hazard by `M_t(k)` at fixed strengths 0.02,
0.05, 0.10, 0.20 and 0.40. It is globally shared, point-local, smooth,
target-blind and prefix causal. A parallel pasture translation uses the same
law only as a secondary falsification of the two-learner pattern.

The primary law fails monotonically in fold 2. At the weakest strength 0.02,
annual gains are `+0.000981500`, `+0.000841445`, `-0.000103437` and
`+0.000124526`; allocation gains are `+0.000001904`, `+0.000000267`,
`-0.000000401` and `-0.000001636`; raw-cycle gains are `+0.000006321`,
`+0.000031402`, `-0.000006451` and `-0.000003766`. Higher strengths deepen
the annual fold-2 loss and the allocation/raw-cycle sign reversals. The pasture
law improves raw cycle in all four folds at every tested strength but also
loses annual and allocation in fold 2 at every strength. Future-half mutation
changes the first 96 output months by exactly zero for both families.

No declared strength clears the annual, normalized-allocation and raw-cycle
gates in all four folds. Exact full-grid candidate proxy scoring and ecological
ratio auditing were therefore correctly skipped. The learned ceiling remains
diagnostic only, and neither clean-cover thermal-coherence law is justified for
canonical, official, optimization, result, progress, input-documentation,
country-audit or thread changes.

The complete numeric evidence is in `annual_rank_strong_oof.json`; the
reproducible learner and law scripts are `annual_rank_strong_oof.py` and
`pasture_thermal_coherence_law.py`; the held-law transcript is
`thermal_coherence_law_output.txt`.
