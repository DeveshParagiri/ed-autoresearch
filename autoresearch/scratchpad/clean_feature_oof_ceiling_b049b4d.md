# Clean exogenous held-block information ceiling

This is a diagnostic-only result from `clean_feature_oof_ceiling_b049b4d.py`.
No learned surface entered `model.py` or the official ledger. Coordinates only
defined 15-degree validation blocks. The 2,234 fitted cells carry 85.002315% of
GFED5 reference-weighted fire. Features contained only current or prefix-causal
state derived from monthly precipitation, dryness, air temperature, LUH2
fractions, and lightning. There was no coordinate, region, calendar, ED state,
modern fire weather, population, future state, or completed-record climatology.

The prepared `annual_precipitation` was excluded after it was shown to be the
completed calendar-year rainfall total repeated into earlier months. Replacing
it everywhere by `12 * EMA12(monthly_precipitation)`, initialized from month
zero, changes the current model from exact 0.71688851 to 0.71740548. The causal
replacement scores bias 0.75480504, RMSE 0.54621417, seasonal 0.86059090, and
spatial 0.87920312. It therefore improves Overall by 0.00051697, despite a
seasonal-score loss, and requires no trailing-sum fallback test.

The clean ecology-suppressed rebuild using the same causal annualized rainfall
scores 0.66937908, with bias 0.71318130, RMSE 0.49630475, seasonal 0.82369306,
and spatial 0.81741154. The previous 0.666015 clean result used the invalid
completed-year rainfall field; the causal replacement is better.

Depth-three held-block correction of the clean causal rebuild reaches
0.68928477 with annual map only, 0.68692647 with normalized cycle only, and
0.71063587 when the two independently learned corrections are combined. The
combined metrics are bias 0.75146427, RMSE 0.54167343, seasonal 0.86037306,
and spatial 0.85799514. This shows that the permitted observables contain
substantial information not represented by the compact clean equation, but
even the diagnostic learner remains below the repaired current model.

Depth-three held-block correction of the repaired current model reaches
0.71844271 with annual map only, 0.72135609 with normalized cycle only, and
0.72269353 when combined. The combined metrics are bias 0.75672444, RMSE
0.55403378, seasonal 0.86568341, and spatial 0.88299224. Thus the maximum
observed gain from the clean features on this protocol is 0.00528805, far too
small to support an honest route from the current state to 0.8.

The current-model annual-map residual repeatedly selects 24-month temperature
crossed with managed-open or rangeland cover, lightning memory crossed with
24-month temperature, and cropland crossed with rangeland. Partial dependence
is strongest in cold long-term climates: the tenth-percentile 24-month
temperature state receives about +1.82 log residual relative to the median.
High primary and pasture fractions receive negative corrections, while the
highest rangeland quantile receives a modest positive correction. These
one-dimensional effects are conditional-support-sensitive and should not be
copied as coefficients.

The normalized-cycle residual repeatedly selects three-month temperature
departure, twelve-month warming, three-month lightning state and departure,
and short dryness departure. Negative short temperature departure reduces the
predicted allocation residual by about 0.0068 relative to neutral; positive
lightning departure raises it by about 0.0086 at the upper sampled quantile.
This supports a physical heating-onset by ignition-opportunity interaction, but
the full clean-feature learner earns only about 0.004 seasonal/cycle Overall
gain, so any smooth translation must be treated as a small repair rather than
the missing 0.8 mechanism.

## Causal annual-rain promotion audit

Commit `75fe945` implements the causal annual-rain replacement directly inside
the mechanistic model. Against `b049b4d`, exact regional Overall deltas are
Australia -0.003060, boreal Asia -0.000185, boreal North America -0.000047,
Central America -0.003655, Central Asia +0.001777, equatorial Asia +0.000019,
Europe -0.001859, Middle East -0.000791, northern Africa +0.001283, northern
South America +0.008859, southeast Asia +0.003193, southern Africa -0.001032,
southern South America +0.003689, and temperate North America +0.000663. Seven
of fourteen regions improve; the largest loss is 0.00366.

Established ecological model/observation ratios change from 0.841654 to
0.901671 in intact tropical closed canopy, 0.838579 to 0.898286 in tropical
closed canopy, 1.028738 to 1.053414 in temperate closed canopy, 1.042410 to
1.041174 in boreal forest, 1.061460 to 1.074389 in tropical open woodland,
1.006302 to 1.047023 in productive rangeland, 0.920423 to 0.924404 in cropland,
and 1.321544 to 1.299257 in arid low fuel. The repair removes future leakage
without introducing a pathological ecological regime.

After every primitive future input from month 96 is halved and causal annual
rain is recomputed from the perturbed monthly series, both the derived annual
state and the full prediction match the unperturbed first 96 months with exact
maximum absolute difference zero. The original opaque annual input also passes
a superficial input-perturbation test because its leaked completed-year value
is already baked into the earlier inputs; that control cannot validate its
upstream construction.
