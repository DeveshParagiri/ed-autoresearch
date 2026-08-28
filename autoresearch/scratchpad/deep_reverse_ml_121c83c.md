# Deep reverse-ML audit at the weak dead-fuel incumbent

This diagnostic is pinned to `121c83c`, whose `autoresearch/model.py` blob `b82c285259f35f0f942ddc8a78663d8d14dd36b1` is identical to the execution checkout. The exact incumbent proxy score is 0.719892388. The audit uses 4,463 cells covering 92.5147 percent of observed fire and 90.4223 percent of incumbent excess fire. Whole cells are assigned to four 15-degree spatial-block folds containing 1,242, 1,110, 986, and 1,125 cells.

The 48 predictors are current coupled-valid inputs, the incumbent fire opportunity, and point-local exponential summaries initialized from and updated only with current or previous timesteps. Coordinates exist only in the fold vector. No coordinate, region, cell identity, calendar label, future summary, benchmark-derived feature, or target feedback enters a predictor or proposed mechanism.

## Separate held-out ceilings

Two depth-four histogram-gradient-boosted ensembles with 180 stages are fit separately. The annual head predicts annual log residual while seeing only row-local current or past-causal predictors; the same cell target is repeated over months, but no cell appears in both training and held data. The cycle head predicts the normalized monthly-allocation log residual. Learned corrections are applied only to their unseen cell blocks and are diagnostic surfaces, never scientific candidates.

The annual OOF R-squared values are 0.63328, 0.61327, 0.65837, and 0.58560 by fold, or 0.62649 jointly. Every annual held loss improves at all tested blends. The best Overall diagnostic is the fixed 0.25 blend at 0.725454849, a gain of 0.005562461 over the mechanistic incumbent. Its score components are bias 0.767629254, RMSE 0.553834239, seasonal 0.858852022, and spatial 0.893124493. A stronger 0.50 blend lowers Overall slightly to 0.725224309 because cycle and spatial costs begin to dominate.

The normalized-cycle OOF R-squared values are 0.28283, 0.32008, 0.35652, and 0.28446, or 0.30584 jointly. Every held block improves both its annual and normalized-cycle loss at every tested blend. The best Overall diagnostic is the fixed 0.50 blend at 0.721929910, a gain of 0.002037522. Its score components are bias 0.756031336, RMSE 0.553316858, seasonal 0.866446949, and spatial 0.880537547. The full blend raises RMSE and seasonal further but loses Overall through bias and spatial displacement.

The deeper learner therefore exposes more annual structure than the earlier depth-two pass, but the score-level ceiling is still modest: about 0.7255 for annual structure and 0.7219 for cycle structure when corrected separately. This is diagnostic headroom, not a promotable learned model.

## Fold-stable interaction evidence

The strongest annual interaction is trailing lightning by trailing temperature variability. Its parent-child counts are 58, 30, 40, and 37, and its held partial-dependence interaction contrasts are -0.18668, -0.39190, -0.15570, and -0.16587. Lightning has a positive main effect and temperature variability a negative main effect in every fold, while their negative interaction says lightning-supported annual capacity collapses in thermally erratic climates.

The strongest normalized-cycle interaction is current relative opportunity by absolute temperature. Its counts are 22, 52, 28, and 55, and its interaction contrasts are +0.04785, +0.30526, +0.06568, and +0.18205. Both main effects are negative in every fold, while the positive interaction is sub-additive: high current opportunity and sustained heat are each associated with overallocated cycle mass, but their overlap should not be penalized twice. Absolute temperature by three-month warming and lightning by three-month warming are also positive in all four folds, but those onset forms already exist in the incumbent rare-ignition family. The opportunity-temperature interaction is the cleaner missing shape.

## One mechanistic translation and result

The tested family is thermal-opportunity saturation. Let monthly incumbent hazard be `h_t = -log(1-p_t)`, its causal twelve-month exponential state be `m_t = EMA12(h)_t`, bounded relative opportunity be `A_t = h_t/(h_t+m_t)`, and heat stress be `H_t = sigmoid((T_t-20)/3)`. The fixed 20 C center is the incumbent ignition center and 3 C is its existing broad thermal scale; neither is copied from a learned split. Refractory pressure uses inclusion-exclusion,

`S_t = A_t + H_t - A_t H_t`,

so high current opportunity or sustained heat can each make fuel temporarily refractory while their overlap cannot consume the same fuel twice. Existing hazard is redistributed with

`F_t(k) = exp(-k S_t) / EMA12(exp(-k S))_t`,

and `p'_t = 1 - exp(-h_t F_t(k))` for fixed `k` in 0.05, 0.10, 0.20, and 0.40. This is globally shared, pointwise and prefix causal; perturbing all inputs after month 96 changes the first 96 output months by exactly zero. It is distinct from the managed-open temperature gate, which broadens the upstream base-rate temperature response only in managed fine fuel, and from surface-seasonality capacity, which uses rainfall variability and moderate annual fire opportunity. This family acts on the final current hazard and represents a refractory redistribution rather than a new temperature response or annual source.

The family fails the held gate monotonically. At `k=0.05`, annual losses worsen by 0.001842766, 0.002368983, 0.001976898, and 0.001709863. Normalized-cycle gain is +0.000096590 in fold zero but -0.000190939, -0.000227143, and -0.000072294 in the other folds. Larger strengths worsen the same three cycle folds and all four annual folds. No fixed bracket clears the every-fold cycle gate, so exact full-grid scoring is correctly skipped and no ecology audit or canonical recommendation is permitted.

## Decision

A genuinely deeper, geography-blind learner confirms that fold-stable structure survives, especially in annual fire ordering, but only about 0.0056 Overall is recoverable by the annual learned surface and about 0.0020 by the cycle surface in isolation. The single most stable unrepresented cycle interaction does not survive translation into one globally shared ecological law. This result supports looking for new coupled-valid state that resolves annual cell ordering rather than adding another empirical temperature or opportunity modifier over the same inputs.

The requested Claude consultation could not run because the installed Claude Code client reported that its OAuth session had expired and could not be refreshed. The analysis above is therefore the local reproducible diagnostic, not a claimed Claude response.
