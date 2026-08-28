# Annual lightning by thermal coherence at the weak dead-fuel incumbent

This scratch test is pinned to `121c83c`, whose `autoresearch/model.py` blob `b82c285259f35f0f942ddc8a78663d8d14dd36b1` remains identical to the execution checkout. The exact incumbent proxy score is 0.719892388. The held audit uses the same 4,463 whole cells and four 15-degree spatial blocks as the deeper reverse-ML diagnosis, covering 92.5147 percent of observed fire and 90.4223 percent of incumbent excess fire. Coordinates assign folds only and never enter the candidate.

## Physical translation

The annual learner at `4fc1b6e` found negative `lightning_ema12` by `temperature_variability12` interaction contrasts in all four folds, with underburn at high lightning and low temperature variability and overburn at low lightning and high variability. The physical translation defines `I = L12 / (L12 + 0.02)` and `V = sigma_T12 / (sigma_T12 + 4)`. Natural/open fine-fuel support `S` is trailing GPP saturation times natural and secondary open cover times the incumbent continuity response. Final hazard is multiplied by

`M(k) = [1 + k S I (1 - V)] exp[-k S V (1 - I)]`.

Lightning therefore expands event footprint only where the trailing thermal season is coherent, while continental variability brakes footprint where natural ignition is insufficient. The two responses meet smoothly and share one global strength. The equation has no learned threshold or coefficient, coordinate, region, neighbour, future reduction, benchmark field, or target feedback.

## Structural distinction

The incumbent annual closure uses high temperature variability and trailing lightning together as a positive carrier for a rare cold-thaw source. This test instead expresses the observed negative interaction in the existing final hazard and has no additive ignition source. The incumbent coherent surface capacity makes high temperature variability a positive warm, rain-supported surface-capacity term but does not condition it on lightning. The local footprint uses lightning or managed access without thermal coherence. The older `c91cc73` annual lightning floor added fuel-supported ignition and globally renormalized annual area, while `4db79e2` added a wet-dry fuel pump gated by lightning. This family neither adds a floor nor conserves a completed-year map; it is a point-local, prefix-causal event-footprint response.

## Held result

At strength 0.05, annual-log gains by fold are `+0.000158200, -0.000298997, -0.000221655, -0.000130173`; normalized-allocation gains are `+0.000002233, -0.000000140, -0.000000195, -0.000001759`; and raw-cycle gains are `-0.000044829, +0.000006049, +0.000005066, +0.000017370`. Strengths 0.10, 0.20, and 0.40 retain the same annual sign reversal and make the aggregate annual loss progressively worse. No strength improves annual, allocation, and raw cycle loss in all four folds.

The weakest and aggregate-best bracket moves held ecological fire ratios only modestly: intact tropical closed `0.82113 -> 0.82560`, temperate closed `0.87517 -> 0.87672`, boreal `1.63161 -> 1.62635`, tropical open `1.02130 -> 1.02750`, productive rangeland `0.66721 -> 0.66803`, cropland `0.95726 -> 0.95761`, and arid low fuel `1.15597 -> 1.15560`. These are held-selection diagnostics, not full-grid ecology claims. Reversing and perturbing every input after month 96 changes the first 96 candidate months by exactly zero.

## Decision

The fold-stable learned interaction does not survive as this compact globally shared physical law. Its ecological displacement is bounded and its prefix behavior is exact, but its annual correction reverses in three of four held blocks and its cycle effects split across the complementary block. Exact full-grid scoring, canonical installation, official evaluation, tuning, and ledger or progress changes are not justified.
