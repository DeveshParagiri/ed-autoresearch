# Corrected limiting-factor aggregation against 121c83c

This scratch experiment pins `autoresearch/model.py` at commit `121c83c`, confirms that the current canonical file has the same model blob `b82c285259f35f0f942ddc8a78663d8d14dd36b1`, and reproduces the incumbent exact proxy `0.719892388`. It screens 4,463 whole cells assigned to four disjoint 15-degree spatial folds; the sample contains 92.5147% of observed-area weight and 90.4223% of incumbent excess-area weight. Coordinates define folds only, and GFED enters only held losses.

## Corrected architecture

Let \(x_1,\ldots,x_4\in(0,1]\) be the canonical dryness, precipitation, GPP-fuel, and managed-open temperature favourability factors. The incumbent base aggregation is their product \(P=\prod_i x_i\). The corrected normalized soft minimum is

\[
S_\beta=-\frac{1}{\beta}\log\left(\frac{1}{4}\sum_{i=1}^{4}e^{-\beta x_i}\right).
\]

The factor of \(1/4\) is essential. It makes \(S_\beta(x,x,x,x)=x\), whereas the historical sum-offset implementation returns \(x-\log(4)/\beta\) and can collapse an equal low-favourability vector to its numerical floor. Stable mean log-sum-exp implementations at \(\beta=2,8,25\), and the hard-min limit all reproduce equal inputs with maximum absolute error exactly `0`.

The product-to-limit blend is geometric,

\[
A_{w,\beta}=P^{1-w}S_\beta^w,
\]

for fixed \(w\in\{0.10,0.25,0.50,1.00\}\). The distinct comparison is the homogeneous generalized mean of order minus one, the harmonic limit

\[
H=\left(\frac{1}{4}\sum_{i=1}^{4}x_i^{-1}\right)^{-1},
\]

used in the same geometric blends. Unlike the additive log-sum-exp soft minimum, the harmonic formulation is scale homogeneous and penalizes multiple jointly weak factors through reciprocal averaging. Its equal-input identity holds to `1.11e-16`.

All equations are globally shared and pointwise. They use only the incumbent coupled-valid inputs and causal base-factor paths. They contain no learned coefficient, target feedback, geographic term, region, neighbour, future statistic, or invalid input. Reversing and perturbing every input after month 96 changes the selected best candidate before month 96 by exactly `0`.

## Held result

No one of the twenty fixed brackets improves annual-log, normalized-allocation, and raw-cycle loss in every fold. This is not a flat response. Every softmin, hard-min, and harmonic bracket improves annual loss in all four folds. The failure is instead a fold-stable waveform conflict: raw-cycle loss worsens in fold 2 for every one of the twenty brackets.

Two normalized softmin forms clear all eight annual and allocation gates. At \(\beta=8,w=1\), allocation gains are `+0.002979473`, `+0.002775266`, `+0.000036630`, and `+0.002016687`, but fold-2 raw-cycle gain is `-0.000523527`. At \(\beta=25,w=1\), allocation gains are `+0.003113836`, `+0.002992045`, `+0.000112066`, and `+0.001542029`, while fold-2 raw-cycle gain remains `-0.000229641`. The sharpest low-impact hard-min bracket at \(w=0.10\) still produces fold-2 raw-cycle gain `-0.000004235`, while the weakest smooth \(\beta=2,w=0.10\) bracket produces `-0.000012408` there.

The harmonic family repeats the same sign conflict. At its weakest blend, annual gains are positive in all folds, but fold-2 raw-cycle gain is `-0.000014566`; stronger blends deepen that failure to `-0.000885192` at full replacement. Thus a meaningfully different generalized mean does not resolve the rejected waveform.

The preregistered held gate rejects the limiting-factor family on current lineage despite its strong annual-map headroom. Exact evaluation and ecology were correctly skipped, so no exact proxy delta is justified and no canonical change is recommended. The complete twenty-bracket screen is reproducible with `autoresearch/scratchpad/corrected_softmin_limiting_121c83c.py`.
