# Compound-Poisson dry-spell falsification against 121c83c

This scratch experiment pins `autoresearch/model.py` at commit `121c83c`, verifies its model blob `b82c285259f35f0f942ddc8a78663d8d14dd36b1` against the current canonical file, and reproduces the incumbent exact proxy `0.719892388`. The held screen uses 4,463 whole cells in four disjoint 15-degree spatial folds. Those cells contain 92.5147% of observed-area weight and 90.4223% of incumbent excess-area weight. Coordinates assign folds only, while GFED observations enter only the held losses.

## First-principles weather state

For monthly precipitation \(P_t\), fixed mean storm depth \(d\), and \(D_t\) calendar days, a compound-Poisson arrival model gives expected storm count \(\lambda_t=P_t/d\) and daily zero-rain probability

\[
p_{0,t}=\exp\left(-\frac{P_t}{dD_t}\right).
\]

An exact finite-state Bernoulli recursion tracks the probability of ending each day with zero through six consecutive dry days without yet reaching seven. Probability leaving the six-day state on another dry day is absorbed, giving \(W_t=\Pr(\text{at least one seven-day event-free run})\). Unlike a smooth monthly rain brake, \(W_t\) represents the discrete within-month opportunity for an uninterrupted dry window. Fixed storm-depth brackets of 5, 10, and 20 mm span shallow, moderate, and deep precipitation events without fitting the benchmark. Their selected-cell mean expected storm counts are 17.8645, 8.9322, and 4.4661, while mean seven-day run probabilities are 0.5242, 0.7047, and 0.8738.

A local 30 mm moisture store receives current rain and recedes with a fixed two-month drydown scaled by current coupled-valid dryness. Its deficit \(B_t\) and the incumbent dryness response \(C_t=\mathrm{dryness}_t/(\mathrm{dryness}_t+500)\) condition the run probability into combustible opportunity

\[
O_t=W_t\sqrt{B_tC_t}.
\]

The equation is globally shared, pointwise, prefix causal, and uses only current monthly precipitation, current dryness, its carried local store, and incumbent hazard. It has no learned coefficient, fitted threshold, geographic term, target feedback, future statistic, wet-day input, or installed dry-spell field. Perturbing and reversing every input after month 96 changes all candidate values before month 96 by exactly `0`.

## Distinct roles

The occurrence role treats \(O_t\) as the probability that a latent combustible window exists. For incumbent hazard \(H_t=-\log(1-p_t)\), the marginal window-conditioned occurrence is

\[
p^{\mathrm{occ}}_t=O_t\left[1-\exp\left(-H_t/O_t\right)\right],
\]

with a fixed 0.10, 0.25, or 0.50 blend with incumbent probability. This is a latent-event mixture and cap rather than multiplication by a monotone rain response.

The conserved-release role stores \(s(1-O_t)H_t\), where \(s\) is the same fixed blend bracket, and releases the finite bank with fraction \(1-(1-O_t)\exp(-1/24)\). For every bracket, summed output hazard plus terminal bank equals summed input hazard to relative error at most `1.36e-15`.

## Held result

No one of the eighteen fixed brackets improves annual-log, normalized-allocation, and raw-cycle losses in every fold. Occurrence at the weakest 0.10 blend worsens annual loss in every fold for all three storm depths. At 20 mm, its annual gains are `-0.002669518`, `-0.002168079`, `-0.000240945`, and `-0.000550912`; allocation improves only fold 3, while raw cycle improves only fold 0.

The strongest aggregate compromise is conserved release with 20 mm storms and blend 0.10. It improves normalized allocation in all four folds by `+0.000631598`, `+0.000313201`, `+0.000208281`, and `+0.000502210`, but annual gains are `-0.000508754`, `-0.000872317`, `-0.000147108`, and `-0.000156770`, while raw-cycle gains are `-0.000079016`, `-0.000414836`, `-0.000335342`, and `-0.000203084`. Stronger blends amplify the annual and raw-cycle losses rather than reversing them.

The family therefore exposes a structural conflict: inferred event-free windows can redistribute the normalized seasonal allocation, but they remove or delay hazard in months whose raw burned-area magnitude the incumbent already needs. The preregistered held gate rejects the family. Exact evaluation and ecology were correctly skipped, so no exact proxy delta is justified and no canonical change is recommended. The complete eighteen-bracket output is reproducible with `autoresearch/scratchpad/compound_poisson_dry_spell_121c83c.py`.
