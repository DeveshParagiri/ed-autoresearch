# Capacity and allocation architecture audit of canonical 121c83c

This is a read-only architecture audit of the complete current `autoresearch/model.py`. The active component tuple is at lines 25–30 and the executed stage order is at lines 2678–2744. The file remains byte-identical to the pinned `121c83c` blob `b82c285259f35f0f942ddc8a78663d8d14dd36b1` at audit time.

## Active mechanism map

“Slow capacity” means a mechanism primarily changes the local multi-month or annual hazard envelope. “Allocation” means it primarily moves that envelope among months. “Both” means the same equation simultaneously changes magnitude and monthly shape, or its output-dependent gate makes those effects inseparable.

| Executed stage | Active component or gate | Classification | Architectural reason |
|---|---|---|---|
| Causal annual-rain preparation, lines 2661–2673 | Always active | Slow capacity input | Replaces completed-year rain with a prefix-causal twelve-month precipitation state used throughout later stages. |
| Base dryness factor, lines 204–209 | `dryness` | Both | The instantaneous hump shapes the fire season, while its cellwise mean controls annual support. |
| Base precipitation factor, lines 210–217 | `precipitation` | Both | Multiplies a slow annual-rain support term by an instantaneous monthly-rain brake in one factor. |
| Base GPP-fuel factor, lines 218–221 | `fuel` | Both | Instantaneous GPP supplies both seasonal phenology and the long-run magnitude of the base rate. |
| Managed-open temperature gate, lines 139–194 and 222–225 | `temperature` | Both | Current temperature timing is blended using twelve-month fine fuel, annual rain, land cover, canopy, and biomass. |
| Pre-transform cropland fragmentation, lines 2680–2689 | `cropland` | Slow capacity | A mostly structural land-cover brake scales rate before the nonlinear burned-fraction transform. |
| Curing multiplier, lines 318–339 and 2690–2691 | `curing` | Allocation with leakage | A current flammability ratio is divided by its causal EMA. It is mean-relative, not exactly mass-conserving. |
| Poisson transform and month duration, lines 260–278 | Always active | Both and nonlinear | `1-exp(-rate)` plus calendar duration changes peak-to-shoulder contrast whenever upstream magnitude changes. |
| Pathway event scaling, lines 388–508 | `pathway_hazards` | Both | Annual-scale event amplification and current surface, woody, and crop availability are mixed in the same hazard equation. |
| Cool-crop and humid-forest brakes, lines 350–385 | `cropland` and `fuel` | Both | Slow cover and annual-rain structure multiply current temperature and leaf-area states. |
| Ecological fire capacity, lines 511–556 | `regime_capacity` | Slow capacity | Twenty-four-month temperature, annual rain, biomass, canopy, and land cover set cold-forest amplification and productive-range suppression. |
| Seasonal rainfall capacity, lines 1481–1509 | `regime_capacity` | Both | A rolling rain-variability capacity factor is gated by the already-realized trailing fire prediction and applied only to the current month. |
| State-dependent fire season, lines 1198–1231 | `phenology` | Allocation with leakage | Current dryness is amplified conditional on trailing predicted fire, then divided by a causal EMA rather than routed through a conserved stock. |
| Rare lightning, rain-pulse, and natural-onset sources, lines 973–1195 | `rare_ignition` | Both | Current ignition windows add new hazard, while opportunity gaps depend on trailing predicted fire. |
| Crop residue event, lines 1234–1328 | `cropland` with `crop_residue_event_scale=0.25` | Allocation with leakage | Adds a current drying/warming residue event and rescales through separate baseline and adjusted EMAs. The rain-management brake subterm is inactive because its strength is zero. |
| Dead-fuel pool, lines 1331–1478 | `dead_fuel_pool` | Both | Slow litter stocks control current availability; the current prediction consumes those stocks, and an EMA ratio attempts seasonal redistribution. |
| Conditional fire allocation, lines 1551–1611 | Inactive | Dormant | It is called, but its gate requires `conditional_allocation`, which is not an allowed active component. |
| Live-fuel green-up brake, lines 1512–1548 | `phenology` | Allocation with leakage | Current green-up suppression is normalized by separate causal EMAs, not a conserved hazard bank. |
| Surface opportunity bank, lines 1614–1721 | `surface_opportunity_bank` | Allocation with terminal stock | Stores a surface share of current hazard and releases it using current relative hazard and weather readiness. |
| Local fire footprint, lines 1724–1805 | `pathway_hazards` | Slow capacity | Multiplies post-bank hazard by a surface footprint derived mainly from trailing lightning, fine fuel, and structural access. |
| Annual regime closure, lines 1880–2041 | `annual_regime_closure` | Both | Suppresses warm persistent fire and adds cold thaw fire using trailing predicted fire plus current thaw and combustion. |
| Multi-pathway opportunity bank, lines 2171–2337 | `surface_opportunity_bank` | Allocation with slow feedback | Four pathway banks use weather readiness, but managed storage also depends on trailing annual hazard. |
| Pathway fuel-recovery reservoir, lines 2340–2441 | `surface_opportunity_bank` | Both | Slow recovery stocks and finite banks redistribute hazard, while the amount immediately burnable changes repeat-fire capacity. |
| Secondary litter banks, lines 2445–2594 | `surface_opportunity_bank` | Allocation with terminal stock | Secondary live-to-litter states control release of finite secondary pathway hazard. |
| Secondary-open footprint, lines 864–970 | `secondary_open_footprint` | Slow capacity | A late hazard multiplier uses twelve-month fuel, rain and temperature support, and structural secondary-open share. |
| Fragmented managed recurrence brake, lines 2597–2645 | `pathway_hazards` | Slow capacity with feedback | Structural fragmentation suppresses current hazard in proportion to trailing realized hazard. |
| Surface-seasonality capacity, lines 559–699 | `regime_capacity` | Both | Despite its capacity label and late placement, its multiplier contains current rain and dryness combustion, so it selectively amplifies particular months. |
| Arrival-order factor, lines 702–861 | `arrival_order` | Allocation with leakage | A one-month lightning/combustion ordering signal is divided by its causal EMA and applied to final hazard. |

The dormant softmin at lines 236–249, lag at lines 281–297, vegetation branch at lines 226–235, and conditional allocator at lines 1551–1611 are not in the active component set. Their parameters may exist, but they do not run in canonical prediction.

## Precise capacity-to-waveform coupling

The first coupling occurs before most ecological logic. An annual-oriented change to the base rate passes through `p_t=1-exp(-m_t r_t)` at lines 260–278, so a scalar rate change is not a scalar burned-fraction change: peak months saturate and shoulder months expand differently. Pathway event scaling then computes `connected_t=p_t/(p_t+0.003)` at lines 402–406 and uses it both in the annual multiplier and the surface pathway multiplier at lines 497–507. The candidate’s magnitude therefore changes its own month-specific event-scale law.

The more consequential coupling is downstream feedback. `_rare_lightning_ignition` uses trailing prediction in its opportunity gaps at lines 1055–1063, 1093–1098, and 1166–1176. `_state_dependent_fire_season` uses trailing prediction to set recurrent-season strength at lines 1214–1228. `_seasonal_rainfall_capacity` gates current amplification by trailing prediction at lines 1493–1508. `_dead_fuel_pool_response` consumes litter using current prediction at lines 1429–1461. `_surface_fire_opportunity_bank` releases according to current hazard relative to its trailing state at lines 1688–1718. `_annual_regime_closure` conditions warm suppression and cold supply on trailing prediction at lines 1904–1908, 1972–1980, and 2013–2037. `_multi_pathway_opportunity_bank` makes both release and managed storage depend on current or trailing hazard at lines 2301–2333. `_pathway_fuel_recovery_reservoir` changes stock consumption and bank release using current hazard at lines 2403–2436. `_fragmented_managed_recurrence_brake` uses trailing final hazard at lines 2618–2641.

These operators are serial and do not commute. A capacity increase before a bank changes how much enters the bank and when it is released. A late capacity multiplier then selectively amplifies the released months without amplifying the remaining bank. The pipeline alternates these roles: early capacity, allocation, source, allocation, capacity, closure, three more allocation or stock stages, two late capacity stages, and a final allocation stage. Consequently, an annual correction changes the input to the timing machinery, the timing machinery changes its own gates in response, and late capacity stages amplify the altered release pattern.

The causal-EMA “conservation” pattern is a second, exact source of leakage. Curing, crop residue, dead fuel, live green-up, state-dependent season, and arrival order divide a current adjusted value by a causal running reference. This preserves neither a calendar-year total nor a finite local hazard stock. When upstream magnitude or phase changes, the baseline and adjusted EMAs follow different trajectories, so the normalization itself changes annual magnitude and carries the phase error forward. This explains the repeated empirical signature: a mechanism improves annual-log loss while one raw-cycle fold reverses, and increasing its strength initially improves bias or seasonal score before spatial or raw-cycle skill collapses.

## Minimal prefix-causal refactor

The smallest useful change is to project any proposed annual mechanism onto a slow hazard envelope after the existing timing pipeline, rather than feeding its instantaneous correction back through all output-dependent stages. Let \(h_t=-\log(1-p_t)\) be final incumbent hazard and let \(\tilde h_t\) be the hazard from a raw mechanistic annual candidate such as direct litter-load replacement. Define causal twelve-month states

\[
B_t=(1-\alpha)B_{t-1}+\alpha h_t,\qquad
\widetilde B_t=(1-\alpha)\widetilde B_{t-1}+\alpha\tilde h_t,
\qquad \alpha=1-e^{-1/12}.
\]

The capacity-only projection is

\[
c_t=\frac{\widetilde B_t}{B_t+\varepsilon},\qquad
h^{\mathrm{new}}_t=c_t h_t,qquad
p^{\mathrm{new}}_t=1-e^{-h^{\mathrm{new}}_t}.
\]

This keeps the incumbent’s fast monthly allocation \(h_t/B_t\) and imports only the candidate’s slow local capacity ratio. It is globally shared, pointwise, prefix causal, and contains no target, region, learned coefficient, or future normalization. It deliberately does not claim exact calendar-year conservation; it is a causal slow-envelope projection. Because it runs after the current timing stages, the capacity change cannot alter rare-source gaps, release fractions, fuel consumption, recurrence, or arrival-order normalization during the same prediction call.

An implementation sketch is:

```python
def _slow_capacity_projection(baseline, raw):
    alpha = 1.0 - np.exp(-1.0 / 12.0)
    base_h = -np.log1p(-np.clip(baseline, 0.0, 1.0 - 1e-7))
    raw_h = -np.log1p(-np.clip(raw, 0.0, 1.0 - 1e-7))
    base_slow = _antecedent(base_h, alpha)
    raw_slow = _antecedent(raw_h, alpha)
    capacity = np.maximum(raw_slow, 0.0) / (base_slow + 1e-8)
    projected = base_h * capacity
    return -np.expm1(-np.clip(projected, 0.0, 50.0))
```

The first candidate should be the already supported direct litter-load replacement at blend 0.10, because it has exact Overall `+0.000050448` but a spatial tradeoff. Compute its raw final hazard exactly as in the existing scratch experiment, then pass the incumbent and raw candidate through `_slow_capacity_projection`. Do not re-run downstream timing stages on the projected result.

## Falsification protocol

The test should pin `121c83c` and reuse `held_losses` plus the same 4,463 whole cells and four 15-degree folds. The preregistered comparison should contain exactly the incumbent, raw direct-load blend 0.10, and its twelve-month slow projection. Fixed six- and twenty-four-month projections may be included only as sensitivity brackets, not selected by the target.

The architecture claim fails if the projected form does not retain positive annual-log gain in every held fold, does not improve raw-cycle loss relative to the raw litter candidate in every fold, or introduces a new allocation-loss reversal. A direct stage-decoupling diagnostic should also compare the normalized monthly cycle before and after multiplying all input hazards by a constant; the projected operator should preserve that allocation up to numerical tolerance apart from the deliberately slow envelope drift. Reversing and perturbing all inputs after month 96 must leave the first 96 projected months exactly unchanged.

Only a held survivor should receive full exact replay. Under the Overall-first rule it must keep exact Overall above `0.719892388`, reduce the raw candidate’s `-0.000573012` spatial-component loss, retain nonnegative seasonal delta, move global area ratio no farther from one, and create no severe ecology or Congo pathology under the existing thresholds. If the slow projection loses the raw candidate’s exact gain, the annual improvement depended on its fast monthly residual and the proposed separation is falsified. If it retains Overall while repairing waveform or spatial loss, it is evidence that the current failure is architectural coupling rather than missing physical information.
