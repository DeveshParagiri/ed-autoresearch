# Live-to-dead litter mass balance falsification against 121c83c

This scratch experiment pins `autoresearch/model.py` at commit `121c83c`, whose model blob is `b82c285259f35f0f942ddc8a78663d8d14dd36b1`, and confirms the current canonical file has that same blob. The incumbent exact proxy is `0.719892388`. The test uses 4,463 whole cells assigned to four disjoint 15-degree spatial folds; these cells cover 92.5147% of observed-area weight and 90.4223% of incumbent excess-area weight. Coordinates are used only for fold assignment. GFED observations are used only in held losses.

## Mechanism

For pathway \(i\in\{f,s\}\), the globally shared, point-local state is

\[
L_{i,t+1}=L_{i,t}+I_{i,t}-T_{i,t},\qquad
D_{i,t+1}=D_{i,t}(1-d_{i,t})+T_{i,t}-C_{i,t}.
\]

The fast herbaceous pathway uses three-month live turnover and six-month dead decay. The slow woody pathway uses eighteen-month live turnover and thirty-six-month dead decay. Its inputs are fixed functions of current coupled-valid GPP, LAI, biomass, and natural or secondary canopy. Causal LAI decline increases turnover, current rain and temperature increase decomposition, and current dryness, rain, temperature, and local incumbent hazard govern combustion and dead-mass consumption. The state uses no target feedback, geographic term, learned coefficient, fitted threshold, future summary, or information outside the local monthly prefix.

The direct-load form replaces the incumbent fine-fuel proxy \(F_t=\operatorname{EMA}_{12}(GPP)_t/(\operatorname{EMA}_{12}(GPP)_t+0.35)\) with the explicit dead-to-total litter fraction

\[
Q_t=0.75\frac{D_{f,t}}{D_{f,t}+L_{f,t}+0.05}
+0.25\frac{D_{s,t}}{D_{s,t}+L_{s,t}+0.10}.
\]

The relative-allocation form instead applies the causal seasonal ratio \((R_t+0.05)/(F_t+0.05)\), divided by its local twelve-month antecedent, where \(R_t\) is current combustion readiness times \(Q_t\). The finite-release form stores the fraction of current fine-path hazard unavailable under \(R_t\) and releases that bank at a fixed rate \(1-\exp[-(1/24+8R_t)]\). Each form was tested at fixed blend fractions 0.25, 0.50, and 1.00. These are mechanistically distinct uses of the mass-balance state rather than scalar rescues of one surface.

The integrated state closes input minus decomposition, combustion, and terminal mass to relative error `4.27224730456e-16`. Reversing and perturbing every input after month 96 changes the selected formulation's predictions before month 96 by exactly `0`, confirming prefix causality.

## Held-block result

No formulation clears the required annual-log, normalized-allocation, and raw-cycle losses in every fold. The weakest direct-load blend, 0.25, improves annual and allocation losses in all four folds, but raw-cycle gains are `+0.000067940`, `+0.000253252`, `-0.000334826`, and `-0.000198488`; the waveform reverses in folds 2 and 3. The relative allocator at 0.25 improves annual loss in all folds by `+0.005332928`, `+0.003331365`, `+0.002925865`, and `+0.002461274`, but allocation reverses in folds 0 and 3 and raw cycle reverses in fold 0. The finite-release form at 0.25 improves allocation in every fold, but annual losses worsen by `-0.001118387`, `-0.001115793`, `-0.000382773`, and `-0.000759263`, while raw-cycle losses also worsen in every fold. Stronger blends do not repair those conflicting signs.

The aggregate held compromise is relative allocation at blend 0.50, but its ecology audit worsens the already high boreal ratio from `1.63161` to `1.76859`, tropical open from `1.02130` to `1.08803`, and arid low-fuel from `1.15597` to `1.16669`. Because every candidate fails at least one held metric in at least one spatial fold, the preregistered gate rejects the family. Exact evaluation was not run, so no exact proxy delta is justified and no canonical change is recommended.

The reproducible implementation and complete nine-bracket output are produced by `autoresearch/scratchpad/live_dead_litter_mass_balance_121c83c.py`.
