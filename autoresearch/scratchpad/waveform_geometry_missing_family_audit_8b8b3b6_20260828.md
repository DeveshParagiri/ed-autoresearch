# Waveform-geometry family audit after Entry 236

## Conclusion

The canonical model and closed experiments cover the major one-dimensional waveform operators suggested by onset, peak, tail, hysteresis, curing, and fuel storage. The only defensible current-input family I could not find in code, scratch experiments, or the research thread is the signed circulation of fuel production and combustibility through local state space. It asks whether productive fuel state precedes combustion or combustion precedes renewed production. That is a direction variable, not another level, amplitude, lag, or univariate derivative.

The recommended next probe is therefore a fuel-production/combustibility arrival-order state applied only to the release selector of the existing finite surface bank. It should not be installed as a new hazard multiplier or a second bank. Its predeclared comparison against an unsigned circulation control is essential: without that control, a nominal success could be another generic seasonal concentrator.

## Prior family to current-canonical map

| Waveform family | Prior evidence and disposition | Current canonical representation at `8b8b3b6` |
|---|---|---|
| Dry-season onset and attack | Antecedent-rain curing produced the first large gain in Entries 3–7. A broad missed-window pulse and stronger onset were neutral or harmful in Entries 104–105. Rapid warming, drying onset, and lightning arrival survived as a rare source in Entries 136–140. Thermal-curing onset failed a held block monotonically in Entry 213, and the dryness/temperature EMA window failed every held loss in Entry 235. | `_curing` is active at `model.py:320-341`; `_state_dependent_fire_season` concentrates recurrent hazard into absolute dry phase at `1200-1233`; `_rare_lightning_ignition` contains the successful dry-warm-lightning onset at `1118-1197`; crop residue has a separate drying/warming shoulder at `1288-1329`.
| Peak timing, peak contrast, and simple phase shifts | The one-month assembled-rate lag helped shoulders but did not move peak month; a two-month kernel and rain-dependent lag failed in Entries 30–31. Repeated power, flatness-targeted, VPD, and peak-above-median sharpening failed in Entries 20–24, 72, and 88–91. Warm-open one-to-four-month dry clocks sharpened the wrong phase in Entry 121. Phase-anchored falling-limb depletion gained only `0.000006854` in Entry 234. | The historical `_lag` function remains at `model.py:283-299`, but `lag` is absent from `COMPONENTS` and does not execute. Peak contrast now arises jointly from the nonlinear transform at `262-280`, the dry-phase allocator, pathway readiness, finite banks, and final litter timing rather than a direct peak operator.
| Tail persistence and post-peak suppression | Generic depletion was tuned off in Entries 3–5 and re-refuted after the transform fix in Entry 67. The one-sided max/relaxation hysteresis added only to the tail and declined monotonically in Entry 93. Output-mean-neutral within-season consumption protected the spatial map but pulled peak phase earlier and collapsed seasonal skill in Entry 94. The phase-selective successor in Entry 234 obtained the intended centered-cycle sign but negligible exact leverage and raw-cycle reversals. | There is no active generic rate hysteresis or generic tail suppressor. Tail behavior instead comes from live-fuel green-up suppression at `model.py:1645-1683`, burn consumption within the dead pools at `1431-1463` and `1566-1605`, and the storage/release stack.
| Bimodality and management shoulders | Temperature-derived spring stubble and hot-season rangeland calendars produced historical gains in Entries 77–84, but their extra autumn shoulder and moisture refinements failed. Spring crop-residue ignition failed ecological breadth in Entry 108. TENA's April primary and October secondary peaks remained unrecoverable from monthly GPP and LAI; a finite residue allocator improved L1 slightly without moving the August peak in Entry 171. Current LUH2 is annual and repeated monthly, so it cannot identify planting or harvest dates. | The remaining globally shared crop timing is `_rain_conditioned_crop_management` at `model.py:1236-1330` and the crop readiness/recovery branches at `2419-2452` and `2473-2574`. They can create a distinct managed shoulder, but the missing two-season management calendar is not an honest current-input family.
| Curing and live-to-dead conversion | Accumulated GPP/LAI fuel load fought antecedent-rain curing and was discarded in Entry 13. Root-water drawdown, fixed-age maturation, accumulated dry-degree curing, and a direct live/dead SPITFIRE proxy all failed in Entry 144. A reconstructed live/dead litter balance lacked phase information in Entry 221; exact factorization then isolated its fast timing channel in Entries 229–231, producing the current exact gain. Surface-area-weighted live/dead fuel retained the same annual-versus-cycle conflict in Entry 232. | The base rain-curing factor remains active at `320-341`. `_dead_fuel_pool_response` uses GPP and LAI decline, separate fine and woody stocks, decomposition, combustion, and consumption at `1333-1480`. `_live_dead_litter_timing` adds the accepted timing-only residual at `1483-1612`.
| Finite fuel and opportunity banks | The clean surface bank was promoted in Entry 114, separate managed/crop/woody/background banks in Entry 116, pathway fuel recovery in Entry 117, and secondary litter banks in Entry 119. Produced-fuel stocks, unified live/dead rebuilds, dry-window clocks, prognostic connectivity stocks, recoverable opportunity, and inferred dry spells were rejected in Entries 121–123, 144, 187–194, 221, and 224. | `_surface_fire_opportunity_bank` at `1747-1855`, `_multi_pathway_opportunity_bank` at `2304-2470`, `_pathway_fuel_recovery_reservoir` at `2473-2574`, and `_secondary_fuel_litter_banks` at `2578-2727` already cover finite storage, release, recovery, and pathway-specific residence times. A new bank would be a duplicate.
| Direction and ordering | Separate positive temperature, dryness, lightning, and GPP-decline departures have been tested repeatedly. The only signed two-process order state found in the repository compares lightning ignition with combustibility, introduced in Entries 184–189 and retained after Entries 202 and 208. It does not compare fuel production with combustion. | `_ignition_combustibility_arrival_order` computes the signed cross-lag of lightning and combustion at `model.py:704-863`. The other active pathway equations use current or one-sided fuel decline and combustion levels, but none computes the orientation of the fuel-production/combustibility trajectory.

This map also explains why a new bimodal or hysteretic label is insufficient. A second peak derived from the same crop residue, GPP decline, or temperature shoulder repeats the closed management-calendar work. A slow-decay state repeats rate hysteresis or a litter bank. A positive departure repeats rare onset. A factor divided by an EMA repeats the non-neutral sharpening failure in Entry 235.

## The untested family: fuel-production/combustibility circulation

Define bounded current productivity and combustibility using scales already active in the model,

\[
P_t=\frac{\mathrm{GPP}_t}{\mathrm{GPP}_t+0.35},\qquad
C_t=\frac{D_t}{D_t+500}\frac{1}{1+R_t/35}\,\sigma\!\left(\frac{T_t-5}{3}\right).
\]

The one-step oriented circulation is

\[
J_t=P_{t-1}C_t-C_{t-1}P_t,\qquad -1\leq J_t\leq1.
\]

Positive `J` means productive fuel state precedes increasing combustibility, the production-to-drydown branch. Negative `J` means combustion precedes renewed production, the rewetting or green-up branch. Equal current `P` and `C` can therefore receive different timing treatment depending on trajectory direction. No active component has that information. The rare-onset term uses separate positive departures, the final litter law uses mass and its slow residual, and arrival order uses lightning rather than fuel production.

The first probe should reuse the incumbent surface bank rather than add another operator. If its current release opportunity is `O_t`, replace only the release selector by

\[
O^{\mathrm{dir}}_t=O_t\,\operatorname{clip}(1+\kappa S_tJ_t,0.75,1.25),
\]

where `S_t` is the already computed surface pathway share and `kappa` takes fixed weak values `0.10`, `0.25`, and `0.50`. Positive production-to-combustion circulation advances release; reverse circulation defers it into the same finite bank. The pathway hazard entering the bank is unchanged, no hazard is created, all constants are global, and every input is current or one-step antecedent. This is pointwise and prefix-causal by construction.

The mandatory amplitude control is

\[
O^{\mathrm{unsigned}}_t=O_t\,\operatorname{clip}(1+\kappa S_t|J_t|,0.75,1.25).
\]

The directional family is informative only if the signed law beats both the incumbent and this unsigned control on held raw-cycle and normalized-allocation losses without buying its result through annual area. An inverted-sign control should be reported once, not tuned, because the physical hypothesis predeclares production before combustion as the supported orientation.

## Why this is distinct and what would falsify it

This law does not sharpen high states, move everything one month, prolong all tails, suppress all falling limbs, infer a planting month, or construct another fuel pool. It measures an oriented loop between two existing physical processes and uses an already accepted conserved allocator. The repository-wide overlap search found previous-GPP pulses and one-sided GPP decline, but no antisymmetric `P_(t-1) C_t - C_(t-1) P_t` state or equivalent fuel/combustibility cross-lag.

The highest-information screen is the standard four disjoint whole-cell folds with annual-log, normalized-allocation, and raw-cycle losses, plus bank closure and a future-prefix mutation. Exact replay should follow only if the signed physical orientation is broadly stable and better than the unsigned control. The exact audit must include all Overall components, all fourteen regions, global area, peak-month changes, intact tropical, temperate closed, boreal, tropical open, productive rangeland, crop, arid low-fuel, and Congo ratios.

The likely failure is ecological sign heterogeneity: wet-season production can precede fire in seasonal grasslands, while woody, crop, or ignition-limited systems may traverse a different loop. Restricting the test to the existing surface share is therefore part of the hypothesis, not post-hoc rescue. A second failure is observational: monthly GPP is production rather than dead fine-fuel load, so `J` may be too phase-blurred or collapse to the curing signal. If the signed and unsigned forms behave alike, the direction contains no incremental information and the family is closed. If only a fitted sign, region-specific sign, calendar phase, or target-selected mask works, the family is scientifically rejected.

The packaged GPP provenance also remains unresolved as documented in `autoresearch/inputs/README.md:21-26`. This family is coupled-valid only when GPP is generated online by a provenance-clean ED run; the present offline score cannot establish leakage-free causality.
