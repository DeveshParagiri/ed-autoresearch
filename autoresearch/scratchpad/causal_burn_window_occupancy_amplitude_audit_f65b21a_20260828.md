# Causal burn-window occupancy amplitude audit at `f65b21a`

## Verdict

The proposed law

\[
W_t=\sigma((D_t-E[D]_t)/s_D)\,\sigma((T_t-E[T]_t)/4),\qquad
h'_t=h_t\left(W_t/E[W]_t\right)^\gamma
\]

is prefix-causal if every exponential state is initialized from the first available month and updated only from the current prefix, but it is not a new burn-window occupancy mechanism, is not mean-neutral, and cannot presently be distinguished from state-selected temporal sharpening. It should not receive another held or exact test in this form.

## Duplication audit

The closest active structural duplicate is `_state_dependent_fire_season` in `autoresearch/model.py:1200-1233`. It converts current dryness into a dry-phase gate, exponentiates a locally gated signal, divides the factor by its causal twelve-month running state, and multiplies the prediction by the relative factor. Its nonzero strength is active at `autoresearch/model.py:82-84`; later operating-point audits found the phenology component load-bearing rather than dead (`autoresearch/scratchpad/thread.md:4760-4774` and `5917-5939`). The proposed law changes the physical signal from absolute dry phase plus recurrence to paired dryness and temperature departures, but preserves the same leaky causal-EMA allocation architecture.

The paired departures are themselves already active. `_rare_lightning_ignition` computes `heat_onset = sigmoid(T-T3)` and a scale-free `dryness_departure=(D-D3)/(D+D3+100)` before combining them with absolute combustion, lightning arrival, antecedent fuel, natural share, and low prior fire opportunity at `autoresearch/model.py:1118-1197`. That rare source was selected precisely because short warming and drying onset were stable in held blocks (`autoresearch/scratchpad/thread.md:4645-4658`) and was promoted after the eligible exact test (`autoresearch/scratchpad/thread.md:4691-4716`). `_multi_pathway_opportunity_bank` also uses three- and twelve-month warming departures in crop and woody readiness at `autoresearch/model.py:2341-2357` and `2417-2433`.

A nearly identical causal-relative temperature-departure experiment is already closed. Entry 213 used a surface-share, absolute-thermal, three-month-warming, and GPP-curing signal with `F=exp(kS)/EMA12(exp(kS))` (`autoresearch/scratchpad/thread.md:6087-6097`). Fixed strengths improved annual loss broadly but worsened normalized cycle in the same held block monotonically, so the family was falsified without exact evaluation (`autoresearch/scratchpad/thread.md:6099-6106`). The surviving difference here is a dryness departure substituted for the fuel-curing gate, but that departure is already in the active rare-onset term.

The repository also contains a literal dry-window occupancy test. `autoresearch/scratchpad/dry_window_occupancy_b867ed7.py:63-127` forms a smooth current dry-month indicator from rain, carries a causal twelve-month occupancy state, and gates it with rain memory, temperature memory, natural/open cover, fuel, and continuity. Its best wet-window brake gained only 0.000130 and the paired semi-arid gain was negative at every strength, so it was not promoted (`autoresearch/scratchpad/thread.md:4511-4527`). This matters because that state measures persistence or duration; the proposed product of positive departures measures onset.

Finally, the active model already has physically explicit finite allocators. `_surface_fire_opportunity_bank` stores surface hazard and releases it under fine fuel, absolute combustion, rain deficit, curing, and relative opportunity at `autoresearch/model.py:1821-1850`. `_multi_pathway_opportunity_bank` repeats the finite-stock logic for managed, crop, woody, and background pathways at `autoresearch/model.py:2200-2223` and `2309-2445`. Recasting the proposed multiplier as a conserved stock would therefore duplicate existing machinery rather than open a new mechanism family.

## Dryness semantics and circularity

The installed `dryness` is not instantaneous fine-fuel moisture. `climate.nc` declares units of millimetres and `long_name = "running Thornthwaite moisture deficit"`; the repository describes it as accumulated dryness (`README.md:16`). It is a carried water-balance state, while air temperature is a monthly mean in degrees Celsius. Therefore `D-E[D]` is the acceleration or high-pass departure of an already accumulated hydroclimatic deficit, not occupancy of a physically defined combustible interval. Unless `s_D` is explicitly defined it is an unidentified millimetre scale with no globally fixed fuel-class interpretation; making it a causal standard deviation would yield a standardized anomaly, not new physics.

This is not target circularity: neither burned area nor a prediction-derived field appears in `W`, and the equation can satisfy the pointwise and prefix rules in `autoresearch/research.md:7-13`. It is nevertheless process-redundant. A Thornthwaite moisture deficit already embeds precipitation history and temperature-based evaporative demand, so multiplying its positive departure by a separate positive temperature departure double-conditions the same warming-and-drying transition. The active base rate already applies absolute dryness, monthly rain, fuel, and temperature together at `autoresearch/model.py:199-227`; the proposed law then reuses two of those drivers downstream without introducing a new fuel pool, ignition process, spread process, or weather-duration observation.

The sigmoids also do not define occupancy. At zero departure each returns 0.5, giving `W=0.25`; on a persistently hot, dry plateau both departures relax toward zero even though combustion can remain maximal. The product selects the rising shoulder of a season and suppresses sustained extremes. Calling it burn-window occupancy therefore gives an onset detector a duration interpretation it does not have.

## The EMA division is not mean-neutral

Let `E_t=(1-alpha)E_{t-1}+alpha W_t` and `Q_t=W_t/E_t`. No identity makes the arithmetic mean of `Q` equal to one. Using the model's twelve-month `alpha=1-exp(-1/12)`, initialize `E_0=W_0=0.2` and let `W_1=0.8`. Then `Q_0=1`, `E_1=0.247973`, and `Q_1=3.226153`, so the two-month mean ratio is 2.113077, not one. A falling sequence produces the opposite bias, and initialization makes the imbalance path-dependent.

Raising the ratio to `gamma` adds another nonlinear mismatch. More importantly, even an unweighted mean-one factor would not conserve fire: `sum(h_t Q_t^gamma)` differs from `sum(h_t)` whenever the factor covaries with hazard, and the subsequent `1-exp(-h)` transform, month-duration weights, calendar-year boundaries, downstream states, and clipping introduce further leakage. The repository's architecture audit reaches the same conclusion for all causal-EMA normalization stages at `autoresearch/scratchpad/architecture_capacity_allocation_audit_121c83c_20260828.md:44-50`. The `_curing` docstring's claim that its causal ratio has per-cell time mean one at `autoresearch/model.py:320-341` is therefore mathematically false for a finite sequence.

## Physics decision

The direct law is generic sharpening with a physical-state mask. In log space it is simply `log(h'/h)=gamma[log(W)-log(E[W])]`; it has no conserved quantity, fuel residence time, ignition count, or spread length. The fixed temperature width of 4 degrees C does not repair the unidentified millimetre scale `s_D`, and applying it to total hazard would mix surface, woody, crop, and background pathways that have different fuel-moisture and event-size physics.

A defensible seasonal-allocation equation would defer a bounded fraction of pathway hazard into a nonnegative stock and release that stock using an absolute combustibility probability. With input hazard `h_t`, storage fraction `f_t`, stock `B_t`, and release `r_t`, the minimal closure is

\[
A_t=B_{t-1}+f_t h_t,\qquad h'_t=(1-f_t)h_t+r_t A_t,\qquad B_t=(1-r_t)A_t.
\]

This conserves `sum(h')+B_final=sum(h)+B_initial` and gives every term a physical role. However, this is already the surface and multi-pathway opportunity-bank architecture cited above. With current valid inputs, a new fixed equation cannot honestly be called an independent occupancy mechanism.

The high-information recommendation is therefore no-probe for Claude's proposed ratio. A genuinely distinct successor requires an independently observable duration state such as daily VPD-threshold occupancy, dry-spell length, or pre-fire ED fine-fuel moisture. The first two prepared diagnostics remain coupled-invalid until the documented 1850 bridge is installed (`autoresearch/inputs/README.md:7-15`), and the last is absent from the current export. Until one of those states becomes eligible, another `departure/EMA(departure)` multiplier would only retest the closed sharpening and onset families.
