// Auto-generated from paper.md by build.py — do not edit by hand.
// Edit paper.md, then run: python build.py   (or python watch.py)

#set document(
  title: "Development and Optimization of a Global Fire Model Using Autoresearch AI",
  author: ("Richard Owusu-Ansah",
    "George Hurtt",
    "Lei Ma",
    "Devesh Paragiri",
    "Janna Chapman"),
)

#set page(
  paper: "us-letter",
  margin: (x: 1in, y: 1in),
  numbering: "1",
  number-align: center,
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
  lang: "en",
)

// More air between paragraphs
#set par(
  justify: true,
  leading: 0.78em,
  first-line-indent: 1.2em,
  spacing: 1.05em,
)

#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  set text(size: 12pt, weight: "bold")
  v(2.0em, weak: true)
  block(below: 1.15em, it)
}
#show heading.where(level: 2): it => {
  set text(size: 11pt, weight: "bold")
  v(1.55em, weak: true)
  block(below: 0.95em, it)
}

#show figure: set block(breakable: false, spacing: 1.6em)
#show figure.caption: set text(size: 9pt)
#set figure(gap: 0.7em)
#set math.equation(numbering: "(1)")

// Title block
#align(center)[
  #text(size: 14pt, weight: "bold")[
    Development and Optimization of a Global Fire Model Using\
    Autoresearch AI
  ]
  #v(1.2em)
  #text(size: 10.5pt)[
    Richard Owusu-Ansah,
    George Hurtt,
    Lei Ma,
    Devesh Paragiri,
    Janna Chapman
  ]
  #v(0.6em)
  #text(size: 9.5pt)[
    Department of Geographical Sciences, University of Maryland, College Park, MD, USA
  ]
]

#v(1.6em)

// Abstract: centered label, then body
#align(center)[
  #text(size: 11pt, weight: "bold")[Abstract]
]
#v(0.7em)
#par(first-line-indent: 0em, leading: 0.78em, spacing: 1.05em)[
  Fire returns a large flux of carbon to the atmosphere each year, yet the land models used to estimate the terrestrial carbon balance reproduce it poorly, the Ecosystem Demography model among them. We develop and optimize a fire submodule for this model with an automated, AI-assisted loop, which we call autoresearch, that changes the model one step at a time, either its functional form or the criterion by which it is scored, fits each version, and evaluates it against the fifth Global Fire Emissions Database using the official ILAMB benchmarking system. The loop produces a family of versions whose differences can be attributed to single causes. In the base version, Model C, a single global formula fit to an aggregate benchmark score reproduces the broad geography of fire but under-represents the intense regional burning where the model is weakest. Model D changes only the goodness-of-fit criterion and stays near this baseline, whereas Model E changes the functional form, raising the spatial distribution score from 0.79 to 0.88 and the peak burned fraction on the most fire-prone cells from Model D's 0.04 to 0.075 against the 0.10 observed, and bringing the total burned area to 816 million hectares per year, close to the 793 recorded by GFED5. The functional form, not the goodness-of-fit criterion, is therefore the binding constraint on burned-area skill. Held-out tests in time and space confirm the gain as genuine structure, and the burned area yields fire carbon emissions consistent with observations.
]

#v(1.5em)

= Introduction

Fire is a major flux in the global carbon cycle. Landscape fires emit about 3.4 Pg of carbon to the atmosphere each year, averaged over 2002 to 2022 in the fifth version of the Global Fire Emissions Database @van-der-werf-2025. These are gross emissions, and much of the carbon is returned to the land as vegetation regrows after a fire. Even so, it remains large relative to the other terms of the land carbon budget, in which net emissions from land-use change and the net land sink average about 1.4 and 2.4 Pg of carbon per year over the same recent decade @friedlingstein-2025. Fire is also a persistent global process rather than a local or occasional disturbance. It recurs each year across savannas, tropical and boreal forests, and other fire-prone biomes, and its controls differ among them, from fuel-limited grasslands to deforestation-driven tropical fires @jones-2022 @chen-2023. A flux of this size, redistributed across the globe every year, means that how a land model represents fire bears directly on its estimate of the land carbon balance. Fire-enabled vegetation models still meet that standard with widely varying success, spanning an order of magnitude in simulated global burned area @hantson-2020.

Fire has long been poorly represented in the land models used to estimate the terrestrial carbon sink. The dynamic global vegetation models of the TRENDY ensemble, which supplies the land component of the global carbon budget, do not all include a fire scheme, and fire is not among the variables against which these models are routinely evaluated @sitch-2024. Where fire schemes have been evaluated against satellite burned area, their skill is limited and varies widely from one model to the next. Across the nine models of the Fire Model Intercomparison Project, simulated global burned area spans an order of magnitude, from 39 to 536 Mha per year, and no single model performs best across the benchmarks @hantson-2020. The spread persists in the current generation of Earth system models. Nineteen CMIP6 models submitted fire simulations, and among the nine that report burned area, evaluated against GFED5 and other satellite products, global values span about 170 to 760 Mha per year, their spatial correlation with the observed pattern reaches only 0.28 to 0.70, and none reproduces the observed multi-decadal decline in burning @li-2024. The difficulty is structural rather than a matter of calibration. Even where models capture the broad geography of fire, they misrepresent how burned area responds to its drivers, in particular its sensitivity to vegetation productivity and fuel @forkel-2019, and fire remains an acknowledged challenge as the community designs the next round of model intercomparisons @li-2026. The Ecosystem Demography model, a vegetation model of the same class, shares this limitation. Its native fire module greatly overestimates global burned area and reproduces little of the observed spatial pattern. That the same limitation recurs across otherwise independent models points to a difficulty rooted in fire itself, not in any single implementation.

Fire is difficult to model because it depends on several factors that must coincide. A fire needs fuel, that fuel must be dry enough to burn, and it must be ignited by lightning or by people @jones-2022. None of these acts uniformly across the globe, and the factor that limits fire changes from one biome to another. Savannas and grasslands are fuel-limited. Their grass is abundant only after a growing season, so their fire tracks the production and curing of that fuel and rises with productivity @jones-2022 @son-2024. Moist tropical forests are the opposite. Their fuel is rarely dry enough to burn on its own, so their fire follows drought and deforestation rather than fuel supply @abatzoglou-2025. Whether such a forest persists or gives way to flammable grassland turns on a grass-fire feedback that a bulk description of burning cannot capture @shuman-2024. Boreal and temperate forests are flammability-limited, and they burn in the rarer years when fire weather turns extreme @abatzoglou-2025. These regimes are held back by opposing constraints, fuel in one place and moisture in another @mccoll-2022. The same shift in climate can then raise burning in one region and lower it in another. The dominant control itself reorders, from fuel and soil moisture in the tropics to meteorology at high latitudes @son-2024. A single global formula has to reconcile all of this, which is why a globally applicable description of fire is so hard to build.

The fire submodule at the center of this work is exactly such a formula. One parameter set is applied uniformly to every land cell, and it is fit by numerical optimization against GFED5. The optimizer targets the aggregate Overall score of the International Land Model Benchmarking framework, which combines bias, root-mean-square error, seasonal cycle, and spatial distribution into a single weighted sum, with the error term counted twice @collier-2018. That framework is the community standard for land-model evaluation and remains in active use in current benchmarking @massoud-2026. Its designers note that a higher aggregate score does not necessarily reflect a more process-faithful model, because such scores are the same scalars against which many models are calibrated in the first place @collier-2018. The distortion is sharp for burned area. The score integrates across the whole land surface, and because most cells burn little or nothing in any given year, it is dominated by the broad zero-fire background rather than by the small fraction of cells where fire is a major flux. Fitting against that score rewards a model that gets the broad field roughly right and does not strongly punish one that misses the intense regional burning where the model is weakest. A related Earth-system study found that the parameter set with the highest goodness-of-fit metric underestimated the true physical response by 79 percent @boardman-2025. A formula fit against such an aggregate metric is steered toward the average and away from the extremes, and re-tuning within the same formulation does not undo that steering. Recent CMIP6 fire evaluations reach the same conclusion, that both calibration revisions and a re-examination of the underlying parameterizations are needed @li-2024. In recent SPITFIRE development the main improvements have come from structural fixes rather than parameter tuning @oberhagemann-2025. Closing the gap requires changes in both what the model is and how it is scored.

This paper builds both changes into one automated loop, an AI-driven procedure that searches over both the functional form of the fire submodule and the goodness-of-fit criterion by which it is judged. We refer to the loop as autoresearch. The loop belongs to a wider effort to bring machine learning to Earth-system models while keeping them physically interpretable, an effort in which burned area is named among the target problems @reichstein-2019. Machine learning has already been applied to fire in this setting, but as a black box. A deep neural network has been used as a surrogate for the fire scheme of E3SM @zhu-2022, and a data-driven fire model has been coupled into JSBACH4 within the ICON Earth-system model @son-2024. Autoresearch differs in what it produces. Its output is a physical formula whose terms can be read and reasoned about, not a neural surrogate. Its closest precedent in spirit is recent cloud-cover work that pairs an interpretable data-driven equation with automatic tuning and separates the effect of the equation from the effect of the tuning by a controlled ablation @grundner-2025. Autoresearch brings the same discipline to a burned-area submodule scored against GFED5, and it searches over the goodness-of-fit criterion in addition to the form. The loop produces a family of versions of the model in which each version differs from the one before by exactly one change, in either the functional form or the goodness-of-fit criterion, and every version is evaluated against GFED5 on the same grid and period. The paper's contributions follow from this design. The first is the autoresearch loop itself. The second is the family of improved fire submodules that it yields. The third is a goodness-of-fit criterion that targets the intense burning the aggregate score under-rewards. The fourth is the identification, and partial relief, of a structural ceiling that keeps the model below observed fire intensity even when its parameters are fit as well as the form permits.

= Methods

== The fire model setup

The fire submodule predicts the fraction of each grid cell that burns in a given month. It is a closed-form function of a small set of environmental drivers, it carries no vegetation state of its own, and it is run offline so that its behaviour can be studied apart from feedbacks in the host model. It computes on a one-degree grid, matching the resolution of its drivers, and its output is mapped to the half-degree grid of the reference for scoring, over the period 2001 to 2016.

The fire model is driven by four inputs, namely dryness, precipitation, air temperature, and gross primary production as a proxy for available fuel. Dryness is an accumulated climatic water deficit, the running sum of monthly potential evapotranspiration minus precipitation, reset after a wet month, so it measures how much moisture deficit has built up since the last substantial rain. Dryness, precipitation, and temperature are derived from the CRUJRA reanalysis @harris-2020, and productivity comes from the output of the coupled ED model @ma-2022. This hybrid sourcing is deliberate. Observed climate lets the fire model be judged against the reference on its own skill, and taking productivity from the coupled model keeps it consistent with the run it is meant to serve.

Every version of the fire model is evaluated in the same way. The reference is the GFED5 burned-area product @chen-2023, and scoring uses version 2.7.3 of the official International Land Model Benchmarking system @collier-2018, configured for the global region, which reports separate scores for bias, root-mean-square error, the seasonal cycle, and the spatial distribution, together with an aggregate Overall score that combines them. Every version is scored on the same half-degree grid and over the same 2001 to 2016 period, so that every comparison between versions rests on one common yardstick.

== Automated model development and optimization

The fire model was developed and calibrated by an automated loop (@fig:loop). At each step the loop makes one change to the model, either to its functional form or to the criterion by which it is scored, fits the parameters of the resulting version, and evaluates that version against the GFED5 reference. We call this loop autoresearch. It combines two practices already established in Earth-system modelling, the automated calibration of model parameters and the automated exploration of model structure, and applies them together to a burned-area model. The change at each step is proposed by an AI agent from the diagnostics of the previous version, and the parameters of each version are fit by numerical optimization.

#figure(
  image("figures/fig1_autoresearch_loop.jpg", width: 72%),
  caption: [The autoresearch loop used to develop and calibrate the fire model. The loop searches on two levels. At the outer level an AI agent proposes one change per step, either to the functional form or to the goodness-of-fit criterion, from the diagnostics of the previous version. At the inner level, for each proposed form, a numerical optimizer (optuna with NSGA-II) fits the parameters. Each version predicts burned area from the CRUJRA climate and GPP drivers and is scored against GFED5 under the selected criterion, aggregate or dynamic-range. Iterating produces the model versions C, D, and E, each differing from the one before by a single lever. The native ED-stock fire module (dashed) is the floor the loop improves on and is not one of its products. Model E is checked on held-out years and cells.],
  placement: top,
) <fig:loop>

The loop works on two levels. The functional form sets which mechanism terms enter the model and how they combine, and it is changed between steps by the AI agent, which proposes a new term or a new structure from the residual errors of the current version and keeps it only if it improves the fit. Within a fixed form, the parameters are fit by numerical optimization. We use a multi-objective evolutionary optimizer that samples the parameters on a logarithmic scale and does not reduce the problem to a single number. It trades off two objectives, the goodness-of-fit score and the rate of false-positive fire, and returns a set of non-dominated solutions, subject to a constraint that holds the global burned area within a set band of the observed total.

Two features of the scoring are deliberate. First, the goodness-of-fit criterion is not fixed but is itself an input to the loop, so the same machinery can be pointed at the aggregate ILAMB score or at a criterion that rewards the spatial pattern on the cells that actually burn, and the difference between the two can be measured directly. Second, the chosen versions are checked for overfitting on data they were not fit to. One test holds out time, fitting on 2001 to 2012 and scoring on the unseen 2013 to 2016. The other holds out space, fitting on a checkerboard of ten-degree tiles and scoring on the complementary tiles, blocked so that neighbouring cells cannot leak the answer. The test is passed when skill on the held-out data stays close to skill on the fitted data, which points to genuine structure rather than a memorized fit.

== Model versions and experimental design

The paper builds an ordered sequence of model versions that all share one setup, the same drivers, grid, period, and GFED5 evaluation, so that each improvement can be traced to a single cause. The sequence is anchored at its lower end by a floor baseline, the fire submodule native to the ED model, which we refer to as ED-stock and which is the model this work set out to improve. Above that floor sits the offline fire model, the fire submodule that the automated development loop began from, which we refer to as Model C. From Model C onward, each version changes exactly one of two things, either the functional form or the goodness-of-fit criterion, and holds the other fixed. Because only one of the two moves at each step, the difference in skill between two adjacent versions can be attributed to that lever, so the sequence functions as a controlled comparison rather than a leaderboard.

Model C sets the fire rate of a cell as a product of six bounded mechanism terms, each a function of a single driver and each lying between zero and one, so that any one term can only hold the rate below the level the others permit. Two of the terms act on dryness: an onset term that switches fire on as accumulated dryness climbs past a low threshold and a suppression term that turns it down again where dryness is so extreme that fuel itself runs short. Two act on precipitation, an annual floor that demands enough yearly rainfall to grow fuel at all and a monthly dampening term that suppresses fire in months that are themselves wet. One term acts on productivity, a humped response that peaks at intermediate fuel and falls away where fuel is scarce or so dense that the cell is seldom dry enough to carry fire. The last acts on air temperature, an ignition response that rises with warmth. The product of the six is raised to a shape exponent that sets how sharply the rate responds to its drivers, and the result is read as the annual fire rate of the cell.

With dryness as $D$, annual and monthly precipitation as $P#emph["ann"$ and $P]"mon"$, air temperature as $T$, and productivity as $G$, the annual rate $R$ of a cell is

$
R = [
  sigma(D; k_1, D_"low")
  dot s(D; k_2, D_"high")
  dot frac(P_"ann", P_"ann" + P_"half")
  dot frac(1, 1 + P_"mon" / kappa)
  dot h(a G; b, d)
  dot sigma(T; k_"ign", c_"ign")
]^p
$ <eq:rate>

where

$
sigma(x; k, c) = frac(1, 1 + e^(-k(x - c))),
quad
s(x; k, c) = frac(1, 1 + e^(k(x - c))),
quad
h(x; b, d) = (1 - e^(-x \/ b)) e^(-x \/ d)
$

are a rising sigmoid, its falling counterpart, and a humped response, with $kappa$ the monthly-precipitation scale, $a$ the productivity gain, and $p$ the shape exponent. Because each of the six terms lies between zero and one, so does their product, and a positive shape exponent cannot lift it above one, so in this base form the annual rate is bounded by one. The annual rate is turned into an annual burned fraction by the saturating step $1 - exp(-R)$ and spread across the twelve months to give the monthly fraction the model is scored on. In this form the model carries twelve fitted parameters, and Model C is the version obtained by fitting them against the aggregate goodness-of-fit score, the single number that combines the benchmark's components, which makes it the starting point of the ladder.

Model D keeps the equation of Model C unchanged and alters only the goodness-of-fit criterion the parameters are fit to. The aggregate score that shaped Model C spreads its weight across every cell, so the great majority that carry little or no fire dominate it. Model D replaces that score with a criterion confined to the cells where fire actually occurs, one that rewards a model both for placing fire in the right cells and for reproducing the range of burned fractions those cells reach. The functional form is untouched and only the criterion differs, so the change in skill from Model C to Model D reflects the goodness-of-fit criterion alone.

Model E holds the goodness-of-fit criterion of Model D and changes the functional form. Its central change is that the formula is no longer global. Model E fits the parameters separately for each major fire continent and combines the regional fits into one field, so that savanna, boreal forest, and tropical forest are each governed by their own parameter values. The form is extended at the same time in three further ways. A per-region amplitude, tied to fuel where fuel is the limit on fire, lets the fire rate exceed one, which the bounded product of the earlier form could not reach. A suppression term removes fire from the wet closed-canopy cells of the tropical forests. And the annual rate is spread into months by a per-month disturbance form in place of the even spread used before. All of these are changes to what the model is rather than to how it is judged, so the criterion is held exactly at Model D's, and the step from Model D to Model E isolates the effect of the functional form. This is the continental model.

= Results

== Burned-area evaluation across model versions

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right, right),
    stroke: none,
    inset: (x: 0.55em, y: 0.5em),
    table.hline(stroke: 0.8pt),
    table.header(
      [*Model*], [*BA*], [*×GFED5*], [*Bias*], [*RMSE*], [*Seas.*], [*Spat.*], [*Overall*],
    ),
    table.hline(stroke: 0.5pt),
    [GFED5], [792.9], [], [], [], [], [], [],
    [ED-stock], [2500.3], [3.15], [0.5437], [0.4652], [0.4298], [0.2085], [0.4225],
    [C], [1001.0], [1.26], [0.6977], [0.4754], [0.8246], [0.7691], [0.6485],
    [D], [1218.7], [1.54], [0.6951], [0.4662], [0.7914], [0.7864], [0.6411],
    [E], [815.6], [1.03], [0.7514], [0.4753], [0.7455], [0.8756], [0.6646],
    table.hline(stroke: 0.8pt),
  ),
  caption: [Global burned-area evaluation of the model versions against the GFED5 reference (2001–2016), using ILAMB. Total BA is in Mha yr#super[−1]. ILAMB scores range from 0 to 1, with higher values indicating closer agreement with GFED5.],
) <tab:scores>

The Overall score rises steeply down the ladder, from 0.42 at the ED-stock floor to 0.66 at Model E, and the total annual burned area falls with it, from 2500 Mha per year at the floor to 816 Mha at Model E, or from 3.15 to 1.03 times the observed total of 793 Mha. The four component scores, however, do not all move in step, and no single version is best on every one. Model C has the strongest seasonal cycle, Model D the lowest root-mean-square error, and Model E the best bias and spatial distribution (@fig:maps), and the total burned area does not approach the observed amount steadily but rises further above it at Model D, to 1.54 times observed, before Model E brings it back. The aggregate Overall score therefore reflects the combined effect of these changes, and the contribution of each one is resolved only by comparing the versions in sequence.

#figure(
  image("figures/fig2_ba_maps.jpg", width: 72%),
  caption: [Mean annual burned-area fraction (% yr#super[−1]) over 2001–2016 for GFED5 (observed) and the four model versions, shown on a common colour scale.],
  placement: top,
) <fig:maps>

ED-stock places fire across the Sahara and the Arabian peninsula, where GFED5 records almost none and where there is little vegetation to carry it, so its over-prediction is fire in the wrong places. Models C and D burn diffusely across the Amazon basin, where GFED5 shows only scattered fire, and D burns more widely and more intensely than C, consistent with its higher total of 1.54 times observed. Model E removes most of the Amazon signal and concentrates burning into the African savanna belt and northern Australia, where GFED5 places most of the observed burning, and its differences against GFED5 are the smallest of the four versions. Those differences do not vanish, with fire over-predicted along the Sahel and across boreal Eurasia and under-predicted in the southern African savanna and northern Australia (@fig:diff), and because they are opposite in sign they largely offset within the global sum, so the closeness of the global total to the observed amount overstates how closely the map is reproduced.

#figure(
  image("figures/extfig1_diff.jpg", width: 72%),
  caption: [Difference in mean annual burned fraction between each model and GFED5 (model minus GFED5, percent per year) for Models ED-stock, C, D, and E, on a common diverging scale, 2001 to 2016. Red shows over-prediction and blue under-prediction.],
  placement: top,
) <fig:diff>

== Isolating the effects of functional form and goodness-of-fit criteria

Replacing the aggregate score with a criterion that rewards the spatial pattern on the actively burning cells, at fixed functional form, produces only a marginal change in skill. From Model C to Model D the overall score declines slightly, from 0.65 to 0.64, and the spatial distribution score rises only modestly, from 0.77 to 0.79, while the total burned area diverges further from the observation, increasing from 1.26 to 1.54 times the observed total. The revised criterion concentrates burning in the most active cells, which slightly improves the spatial pattern and raises the global total, but within the functional form of Model C it cannot increase the per-cell burned fraction in the most fire-prone regions, and aggregate skill does not improve. A change in the goodness-of-fit criterion alone, with the functional form held fixed, is therefore insufficient to improve the model.

Changing the functional form instead, with the new criterion held fixed, produces a large improvement. From Model D to Model E, in which the formula is fit separately by continent and its fire rate is allowed to exceed one, the Overall score rises from 0.64 to 0.66, the spatial distribution score rises from 0.79 to 0.88, and the total burned area returns to near the observed magnitude, falling from 1.54 to 1.03 times the observed total. The per-cell relationship improves in step, the slope of modelled against observed burned fraction rising from 0.42 to 0.66 (@fig:scatter). Set against the marginal effect of the criterion change, this large effect of the form change indicates that, across this ladder, the functional form rather than the goodness-of-fit criterion is the binding constraint on burned-area skill. Because no version combines the new form with the original aggregate criterion, the comparison isolates the effect of each lever but does not establish that the new criterion was required for the form change to succeed.

#figure(
  image("figures/fig3_scatter.jpg", width: 65%),
  caption: [Per-grid-cell burned-area fraction of models C, D, and E versus GFED5 on active-fire cells (GFED5 annual $>$ 0.1%), 2001–2016. The dashed line is 1:1, and the solid line the fitted slope.],
  placement: top,
) <fig:scatter>

== Reproducing fire intensity and grid-cell dynamic range

The aggregate and component scores quantify how well a version reproduces the broad geography of fire, but they do not indicate whether it attains the intensity of observed fire in the cells that burn. Evaluated on those cells, the relationship between modelled and observed burned fraction decomposes into a spatial correlation and a ratio of standard deviations, the product of which is the slope of that relationship. A high correlation combined with a low standard-deviation ratio locates fire correctly but underestimates its intensity, reaching the right cells yet not the high burned fractions those cells attain, whereas a slope approaching unity requires agreement in both the location and the magnitude of burning. This decomposition therefore isolates the intensity of simulated fire, which the aggregate scores cannot resolve.

Models C and D reproduce the location of fire but fall well short of its intensity. In both, the spatial correlation is moderate, near 0.5. The standard-deviation ratio stays below one, however, at 0.76 for Model C and 0.85 for Model D, so the slope of the relationship remains near 0.4. The peak per-cell burned fraction reaches only about 0.04 in each, less than half the 0.10 that GFED5 records on the most fire-prone cells. Changing the goodness-of-fit criterion from Model C to Model D raises the standard-deviation ratio slightly but leaves the correlation, the slope, and the peak fraction almost unchanged, consistent with the marginal effect of the criterion change on the aggregate scores.

The functional-form change lifts this limit. In Model E the standard-deviation ratio rises to 0.93, close to one, and the correlation rises to 0.71, so the slope of the relationship increases to 0.66 and the points climb toward the one-to-one line. The peak per-cell burned fraction rises to 0.075, well above the 0.04 of Models C and D. Model E does not, however, close the gap entirely, its peak remaining below the 0.10 that GFED5 reaches on the most fire-prone cells. The intensity limit is therefore substantially reduced but not eliminated. Consistent with the aggregate scores, this improvement is attributable to the change in functional form rather than to the change in goodness-of-fit criterion.

== Robustness to held-out years and regions

Fitting the formula separately for each continent gives Model E many more free parameters than the single global set used by the earlier versions, and this added flexibility could in principle reproduce the fitted data by capturing noise rather than genuine structure. Because this risk applies chiefly to Model E, it is evaluated on data withheld from the fit. In both tests the spatial correlation on the actively burning cells, near 0.70 on the fitted data, falls by only 0.05 on the withheld years and by 0.015 on the withheld cells (@tab:holdout). Model E therefore generalizes across both time and space, and its improvement reflects genuine structure rather than overfitting.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: none,
    inset: (x: 0.55em, y: 0.5em),
    table.hline(stroke: 0.8pt),
    table.header(
      [*Held-out test*], [*r, fit*], [*r, held-out*], [*Δr*],
    ),
    table.hline(stroke: 0.5pt),
    [Unseen years], [0.695], [0.645], [−0.050],
    [Unseen cells], [0.700], [0.685], [−0.015],
    table.hline(stroke: 0.8pt),
  ),
  caption: [Held-out validation of Model E. Spatial correlation #emph[r] on active-fire cells, on the data used to fit versus unseen held-out data ($Delta r =$ held-out − fit). Unseen years: model fit on 2001–2012 and tested on 2013–2016. Unseen cells: blocked 10° spatial cross-validation.],
) <tab:holdout>

== Fire emissions

Fire carbon emissions are the product of the area burned and the carbon released per unit of that area. The carbon released is the fuel present in each burning cell, held in the vegetation and litter pools, multiplied by the fraction of each pool that combustion consumes, a fraction that rises with dryness. These combustion fractions are calibrated against the GFED5 fire carbon product separately for each continent. The emissions of Model E total 3.15 Pg of carbon per year, close to the 3.4 Pg recorded by GFED5, and they score 0.64 against the benchmark, comparable to the model's burned-area score of 0.66. The emissions concentrate in the African savanna belt, the global emissions maximum, which Model E reproduces, while the boreal and South Asian sources are under-represented (@fig:ffire). The burned-area model therefore reproduces fire carbon emissions of realistic magnitude and of comparable skill to the burned area itself.

#figure(
  image("figures/fig4_ffire.jpg", width: 58%),
  caption: [Mean annual fire carbon emissions (g C m#super[−2] yr#super[−1]) for GFED5 and Model E, on a common colour scale, 2001 to 2016.],
  placement: top,
) <fig:ffire>

#figure(
  image("figures/extfig2_seasonal.jpg", width: 70%),
  caption: [Seasonal cycle of burned area for six fire regions, GFED5 against Models C, D, and E. Each curve is the area-weighted mean monthly burned fraction, averaged over 2001 to 2016.],
  placement: top,
) <fig:seasonal>

= Discussion

This result is not specific to the model studied here, and what makes it useful is that it has been isolated. Model development ordinarily changes form and calibration together, so the factor that limits skill is hard to identify. Isolating the two calls for a controlled comparison, of the kind used to separate the contribution of a data-driven cloud-cover scheme from its tuning @grundner-2025, and the version ladder here is one. It shows that within a form unable to express the observed intensity, neither recalibration nor a criterion built to reward that intensity produces it. This substantiates, in a direct experiment, the concern that a model optimized to a benchmark score can be steered toward the average of the field without acquiring the process behaviour the score is taken to represent. For a structurally limited component, effort spent refining the scoring criterion or retuning the parameters yields little, and the productive lever is the functional form. The new criterion was not without value, however, since by rewarding an intensity the form could not produce, it made the structural limit visible and motivated the change of form that followed.

The ceiling has a specific origin in the structure of the base form. Each mechanism term is bounded between zero and one, so their product is too, and raising it to a positive exponent cannot lift it above one. The annual fire rate is therefore capped at one, and the conversion to a monthly burned fraction holds the per-cell fraction near the 0.04 that Models C and D reach, below the 0.10 recorded on the most fire-prone cells of GFED5. No choice of the fitted parameters can exceed this bound, so the intensity limit is a structural property of the model class rather than a shortfall of calibration, and neither retuning nor a change of scoring criterion could close it. The fuel-driven amplitude of Model E removes the bound where fuel is abundant, allowing the rate to exceed one and lifting the ceiling. Model E's peak per-cell fraction still falls short of the observed value, so the amplitude, as formulated, raises the ceiling without reaching it, and the residual gap marks the limit of the present form rather than of the data.

The per-continent fit improves a region only where that region has been fitted in its own right. Boreal Eurasia, which the fit covers, gains the fire that Models C and D had under-represented, though it now overshoots, reaching 2.65 percent of area burned each year against the observed 1.63. Boreal North America is not separately fitted, having been left on the global parameters when its regional fit failed to improve on them, and the model produces almost no fire there, against an observed 0.38 percent per year over a wide area. The improvement is thus real where a region is fitted and absent where the model reverts to the global form. Joining the separately fitted regions carries a second cost. Because each region uses its own parameters up to a hard border, the predicted burned area can change abruptly where two regions meet, leaving straight discontinuities across the map. The fit also corrects the amount of fire more readily than its timing, and in boreal Eurasia and Southeast Asia the modelled season is too broad and peaks in the wrong months, leaving the seasonal cycle the weakest part of the fit (@fig:seasonal). None of these undermines the result, but together they mark the regional model as a first step, to be smoothed at its borders, corrected in its seasonal timing, and extended to the regions still on the global form.

The emissions are a downstream product of the burned-area model rather than a target of it. They follow from combining the burned area with a separate combustion step, so they depend on the carbon available to burn and on how completely it is consumed, not only on where and how much fire occurs. Model E was optimized for burned area alone, and its emissions, though close to the observed total and of comparable skill, inherit both the strengths and the errors of that burned area and add the assumptions of the combustion step, whose per-pool coefficients and dryness dependence are themselves calibrated and uncertain. Emissions consistent with the observations are therefore a check that the burned area carries sensible carbon, not evidence that the model was built to reproduce fire carbon. Targeting fire carbon directly, by optimizing the burned area and the combustion step against the emissions themselves, is a natural next step and would test whether the agreement reached here through burned area can be improved upon.

Beyond this model, the approach itself is what the study tests. An automated loop that changes the functional form and the goodness-of-fit criterion one at a time, fits each version, and scores it against a common reference is a workable way to develop a model component, and here it produced a family of versions whose differences could be traced to single causes rather than to the usual tangle of simultaneous changes. Its limits are equally clear. The loop searches only within the forms and criteria the researcher supplies, so it accelerates and organizes the exploration rather than conducting it autonomously. The choice of goodness-of-fit criterion shapes what the search rewards and therefore what it finds, which is a strength when the criterion is chosen deliberately and a hazard when it is not. A form with more parameters, moreover, can fit noise as readily as structure, so held-out validation is not optional but part of the method. Used this way, with its criteria and search space supplied and its results validated, an automated loop is a means of structuring model development, not a substitute for the judgement that guides it.

= Conclusion

This paper developed a global fire submodule with an automated loop that searches over both the functional form of the model and the criterion by which it is scored, producing a family of versions each differing from the last by one change and all evaluated against GFED5. The controlled comparison gives the central result. The functional form, not the scoring criterion, is the binding constraint on burned-area skill, since changing the criterion alone left the model at its baseline while changing the form raised the Overall score from 0.64 to 0.66, lifted the spatial distribution score from 0.79 to 0.88, and brought the total burned area from over-prediction back to within a few percent of the observed magnitude. Held-out tests in time and space confirm that this gain reflects genuine structure rather than overfitting.

The value of the controlled design is that it both improved the model and identified why, separating the effect of the form from the effect of the criterion rather than leaving the two confounded. Clear limitations remain, in the regions the per-continent fit does not yet cover and in the mapping from burned area to fire carbon, which the model reaches only indirectly. The submodule is developed for the coupled ED land model, and its transfer into that coupled setting, where the fire it predicts feeds back on the vegetation and fuel that drive it, is the natural next step.

#pagebreak()

#bibliography("refs.bib", title: "References", style: "apa")
