# What we already know, 2026-08-14

Everything here was measured, not assumed. Where something is a hypothesis it says so.

## The setup

Lei ran two coupled ED simulations, GCB2026 S3, 1700 to 2025, identical except for the
fire scheme. One uses our **Model F**, one uses ED's **native** fire. We scored both with
official ILAMB against GFED5 and GFED4.1s on a common 2001-2016 window.

## Result 1. Model F beats the default, cleanly

```
official ILAMB, global, 2001-2016

vs GFED5              Model F   default    EDv3
Bias                   0.6423    0.6221   0.5437
RMSE                   0.4661    0.4660   0.4652
Seasonal               0.4858    0.4832   0.4298
Spatial                0.2161    0.1709   0.2085
Overall                0.4553    0.4416   0.4225

vs GFED4.1s           Model F   default    EDv3
Bias                   0.7142    0.6815   0.5315
RMSE                   0.4889    0.4888   0.4872
Seasonal               0.4550    0.4524   0.3956
Spatial                0.3772    0.2980   0.1116
Overall                0.5049    0.4819   0.4026
```

Model F wins all twenty component scores. The offline development transfers. `EDv3` is ED's
submitted TRENDY model and is a **different vintage** with the opposite magnitude error, so
treat that column as context, not proof.

## Result 2. But everything coupled is burning far too little

```
GFED5 observed                793 Mha/yr
GFED4.1s observed             467
EDv3, TRENDY submission      2500        over-burns, wrong in the other direction
coupled ED, default fire      177
coupled ED, Model F           168
```

Both coupled runs burn about a fifth of observed. **Because the default run does it too,
this is upstream of our fire scheme.** That is the thing to diagnose.

## Result 3. The fire cap is confirmed as one cause, and measured

`ED_params.defaults.cfg` line 137 sets

```
fire_max_disturbance_rate = 0.2
```

The offline model runs at 5.0. The annual-to-monthly transform is
`burned_frac_month = 1 - exp(-min(rate, FIRE_MAX) / 12)`, so every cell whose fire rate
exceeds the cap is clipped, and savanna cells are exactly the ones that exceed it.

Running the offline Model F under each cap (`scripts/diag_fire_cap.py`):

```
cap 5.0  ->  793 Mha/yr    the offline setting, and exactly GFED5
cap 2.0  ->  793
cap 1.0  ->  793
cap 0.5  ->  696
cap 0.2  ->  494           what the coupled run uses
```

**So the cap costs 300 Mha, about 38 percent of the gap. Raising it to 1.0 recovers all of
it.** CLAUDE.md in the main repo already recorded that the coupled run needs at least 1.0.

Also worth knowing, **1.3 percent of cells carry 60 percent of the total fire rate**. Fire is
extremely concentrated, which is why capping the top cells hurts so much.

## The gap that is still unexplained

The cap takes 793 down to 494. The coupled run gives 168. **Something else costs another
326 Mha.** Three hypotheses, none tested.

**H1, the dryness scale.** Model F has a near-step function in it.

```
D_high = 2.2134e6   k2 = 0.0342

D_bar 2.0e6  -> suppression 1.0000   fire fully on
D_bar 2.21e6 -> suppression 0.5501
D_bar 2.3e6  -> suppression 0.0000   fire fully off

in the offline dump, only 0.48 percent of cells ever exceed D_high
```

The model goes from full fire to zero fire over about 100,000 units of dryness. Offline that
cliff was harmless because almost nothing reached it. If ED's live `dryness_index_avg` runs
hotter than the dump's, cells fall off the cliff and fire dies. **This is the leading
hypothesis and the reason item 3 in `CLAUDE.md` matters.** We flagged this risk ourselves in
`reference/COUPLING_SPEC_for_Lei.md` section 2 and nobody ever confirmed it.

**H2, the vegetation feedback.** Offline, vegetation is prescribed and fire cannot reduce the
fuel. Coupled, it can. Some damping is expected and correct. Whether it accounts for 326 Mha
is unknown.

**H3, masking.** `fire.cc:33` excludes cropland. The old `fire_suppression` block at
`fire.cc:78-127` applies hardcoded regional floors and is normally off, `ED_params` line 19
sets `fire_suppression = 0`, but the HPC copy may differ. Diff it.

## Confirmed along the way

ED is forced by **CRUJRA v2.4**, from `climate_input_list_TRENDY_S3.txt`, the same product
the offline fire model reads. Only `D_bar` differs, because it is a diagnostic ED computes
rather than a forcing it passes through.

## What Model F is

`reference/params.coupledE_gdp.json`, official ILAMB 0.6783 offline. It carries a GDP human
term which George has since **disqualified** for coupling because GDP does not reach 1850.
`reference/params.coupledFW.json` is the current coupling-legal replacement, 0.6532, no
external fields needed. Model F is what is already in ED, so diagnose with F and swap later.
