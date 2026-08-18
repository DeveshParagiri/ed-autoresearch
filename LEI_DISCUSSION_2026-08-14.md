# Talking points for Lei, in person. Written 2026-08-14

Not an email. This is for a conversation, so it runs in the order the conversation should go.
Every number here was measured this week and is checkable.

---

## 1. What he gave me, and what I did with it

Two coupled ED runs, GCB2026 S3, handed over last week.

```
GCB2026_coupled_model_F_EDv3_S3_burntArea.nc     8.1 GB
GCB2026_coupled_default_EDv3_S3_burntArea.nc     8.1 GB
1700 to 2025, monthly, 0.5 degree, fraction month-1
```

Identical runs except for the fire scheme. One uses our Model F, one uses ED's native fire.
His test was, if F beats the default in the coupled model, the offline development is
working and should continue.

I scored both with **official ILAMB** against **GFED5 and GFED4.1s**, on each reference's own
window and again on a common 2001-2016 window so nothing is explained by a difference in
period.

## 2. His question, answered. Yes

```
official ILAMB, global, 2001-2016

vs GFED5              Model F   default
Bias                   0.6423    0.6221
RMSE                   0.4661    0.4660
Seasonal               0.4858    0.4832
Spatial                0.2161    0.1709
Overall                0.4553    0.4416      +0.0137

vs GFED4.1s           Model F   default
Bias                   0.7142    0.6815
RMSE                   0.4889    0.4888
Seasonal               0.4550    0.4524
Spatial                0.3772    0.2980
Overall                0.5049    0.4819      +0.0230
```

**Twenty component scores, Model F wins all twenty.** No exception on either reference.

Two things to point out.

The gain is concentrated in **spatial**, which is what the offline work targeted. It is
about *where* it burns, not how much.

And the margin is **larger on GFED4.1s**, the reference we did not tune to. That is the
reassuring direction. If it were a tuning artefact the independent reference would show less,
not more.

## 3. Now the awkward part, and it is not our fault

```
GFED5 observed                793 Mha/yr
GFED4.1s observed             467
coupled ED, default fire      177
coupled ED, Model F           168
```

Both coupled runs burn about **a fifth of the observed area**. Offline, the same Model F
burns 793, matching GFED5 exactly.

**The default run under-burns just as badly, so this predates our fire scheme.** That is the
key sentence. It is not something we introduced.

And it caps the whole experiment. Offline the same model gains 0.26 over ED's native scheme.
Coupled it gains 0.014. A model burning a fifth of the world's fire has almost no room to
express a better spatial pattern.

## 4. I found one cause and measured it

`ED_params.defaults.cfg` line 137.

```
fire_max_disturbance_rate = 0.2        the coupled run
FIRE_MAX_RATE             = 5.0        the offline model
```

The transform is `burned_frac_month = 1 - exp(-min(rate, FIRE_MAX)/12)`, so every cell above
the cap is clipped, and savanna cells are exactly the ones above it. Applying each cap to the
offline Model F:

```
cap 5.0  ->  793 Mha/yr     the offline setting, and exactly GFED5
cap 2.0  ->  793
cap 1.0  ->  793
cap 0.5  ->  696
cap 0.2  ->  494            what the coupled run uses
```

**The cap costs 300 Mha, about 38 percent of the gap. Raising it to 1.0 recovers all of it.**

Related and worth saying, **1.3 percent of cells carry 60 percent of the total fire rate**.
Fire is extremely concentrated, which is why clipping the top cells hurts this much.

**Ask him. Can it go to 1.0, and is there a reason it sits at 0.2?**

## 5. The rest of the gap, and my leading suspect

The cap takes 793 to 494. The coupled run gives 168. **Something else costs 326 Mha.**

My suspect is the dryness scale. Model F has a near-step function in it.

```
D_high = 2.2134e6   k2 = 0.0342

D_bar 2.0e6  -> suppression 1.0000     fire fully on
D_bar 2.21e6 -> suppression 0.5501
D_bar 2.3e6  -> suppression 0.0000     fire fully off

offline dump, cells above D_high    0.48 percent
```

Offline that cliff is harmless because almost nothing reaches it. If ED's live
`dryness_index_avg` runs hotter than the dump's, cells fall off it and fire dies.

**We flagged this exact risk ourselves in section 2 of the coupling spec and nobody ever
confirmed it.** Same open question for GPP in section 6.

**Ask him. Can ED write its live `dryness_index_avg`, GPP and AGB alongside burned area,
monthly, 2001-2016?** With that I can drive the offline model on ED's own state and settle
implementation-bug versus driver-shift in an afternoon. Nothing else distinguishes those two
cases, and they need completely different fixes.

## 6. The revelations, which are more interesting than the pass mark

**A. Model F improves by burning less, not more.** 168 against the default's 177, a net loss
of 8 Mha, and it still wins every component. The whole gain is spatial redistribution. It
takes fire out of the Amazon, southern South America, boreal North America and Australia, and
puts it into the African savanna belt, which is where both observations put the world's fire.
The map is `paper_gmd/figures/coupled_ba_maps.png`, panel f.

**B. ED can be wrong by a factor of three in either direction and score the same.** The EDv3
run in the TRENDY leaderboard burns 2500 Mha, over by 3.2 times, and scores 0.4225. The new
coupled default burns 177, under by 4.5 times, and scores 0.4416. One burns the Sahara, the
other barely burns at all, and ILAMB calls them equally good. **That is the thesis of my paper
turning up inside his coupled model.** It is now the fourth independent place it has appeared.

**C. The two observations disagree by 70 percent.** GFED4.1s says 467 Mha, GFED5 says 793.
Same spatial pattern, very different amount. Worth knowing before either is treated as truth,
and it is why scoring against both was worth doing.

**D. Where we would sit on the leaderboard, honestly.** Coupled Model F at 0.4587 ranks about
seventh of nine. Behind CLM6 at 0.6562, ELM-FATES at 0.6502, CLASSIC, CLM-FATES, VISIT and
E3SM. Ahead of JSBACH, SDGVM, and ahead of EDv3 as submitted. Offline the same model scores
0.6783, which would sit above CLM6. **So the leaderboard gap is the magnitude problem, not the
fire scheme.** That is the argument for continuing, and it is specific and testable rather
than a request for faith.

**E. One regression to own before he finds it.** Australia. Model F removes fire there and
both observations show Australia burning heavily.

## 7. What I want out of the conversation

1. `fire_max_disturbance_rate` raised to 1.0, or a reason it cannot be.
2. The live driver dump, `dryness_index_avg`, GPP, AGB, Africa or global, monthly.
3. A spun-up restart I can use, so I can run Africa only and iterate in hours. `restart_dir`
   points at `/gpfs/data1/hurttgp/gel1/leima/AssignTask/gED/Result/`.
4. The GCB2026 S3 run directory, so I copy his configs rather than reconstruct them.

## 8. One thing to tell him, not ask

Model F carries the GDP term and George has ruled GDP out for coupling because it does not
reach 1850. So the version that eventually ships should be the GDP-free one,
`models/C/params.coupledFW.json`, official ILAMB 0.6532, which needs no extra input fields in
ED at all. Diagnose with F because F is what is already implemented, then swap.

---

## Numbers I should not get wrong in the room

| | |
|---|---|
| coupled F vs default, GFED5 | 0.4553 against 0.4416 |
| coupled F vs default, GFED4.1s | 0.5049 against 0.4819 |
| components won | 20 of 20 |
| coupled burned area | F 168, default 177 Mha/yr |
| observed | GFED5 793, GFED4.1s 467 |
| offline Model F | 793 Mha/yr, ILAMB 0.6783 |
| the cap | 0.2 in ED, 5.0 offline, costs 300 Mha |
| unexplained gap | 494 down to 168 |
| the cliff | D_high 2.2134e6, 0.48 percent of offline cells above it |
