# For Lei, when he is back

Add to this as discovery turns up blockers. Send it as one message rather than several.

## The result he asked for, first

His test is answered. Coupled ED with Model F beats coupled ED with the default fire on
**both** GFED5 and GFED4.1s, and on all four ILAMB components in each, twenty out of twenty
with no exception. Overall goes 0.4416 to 0.4553 against GFED5 and 0.4819 to 0.5049 against
GFED4.1s. The offline development transfers.

The margin is larger against GFED4.1s, which is the reference we did **not** tune to, so it
is the closer thing to an independent check.

## The problem that came with it

Both coupled runs burn about 170 Mha/yr. GFED5 says 793. The default run does it too, so it
is not our fire scheme.

We found one cause and measured it. `ED_params.defaults.cfg` line 137 has

```
fire_max_disturbance_rate = 0.2
```

while the offline model runs at 5.0. Applying each cap to the offline Model F gives 793 Mha
at 5.0, 793 at 1.0, and 494 at 0.2. So the cap alone costs 300 Mha and raising it to 1.0
recovers all of it.

**Question 1. Can `fire_max_disturbance_rate` be raised to 1.0 for the fire runs, and was
there a reason it is at 0.2?** If it is holding something else together we need to know.

That still leaves 494 against the coupled 168, so something costs another 326 Mha.

## What would let us find the rest ourselves

**Question 2. Can ED write its live `dryness_index_avg`, GPP and AGB alongside burned area?**
Monthly, 2001-2016 is enough. Our fire equation has a near-step suppression at
`D_high = 2.2134e6` which, offline, only 0.48 percent of cells ever cross. If ED's live
dryness runs hotter than the dump's, that switch is killing fire everywhere, and this is the
risk we flagged ourselves in section 2 of the coupling spec and never confirmed. With that
dump we can drive the offline model on ED's own state and settle it in an afternoon.

**Question 3. Is there a spun-up restart we can use, ideally Africa or global?**
`ED_params.defaults.cfg` points `restart_dir` at
`/gpfs/data1/hurttgp/gel1/leima/AssignTask/gED/Result/`. Richard wants to run Africa only so
he can iterate in hours rather than days. A usable restart is the difference between that and
a cold spin-up.

**Question 4. Which directory holds the GCB2026 S3 run**, so we can copy the exact configs
rather than reconstruct them, and confirm nothing drifted from the copies on the drive.

## Worth telling him, not asking

Model F carries the GDP term, and George has ruled GDP out for coupling because it does not
reach 1850. So whatever we learn from diagnosing F, the version that eventually ships should
be the GDP-free one, `params.coupledFW.json`, official ILAMB 0.6532, which needs no extra
input fields in ED at all. Diagnose with F because F is what is already implemented, then
swap.
