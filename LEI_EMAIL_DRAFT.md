# Draft email to Lei (lma6@umd.edu) — not yet sent

Purpose: get the coupled-side answers we need before any of the continental Model C
work can be promoted into the coupled ED run. (Phrased as direct questions; Lei wrote
the ED code, so no need to walk him through it.)

---

**To:** Lei (lma6@umd.edu)
**Subject:** Model C offline results + a few coupled-run questions

Hi Lei,

Quick update and then a few questions for you on the coupled side.

Offline against GFED5 we made good progress on the spatial pattern George has been asking about. The main move was fitting the fire parameters per continent and adding a fuel-driven term for savanna. The per-grid-cell match to GFED5 improved a lot, the global burned-area total is now about 1.03 times GFED5, and the official ILAMB burned-area score came up to around 0.67. It also held up on a held-out-years test, so it is not just overfitting.

To plan how this carries into the coupled run, a few questions for you.

1. Is the coupled run handling fire monthly now, or still annually. Our offline version concentrates fire into the dry-season months, so the monthly timing matters.

2. What is the current cap on the fire disturbance rate, and can it go higher. Savanna needs a higher rate than we think the current setting allows, and I want to know what has been stable for you.

3. Can the coupled run use different fire parameters per continent in a single global run, or is it one global parameter set right now. Our result depends on regional parameters.

4. You mentioned the coupled global burn came out about two times high and you refit the parameters. Could you share what you changed and the current coupled fire setup. We expect the parameters to need recalibrating once the live GPP and fuel feedback is in, so it would help to start from your coupled baseline.

Happy to hop on a call if that is easier, and I can send the per-continent parameters and figures whenever useful.

Thanks,
Richard

---

Context for the 4 questions (from reading ED_Source_Code/GlobalED; for Richard, not for the email):
1. Monthly timing -> ED's PATCH_FREQ (edmodels.h:134). Our SEASONAL_TRANSFORM assumes monthly accumulation.
2. The cap -> `fire_max_disturbance_rate` (default 0.2 in ED_params.defaults.cfg). Needs raising to reach savanna burned fractions.
3. Per-continent -> ED branches fire by `data->region` (one region per run); a global run needs per-site region/zone tagging.
4. The 2x over-burn -> the GPP/biomass feedback; we need his coupled baseline to recalibrate against.
