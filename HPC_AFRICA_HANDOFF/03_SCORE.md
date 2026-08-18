# Scoring what comes back

Two things need scoring, and only the first needs ILAMB.

## The quick check, do this first

Before any benchmarking, get the Africa-total burned area for each run and compare it to the
observation over the same box. Africa is lat -40 to 40, lon -20 to 60. GFED5 over that box is
roughly 496 Mha/yr, which is most of the world's fire.

If run A, at cap 0.2, comes back well under that and run B, at cap 1.0, comes back much
closer, the trip has already answered its question and everything else is confirmation.

```python
# area-weighted total, the only magnitude that means anything.
# the unweighted mean of percent over-counts small high-latitude cells.
R = 6371000.0
dlon = np.deg2rad(abs(lon[1] - lon[0])); h = abs(lat[1] - lat[0]) / 2
a = (R**2) * dlon * (np.sin(np.deg2rad(lat + h)) - np.sin(np.deg2rad(lat - h)))
AREA = np.abs(a)[:, None] * np.ones((1, len(lon)))
mha = (annual_fraction * AREA).sum() / 1e10
```

Check units before doing anything with them. The references are in `%`, ED writes
`fraction month-1`, and a factor of 100 error here has bitten this project before.

## ILAMB, if the run is worth benchmarking

`scripts/prep_coupled_for_ilamb.py` already does the conversion. It slices to the reference
window, renames `latitude`/`longitude` to `lat`/`lon`, multiplies fraction to percent, adds
CF bounds, and prints the Mha total for every file it writes so the conversion can be checked.

**One trap it fixes and you must not undo.** The time axis and the time bounds have to be
written on the same units string. Left alone, xarray picks different epochs for each and
ILAMB then builds 193 edges for 192 months and silently drops the model from the run with a
broadcast error. Both are pinned to `days since <y0>-01-01`.

The scoring recipe itself, from the main repo.

```bash
export ILAMB_ROOT=ilamb_ref_official
OUT="$PWD/ilamb_out_run"; rm -rf "$OUT"; mkdir -p "$OUT"
cp reference/burntArea_gfed5.cfg "$OUT/ilamb.cfg"     # or burntArea_gfed4.cfg
ilamb-run --config "$OUT/ilamb.cfg" --model_root "$PWD/<model dir>" \
  --regions global --build_dir "$OUT"
```

Read scores from `$OUT/scalar_database.csv`, filtering `Region == global` and
`ScalarName == 'Overall Score'`. ILAMB needs its own copy of `ilamb.cfg` inside the build
directory. Remove any stale `burntArea.*.nc` from a model folder first or ILAMB tries to
merge every `.nc` in the directory and raises `MonotonicityError`.

**Note.** ILAMB and the reference data live in the main repo on the external drive, not here.
If the drive is not attached, do the quick check on the HPC and leave ILAMB for later. The
magnitude answer is what this trip is for. The benchmark score can wait.

## The dryness diagnostic, which is the real prize

If `02_RUN_AFRICA.md` step 5 succeeded and ED wrote its live `dryness_index_avg`, this is the
comparison that settles hypothesis H1.

```
Model F     D_high = 2.2134e6   k2 = 0.0342

offline dump, fraction of cells above D_high    0.48 percent
coupled live, fraction of Africa cells above    ?
```

Report the percentiles of the live field, p50, p90, p99 and max, beside the offline dump's,
which are 9.4e3, 2.3e5, 1.0e6 and 4.8e6. If the live field sits an order of magnitude higher,
the suppression term is switching off fire across the continent and the fix is to rescale
`D_low` and `D_high`, not to retune the whole model.

## What to bring back to the external drive

Keep it small. Slice to 2001-2016 before transferring.

```
burntArea for run A and run B          Africa only, monthly
the live driver dump                   dryness_index_avg, GPP, AGB, Africa only, monthly
DISCOVERED.md                          every path, every module, every job script that worked
a note of what changed in the source   so it can be reproduced
```

`DISCOVERED.md` is the one that matters most. Without it the next trip starts from zero.
