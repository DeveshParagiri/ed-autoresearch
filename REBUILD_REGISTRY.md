# REBUILD REGISTRY — how every scored version is regenerated, and whether that was checked

Written 2026-08-24. **Every version in the paper's burned-area ladder has been rebuilt from what is
in this repository and compared against the file that was actually scored.** The result of that
comparison is in the table. Nothing here is asserted from a log or a note.

## Why this file exists

Dev rebuilt Model H on 2026-08-22, scored 0.5636 through official ILAMB, and asked whether our
recorded 0.6819 was really an optimizer internal score. It was not. Official ILAMB returns 0.681896
for Model H, today and in `paper_gmd/scoring/ba_final/`.

But he could not have got the right answer. `scripts/reproduce_modelC.py` implements Model C and has
no GDP term, so pointing it at `params.H.json` reads `gdp_gamma`, prints it in its own log line, and
never uses it. The result is Model C driven by H's parameters, scoring 0.5582, with no error raised.
Ten env vars change the model equation and none of them were recorded in the params files, so a
params file did not define a model. Run scripts existed for two versions out of ten.

That is now fixed in three places: `scripts/model_mechanisms.py` holds one copy of every mechanism
term, the optimizer stamps an `environment` block into every params file it writes, and
`scripts/reproduce_model.py` rebuilds from that stamp and refuses to guess when it is absent.

## What "verified" means here

Rebuild the version from this repo, then compare the resulting array against the `.nc` that was
scored, cell by cell, month by month.

- **exact** — every value identical. The rebuild is the same file.
- **1e-8** — differs only at float32 rounding, six orders of magnitude below the smallest value that
  matters. Model C was carried through official ILAMB to confirm this is not a real difference: the
  rebuild scores 0.648445 against the canonical 0.648453, apart at the fifth decimal, and the paper
  reports 0.6485 either way.

| Version | Rebuilt by | Verified | Notes |
|---|---|---|---|
| ED-stock | not ours | **n/a** | Stock ED output, not produced by this repo. Nothing to rebuild. |
| C | `reproduce_model.py --params models/C/params.nsga2.json` | **1e-8** | Also confirmed through official ILAMB, 0.648445 vs 0.648453. |
| D | `reproduce_model.py --params models/C/params.paperD.json` | **exact** | Internal Overall reproduces at 0.5908. |
| E (E-clean) | `ASSEMBLY=clean SEASONAL_TRANSFORM=1 assemble_continental.py` | **1e-8** | Fallback `params.spatial.k1.json`. |
| F | `add_gdp_regional.py` | **1e-8** | Coordinate descent reconverges to the saved gamma exactly, Africa 1.60 Boreal 0.50 S.America 0.30 SEAsia 0.10 Europe 0.70 N.America 0.60 Australia 0.00. |
| G | `SEASONAL_TRANSFORM=0 ASSEMBLY=G ASSEMBLE_FALLBACK=params.nsga2.json` | **exact** | |
| G6 | as G, `ASSEMBLY=G6` | **exact** | |
| G7 | as G, `ASSEMBLY=G7` | **exact** | The best version in the paper. |
| H | `reproduce_model.py --params models/C/params.H.json` | **exact** | md5 `b162c7e2280b8222b41e1f41b1d8e995`, identical file, so official 0.681896 by construction. |
| I | as G, `ASSEMBLY=I` | **exact** | |
| Ibest | as G, `ASSEMBLY=Ibest` | **exact** | |

Assemblies take their output directory from `ASSEMBLE_OUTDIR`. Use it. The default writes into
`ilamb/MODELS_CONTINENTAL*`, and `reproduce_modelC.py` writes into the canonical
`ilamb/MODELS/ED-ModelC-final/`, which is how that folder came to hold sixty stale `burntArea.*.nc`
files and is the MonotonicityError hazard `CLAUDE.md` warns about.

## THE PAPER'S MODEL C IS NOT THE SHIPPED MODEL C

`paper_gmd/models/C/params.json` matches **`models/C/params.nsga2.json`**, the 12-parameter
pre-tropfix2 NSGA-II fit, official ILAMB 0.6485.

`models/C/params.json`, the shipped canonical, is the 14-parameter tropfix2 k4 refit, official ILAMB
0.6473, and it is a **different model**. It adds the gated tropical closed-canopy suppression that
cut the magnitude from 1.26x to 1.11x GFED5.

Both are correct in their own place. The paper ladder is built on the 12-parameter C so that C to G
differs in exactly one attribute, and G's unfitted cells fall back to those same parameters. Anyone
rebuilding "Model C" from `models/C/params.json` and comparing against the paper will be 0.0012 out
and will not know why. The stamps in both files now say which is which.

## What is stamped

84 params files carry an `environment` block and an `environment_note` naming the source of the
flags and the verification result.

- `params.H.*` — from `scripts/run_modelH.sh`, `GDP_TERM=1 TROP_MASK=0`.
- `params.nsga2.json`, `params.PRE-tropfix2.json` — paper Model C, no mechanism terms.
- `params.paperD.*` — paper Model D, no mechanism terms.
- `params.G_*` — from `scripts/run_modelG.sh`, `TROP_MASK=0` plus the region.
- `params.Gtrop_*` — from `scripts/run_Gtrop_rest.sh`, `TROP_MASK=1` plus the region.

The per-region files are **inputs to an assembly, not global models**. Rebuilding one on its own
gives that region's parameters applied worldwide, which is not any version in the paper. The
reproducible unit for G, G6, G7, I and Ibest is the assembly command in the table above.

Files fitted before the stamp existed and not listed here still have none. `reproduce_model.py`
refuses to run on them rather than assume Model C, which is the whole point.

## What has NOT been done

- **The fFire (emissions) versions are not covered.** This registry is burned area only. The
  combustion step has its own betas and its own scripts and has not been through this exercise.
- **The E-clean region fits are not individually stamped.** Their environments are recorded in prose
  in `paper_gmd/models/E-clean/PROVENANCE.md`, and the assembly is verified, but the individual
  files would need the same treatment as the G ones.
- **`ilamb/MODELS/ED-ModelC-final/` has not been cleaned.** Sixty-odd stale outputs still sit beside
  the canonical `burntArea.nc`.
