# CLAUDE.md - Africa coupled ED run, HPC session brief

You are on Richard's **work machine**, which has access to the UMD group HPC. The main
project repo lives on an external drive that is **not here**. This folder is the whole
context you get. Read this file, then `00_FINDINGS.md`, then work through `01_DISCOVERY.md`.

Richard is Richard Owusu-Ansah, `rowusuan@terpmail.umd.edu`, PhD student in George Hurtt's
group. He built the fire submodel. Lei Ma (`lma6@umd.edu`) runs coupled ED and is
**unavailable right now**, so everything here has to be discovered rather than asked for.

## The one question this trip answers

Coupled ED burns about **170 Mha/yr**. The observations say **793**. Offline, the exact same
fire model burns 793. Something between the offline model and the coupled model is
destroying three quarters of the world's fire, and we do not know what.

We already found one culprit and measured it (see `00_FINDINGS.md`), but it only accounts
for part of the gap. The rest is unknown.

**A global coupled run is far too slow to iterate on. An Africa-only run is not.** Africa
holds most of the world's fire and is where our fire scheme's changes concentrate. So the
goal is a working Africa-only coupled ED run that Richard controls, fast enough to change
one thing and see the answer the same day.

## What success looks like, in order

1. An Africa-only ED run that completes and writes burned area. Nothing else matters until
   this works.
2. That run repeated at `fire_max_disturbance_rate` = 0.2 and 1.0. If burned area rises the
   way `00_FINDINGS.md` predicts, the cap is confirmed as a real cause in the coupled model.
3. A dump of ED's **live** `dryness_index_avg`, GPP and AGB from that run, so the offline
   model can be driven with them and the remaining gap diagnosed.

Item 3 is the most scientifically valuable and the one nobody has ever done.

## Ground rules

- **Do not modify anything under Lei's directories.** Copy out, never write in. His paths
  appear in `reference/climate_input_list_TRENDY_S3.txt` and in `ED_params.defaults.cfg`.
  Everything we do goes in Richard's own scratch or home space.
- **Do not start a global run.** If a job looks like it will take more than a few hours,
  stop and reconsider the domain.
- **Write down every path you discover** into `DISCOVERED.md` as you go. Richard is moving
  between machines and that file is the only thing that survives.
- Richard has said he does not know the HPC well. Explain what you are doing and why, and
  do not assume he knows the scheduler, the module system, or the queue names.
- **Commit nothing to git from here.** This folder is a copy, not the repo.

## Writing style for anything you author

No em-dashes, en-dashes, semicolons or colons in body prose. Plain sentences.

## What is in this folder

| Path | What |
|---|---|
| `00_FINDINGS.md` | What we already know and proved, with numbers. Read before acting. |
| `01_DISCOVERY.md` | First hour on the HPC. Find the machine, the data, the restarts. |
| `02_RUN_AFRICA.md` | Build, configure the Africa domain, launch. |
| `03_SCORE.md` | Bring the output back and score it against GFED5 and GFED4. |
| `QUESTIONS_FOR_LEI.md` | Everything that needs him, so Richard can send one message. |
| `reference/` | The ED configs, the Makefile, Model F's parameters, the coupling spec. |
| `scripts/` | The diagnostic and scoring scripts from the main repo. |

`reference/` files are **copies for reading**, taken from `ED_Source_Code/GlobalED/` in the
main repo. The real source on the HPC may have drifted from them, so diff before trusting.
