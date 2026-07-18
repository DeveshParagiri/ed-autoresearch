# Development and Optimization of a Global Fire Model Using Autoresearch AI

**Authors:** Richard Owusu-Ansah, George Hurtt, Lei Ma, Devesh Paragiri, Janna Chapman

Physically grounded formula replacements for **ED v3.0**, derived via LLM-driven **autoresearch**.

## What this is

The Ecosystem Demography model (**ED**) is a dynamic global vegetation model used in the TRENDY land-carbon ensemble. Its stock fire module over-burns and poorly matches satellite burned area. This repo is the offline development and evaluation stack for a replacement fire submodule: a closed-form formula in climate and productivity drivers that can be scored against observations without running full coupled ED at every trial.

We do **not** train a black-box neural fire scheme. The product is an interpretable equation (dryness onset and suppression, precipitation controls, GPP fuel response, temperature ignition, and later regional structure). Parameters are fit by numerical optimization; the **form** and the **scoring criterion** are explored by an automated, AI-assisted loop we call autoresearch.

Each loop step changes one lever only, either functional form or goodness-of-fit criterion, fits parameters, and evaluates against the **fifth Global Fire Emissions Database (GFED5)** with the official **ILAMB** benchmarking system. Adjacent versions therefore differ by a known cause.

The work is **offline** first (fixed drivers from CRUJRA climate and a coupled-ED GPP dump) so structure and skill can be diagnosed cleanly. Transfer into live ED is the intended next step; a C++ drop-in sketch lives under `patches/`.

## What the paper shows

**Model C** is the base global formula fit to ILAMB’s aggregate Overall score. It gets the broad geography of fire roughly right but under-represents intense regional burning. **Model D** keeps that formula and changes only the fit criterion (spatial pattern on cells that actually burn). Aggregate skill barely moves. **Model E** changes the form (per-continent parameters, fuel-scaled amplitude, and related structure). Spatial skill and peak burned fraction rise substantially, and global burned area approaches the GFED5 total. Held-out tests in years and space support genuine structure rather than overfitting. A separate combustion step maps burned area to fire carbon emissions consistent with GFED5.

| Version | Role |
|---|---|
| ED-stock | Native ED fire (floor) |
| Model C | Global formula, aggregate ILAMB fit |
| Model D | Same form as C, spatial / active-fire criterion |
| Model E | Form change: continental + fuel amplitude |

Draft: [`paper/paper.pdf`](paper/paper.pdf) · figures: [`paper/figures/`](paper/figures/) · fitted params: [`models/paper/`](models/paper/)

## What’s in the repo

| Path | Purpose |
|---|---|
| `models/paper/` | Parameter files for Models C, D, and E (colleague drop-in: see that folder’s README) |
| `models/combustion/` | Combustion betas that turn burned area into fFire |
| `models/formula.md` | Base formula documentation |
| `scripts/` | Offline run, assembly, ILAMB verify, optimization, figures |
| `paper/` | Working draft PDF, generated figures, official score CSV |
| **`patches/`** | **C++ source meant to replace ED’s native fire module** (`fire_modelC.cc`): same Model C formula, written against ED site state (`dryness_index_avg`, temp, precip, GPP). Not a full ED tree. For coupling notes and caveats (especially D̄), see [`docs/ed-coupling.md`](docs/ed-coupling.md). |
| `ilamb/` | ILAMB configs and scoreable NetCDF outputs |
| `docs/` | Setup, data layout, methods, scripts index |

## Reproduce

```bash
conda activate edfire

python scripts/reproduce_paper.py      # burned area C/D/E, emissions, figures
bash scripts/verify_paper_ilamb.sh     # official ILAMB vs GFED5
```

Environment, data paths, drivers, and full script list: [`docs/`](docs/).  
Bit-exact Table 1 when colleague params arrive: [`models/paper/README.md`](models/paper/README.md).
