# Development and Optimization of a Global Fire Model Using Autoresearch AI

**Authors:** Richard Owusu-Ansah, George Hurtt, Lei Ma, Devesh Paragiri, Janna Chapman

This repository holds the offline fire submodule for the Ecosystem Demography (ED) vegetation model and the code used to develop and evaluate it for the paper above.

Fire is a large term in the land carbon budget, yet ED and other TRENDY-class models reproduce satellite burned area poorly. Here we build a closed-form fire model and improve it with an automated, AI-assisted loop (**autoresearch**). At each step the loop changes exactly one lever, either the functional form or the goodness-of-fit criterion, fits the parameters, and scores the result against **GFED5** with official **ILAMB**. Comparing adjacent versions attributes each skill change to a single cause.

## What the paper shows

The base version (**Model C**) is a single global formula fit to the aggregate ILAMB Overall score. It captures the broad geography of fire but under-represents intense regional burning. **Model D** keeps that formula and changes only the scoring criterion (spatial pattern on cells that burn). Skill barely moves. **Model E** changes the functional form (per-continent parameters, fuel-scaled amplitude, and related structure). Spatial skill and peak burned fraction rise substantially, and global burned area approaches the GFED5 total. Held-out tests in time and space support genuine structure rather than overfitting. A separate combustion step turns burned area into fire carbon emissions consistent with GFED5.

| Version | Role |
|---|---|
| ED-stock | Native ED fire (floor) |
| Model C | Global formula, aggregate ILAMB fit |
| Model D | Same form as C, spatial / active-fire criterion |
| Model E | Form change: continental + fuel amplitude |

Draft: [`paper/paper.pdf`](paper/paper.pdf) · figures: [`paper/figures/`](paper/figures/) · params: [`models/paper/`](models/paper/)

## Reproduce

```bash
conda activate edfire

python scripts/reproduce_paper.py      # burned area C/D/E, emissions, figures
bash scripts/verify_paper_ilamb.sh     # official ILAMB vs GFED5
```

Environment and data setup, directory layout, drivers, ED coupling notes, and script index live under [`docs/`](docs/).

To swap in colleague parameter files for bit-exact Table 1, see [`models/paper/README.md`](models/paper/README.md).
