# Development and Optimization of a Global Fire Model Using Autoresearch AI

**Authors:** Richard Owusu-Ansah, George Hurtt, Lei Ma, Devesh Paragiri, Janna Chapman

Physically grounded formula replacements for **ED v3.0**, derived via LLM-driven **autoresearch**.

## Problem

Fire is a major flux in the global carbon cycle, on the order of a few petagrams of carbon per year, yet the land models used to estimate the terrestrial carbon balance still reproduce it poorly. The Ecosystem Demography model (**ED**), part of the TRENDY land ensemble, is no exception: its stock fire module over-burns and captures little of the observed spatial pattern. Closing that gap matters for any carbon budget that pretends to include fire.

## Approach

This project builds a **closed-form fire formula** for ED, not a neural surrogate. The equation is written in terms of physical drivers (accumulated dryness, precipitation, air temperature, productivity as a fuel proxy) so every term can be read and argued about. Development is **offline**: fixed climate from CRUJRA and GPP from a coupled-ED dump, so each candidate can be scored without a full coupled run.

Improvement is driven by **autoresearch**, an automated loop that at each step changes exactly one thing, either the functional form or the goodness-of-fit criterion, fits the parameters, and evaluates the version against **GFED5** with official **ILAMB**. Because only one lever moves, skill differences between versions can be attributed to a known cause.

## Model ladder

| Version | What changes |
|---|---|
| **ED-stock** | Native ED fire (the floor this work improves on) |
| **Model C** | Global formula, fit to the aggregate ILAMB Overall score |
| **Model D** | Same formula as C; only the fit criterion changes (spatial pattern on cells that burn) |
| **Model E** | Functional form changes: per-continent parameters, fuel-scaled amplitude, and related structure |

**C** gets the broad geography of fire roughly right but under-represents intense regional burning. **D** shows that retargeting the score alone is not enough. **E** raises spatial skill and peak burned fraction and brings global burned area near the GFED5 total. Held-out tests in time and space support genuine structure rather than memorization. A separate combustion step maps burned area to fire carbon emissions consistent with GFED5.

## Reproduce

```bash
conda activate edfire
python scripts/reproduce_paper.py
bash scripts/verify_paper_ilamb.sh
```

Manuscript (Typst): [`paper/`](paper/). Fitted parameters: [`models/paper/`](models/paper/). Setup and scripts: [`docs/`](docs/).
