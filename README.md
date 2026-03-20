# Autoresearch for ED v3.0

Physically grounded formula replacements for the [Ecosystem Demography model (ED v3.0)](https://gmd.copernicus.org/articles/15/1971/2022/), derived using LLM-driven structural diagnosis and Bayesian optimization against [iLAMB](https://www.ilamb.org/) observational benchmarks.

Inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) framework.

**Blog post:** [Autoresearch for Earth System Models](https://deveshparagiri.com/blog/2026/autoresearch-earth-system-models/)

## What This Is

ED v3.0 is an individual-based terrestrial biosphere model that simulates plant growth, fire, soil carbon decomposition, hydrology, and phenology globally. Many of its parameterized formulas have remained unchanged since the 1990s.

This repository contains replacement formulas for three ED source files, identified by systematically searching over both equation structure and parameters. Every replacement maps to a named physical mechanism with published justification. No neural networks or black boxes.

## What Changed

| Module | File | Before → After | Spatial Correlation |
|--------|------|----------------|-------------------|
| Fire | `fire.cc` | Power-law → sigmoid × fuel hump | 0.09 → 0.65 (vs GFED) |
| Soil carbon temperature | `belowgrnd.cc` | Q10 → Lloyd-Taylor (1994) | 0.43 → 0.48 (vs HWSD) |
| Soil carbon moisture | `belowgrnd.cc` | Piecewise linear → log-parabolic | bias: -50% → 0% |
| Phenology | `phenology.cc` | Cold threshold 10°C → 0°C | 0.76 → 0.80 (vs MODIS LAI) |

See [CHANGES.md](CHANGES.md) for detailed before/after formulas with physical justifications.

## Approach

For each module:

1. Extract the parameterized formula from ED's C++ source
2. Replicate in Python for fast evaluation (~1s per global 0.5° grid)
3. Evaluate against gridded observations (iLAMB benchmarks)
4. Use an LLM to diagnose why the formula structurally disagrees with observations
5. Propose alternative formulas along physical axes (e.g. Q10 vs Lloyd-Taylor vs Arrhenius)
6. Bayesian optimization ([Optuna](https://optuna.org/) TPE) over the joint structure × parameter space
7. Validate with 5-fold spatial cross-validation

Modules are optimized following ED's dependency graph (photosynthesis → growth → soil carbon → fire) since upstream improvements cascade downstream. A joint optimization phase enforces cross-module consistency (e.g. water balance closure: P = ET + R).

Each module uses a tailored multi-metric objective, not just spatial correlation. Soil carbon optimizes for both stock accuracy (HWSD) and flux accuracy (Hashimoto Rh). Hydrology requires water balance closure. GPP requires biome-level skill.

## Repository Structure

```
src/                    Full patched source files (drop-in replacements)
  belowgrnd.cc          Soil carbon decomposition + ET
  fire.cc               Fire disturbance
  phenology.cc          Leaf phenology thresholds
patches/                Unified diffs against ED v3.0 originals
configs/
  optimized_params.json All optimized parameter values with references
results/
  MASTER_RESULTS.md     Cross-module summary
  per_module/           Detailed optimization logs per phase
CHANGES.md              Formula-by-formula changelog with citations
```

## How to Use

Replace the corresponding files in your ED v3.0 source directory:

```bash
cp src/belowgrnd.cc  /path/to/EDv3_code/belowgrnd.cc
cp src/fire.cc       /path/to/EDv3_code/fire.cc
cp src/phenology.cc  /path/to/EDv3_code/phenology.cc
```

Or apply patches to your existing source:

```bash
cd /path/to/EDv3_code
patch -p0 < /path/to/ed-autoresearch/patches/belowgrnd.cc.patch
patch -p0 < /path/to/ed-autoresearch/patches/fire.cc.patch
patch -p0 < /path/to/ed-autoresearch/patches/phenology.cc.patch
```

Recompile and run as usual. No new dependencies or data files required.

## Observation Data Sources

Optimization was performed against the following gridded datasets, all publicly available through [iLAMB](https://www.ilamb.org/datasets.html):

| Dataset | Variable | Source |
|---------|----------|--------|
| [HWSD](https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/harmonized-world-soil-database-v12/en/) | Soil carbon stocks | FAO |
| [NCSCD](https://bolin.su.se/data/ncscd/) | Northern circumpolar soil carbon | Bolin Centre |
| [Hashimoto 2015](https://zenodo.org/records/4708444) | Heterotrophic respiration | Zenodo |
| [GFED4.1s](https://www.globalfiredata.org/) | Burned area | Global Fire Data |
| [GLEAM v3.3a](https://www.gleam.eu/) | Evapotranspiration | GLEAM |
| [MODIS](https://modis.gsfc.nasa.gov/) | LAI, ET | NASA LP DAAC |
| [FLUXCOM](https://www.fluxcom.org/) | GPP | MPI-BGC |
| [LORA](https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f9617_9854_8096_5291) | Runoff | NCI Australia |
| [CRU TS v4.09](https://crudata.uea.ac.uk/cru/data/hrg/) | Temperature, precipitation, PET | UEA CRU |

ED simulation output (Ma et al. 2022) from [Zenodo](https://zenodo.org/records/5765486).

## Limitations

These formulas were optimized against steady-state approximations, not a coupled ED re-run. The actual improvement in iLAMB scores requires patching the C++ source, recompiling, running a 500-year spinup, and re-evaluating. Nonlinear feedbacks between modules may alter results.

The steady-state assumption for soil carbon is particularly questionable at high latitudes where permafrost soils are out of equilibrium. The boreal GPP problem (no single formula captures all biomes) remains partially unresolved.

## References

- Ma, L., et al. (2022). Global evaluation of the Ecosystem Demography model (ED v3.0). GMD 15: 1971-1994.
- Collier, N., et al. (2018). The International Land Model Benchmarking (ILAMB) system. JAMES 10: 2731-2754.
- Lloyd, J. & Taylor, J.A. (1994). On the temperature dependence of soil respiration. Functional Ecology 8: 315-323.
- Moyano, F.E., et al. (2013). Responses of soil heterotrophic respiration to moisture availability. Biogeosciences 10: 3961-3981.
- Pausas, J.G. & Ribeiro, E. (2013). The global fire-productivity relationship. GEB 22: 728-736.
- Jolly, W.M., et al. (2005). A generalized, bioclimatic index to predict foliar phenology. GCB 11: 619-632.
- Zhang, L., et al. (2001). Response of mean annual ET to vegetation changes. WRR 37: 701-708.
- Hargreaves, G.H. & Samani, Z.A. (1985). Reference crop evapotranspiration from temperature. AEA 1: 96-99.
