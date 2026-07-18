# ED offline fire submodule (paper)

Closed-form fire model for ED, developed with an autoresearch loop (form or scoring criterion changes one at a time), fit offline, scored against **GFED5** with **official ILAMB**.

Draft: `paper/paper.pdf`  
Figures: `paper/figures/`  
Params: `models/paper/` (see `models/paper/README.md` to drop colleague files for bit-exact Table 1)

## Model ladder

| Version | What | Params |
|---|---|---|
| **ED-stock** | Native ED fire (floor) | TRENDY BA product |
| **Model C** | Global formula, aggregate ILAMB fit | `models/paper/C.json` |
| **Model D** | Same form as C, spatial / active-fire criterion | `models/paper/D.json` |
| **Model E** | Per-continent + fuel amplitude + seasonal transform | `models/paper/E/` |

## Reproduce

```bash
conda activate edfire   # conda create -n edfire -c conda-forge python=3.10 ilamb xarray netcdf4 numpy matplotlib h5py pandas cftime cartopy optuna

# Drivers (not in git): data/crujra/*.npy, global_baseline_modelC_inputs_1997-2016.nc,
#   global_baseline_modelCfuel_inputs_1997-2016.nc (for fFire),
#   data/gfed/4.1/GFED4.1s_YYYY.hdf5,
#   ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc (+ fFire if scoring emissions)

python scripts/reproduce_paper.py          # BA C/D/E + fFire + figures
bash scripts/verify_paper_ilamb.sh         # official ilamb-run → paper/official_ilamb_scores.csv
```

### Outputs

| Path | Content |
|---|---|
| `ilamb/MODELS/paper/Model-{C,D,E}/burntArea.nc` | Burned area |
| `ilamb/MODELS/paper/Model-E-fFire/fFire.nc` | Fire carbon (if fuel + cSoil present) |
| `paper/figures/*.png` | Maps, scatter, seasonal, score bars |
| `paper/official_ilamb_scores.csv` | Official ILAMB Overall + components |

## Data layout

```
data/gfed/4.1/   GFED4.1s hdf5 (land / fire mask for offline run)
data/gfed/5/     GFED5.1 monthly nc (raw)
data/crujra/     climate drivers
```

ILAMB scoring reference: `ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc`

## Repo map

```
README.md
paper/                 draft PDF, figures, official score CSV
models/paper/          C.json, D.json, E/  (+ README for colleague drop-in)
models/combustion/continental/   per-continent fFire betas
models/formula.md
patches/fire_modelC.cc
scripts/
  reproduce_paper.py           one-shot paper rebuild
  reproduce_modelC.py          single param set → burntArea.nc
  assemble_continental.py      Model E stitch
  assemble_combustion_continental.py
  paper_figures.py
  verify_paper_ilamb.sh        official ilamb-run
  optimize_modelC_coupled.py   re-fit (lab)
  validate_holdout*.py
```

## Drivers

- **Climate** (dryness, T, precip): CRUJRA  
- **GPP**: Lei coupled ED dump  
- Compute 1°, score 0.5°, years 2001–2016  

## ED coupling

Offline rate → ED monthly disturbance transform. Coupled transfer still needs rate cap, monthly patches, and region tags for Model E. See `patches/fire_modelC.cc`.
