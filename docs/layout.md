# Repository and data layout

```
README.md                 # paper-facing overview
docs/                     # this folder
paper/
  paper.typ               # Typst manuscript
  refs.bib
  figures/                # paper figure panels
  README.md               # how to compile
models/
  paper/                  # C.json, D.json, E/ continent params
  combustion/continental/ # per-continent fFire betas
  formula.md              # base formula notes
scripts/                  # reproduce / score / fit (see scripts.md)
ilamb/                    # ILAMB configs + scoreable NetCDF outputs
data/                     # local drivers (gitignored)
  crujra/
  gfed/4.1/               # GFED4.1s hdf5
  gfed/5/                 # GFED5.1 monthly nc (raw)
```

ILAMB scoring reference for the paper:

`ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc`
