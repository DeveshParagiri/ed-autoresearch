# Scripts

| Script | Role |
|---|---|
| `reproduce_paper.py` | One-shot rebuild: BA C/D/E, fFire, figures |
| `reproduce_modelC.py` | Single param JSON → `burntArea.nc` |
| `assemble_continental.py` | Model E per-continent stitch |
| `assemble_combustion_continental.py` | Per-continent combustion → fFire |
| `paper_figures.py` | Write `paper/figures/` |
| `verify_paper_ilamb.sh` | Official `ilamb-run` vs GFED5 |
| `optimize_modelC_coupled.py` | Parameter re-fit (lab) |
| `validate_holdout.py` | Held-out years |
| `validate_holdout_cells.py` | Held-out spatial tiles |
| `download_inputs.sh` | Optional pinned input bundle |
| `prep_monthly_inputs.py` | Raw climate → monthly npy |
| `tune_combustion_*.py` | Retune combustion betas |
| `score_spatial.py` | Per-cell r / σ / slope diagnostics |

Params for the paper versions: `models/paper/` (see that folder’s README for colleague drop-in).
