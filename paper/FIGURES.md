# Paper figures

All under `paper/figures/`. Rebuild:

```bash
conda activate edfire
python scripts/paper_figures.py
# or full stack:
python scripts/reproduce_paper.py --figures-only
```

| File | Content |
|---|---|
| `fig_maps_ladder.png` | Annual BA: GFED5, ED-stock, C, D, E |
| `fig_diff_ladder.png` | Model − GFED5 for C, D, E |
| `fig_scatter.png` | Per-cell 1:1 on active-fire cells |
| `fig_seasonal.png` | Regional seasonal cycles |
| `fig_ilamb_scores.png` | Official ILAMB bars (last verify run) |

Scores: `official_ilamb_scores.csv`  
Verify: `bash scripts/verify_paper_ilamb.sh`
