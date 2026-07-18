# Paper model params

Drop files here to reproduce the paper ladder. Expected layout:

| File | Paper version | Notes |
|---|---|---|
| `C.json` | Model C | Global formula, aggregate ILAMB fit |
| `D.json` | Model D | Same form as C, **spatial / active-fire objective only** (no `fire_amp` / `fuel_k`) |
| `E/assembly.json` | Model E region map | Which continent JSON + fallback |
| `E/*.json` | Per-continent params | Africa (fuel form), Boreal, S.America, SEAsia, Europe, `fallback.json` |

## After you get params from a colleague

```bash
# replace C.json / D.json / E/* as needed, then:
conda activate edfire
python scripts/reproduce_paper.py
bash scripts/verify_paper_ilamb.sh
```

Official scores land in `paper/official_ilamb_scores.csv` and `ilamb_out_paper_verify/`.

## Current status

- **C.json** — present (global, no amplitude). Used for Model-C BA.
- **D.json** — provisional (may still be the amp-enabled spatial-k1 product). Refit may update `models/work/params.paperD.json`; promote into `D.json` only after official ILAMB is close to Table 1, or replace with colleague file.
- **E/** — pre-seasonal continental assembly for Table 1 style Model E.
