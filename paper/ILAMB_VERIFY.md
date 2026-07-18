# Official ILAMB check

Last run used `ilamb-run` (ILAMB 2.7, GFED5, global) via `bash scripts/verify_paper_ilamb.sh`.

Raw scalars: `../ilamb_out_paper_verify/scalar_database.csv`  
Summary CSV: `official_ilamb_scores.csv`

Replace `models/paper/{C,D}.json` and `models/paper/E/*` with colleague files, rebuild with `python scripts/reproduce_paper.py`, then re-run the verify script for Table 1 lock-in.
