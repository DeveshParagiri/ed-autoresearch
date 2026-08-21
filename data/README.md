# Data

`catalog.toml` gives each scientific asset one stable project path and `sources.toml` records its source, coverage, units, preprocessing, and recovery route. Large files remain outside Git at those canonical paths.

`inputs/` contains climate, ecosystem, coupled-state, human, and candidate-driver inputs. LUH2-GCB2026 is `input.candidate.luh2`. `benchmarks/` contains the fixed GFED and native ED files consumed only by `scripts/evaluate_candidate.py`.

Run `uv run python scripts/check_workspace.py` for a read-only integrity check. Run `bash scripts/install_all_data.sh` for procurement, and pass `--fetch-public` only when a required public input is missing. Restricted, lab-derived, coupled, and TRENDY assets must not be replaced by superficially similar public files.
