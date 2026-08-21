# ED-Fire

This repository develops and evaluates improved fire mechanisms for Ecosystem Demography. Start with [`research.md`](research.md); it is the only research-control document.

The retained parameter sets for the existing models live in `model/parameters/`, with their minimal mapping in `model/registry.toml`. Candidate inputs and benchmarks live under `data/`. The sole evaluation contract is `evals/burned-area.json`, and the sole evaluation entry point is `scripts/evaluate_candidate.py`.

Evaluate a candidate with:

```bash
uv run python scripts/evaluate_candidate.py path/to/burntArea.nc --output results/<run-name>
```

The candidate must provide monthly `burntArea` fractions on the global 0.5-degree grid for 2001–2016. The command runs the fixed ILAMB suite against GFED5 and GFED4.1s and writes metrics, evaluator records, the candidate artifact, logs, and figures under the requested output directory.

`scripts/run_optuna.py` runs a declared parameter search and replays the best trial. `scripts/run_shapley.py` evaluates every subset of a small set of mechanisms and writes exact grouped Shapley values plus drop-one effects. Both call the same evaluator above.

The local research viewer lives in `viewer/`. Run `pnpm dev` there and open `http://127.0.0.1:4173`; it reads the current `research.md` directly.

Run the lightweight repository checks with:

```bash
uv run python scripts/check_workspace.py
```

Public and project-local data identities remain in `data/catalog.toml` and `data/sources.toml`. Data retrieval is handled by `scripts/install_all_data.sh`; pass `--fetch-public` only when a declared input is absent.
