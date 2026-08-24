# ED-Fire

ED-Fire is an autoresearch experiment for improving the fire module in the [Ecosystem Demography model](https://gel.umd.edu/ed.php). Global fire models still struggle to reproduce where fire occurs, when it peaks, and how much land burns. This project tests whether an LLM-driven research loop can improve those mechanisms with interpretable equations rather than a black-box predictor. More context is available on the [Exaforge Earth System Models project page](https://exaforgelabs.com/research/projects/earth-system-models/).

The loop edits one scientific artifact, [`autoresearch/model.py`](autoresearch/model.py). It can change how climate, vegetation, land use, population, and lightning combine to produce monthly burned area, fit the resulting coefficients with Optuna, and evaluate each committed formulation against GFED5 with ILAMB. Every official experiment records global and 14 regional scores in [`autoresearch/results.tsv`](autoresearch/results.tsv).

## Run the loop

Start inside the agent's working directory:

```sh
cd autoresearch
uv run ar list
```

[`autoresearch/research.md`](autoresearch/research.md) contains the research protocol and `ar --help` documents the available commands. Prepared inputs live beside the model. The host-side implementations live in [`scripts/`](scripts/), and the GFED5 benchmark lives in `evals/`; neither belongs to the loop's normal reading context.

`ar evaluate` updates the single external `progress.png` automatically after recording an experiment. Run `uv run python scripts/progress.py` from the repository root only to rebuild the graph manually from the ledger.

The repository does not retain model copies, experiment directories, diagnostic figures, candidate NetCDF files, or ILAMB output trees. A recorded model is recovered from the Git commit stored in `results.tsv`.
