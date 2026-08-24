# ED-Fire

The research loop runs from [`autoresearch/`](autoresearch/). That directory contains its complete working context: one editable scientific artifact, its instructions, its official result ledger, and its prepared inputs.

The loop has one discoverable command surface. From inside `autoresearch/`, `uv run ar --help` shows every command and `uv run ar list` lists the tools available to the model. Their implementations and the external installer and progress publisher live in [`scripts/`](scripts/). The GFED5 benchmark lives in [`evals/`](evals/). Neither outer directory belongs to the loop's normal working context.

Every official evaluation overwrites the single external `progress.png`. The graph shows the running-best three-decimal GFED5 Overall against the number of official experiments. Run `uv run python scripts/progress.py` only when you want to recreate it manually from the complete ledger.

No model variants, experiment directories, persisted diagnostic figures, candidate NetCDF files, or ILAMB output trees are retained.
