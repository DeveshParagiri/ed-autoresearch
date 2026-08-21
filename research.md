# ED-Fire research

## Goal

Improve the ED fire model through a continuous autoresearch loop: diagnose the existing models and observations, propose mechanisms for new inputs, build candidates, optimize promising parameters, evaluate each candidate, retain or prune mechanisms from the evidence, and repeat.

The existing models are completed prior work. Their retained parameters are in `model/parameters/`, and their previous scores are in `model/registry.toml`.

## Loop

`Existing models + observations → diagnose → propose mechanisms → build and tune candidates → evaluate → retain or prune → repeat`

Explore multiple mechanisms, combinations, and ways of framing the problem. Build every candidate from declared model inputs, then evaluate it with:

```bash
uv run python scripts/evaluate_candidate.py path/to/burntArea.nc --output results/<candidate-name>
```

Use GFED5 Overall as the objective. Use the other metrics to understand why a candidate improved or failed.

## Working memory

Updated: 2026-08-21

The existing models are completed prior work. Tomorrow, start the loop by diagnosing the retained models and proposing new mechanisms for new inputs. Test useful mechanisms alone and in combinations, then use each result to choose what to retain, revise, combine, or drop.

Update this section after every research decision or completed run.

## Results

Append one dated entry after each completed candidate or search. Record the change, result path, main metrics, interpretation, and next step.
