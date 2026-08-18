"""Write portable records for an Optuna study executed inside one run."""

from __future__ import annotations

import json
import os
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


def run_local_storage(run_root: Path) -> str:
    """Return a run-owned SQLite storage URL for a local Optuna study."""
    path = (run_root.resolve() / "artifacts" / "optuna-study.db")
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def export_study(
    study: Any,
    run_root: Path,
    *,
    selected_trial_number: int,
    selection_rule: str,
) -> tuple[Path, Path, Path]:
    """Write the portable Optuna evidence required by a recorded search run."""
    artifacts = run_root.resolve() / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    trials = list(study.get_trials(deepcopy=False))
    selected = next(
        (trial for trial in trials if getattr(trial, "number", None) == selected_trial_number),
        None,
    )
    if selected is None:
        raise ValueError(f"selected Optuna trial does not exist: {selected_trial_number}")

    directions = [str(getattr(direction, "name", direction)).lower() for direction in study.directions]
    summary = {
        "schema": "autoresearch-optuna-study/v1",
        "study_name": study.study_name,
        "directions": directions,
        "sampler": study.sampler.__class__.__name__,
        "pruner": study.pruner.__class__.__name__,
        "trial_count": len(trials),
        "selected_trial": selected_trial_number,
        "selection_rule": selection_rule,
        "study_user_attrs": _json_value(getattr(study, "user_attrs", {})),
    }
    study_path = artifacts / "optuna-study.json"
    trials_path = artifacts / "optuna-trials.jsonl"
    parameters_path = artifacts / "selected-parameters.json"
    _write_json(study_path, summary)
    _write_json(
        parameters_path,
        {
            "schema": "autoresearch-selected-parameters/v1",
            "trial": selected_trial_number,
            "selection_rule": selection_rule,
            "values": _json_value(getattr(selected, "values", None)),
            "parameters": _json_value(getattr(selected, "params", {})),
        },
    )
    temporary = trials_path.with_name(f".{trials_path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w") as handle:
        for trial in sorted(trials, key=lambda item: item.number):
            handle.write(json.dumps(_trial_record(trial), sort_keys=True) + "\n")
    os.replace(temporary, trials_path)
    return study_path, trials_path, parameters_path


def _trial_record(trial: Any) -> dict[str, Any]:
    state = getattr(trial, "state", None)
    return {
        "number": trial.number,
        "state": str(getattr(state, "name", state)).lower(),
        "values": _json_value(getattr(trial, "values", None)),
        "parameters": _json_value(getattr(trial, "params", {})),
        "user_attrs": _json_value(getattr(trial, "user_attrs", {})),
        "intermediate_values": _json_value(getattr(trial, "intermediate_values", {})),
        "started_at": _json_value(getattr(trial, "datetime_start", None)),
        "completed_at": _json_value(getattr(trial, "datetime_complete", None)),
        "duration_seconds": _duration_seconds(getattr(trial, "duration", None)),
    }


def _duration_seconds(value: Any) -> float | None:
    return value.total_seconds() if isinstance(value, timedelta) else None


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
