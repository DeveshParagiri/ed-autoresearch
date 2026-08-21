#!/usr/bin/env python3
"""Optimize candidate parameters and replay the best trial."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import optuna

from autoresearch.optuna_records import export_study, run_local_storage
from autoresearch.run_tools import build_command, metric, read_json, run, write_json


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "evaluate_candidate.py"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and evaluate an Optuna study, then replay its best parameters."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate(config: dict[str, Any]) -> None:
    required = {
        "study",
        "mechanism",
        "parameters",
        "objective",
        "direction",
        "seed",
        "trials",
        "build_command",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("missing config fields: " + ", ".join(missing))
    if config["direction"] not in {"maximize", "minimize"}:
        raise ValueError("direction must be maximize or minimize")
    if not isinstance(config["trials"], int) or config["trials"] < 1:
        raise ValueError("trials must be a positive integer")
    if not isinstance(config["parameters"], dict) or not config["parameters"]:
        raise ValueError("parameters must be a nonempty object")
    template = config["build_command"]
    if not isinstance(template, list) or not all(isinstance(item, str) for item in template):
        raise ValueError("build_command must be a JSON array of command arguments")
    joined = " ".join(template)
    if "{parameters}" not in joined or "{candidate}" not in joined:
        raise ValueError("build_command must contain {parameters} and {candidate}")


def suggest(trial: optuna.Trial, space: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, setting in space.items():
        if not isinstance(setting, dict):
            raise ValueError(f"parameter {name} must be an object")
        kind = setting.get("type")
        if kind == "float":
            values[name] = trial.suggest_float(
                name,
                float(setting["low"]),
                float(setting["high"]),
                step=float(setting["step"]) if "step" in setting else None,
                log=bool(setting.get("log", False)),
            )
        elif kind == "int":
            values[name] = trial.suggest_int(
                name,
                int(setting["low"]),
                int(setting["high"]),
                step=int(setting.get("step", 1)),
                log=bool(setting.get("log", False)),
            )
        elif kind == "categorical":
            choices = setting.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"parameter {name} needs choices")
            values[name] = trial.suggest_categorical(name, choices)
        else:
            raise ValueError(f"unsupported parameter type for {name}: {kind!r}")
    return values


def checked_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output == ROOT or ROOT not in output.parents:
        raise ValueError("output must be inside this repository")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def evaluate(
    *,
    parameters: dict[str, Any],
    template: list[str],
    output: Path,
) -> dict[str, Any]:
    settings = output / "parameters.json"
    candidate = output / "candidate.nc"
    evaluation = output / "evaluation"
    write_json(settings, parameters)
    command = build_command(
        template,
        values={"parameters": settings.resolve(), "candidate": candidate.resolve()},
    )
    run(command, cwd=ROOT, logs=output / "logs" / "build")
    if not candidate.is_file():
        raise FileNotFoundError(f"builder did not write {candidate}")
    run(
        [sys.executable, str(EVALUATOR), str(candidate), "--output", str(evaluation)],
        cwd=ROOT,
        logs=output / "logs" / "evaluate",
    )
    return read_json(evaluation / "metrics.json")


def main() -> int:
    args = arguments()
    config = read_json(args.config.expanduser().resolve())
    validate(config)
    output = checked_output(args.output)
    write_json(output / "config.json", config)

    study = optuna.create_study(
        study_name=str(config["study"]),
        storage=run_local_storage(output),
        direction=str(config["direction"]),
        sampler=optuna.samplers.TPESampler(seed=int(config["seed"])),
    )

    def objective(trial: optuna.Trial) -> float:
        trial_output = output / "trials" / f"{trial.number:06d}"
        parameters = suggest(trial, config["parameters"])
        try:
            metrics = evaluate(
                parameters=parameters,
                template=config["build_command"],
                output=trial_output,
            )
            trial.set_user_attr(
                "metrics",
                str((trial_output / "evaluation" / "metrics.json").relative_to(output)),
            )
            return metric(metrics, str(config["objective"]))
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            raise

    study.optimize(objective, n_trials=config["trials"], catch=(Exception,))
    completed = [trial for trial in study.trials if trial.value is not None]
    if not completed:
        raise RuntimeError("no trial completed successfully")

    best = study.best_trial
    export_study(
        study,
        output,
        selected_trial_number=best.number,
        selection_rule=f"best completed trial by {config['direction']} {config['objective']}",
    )
    winner_metrics = evaluate(
        parameters=dict(best.params),
        template=config["build_command"],
        output=output / "winner",
    )
    replay_value = metric(winner_metrics, str(config["objective"]))
    write_json(
        output / "winner.json",
        {
            "schema": "ed-fire-optuna-winner/v1",
            "trial": best.number,
            "parameters": best.params,
            "objective": config["objective"],
            "study_value": best.value,
            "replay_value": replay_value,
            "reproduced": math.isclose(replay_value, float(best.value), abs_tol=1e-12),
        },
    )
    shutil.copy2(output / "winner" / "candidate.nc", output / "selected-candidate.nc")
    print(f"result={output / 'winner.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        raise
