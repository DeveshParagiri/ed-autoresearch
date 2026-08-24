"""Implementation of ``ar optuna``.

This tool will load ``autoresearch/model.py``, search its declared parameter
space, and score every trial with ``fast_ilamb``. It will not record trials as
experiments or expose the benchmark to the research workspace.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
from collections.abc import Mapping
from typing import Any

import optuna

from scripts.fast_ilamb import GFED5Evaluator
from scripts.runtime import (
    GFED5_PATH,
    ModelError,
    load_inputs,
    load_model,
    predict_current,
    rounded_score,
    score_text,
    validate_model,
)


class OptunaError(RuntimeError):
    """A search-space or optimization error suitable for CLI output."""


def trial_line(
    number: int,
    total: int,
    overall: float,
    best: float,
    params: Mapping[str, Any],
) -> str:
    """Format the live line emitted after every completed trial."""
    values = " ".join(f"{name}={value}" for name, value in sorted(params.items()))
    return (
        f"trial {number}/{total} overall={score_text(overall)} "
        f"best={score_text(best)} {values}"
    ).rstrip()


def _suggest(trial: optuna.Trial, name: str, spec: Mapping[str, Any]) -> Any:
    kind = spec.get("type")
    if kind == "float":
        try:
            low = float(spec["low"])
            high = float(spec["high"])
        except (KeyError, TypeError, ValueError) as error:
            raise OptunaError(f"SEARCH_SPACE[{name!r}] needs numeric low and high") from error
        if not low < high:
            raise OptunaError(f"SEARCH_SPACE[{name!r}] must have low < high")
        step = spec.get("step")
        return trial.suggest_float(
            name,
            low,
            high,
            step=None if step is None else float(step),
            log=bool(spec.get("log", False)),
        )
    if kind == "int":
        try:
            low = int(spec["low"])
            high = int(spec["high"])
        except (KeyError, TypeError, ValueError) as error:
            raise OptunaError(f"SEARCH_SPACE[{name!r}] needs integer low and high") from error
        if low > high:
            raise OptunaError(f"SEARCH_SPACE[{name!r}] must have low <= high")
        return trial.suggest_int(
            name,
            low,
            high,
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    if kind == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            raise OptunaError(f"SEARCH_SPACE[{name!r}] needs nonempty choices")
        return trial.suggest_categorical(name, list(choices))
    raise OptunaError(
        f"SEARCH_SPACE[{name!r}] has unsupported type {kind!r}; "
        "use float, int, or categorical"
    )


class _Progress:
    def __init__(self, total: int, patience: int) -> None:
        self.total = total
        self.patience = patience
        self.completed = 0
        self.stale = 0
        self.best_rounded: float | None = None
        self.stopped_early = False
        self.lock = threading.Lock()

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        with self.lock:
            if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
                print(f"trial {trial.number + 1}/{self.total} failed", flush=True)
                return
            self.completed += 1
            rounded = rounded_score(float(trial.value))
            if self.best_rounded is None or rounded > self.best_rounded:
                self.best_rounded = rounded
                self.stale = 0
            else:
                self.stale += 1
            print(
                trial_line(
                    trial.number + 1,
                    self.total,
                    float(trial.value),
                    float(study.best_value),
                    trial.params,
                ),
                flush=True,
            )
            if self.patience and self.stale >= self.patience:
                self.stopped_early = True
                print(
                    f"early stop: no new three-decimal best in {self.patience} "
                    "completed trials",
                    flush=True,
                )
                study.stop()


def run(args: argparse.Namespace) -> int:
    """Tune the current model and stream one line per completed trial."""
    try:
        if args.trials <= 0:
            raise OptunaError("--trials must be positive")
        if args.patience < 0:
            raise OptunaError("--patience cannot be negative")
        if args.workers <= 0:
            raise OptunaError("--workers must be positive")

        model = load_model()
        inputs, _ = validate_model(model)
        search_space = model.SEARCH_SPACE
        if not search_space:
            raise OptunaError("model.py declares an empty SEARCH_SPACE")
        unknown = sorted(set(search_space) - set(model.PARAMS))
        if unknown:
            raise OptunaError(
                "SEARCH_SPACE parameters need defaults in PARAMS: " + ", ".join(unknown)
            )
        data = load_inputs(inputs)
        evaluator = GFED5Evaluator(GFED5_PATH)

        def objective(trial: optuna.Trial) -> float:
            sampled = {
                name: _suggest(trial, name, specification)
                for name, specification in search_space.items()
            }
            try:
                prediction = predict_current(model, data, params=sampled)
                scores = evaluator.score(prediction)["global"]
                trial.set_user_attr(
                    "metrics",
                    {
                        name: float(scores[name])
                        for name in (
                            "overall_score",
                            "bias_score",
                            "rmse_score",
                            "seasonal_cycle_score",
                            "spatial_distribution_score",
                        )
                    },
                )
                return float(scores["overall_score"])
            finally:
                if "prediction" in locals():
                    del prediction
                gc.collect()

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(
            seed=args.seed,
            constant_liar=args.workers > 1,
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        progress = _Progress(args.trials, args.patience)
        print(
            f"optimizing {len(search_space)} parameter(s); trials={args.trials} "
            f"patience={args.patience} workers={args.workers}",
            flush=True,
        )
        study.optimize(
            objective,
            n_trials=args.trials,
            n_jobs=args.workers,
            callbacks=[progress],
            catch=(ModelError, ValueError, FloatingPointError),
            show_progress_bar=False,
        )
        completed = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            raise OptunaError("all Optuna trials failed")

        best = study.best_trial
        metrics = best.user_attrs["metrics"]
        print(
            "best "
            + " ".join(
                [
                    f"overall={score_text(metrics['overall_score'])}",
                    f"bias={score_text(metrics['bias_score'])}",
                    f"rmse={score_text(metrics['rmse_score'])}",
                    f"seasonal={score_text(metrics['seasonal_cycle_score'])}",
                    f"spatial={score_text(metrics['spatial_distribution_score'])}",
                ]
            )
        )
        print("best_params=" + json.dumps(best.params, sort_keys=True, separators=(",", ":")))
        print(
            f"completed={len(completed)} requested={args.trials} "
            f"stopped_early={str(progress.stopped_early).lower()}"
        )
    except (ModelError, OptunaError, OSError, ValueError) as error:
        print(f"ar optuna: {error}", file=sys.stderr)
        return 2
    return 0
