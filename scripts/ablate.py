"""Exact Shapley attribution for the current model's named components."""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType

import numpy as np

from scripts.fast_ilamb import GFED5Evaluator
from scripts.runtime import (
    GFED5_PATH,
    ModelError,
    load_inputs,
    load_model,
    score_text,
    validate_model,
)


MAX_COMPONENTS = 15
GLOBAL_METRICS = (
    ("overall", "overall_score"),
    ("bias", "bias_score"),
    ("rmse", "rmse_score"),
    ("seasonal", "seasonal_cycle_score"),
    ("spatial", "spatial_distribution_score"),
)

Scores = dict[str, dict[str, float]]
SubsetScores = dict[frozenset[str], Scores]


class AblationError(RuntimeError):
    """A model contract or ablation execution error suitable for CLI output."""


def _validate_model(model: ModuleType) -> tuple[tuple[str, ...], tuple[str, ...]]:
    inputs, components = validate_model(model, require_components=True)
    if len(components) > MAX_COMPONENTS:
        raise AblationError(
            f"exact Shapley is limited to {MAX_COMPONENTS} components "
            f"({1 << MAX_COMPONENTS} subset evaluations); model.py declares {len(components)}"
        )
    return inputs, components


def _subsets(components: Sequence[str]) -> list[frozenset[str]]:
    return [
        frozenset(enabled)
        for size in range(len(components) + 1)
        for enabled in itertools.combinations(components, size)
    ]


def _score_subsets(
    model: ModuleType,
    data: Mapping[str, np.ndarray],
    components: Sequence[str],
    evaluator: GFED5Evaluator,
    emit: Callable[[str], None] = print,
) -> SubsetScores:
    subsets = _subsets(components)
    params = dict(model.PARAMS)
    scored: SubsetScores = {}
    for number, enabled in enumerate(subsets, start=1):
        label = ",".join(name for name in components if name in enabled) or "none"
        try:
            prediction = model.predict(data, params=params, components=enabled)
            scores = evaluator.score(prediction)
        except Exception as error:
            raise AblationError(f"subset {label!r} failed: {error}") from error
        overall = scores["global"]["overall_score"]
        if not np.isfinite(overall):
            raise AblationError(f"subset {label!r} produced a non-finite overall score")
        scored[enabled] = scores
        emit(
            f"subset {number}/{len(subsets)} components={label} "
            f"overall={score_text(overall)}"
        )
    return scored


def shapley_values(
    components: Sequence[str],
    values: Mapping[frozenset[str], float],
) -> dict[str, float]:
    """Average every component's marginal value over all possible orderings."""
    count = len(components)
    denominator = math.factorial(count)
    attribution: dict[str, float] = {}
    for component in components:
        contribution = 0.0
        others = [name for name in components if name != component]
        for subset in _subsets(others):
            weight = (
                math.factorial(len(subset))
                * math.factorial(count - len(subset) - 1)
                / denominator
            )
            contribution += weight * (values[subset | {component}] - values[subset])
        attribution[component] = contribution
    return attribution


def _metric_values(
    scores: SubsetScores,
    region: str,
    metric: str,
) -> dict[frozenset[str], float]:
    return {subset: values[region][metric] for subset, values in scores.items()}


def _format_number(value: float) -> str:
    return score_text(value, signed=True)


def _print_report(components: Sequence[str], scores: SubsetScores) -> None:
    full = frozenset(components)
    empty = frozenset()
    global_shapley = {
        label: shapley_values(components, _metric_values(scores, "global", metric))
        for label, metric in GLOBAL_METRICS
    }

    print()
    print(f"exact Shapley attribution; fixed PARAMS; {len(scores)} subsets; positive means helpful")
    print("component\tshapley_overall\tshapley_bias\tshapley_rmse\tshapley_seasonal\tshapley_spatial\tdrop_one_overall")
    for component in components:
        without = full - {component}
        drop_one = (
            scores[full]["global"]["overall_score"]
            - scores[without]["global"]["overall_score"]
        )
        values = [_format_number(global_shapley[label][component]) for label, _ in GLOBAL_METRICS]
        print("\t".join([component, *values, _format_number(drop_one)]))

    regions = [name for name in scores[full] if name != "global"]
    print()
    print("regional Shapley")
    print("region\tcomponent\tshapley_overall\tshapley_bias\tshapley_rmse\tshapley_seasonal\tshapley_spatial")
    for region in regions:
        attribution = {
            label: shapley_values(components, _metric_values(scores, region, metric))
            for label, metric in GLOBAL_METRICS
        }
        for component in components:
            values = [
                _format_number(attribution[label][component])
                for label, _ in GLOBAL_METRICS
            ]
            print("\t".join([region, component, *values]))

    empty_overall = scores[empty]["global"]["overall_score"]
    full_overall = scores[full]["global"]["overall_score"]
    reconstruction = sum(global_shapley["overall"].values())
    print()
    print(
        f"global overall: empty={score_text(empty_overall)} "
        f"full={score_text(full_overall)} "
        f"change={score_text(full_overall - empty_overall, signed=True)} "
        f"shapley_sum={score_text(reconstruction, signed=True)}"
    )


def run(args: argparse.Namespace) -> int:
    """Run exact fixed-parameter Shapley attribution and leave-one-out diagnostics."""
    del args
    try:
        model = load_model()
        inputs, components = _validate_model(model)
        print(
            f"loading {len(inputs)} input(s); evaluating {1 << len(components)} "
            "component subsets"
        )
        data = load_inputs(inputs)
        evaluator = GFED5Evaluator(GFED5_PATH)
        scores = _score_subsets(model, data, components, evaluator)
        _print_report(components, scores)
    except (AblationError, ModelError) as error:
        print(f"ar ablate: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"ar ablate: {error}", file=sys.stderr)
        return 2
    return 0
