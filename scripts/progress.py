"""Render the single current running-best GFED5 progress graph.

It derives all chart state from the official result ledger, atomically
overwrites ``progress.png``, and never writes into the agent's working context.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ed-fire-matplotlib")
)

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "autoresearch" / "results.tsv"
DEFAULT_OUTPUT = ROOT / "progress.png"
REQUIRED_FIELDS = {
    "commit",
    "overall",
    "bias",
    "rmse",
    "seasonal",
    "spatial",
    "regional",
    "inputs",
    "description",
}


class ProgressError(RuntimeError):
    """A malformed ledger or graph output error."""


def _read_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or set(reader.fieldnames) != REQUIRED_FIELDS:
            raise ProgressError(f"{path} does not match the result ledger schema")
        rows = list(reader)
    for number, row in enumerate(rows, start=2):
        try:
            score = float(row["overall"])
        except ValueError as error:
            raise ProgressError(f"{path}:{number} has an invalid Overall score") from error
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ProgressError(f"{path}:{number} has Overall outside [0,1]: {score}")
    return rows


def render(results: Path, output: Path) -> bool:
    rows = _read_results(results)
    if not rows:
        print("progress: results.tsv has no experiments yet")
        return False

    scores = np.asarray([float(row["overall"]) for row in rows], dtype=np.float64)
    running_best = np.maximum.accumulate(scores)
    experiments = np.arange(1, len(rows) + 1)
    improvements = np.concatenate(([True], running_best[1:] > running_best[:-1]))
    other_experiments = ~improvements

    figure, axis = plt.subplots(figsize=(13, 6.8))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.scatter(
        experiments[other_experiments],
        scores[other_experiments],
        s=18,
        color="#B9BEC3",
        alpha=0.55,
        edgecolor="none",
        label="Other experiments",
        zorder=1,
    )
    axis.step(
        experiments,
        running_best,
        where="post",
        color="#2EAD67",
        linewidth=2.1,
        label="Running best",
        zorder=2,
    )
    axis.scatter(
        experiments[improvements],
        running_best[improvements],
        s=48,
        color="#39C978",
        edgecolor="#176B3A",
        linewidth=0.9,
        label="Overall improvements",
        zorder=3,
    )
    count = len(rows)
    kept = int(improvements.sum())
    axis.set_title(
        f"ED-Fire Autoresearch Progress: {count} Experiments, "
        f"{kept} Overall Improvements",
        fontsize=15,
        fontweight="semibold",
    )
    axis.set_xlabel("Experiment #")
    axis.set_ylabel("GFED5 Overall (higher is better)")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=2))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axis.set_xlim(0.6, max(2, len(rows)) + 0.8)
    spread = max(0.006, float(scores.max() - scores.min()))
    lower_margin = max(0.004, spread * 0.18)
    upper_margin = max(0.008, spread * 0.35)
    axis.set_ylim(
        max(0.0, float(scores.min()) - lower_margin),
        min(1.0, float(scores.max()) + upper_margin),
    )
    axis.grid(color="#D7DDE1", linewidth=0.7, alpha=0.65)
    handles, labels = axis.get_legend_handles_labels()
    order = [0, 2, 1]
    axis.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="best",
        frameon=True,
    )
    figure.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    try:
        figure.savefig(partial, format="png", dpi=160, facecolor="white")
        os.replace(partial, output)
    finally:
        plt.close(figure)
        partial.unlink(missing_ok=True)
    print(f"progress={output}")
    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the external running-best graph from results.tsv."
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    results = args.results.expanduser().resolve()
    output = args.output.expanduser().resolve()
    try:
        render(results, output)
    except (OSError, ProgressError) as error:
        print(f"progress: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
