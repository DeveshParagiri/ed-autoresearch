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
# Official apples-to-apples GFED5 ILAMB check recorded in commit c4a7b5f.
STOCK_ED_BASELINE_OVERALL = 0.4225
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
    improvements = np.concatenate(
        (
            [scores[0] > STOCK_ED_BASELINE_OVERALL],
            running_best[1:] > running_best[:-1],
        )
    )
    other_experiments = ~improvements
    progress_steps = np.arange(0, len(rows) + 1)
    progress_scores = np.concatenate(([STOCK_ED_BASELINE_OVERALL], running_best))

    figure, (detail_axis, baseline_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(13, 6.8),
        gridspec_kw={"height_ratios": (4.8, 1.15), "hspace": 0.06},
    )
    figure.patch.set_facecolor("white")
    detail_axis.set_facecolor("white")
    baseline_axis.set_facecolor("white")
    detail_axis.scatter(
        experiments[other_experiments],
        scores[other_experiments],
        s=18,
        color="#B9BEC3",
        alpha=0.55,
        edgecolor="none",
        label="Other experiments",
        zorder=1,
    )
    detail_axis.step(
        progress_steps,
        progress_scores,
        where="post",
        color="#2EAD67",
        linewidth=2.1,
        label="Running best",
        zorder=2,
    )
    baseline_axis.step(
        progress_steps,
        progress_scores,
        where="post",
        color="#2EAD67",
        linewidth=2.1,
        zorder=2,
    )
    detail_axis.scatter(
        experiments[improvements],
        running_best[improvements],
        s=48,
        color="#39C978",
        edgecolor="#176B3A",
        linewidth=0.9,
        label="Overall improvements",
        zorder=3,
    )
    baseline_axis.scatter(
        [0],
        [STOCK_ED_BASELINE_OVERALL],
        s=62,
        marker="D",
        color="#3D5A80",
        edgecolor="#24364D",
        linewidth=0.9,
        label="Stock ED baseline",
        zorder=4,
    )
    count = len(rows)
    kept = int(improvements.sum())
    figure.suptitle(
        f"ED-Fire Autoresearch Progress: {count} Experiments, "
        f"{kept} Overall Improvements",
        fontsize=15,
        fontweight="semibold",
        y=0.985,
    )
    baseline_axis.set_xlabel("Experiment #")
    figure.text(
        0.017,
        0.5,
        "GFED5 Overall (higher is better)",
        rotation="vertical",
        va="center",
    )

    detail_spread = max(0.006, float(scores.max() - scores.min()))
    detail_axis.set_ylim(
        max(0.0, float(scores.min()) - max(0.004, detail_spread * 0.12)),
        min(1.0, float(scores.max()) + max(0.008, detail_spread * 0.25)),
    )
    baseline_margin = 0.010
    baseline_axis.set_ylim(
        max(0.0, STOCK_ED_BASELINE_OVERALL - baseline_margin),
        min(1.0, STOCK_ED_BASELINE_OVERALL + baseline_margin),
    )
    baseline_axis.set_xlim(-0.45, max(2, len(rows)) + 0.8)
    baseline_axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=2))
    baseline_axis.yaxis.set_major_locator(MaxNLocator(nbins=3))

    for plot_axis in (detail_axis, baseline_axis):
        plot_axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        plot_axis.grid(color="#D7DDE1", linewidth=0.7, alpha=0.65)

    detail_axis.spines["bottom"].set_visible(False)
    baseline_axis.spines["top"].set_visible(False)
    detail_axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    baseline_axis.tick_params(axis="x", which="both", top=False)

    break_size = 0.008
    break_style = {"color": "#4A525A", "clip_on": False, "linewidth": 0.9}
    detail_axis.plot(
        (-break_size, +break_size),
        (-break_size, +break_size),
        transform=detail_axis.transAxes,
        **break_style,
    )
    detail_axis.plot(
        (1 - break_size, 1 + break_size),
        (-break_size, +break_size),
        transform=detail_axis.transAxes,
        **break_style,
    )
    baseline_axis.plot(
        (-break_size, +break_size),
        (1 - break_size, 1 + break_size),
        transform=baseline_axis.transAxes,
        **break_style,
    )
    baseline_axis.plot(
        (1 - break_size, 1 + break_size),
        (1 - break_size, 1 + break_size),
        transform=baseline_axis.transAxes,
        **break_style,
    )

    detail_handles, detail_labels = detail_axis.get_legend_handles_labels()
    baseline_handles, baseline_labels = baseline_axis.get_legend_handles_labels()
    handles_by_label = dict(
        zip(detail_labels + baseline_labels, detail_handles + baseline_handles, strict=True)
    )
    legend_order = [
        "Stock ED baseline",
        "Other experiments",
        "Overall improvements",
        "Running best",
    ]
    detail_axis.legend(
        [handles_by_label[label] for label in legend_order],
        legend_order,
        loc="upper left",
        frameon=True,
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.09)

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
