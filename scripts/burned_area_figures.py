#!/usr/bin/env python3
"""SciencePlots renderer for the canonical ED-Fire burned-area figures."""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FONT_FAMILY = "Basier Square"
BASIER_REGULAR = PROJECT_ROOT / "assets" / "fonts" / "BasierSquare-Regular.otf"
BASIER_SEMIBOLD = PROJECT_ROOT / "assets" / "fonts" / "BasierSquare-SemiBold.otf"
FONT_SHA256 = {
    BASIER_REGULAR.name: "0c621e44ad72820b0f3e3af5b75b3c2dc656d9072dc9b99965be112459af78ce",
    BASIER_SEMIBOLD.name: "0a4d3cb400241c00b65babf13dc10af9e467f24efdd0e37dec15eb38508a180d",
}

SCORE_COMPONENTS = (
    ("bias_score", "Bias"),
    ("rmse_score", "RMSE"),
    ("seasonal_cycle_score", "Seasonal"),
    ("spatial_distribution_score", "Spatial"),
    ("overall_score", "Overall"),
)
FIGURE_NAMES = (
    "01-score-summary.png",
    "02a-mean-burned-area.png",
    "02b-burned-area-differences.png",
    "03-seasonal-cycles.png",
    "04-spatial-distribution.png",
    "05-benchmark-sensitivity.png",
)

# These are the first colors in SciencePlots' colorblind-safe bright cycle.
GFED5 = "#000000"
GFED4 = "#777777"
STOCK = "#0C5DA5"
CANDIDATE = "#FF2C00"
GFED5_HIGHER = "#0C5DA5"
GFED4_HIGHER = "#FF9500"
NEUTRAL = "#3B3B3B"


def configure_plotting(config_root: Path) -> Any:
    """Load the repository-owned journal style and fonts without LaTeX."""
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(config_root)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import scienceplots  # noqa: F401  # registers the named styles

    for font_path in (BASIER_REGULAR, BASIER_SEMIBOLD):
        if not font_path.is_file():
            raise FileNotFoundError(f"missing repository font: {font_path}")
        font_manager.fontManager.addfont(font_path)

    plt.style.use(["science", "no-latex", "bright"])
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY],
            "font.size": 10,
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": FONT_FAMILY,
            "mathtext.bf": f"{FONT_FAMILY}:weight=semibold",
            "mathtext.sf": FONT_FAMILY,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": None,
        }
    )
    return plt


def _save(plt: Any, figure: Any, path: Path, dimensions: dict[str, int]) -> None:
    dpi = 150
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.set_size_inches(dimensions["width"] / dpi, dimensions["height"] / dpi)
    figure.savefig(
        path,
        dpi=dpi,
        metadata={"Software": "ED-Fire SciencePlots renderer"},
    )
    plt.close(figure)

    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"figure is not a valid PNG: {path}")
    actual = struct.unpack(">II", header[16:24])
    expected = (dimensions["width"], dimensions["height"])
    if actual != expected:
        raise ValueError(f"figure has wrong dimensions: {path} is {actual}, expected {expected}")


def _figure_size(dimensions: dict[str, int]) -> tuple[float, float]:
    return (dimensions["width"] / 150, dimensions["height"] / 150)


def _panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.10,
        1.01,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )


def _map_panel_label(axis: Any, label: str) -> None:
    axis.text(
        0.018,
        0.965,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=NEUTRAL,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.6},
        zorder=5,
    )


def _zero_line(axis: Any, *, vertical: bool = False) -> None:
    if vertical:
        axis.axvline(0, color=NEUTRAL, linewidth=0.8, zorder=1)
    else:
        axis.axhline(0, color=NEUTRAL, linewidth=0.8, zorder=1)


def plot_score_summary(
    plt: Any,
    path: Path,
    dimensions: dict[str, int],
    scores: dict[str, dict[str, dict[str, float]]],
    scales: dict[str, float],
    *,
    candidate_label: str,
) -> None:
    figure = plt.figure(figsize=_figure_size(dimensions))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.3, 0.9))
    score_axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]))
    mean_axis = figure.add_subplot(grid[1, :])
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.90,
        wspace=0.16,
        hspace=0.28,
    )

    labels = [label for _, label in SCORE_COMPONENTS]
    positions = np.arange(len(labels))
    width = 0.36
    for panel, (axis, benchmark_id, title) in enumerate(
        zip(
            score_axes,
            ("gfed5", "gfed4_1s"),
            ("Against GFED5", "Against GFED4.1s"),
            strict=True,
        )
    ):
        stock = [scores["ED-stock"][benchmark_id][key] for key, _ in SCORE_COMPONENTS]
        candidate = [scores["Candidate"][benchmark_id][key] for key, _ in SCORE_COMPONENTS]
        axis.bar(
            positions - width / 2,
            stock,
            width,
            label="ED-stock",
            color=STOCK,
            edgecolor="white",
            linewidth=0.5,
        )
        axis.bar(
            positions + width / 2,
            candidate,
            width,
            label=candidate_label,
            color=CANDIDATE,
            edgecolor="white",
            linewidth=0.5,
        )
        axis.set_xticks(positions, labels)
        axis.set_ylim(0, 1)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.set_ylabel("ILAMB score")
        axis.set_title(title, loc="left")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
        axis.legend(loc="upper left", ncols=2)
        _panel_label(axis, chr(ord("a") + panel))

    benchmark_ids = ("gfed5", "gfed4_1s")
    benchmark_labels = ("GFED5", "GFED4.1s")
    centers = np.arange(2)
    mean_width = 0.23
    benchmark_means = [
        scores["Candidate"][benchmark_id]["benchmark_period_mean_percent"]
        for benchmark_id in benchmark_ids
    ]
    stock_means = [
        scores["ED-stock"][benchmark_id]["model_period_mean_percent"]
        for benchmark_id in benchmark_ids
    ]
    candidate_means = [
        scores["Candidate"][benchmark_id]["model_period_mean_percent"]
        for benchmark_id in benchmark_ids
    ]
    benchmark_bars = mean_axis.bar(
        centers - mean_width,
        benchmark_means,
        mean_width,
        label="Benchmark",
        color=(GFED5, GFED4),
        edgecolor="white",
        linewidth=0.5,
    )
    stock_bars = mean_axis.bar(
        centers,
        stock_means,
        mean_width,
        label="ED-stock",
        color=STOCK,
        edgecolor="white",
        linewidth=0.5,
    )
    candidate_bars = mean_axis.bar(
        centers + mean_width,
        candidate_means,
        mean_width,
        label=candidate_label,
        color=CANDIDATE,
        edgecolor="white",
        linewidth=0.5,
    )
    mean_axis.set_ylim(0, float(scales.get("global_mean_percent_max", 1.0)))
    mean_axis.set_xticks(centers, benchmark_labels)
    mean_axis.set_ylabel("global period mean (% month$^{-1}$)")
    mean_axis.set_title("Global means on the ILAMB comparison intersections", loc="left")
    mean_axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
    mean_axis.legend(loc="upper right", ncols=3)
    for bars in (benchmark_bars, stock_bars, candidate_bars):
        mean_axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    _panel_label(mean_axis, "c")

    figure.suptitle(
        f"{candidate_label}: benchmark score summary",
        fontsize=16,
        fontweight="semibold",
    )
    _save(plt, figure, path, dimensions)


def _longitude(value: float, _: int) -> str:
    if value == 0:
        return "0°"
    return f"{abs(int(value))}°{'W' if value < 0 else 'E'}"


def _latitude(value: float, _: int) -> str:
    if value == 0:
        return "0°"
    return f"{abs(int(value))}°{'S' if value < 0 else 'N'}"


def plot_mean_burned_area_fields(
    plt: Any,
    path: Path,
    dimensions: dict[str, int],
    fields: dict[str, dict[str, Any]],
    scales: dict[str, float],
    *,
    candidate_label: str,
) -> None:
    from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, NullLocator

    figure, axes = plt.subplots(2, 2, figsize=_figure_size(dimensions))
    figure.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.20,
        top=0.88,
        wspace=0.10,
        hspace=0.16,
    )

    field_limit = float(scales["annual_percent_max"])
    display_domain = np.isfinite(fields["stock"]["annual_percent"])
    panels = (
        ("GFED4.1s", fields["gfed4_1s"]["annual_percent"]),
        ("GFED5", fields["gfed5"]["annual_percent"]),
        ("ED-stock", fields["stock"]["annual_percent"]),
        (candidate_label, fields["candidate"]["annual_percent"]),
    )
    field_image = None
    for index, (axis, (title, values)) in enumerate(zip(axes.flat, panels, strict=True)):
        display_values = np.where(display_domain, values, np.nan)
        field_image = axis.imshow(
            display_values,
            origin="lower",
            extent=(-180, 180, -90, 90),
            vmin=0,
            vmax=field_limit,
            cmap="magma_r",
            aspect="auto",
            interpolation="nearest",
            rasterized=True,
        )
        axis.set_title(title, pad=5)
        axis.set_xlim(-180, 180)
        axis.set_ylim(-90, 90)
        axis.set_xticks((-120, 0, 120))
        axis.set_yticks((-60, 0, 60))
        axis.xaxis.set_major_formatter(FuncFormatter(_longitude))
        axis.yaxis.set_major_formatter(FuncFormatter(_latitude))
        axis.grid(color="white", alpha=0.28, linewidth=0.45)
        axis.tick_params(labelbottom=index >= 2)
        if index % 2:
            axis.tick_params(labelleft=False)
        _map_panel_label(axis, chr(ord("a") + index))

    for row, label in enumerate(("BENCHMARKS", "MODELS")):
        axes[row, 0].text(
            -0.15,
            0.5,
            label,
            transform=axes[row, 0].transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=8,
            fontweight="semibold",
            color=NEUTRAL,
            clip_on=False,
        )

    if field_image is None:
        raise RuntimeError("absolute map panels were not created")
    left = axes[1, 0].get_position().x0
    right = axes[1, 1].get_position().x1
    color_axis = figure.add_axes(
        (left, axes[1, 0].get_position().y0 - 0.080, right - left, 0.020)
    )
    field_ticks = np.linspace(0, field_limit, 5)
    field_colorbar = figure.colorbar(
        field_image,
        cax=color_axis,
        orientation="horizontal",
        ticks=field_ticks,
    )
    field_colorbar.ax.xaxis.set_major_locator(FixedLocator(field_ticks))
    field_labels = [f"{value:g}" for value in field_ticks]
    field_labels[-1] = f"≥{field_limit:g}"
    field_colorbar.ax.xaxis.set_major_formatter(FixedFormatter(tuple(field_labels)))
    field_colorbar.ax.xaxis.set_minor_locator(NullLocator())
    field_colorbar.ax.tick_params(
        top=False,
        labeltop=False,
        bottom=True,
        labelbottom=True,
    )
    field_colorbar.set_label("Burned area (% grid-cell area yr$^{-1}$)", labelpad=6)
    figure.text(
        0.5,
        0.025,
        f"The end color includes values at and above {field_limit:g}% yr$^{{-1}}$; white is outside the pinned ED-stock display domain or missing.",
        ha="center",
        va="center",
        fontsize=7.5,
        color=NEUTRAL,
    )
    figure.suptitle(
        f"{candidate_label}: benchmarks and model fields",
        fontsize=16,
        fontweight="semibold",
    )
    _save(plt, figure, path, dimensions)


def plot_burned_area_differences(
    plt: Any,
    path: Path,
    dimensions: dict[str, int],
    fields: dict[str, dict[str, Any]],
    scales: dict[str, float],
    *,
    candidate_label: str,
) -> None:
    from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, NullLocator

    figure, axes = plt.subplots(1, 2, figsize=_figure_size(dimensions))
    figure.subplots_adjust(
        left=0.08,
        right=0.97,
        bottom=0.28,
        top=0.82,
        wspace=0.10,
    )

    residual_limit = float(scales["residual_percent_abs"])
    display_domain = np.isfinite(fields["stock"]["annual_percent"])
    panels = (
        (
            f"{candidate_label} − GFED4.1s",
            fields["candidate"]["annual_percent"] - fields["gfed4_1s"]["annual_percent"],
        ),
        (
            f"{candidate_label} − GFED5",
            fields["candidate"]["annual_percent"] - fields["gfed5"]["annual_percent"],
        ),
    )
    residual_image = None
    for index, (axis, (title, values)) in enumerate(zip(axes, panels, strict=True)):
        display_values = np.where(display_domain, values, np.nan)
        residual_image = axis.imshow(
            display_values,
            origin="lower",
            extent=(-180, 180, -90, 90),
            vmin=-residual_limit,
            vmax=residual_limit,
            cmap="RdBu_r",
            aspect="auto",
            interpolation="nearest",
            rasterized=True,
        )
        axis.set_title(title, pad=5)
        axis.set_xlim(-180, 180)
        axis.set_ylim(-90, 90)
        axis.set_xticks((-120, 0, 120))
        axis.set_yticks((-60, 0, 60))
        axis.xaxis.set_major_formatter(FuncFormatter(_longitude))
        axis.yaxis.set_major_formatter(FuncFormatter(_latitude))
        axis.grid(color="white", alpha=0.28, linewidth=0.45)
        if index:
            axis.tick_params(labelleft=False)
        _map_panel_label(axis, chr(ord("a") + index))

    if residual_image is None:
        raise RuntimeError("difference map panels were not created")
    left = axes[0].get_position().x0
    right = axes[1].get_position().x1
    color_axis = figure.add_axes(
        (left, axes[0].get_position().y0 - 0.110, right - left, 0.030)
    )

    residual_ticks = np.linspace(-residual_limit, residual_limit, 5)
    residual_colorbar = figure.colorbar(
        residual_image,
        cax=color_axis,
        orientation="horizontal",
        ticks=residual_ticks,
    )
    residual_colorbar.ax.xaxis.set_major_locator(FixedLocator(residual_ticks))
    residual_labels = [f"{value:g}" for value in residual_ticks]
    residual_labels[0] = f"≤−{residual_limit:g}"
    residual_labels[-1] = f"≥{residual_limit:g}"
    residual_colorbar.ax.xaxis.set_major_formatter(FixedFormatter(tuple(residual_labels)))
    residual_colorbar.ax.xaxis.set_minor_locator(NullLocator())
    residual_colorbar.ax.tick_params(
        top=False,
        labeltop=False,
        bottom=True,
        labelbottom=True,
    )
    residual_colorbar.set_label(
        "Model minus benchmark (% grid-cell area yr$^{-1}$)",
        labelpad=6,
    )
    figure.text(
        0.5,
        0.035,
        f"End colors include differences beyond ±{residual_limit:g}% yr$^{{-1}}$; white is outside the pinned ED-stock display domain or missing.",
        ha="center",
        va="center",
        fontsize=7.5,
        color=NEUTRAL,
    )
    figure.suptitle(
        f"{candidate_label}: difference maps",
        fontsize=16,
        fontweight="semibold",
    )
    _save(plt, figure, path, dimensions)


def plot_seasonal_cycles(
    plt: Any,
    path: Path,
    dimensions: dict[str, int],
    fields: dict[str, dict[str, Any]],
    regions: list[dict[str, Any]],
    scales: dict[str, float],
    *,
    candidate_label: str,
) -> None:
    figure, axes = plt.subplots(
        2,
        4,
        figsize=_figure_size(dimensions),
        layout="constrained",
        sharex=True,
        sharey=False,
    )
    months = np.arange(1, 13)
    styles = (
        ("gfed5", "GFED5", GFED5, "-", 1.5),
        ("gfed4_1s", "GFED4.1s", GFED4, "--", 1.5),
        ("stock", "ED-stock", STOCK, "-.", 1.6),
        ("candidate", candidate_label, CANDIDATE, "-", 2.1),
    )
    for index, (axis, region) in enumerate(zip(axes.flat, regions, strict=True)):
        region_id = region["id"]
        for field_id, label, color, linestyle, linewidth in styles:
            axis.plot(
                months,
                fields[field_id]["seasonal_percent"][region_id],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        axis.set_title(region["label"], loc="left")
        axis.set_xlim(1, 12)
        regional_limits = scales.get("seasonal_percent_max_by_region", {})
        if isinstance(regional_limits, dict) and region_id in regional_limits:
            seasonal_limit = float(regional_limits[region_id])
        else:
            seasonal_limit = float(scales["seasonal_percent_max"])
        axis.set_ylim(0, seasonal_limit)
        axis.set_xticks((1, 3, 5, 7, 9, 11), ("Jan", "Mar", "May", "Jul", "Sep", "Nov"))
        axis.set_yticks(np.linspace(0, seasonal_limit, 4))
        axis.grid(color="#D9D9D9", linewidth=0.55)
        _panel_label(axis, chr(ord("a") + index))

    for axis in axes[1, :]:
        axis.set_xlabel("month")
    for axis in axes[:, 0]:
        axis.set_ylabel("burned area (% month$^{-1}$)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=4,
        handlelength=3.0,
    )
    figure.suptitle(
        f"{candidate_label}: seasonal cycles",
        fontsize=16,
        fontweight="semibold",
    )
    _save(plt, figure, path, dimensions)


def _correlation(reference: np.ndarray, model: np.ndarray) -> float:
    if reference.size < 2 or np.std(reference) == 0 or np.std(model) == 0:
        return float("nan")
    return float(np.corrcoef(reference, model)[0, 1])


def plot_spatial_distribution(
    plt: Any,
    path: Path,
    dimensions: dict[str, int],
    fields: dict[str, dict[str, Any]],
    scales: dict[str, float],
    *,
    candidate_label: str,
) -> None:
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    figure, axes = plt.subplots(2, 2, figsize=_figure_size(dimensions))
    figure.subplots_adjust(
        left=0.08,
        right=0.97,
        bottom=0.13,
        top=0.90,
        wspace=0.22,
        hspace=0.26,
    )
    color_axis = figure.add_axes((0.14, 0.055, 0.72, 0.025))
    maximum = float(scales["scatter_percent_max"])
    panels = (
        ("gfed5", "candidate", f"{candidate_label} vs GFED5"),
        ("gfed4_1s", "candidate", f"{candidate_label} vs GFED4.1s"),
        ("gfed5", "stock", "ED-stock vs GFED5"),
        ("gfed4_1s", "stock", "ED-stock vs GFED4.1s"),
    )
    density = None
    for index, (axis, (reference_id, model_id, title)) in enumerate(
        zip(axes.flat, panels, strict=True)
    ):
        reference = fields[reference_id]["annual_percent"]
        model = fields[model_id]["annual_percent"]
        valid = np.isfinite(reference) & np.isfinite(model) & (reference > 0)
        x = reference[valid]
        y = model[valid]
        shown = (x <= maximum) & (y <= maximum)
        density = axis.hexbin(
            x[shown],
            y[shown],
            gridsize=70,
            extent=(0, maximum, 0, maximum),
            mincnt=1,
            cmap="viridis",
            norm=LogNorm(vmin=1, vmax=10_000),
            linewidths=0,
            rasterized=True,
        )
        axis.plot((0, maximum), (0, maximum), color="white", linewidth=2.2, zorder=3)
        axis.plot(
            (0, maximum),
            (0, maximum),
            color=NEUTRAL,
            linewidth=0.9,
            linestyle="--",
            zorder=4,
            label="one-to-one",
        )
        axis.set_xlim(0, maximum)
        axis.set_ylim(0, maximum)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks(np.linspace(0, maximum, 5))
        axis.set_yticks(np.linspace(0, maximum, 5))
        axis.set_title(title, loc="left")
        axis.set_xlabel("benchmark burned area (% yr$^{-1}$)")
        axis.set_ylabel("model burned area (% yr$^{-1}$)")
        axis.grid(color="#D9D9D9", linewidth=0.5)
        correlation = _correlation(x, y)
        median_bias = float(np.median(y - x))
        shown_percent = float(np.mean(shown) * 100.0) if shown.size else 0.0
        axis.text(
            0.04,
            0.95,
            (
                f"n = {x.size:,}\nr = {correlation:.3f}\n"
                f"median bias = {median_bias:+.3f}\nshown = {shown_percent:.3f}%"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color=NEUTRAL,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
        )
        _panel_label(axis, chr(ord("a") + index))

    if density is None:
        raise RuntimeError("density panels were not created")
    colorbar = figure.colorbar(
        density,
        cax=color_axis,
        orientation="horizontal",
        label="grid cells per hexagon (log scale)",
    )
    colorbar.set_ticks(
        (1, 10, 100, 1_000, 10_000),
        labels=("1", "10", "100", "1,000", "10,000"),
    )
    colorbar.ax.xaxis.set_major_locator(FixedLocator((1, 10, 100, 1_000, 10_000)))
    colorbar.ax.xaxis.set_major_formatter(
        FixedFormatter(("1", "10", "100", "1,000", "10,000"))
    )
    colorbar.ax.xaxis.set_minor_locator(NullLocator())
    colorbar.ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    figure.suptitle(
        f"{candidate_label}: spatial distributions",
        fontsize=16,
        fontweight="semibold",
    )
    _save(plt, figure, path, dimensions)


def plot_benchmark_sensitivity(
    plt: Any,
    path: Path,
    dimensions: dict[str, int],
    scores: dict[str, dict[str, dict[str, float]]],
    fields: dict[str, dict[str, Any]],
    regions: list[dict[str, Any]],
    scales: dict[str, float],
    *,
    candidate_label: str,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=_figure_size(dimensions),
        layout="constrained",
    )
    labels = [label for _, label in SCORE_COMPONENTS]
    positions = np.arange(len(labels))
    for index, (axis, model, title) in enumerate(
        (
            (axes[0, 0], "Candidate", f"{candidate_label}: GFED5 − GFED4.1s"),
            (axes[0, 1], "ED-stock", "ED-stock: GFED5 − GFED4.1s"),
        )
    ):
        delta = np.asarray(
            [
                scores[model]["gfed5"][key] - scores[model]["gfed4_1s"][key]
                for key, _ in SCORE_COMPONENTS
            ]
        )
        colors = [GFED5_HIGHER if value >= 0 else GFED4_HIGHER for value in delta]
        axis.bar(positions, delta, color=colors, edgecolor="white", linewidth=0.5)
        _zero_line(axis)
        axis.set_xticks(positions, labels)
        score_limit = float(scales.get("score_difference_abs", 0.25))
        axis.set_ylim(-score_limit, score_limit)
        axis.set_yticks(np.linspace(-score_limit, score_limit, 5))
        axis.set_ylabel("GFED5 score − GFED4.1s score")
        axis.set_title(title, loc="left")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
        for bar, value in zip(axis.patches, delta, strict=True):
            offset = score_limit * (0.05 if value >= 0 else -0.05)
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:+.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
                color=NEUTRAL,
            )
        _panel_label(axis, chr(ord("a") + index))

    region_labels = [region["label"] for region in regions]
    region_ids = [region["id"] for region in regions]
    candidate = fields["candidate"]["regional_annual_percent"]
    stock = fields["stock"]["regional_annual_percent"]
    gfed5 = fields["gfed5"]["regional_annual_percent"]
    gfed4 = fields["gfed4_1s"]["regional_annual_percent"]
    y = np.arange(len(regions))
    bias5 = np.asarray([candidate[region_id] - gfed5[region_id] for region_id in region_ids])
    bias4 = np.asarray([candidate[region_id] - gfed4[region_id] for region_id in region_ids])
    height = 0.36
    axes[1, 0].barh(y - height / 2, bias5, height, label="vs GFED5", color=GFED5)
    axes[1, 0].barh(y + height / 2, bias4, height, label="vs GFED4.1s", color=GFED4)
    _zero_line(axes[1, 0], vertical=True)
    axes[1, 0].set_yticks(y, region_labels)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim(
        -float(scales["regional_bias_percent_abs"]),
        float(scales["regional_bias_percent_abs"]),
    )
    axes[1, 0].set_xlabel(f"{candidate_label} bias (% yr$^{{-1}}$)")
    axes[1, 0].set_title("Regional candidate bias", loc="left")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(axis="x", color="#D9D9D9", linewidth=0.55, zorder=0)
    regional_limit = float(scales["regional_bias_percent_abs"])
    for bar, value in zip(axes[1, 0].patches, np.concatenate((bias5, bias4)), strict=True):
        if abs(value) <= regional_limit:
            continue
        direction = 1 if value > 0 else -1
        axes[1, 0].text(
            direction * regional_limit * 0.96,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}",
            ha="right" if direction > 0 else "left",
            va="center",
            fontsize=7.5,
            color="white",
            fontweight="bold",
        )
    _panel_label(axes[1, 0], "c")

    candidate_delta = np.asarray([candidate[region_id] - stock[region_id] for region_id in region_ids])
    axes[1, 1].barh(y, candidate_delta, color=CANDIDATE)
    _zero_line(axes[1, 1], vertical=True)
    axes[1, 1].set_yticks(y, region_labels)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim(
        -float(scales["regional_bias_percent_abs"]),
        float(scales["regional_bias_percent_abs"]),
    )
    axes[1, 1].set_xlabel(f"{candidate_label} − ED-stock (% yr$^{{-1}}$)")
    axes[1, 1].set_title("Regional change from native ED", loc="left")
    axes[1, 1].grid(axis="x", color="#D9D9D9", linewidth=0.55, zorder=0)
    for bar, value in zip(axes[1, 1].patches, candidate_delta, strict=True):
        if abs(value) <= regional_limit:
            continue
        direction = 1 if value > 0 else -1
        axes[1, 1].text(
            direction * regional_limit * 0.96,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}",
            ha="right" if direction > 0 else "left",
            va="center",
            fontsize=7.5,
            color="white",
            fontweight="bold",
        )
    _panel_label(axes[1, 1], "d")

    figure.suptitle(
        f"{candidate_label}: benchmark sensitivity",
        fontsize=16,
        fontweight="semibold",
    )
    _save(plt, figure, path, dimensions)


def render_suite(
    plt: Any,
    output: Path,
    dimensions: dict[str, dict[str, int]],
    scores: dict[str, dict[str, dict[str, float]]],
    fields: dict[str, dict[str, Any]],
    regions: list[dict[str, Any]],
    scales: dict[str, float],
    *,
    candidate_label: str,
) -> tuple[Path, ...]:
    """Render the six canonical figures with one immutable presentation shape."""
    paths = tuple(output / filename for filename in FIGURE_NAMES)
    plot_score_summary(
        plt,
        paths[0],
        dimensions[paths[0].name],
        scores,
        scales,
        candidate_label=candidate_label,
    )
    plot_mean_burned_area_fields(
        plt,
        paths[1],
        dimensions[paths[1].name],
        fields,
        scales,
        candidate_label=candidate_label,
    )
    plot_burned_area_differences(
        plt,
        paths[2],
        dimensions[paths[2].name],
        fields,
        scales,
        candidate_label=candidate_label,
    )
    plot_seasonal_cycles(
        plt,
        paths[3],
        dimensions[paths[3].name],
        fields,
        regions,
        scales,
        candidate_label=candidate_label,
    )
    plot_spatial_distribution(
        plt,
        paths[4],
        dimensions[paths[4].name],
        fields,
        scales,
        candidate_label=candidate_label,
    )
    plot_benchmark_sensitivity(
        plt,
        paths[5],
        dimensions[paths[5].name],
        scores,
        fields,
        regions,
        scales,
        candidate_label=candidate_label,
    )
    return paths
