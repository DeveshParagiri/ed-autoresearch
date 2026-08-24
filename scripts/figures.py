"""Create the two ephemeral diagnostics for the current model."""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from scripts.fast_ilamb import GFED5Evaluator
from scripts.runtime import (
    GFED5_PATH,
    ModelError,
    load_inputs,
    load_land_mask,
    load_model,
    predict_current,
    validate_model,
)


GFED5_COLOR = "#000000"
MODEL_COLOR = "#FF2C00"
NEUTRAL = "#3B3B3B"
ANNUAL_PERCENT_MAX = 80.0
DIFFERENCE_PERCENT_ABS = 60.0
OCEAN_COLOR = "#dce7ec"
REGION_LABELS = {
    "global": "Global",
    "bona": "Boreal North America",
    "tena": "Temperate North America",
    "ceam": "Central America",
    "nhsa": "Northern South America",
    "shsa": "Southern South America",
    "euro": "Europe",
    "mide": "Middle East",
    "nhaf": "Northern Africa",
    "shaf": "Southern Africa",
    "boas": "Boreal Asia",
    "ceas": "Central Asia",
    "seas": "Southeast Asia",
    "eqas": "Equatorial Asia",
    "aust": "Australia",
}
SEASONAL_LIMITS = {
    "global": 0.5,
    "bona": 0.1,
    "tena": 2.5,
    "ceam": 1.2,
    "nhsa": 2.2,
    "shsa": 1.8,
    "euro": 1.6,
    "mide": 0.9,
    "nhaf": 5.0,
    "shaf": 4.5,
    "boas": 0.6,
    "ceas": 1.7,
    "seas": 2.0,
    "eqas": 0.25,
    "aust": 1.0,
}


def _configure_plotting(config_root: Path) -> Any:
    """Use the retained dev SciencePlots style without its custom font."""
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(config_root)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex", "bright"])
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
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


def _save(plt: Any, figure: Any, path: Path, dimensions: tuple[int, int]) -> None:
    dpi = 150
    figure.set_size_inches(dimensions[0] / dpi, dimensions[1] / dpi)
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
    if actual != dimensions:
        raise ValueError(f"figure has dimensions {actual}, expected {dimensions}")


def _panel_label(axis: Any, label: str, *, map_panel: bool = False) -> None:
    if map_panel:
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
    else:
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


def _longitude(value: float, _: int) -> str:
    if value == 0:
        return "0°"
    return f"{abs(int(value))}°{'W' if value < 0 else 'E'}"


def _latitude(value: float, _: int) -> str:
    if value == 0:
        return "0°"
    return f"{abs(int(value))}°{'S' if value < 0 else 'N'}"


def _spatial_cycle(cycle: np.ndarray, area: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = np.where(mask, 0.0, area)
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError("regional figure mask has zero area")
    return np.sum(cycle * weights[None, ...], axis=(1, 2)) / denominator


def _comparison_figure(
    plt: Any,
    path: Path,
    observed: np.ndarray,
    model: np.ndarray,
    land: np.ndarray,
) -> None:
    from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, NullLocator

    observed = np.where(land, observed, np.nan)
    model = np.where(land, model, np.nan)
    difference = model - observed
    burned_cmap = plt.get_cmap("Reds").copy()
    burned_cmap.set_bad(OCEAN_COLOR)
    difference_cmap = plt.get_cmap("RdBu_r").copy()
    difference_cmap.set_bad(OCEAN_COLOR)
    figure = plt.figure(figsize=(14, 11.5))
    grid = figure.add_gridspec(
        5,
        2,
        height_ratios=(1.0, 0.065, 0.16, 2.0, 0.065),
        left=0.065,
        right=0.98,
        bottom=0.07,
        top=0.92,
        hspace=0.16,
        wspace=0.08,
    )
    field_axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]))
    difference_axis = figure.add_subplot(grid[3, :])
    field_image = None
    panels = (("GFED5", observed), ("Current model", model))
    for index, (axis, (title, values)) in enumerate(zip(field_axes, panels, strict=True)):
        field_image = axis.imshow(
            values,
            origin="lower",
            extent=(-180, 180, -90, 90),
            vmin=0,
            vmax=ANNUAL_PERCENT_MAX,
            cmap=burned_cmap,
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
        axis.contour(
            np.linspace(-179.75, 179.75, land.shape[1]),
            np.linspace(-89.75, 89.75, land.shape[0]),
            land.astype(np.uint8),
            levels=(0.5,),
            colors="#75858d",
            linewidths=0.35,
        )
        if index:
            axis.tick_params(labelleft=False)
        _panel_label(axis, chr(ord("a") + index), map_panel=True)

    difference_image = difference_axis.imshow(
        difference,
        origin="lower",
        extent=(-180, 180, -90, 90),
        vmin=-DIFFERENCE_PERCENT_ABS,
        vmax=DIFFERENCE_PERCENT_ABS,
        cmap=difference_cmap,
        aspect="auto",
        interpolation="nearest",
        rasterized=True,
    )
    difference_axis.set_title("Model − GFED5", pad=5)
    difference_axis.set_xlim(-180, 180)
    difference_axis.set_ylim(-90, 90)
    difference_axis.set_xticks((-120, 0, 120))
    difference_axis.set_yticks((-60, 0, 60))
    difference_axis.xaxis.set_major_formatter(FuncFormatter(_longitude))
    difference_axis.yaxis.set_major_formatter(FuncFormatter(_latitude))
    difference_axis.grid(color="white", alpha=0.28, linewidth=0.45)
    difference_axis.contour(
        np.linspace(-179.75, 179.75, land.shape[1]),
        np.linspace(-89.75, 89.75, land.shape[0]),
        land.astype(np.uint8),
        levels=(0.5,),
        colors="#75858d",
        linewidths=0.35,
    )
    _panel_label(difference_axis, "c", map_panel=True)

    if field_image is None:
        raise RuntimeError("absolute map panels were not created")
    absolute_axis = figure.add_subplot(grid[1, :])
    field_ticks = np.linspace(0, ANNUAL_PERCENT_MAX, 5)
    absolute_bar = figure.colorbar(
        field_image,
        cax=absolute_axis,
        orientation="horizontal",
        ticks=field_ticks,
    )
    absolute_bar.ax.xaxis.set_major_locator(FixedLocator(field_ticks))
    absolute_labels = [f"{value:g}" for value in field_ticks]
    absolute_labels[-1] = f"≥{ANNUAL_PERCENT_MAX:g}"
    absolute_bar.ax.xaxis.set_major_formatter(FixedFormatter(tuple(absolute_labels)))
    absolute_bar.ax.xaxis.set_minor_locator(NullLocator())
    absolute_bar.set_label("Burned area (% grid-cell area yr$^{-1}$)", labelpad=6)

    difference_color_axis = figure.add_subplot(grid[4, :])
    difference_ticks = np.linspace(-DIFFERENCE_PERCENT_ABS, DIFFERENCE_PERCENT_ABS, 5)
    difference_bar = figure.colorbar(
        difference_image,
        cax=difference_color_axis,
        orientation="horizontal",
        ticks=difference_ticks,
    )
    difference_bar.ax.xaxis.set_major_locator(FixedLocator(difference_ticks))
    difference_labels = [f"{value:g}" for value in difference_ticks]
    difference_labels[0] = f"≤−{DIFFERENCE_PERCENT_ABS:g}"
    difference_labels[-1] = f"≥{DIFFERENCE_PERCENT_ABS:g}"
    difference_bar.ax.xaxis.set_major_formatter(FixedFormatter(tuple(difference_labels)))
    difference_bar.ax.xaxis.set_minor_locator(NullLocator())
    difference_bar.set_label("Model minus GFED5 (% yr$^{-1}$)", labelpad=6)
    figure.suptitle("Current model: GFED5 burned-area comparison", fontsize=16, fontweight="semibold")
    _save(plt, figure, path, (2100, 1725))


def _seasonal_figure(
    plt: Any,
    path: Path,
    evaluator: GFED5Evaluator,
    observed_cycle: np.ndarray,
    model_cycle: np.ndarray,
) -> None:
    figure, axes = plt.subplots(3, 5, figsize=(16, 10), layout="constrained", sharex=True)
    months = np.arange(1, 13)
    for index, (axis, region) in enumerate(
        zip(axes.flat, evaluator.regions, strict=True)
    ):
        observed = _spatial_cycle(observed_cycle, evaluator.area, evaluator.regions[region])
        modeled = _spatial_cycle(model_cycle, evaluator.area, evaluator.regions[region])
        axis.plot(months, observed, label="GFED5", color=GFED5_COLOR, linewidth=1.5)
        axis.plot(months, modeled, label="Current model", color=MODEL_COLOR, linewidth=2.1)
        axis.set_title(REGION_LABELS[region], fontsize=9.5)
        axis.set_xlim(1, 12)
        axis.set_ylim(0, SEASONAL_LIMITS[region])
        axis.set_xticks((1, 3, 5, 7, 9, 11), ("Jan", "Mar", "May", "Jul", "Sep", "Nov"))
        axis.set_yticks(np.linspace(0, SEASONAL_LIMITS[region], 4))
        axis.grid(color="#D9D9D9", linewidth=0.55)
        _panel_label(axis, chr(ord("a") + index))
    for axis in axes[2, :]:
        axis.set_xlabel("month")
    for axis in axes[:, 0]:
        axis.set_ylabel("burned area (% month$^{-1}$)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=2, handlelength=3.0)
    figure.suptitle("Current model: seasonal cycles", fontsize=16, fontweight="semibold")
    _save(plt, figure, path, (2400, 1500))


def generate_figures(
    prediction: np.ndarray,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Render the retained dev-style diagnostics into temporary storage."""
    destination = output_dir or Path(tempfile.mkdtemp(prefix="ed-fire-figures-"))
    destination.mkdir(parents=True, exist_ok=True)
    mplconfig = destination / ".mplconfig"
    plt = _configure_plotting(mplconfig)
    comparison_path = destination / "gfed5-model-difference.png"
    seasonal_path = destination / "seasonal-cycle.png"

    evaluator = GFED5Evaluator(GFED5_PATH)
    land = np.repeat(np.repeat(load_land_mask(), 2, axis=0), 2, axis=1)
    model_annual = np.repeat(
        np.repeat(np.asarray(prediction).mean(axis=0) * 1200.0, 2, axis=0),
        2,
        axis=1,
    )
    observed_annual = np.ma.filled(evaluator.reference_mean * 12.0, np.nan)
    _comparison_figure(plt, comparison_path, observed_annual, model_annual, land)

    model_cycle = np.repeat(
        np.repeat(
            np.asarray(prediction).reshape(16, 12, 180, 360).mean(axis=0) * 100.0,
            2,
            axis=1,
        ),
        2,
        axis=2,
    )
    observed_cycle = np.ma.filled(evaluator.reference_cycle, np.nan)
    _seasonal_figure(plt, seasonal_path, evaluator, observed_cycle, model_cycle)
    shutil.rmtree(mplconfig, ignore_errors=True)
    return comparison_path, seasonal_path


def run(args: argparse.Namespace) -> int:
    """Recreate the current model's two ephemeral diagnostic figures."""
    del args
    try:
        model = load_model()
        inputs, _ = validate_model(model)
        data = load_inputs(inputs)
        prediction = predict_current(model, data)
        comparison, seasonal = generate_figures(prediction)
        print(f"comparison={comparison}")
        print(f"seasonal_cycle={seasonal}")
    except (ModelError, OSError, ValueError) as error:
        print(f"ar figures: {error}", file=sys.stderr)
        return 2
    return 0
