"""Fast in-memory recreation of the locked GFED5 ILAMB evaluation.

This is host-side infrastructure used internally by Optuna and ablation. It
reproduces the ILAMB 2.7.3 ``ConfBurntArea`` mean-state scalar scores without
invoking ILAMB or writing candidate NetCDF files. It is not a model-facing
tool and never records an experiment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

import numpy as np
from netCDF4 import Dataset


GFED5_SHA256: Final = "46594753a3f111e0ddc5526370708b0648f7f7b67458b42153dc04b7d035051b"
EARTH_RADIUS_METRES: Final = 6.371e6
MONTH_MIDPOINTS: Final = np.asarray(
    [15.5, 45.0, 74.5, 105.0, 135.5, 166.0, 196.5, 227.5, 258.0, 288.5, 319.0, 349.5]
)
OVERALL_WEIGHTS: Final = {
    "bias_score": 1.0,
    "rmse_score": 2.0,
    "seasonal_cycle_score": 1.0,
    "spatial_distribution_score": 1.0,
}

# These are the regions registered by ILAMB 2.7.3's ILAMB.Regions module.
# Bounds use ILAMB's lower-exclusive, upper-inclusive cell-centre convention
# and are stored in (south, north, west, east) order.
GFED_REGIONS: Final = {
    "global": None,
    "bona": (49.75, 79.75, -170.25, -60.25),
    "tena": (30.25, 49.75, -125.25, -66.25),
    "ceam": (9.75, 30.25, -115.25, -80.25),
    "nhsa": (0.25, 12.75, -80.25, -50.25),
    "shsa": (-59.75, 0.25, -80.25, -33.25),
    "euro": (35.25, 70.25, -10.25, 30.25),
    "mide": (20.25, 40.25, -10.25, 60.25),
    "nhaf": (0.25, 20.25, -20.25, 45.25),
    "shaf": (-34.75, 0.25, 10.25, 45.25),
    "boas": (54.75, 70.25, 30.25, 179.75),
    "ceas": (30.25, 54.75, 30.25, 142.58),
    "seas": (5.25, 30.25, 65.25, 120.25),
    "eqas": (-10.25, 10.25, 99.75, 150.25),
    "aust": (-41.25, -10.5, 112.0, 154.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _time_mean(values: np.ma.MaskedArray, weights: np.ndarray) -> np.ma.MaskedArray:
    shaped = weights.reshape((-1,) + (1,) * (values.ndim - 1))
    valid_weight = (~np.ma.getmaskarray(values)) * shaped
    return np.ma.sum(values * shaped, axis=0) / np.sum(valid_weight, axis=0)


def _inferred_time_weights(years: int) -> np.ndarray:
    """Match the bounds ILAMB Variable.rms() infers from monthly centres."""
    centres = np.concatenate([MONTH_MIDPOINTS + 365.0 * year for year in range(years)])
    bounds = np.zeros((centres.size, 2), dtype=np.float64)
    bounds[1:, 0] = 0.5 * (centres[:-1] + centres[1:])
    bounds[:-1, 1] = 0.5 * (centres[:-1] + centres[1:])
    bounds[0, 0] = centres[0] - 0.5 * (centres[1] - centres[0])
    bounds[-1, 1] = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    return np.diff(bounds, axis=1)[:, 0]


def _cell_areas(lat_bounds: np.ndarray, lon_bounds: np.ndarray) -> np.ndarray:
    return EARTH_RADIUS_METRES**2 * np.outer(
        np.sin(np.deg2rad(lat_bounds[:, 1])) - np.sin(np.deg2rad(lat_bounds[:, 0])),
        np.deg2rad(lon_bounds[:, 1] - lon_bounds[:, 0]),
    )


def _region_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    bounds: tuple[float, float, float, float] | None,
) -> np.ndarray:
    if bounds is None:
        return np.zeros((lat.size, lon.size), dtype=bool)
    south, north, west, east = bounds
    inside = (
        (lat[:, None] > south)
        & (lat[:, None] <= north)
        & (lon[None, :] > west)
        & (lon[None, :] <= east)
    )
    return ~inside


def _spatial_mean(values: np.ma.MaskedArray, area: np.ma.MaskedArray) -> float:
    return float(np.ma.sum(values * area) / np.ma.sum(area))


def _spatial_distribution(
    reference: np.ma.MaskedArray,
    candidate: np.ma.MaskedArray,
    region_mask: np.ndarray,
) -> float:
    ref = np.ma.array(reference, mask=np.ma.getmaskarray(reference) | region_mask)
    com = np.ma.array(candidate, mask=np.ma.getmaskarray(candidate) | region_mask)
    ref_std = float(ref.std())
    com_std = float(com.std())
    if not np.isfinite(ref_std) or ref_std <= 1e-12:
        return 0.0
    if not np.isfinite(com_std) or com_std <= 1e-12:
        # A spatially constant null model has no spatial-distribution skill.
        # Defining that skill as zero gives exact Shapley a finite empty-set
        # baseline without changing any non-degenerate model score.
        return 0.0
    normalized_std = com_std / ref_std

    # This intentionally matches ILAMB 2.7.3 Variable.correlation(), including
    # its use of the full grid-cell count when regional values are masked.
    count = ref.size
    ref_mean = ref.sum() / count
    com_mean = com.sum() / count
    cross = (ref * com).sum()
    ref_square = (ref * ref).sum()
    com_square = (com * com).sum()
    correlation = (cross - count * ref_mean * com_mean) / (
        np.sqrt(ref_square - count * ref_mean * ref_mean)
        * np.sqrt(com_square - count * com_mean * com_mean)
    )
    if np.ma.is_masked(correlation) or not np.isfinite(float(correlation)):
        return 0.0
    return float(
        4.0
        * (1.0 + correlation)
        / ((normalized_std + 1.0 / normalized_std) ** 2 * 2.0)
    )


class GFED5Evaluator:
    """Preload GFED5 once and score repeated model predictions in memory."""

    def __init__(self, reference_path: str | Path, *, verify_checksum: bool = True) -> None:
        path = Path(reference_path).expanduser().resolve()
        if verify_checksum and _sha256(path) != GFED5_SHA256:
            raise ValueError(f"GFED5 checksum does not match the locked benchmark: {path}")

        with Dataset(path) as dataset:
            variable = dataset.variables["burntArea"]
            if variable.dimensions != ("time", "lat", "lon") or variable.units != "%":
                raise ValueError("GFED5 must expose burntArea(time, lat, lon) in percent")
            reference = np.ma.asarray(variable[:192])
            self.lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
            self.lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
            lat_bounds = np.asarray(dataset.variables[dataset.variables["lat"].bounds][:])
            lon_bounds = np.asarray(dataset.variables[dataset.variables["lon"].bounds][:])
            time_bounds = np.asarray(dataset.variables[dataset.variables["time"].bounds][:192])

        if reference.shape != (192, 360, 720):
            raise ValueError(f"GFED5 has unexpected comparison shape {reference.shape}")
        if np.ma.getmaskarray(reference).any():
            raise ValueError("the locked GFED5 benchmark unexpectedly contains missing cells")

        self.area = _cell_areas(lat_bounds, lon_bounds)
        self.month_lengths = np.diff(time_bounds, axis=1)[:, 0]
        self.reference_mean = _time_mean(reference, self.month_lengths)
        temporal_weights = _inferred_time_weights(years=16)
        self.reference_temporal_std = np.ma.sqrt(
            _time_mean((reference - self.reference_mean[None, ...]) ** 2, temporal_weights)
        )
        self.reference_cycle = reference.reshape(16, 12, 360, 720).mean(axis=0)
        self.reference_phase = MONTH_MIDPOINTS[np.argmax(self.reference_cycle, axis=0)]
        self.regions = {
            name: _region_mask(self.lat, self.lon, bounds)
            for name, bounds in GFED_REGIONS.items()
        }

    @staticmethod
    def _candidate_percent(prediction: np.ndarray) -> np.ma.MaskedArray:
        candidate = np.asarray(prediction)
        if candidate.shape != (192, 180, 360):
            raise ValueError(
                "predict() must return monthly 1-degree burned fractions with shape "
                f"(192, 180, 360), got {candidate.shape}"
            )
        if not np.isfinite(candidate).all():
            raise ValueError("predict() returned non-finite burned fractions")
        minimum = float(candidate.min())
        maximum = float(candidate.max())
        if minimum < -1e-7 or maximum > 1.0 + 1e-7:
            raise ValueError(f"predict() returned burned fractions outside [0, 1]: {minimum}, {maximum}")

        # Official candidates are stored as float32 and repeated into the four
        # 0.5-degree GFED5 cells. Repeating a fraction preserves burned area.
        candidate = np.asarray(candidate, dtype=np.float32)
        candidate = np.repeat(np.repeat(candidate, 2, axis=1), 2, axis=2)
        candidate *= np.float32(100.0)
        return np.ma.asarray(candidate)

    def score(self, prediction: np.ndarray) -> dict[str, dict[str, float]]:
        candidate = self._candidate_percent(prediction)
        candidate_mean = _time_mean(candidate, self.month_lengths)
        candidate_cycle = candidate.reshape(16, 12, 360, 720).mean(axis=0)

        bias = candidate_mean - self.reference_mean
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            bias_score = np.ma.exp(-np.ma.abs(bias / self.reference_temporal_std))
        bias_score[bias_score < 1e-16] = 0.0

        reference_anomaly = self.reference_cycle - self.reference_cycle.mean(axis=0)
        candidate_anomaly = candidate_cycle - candidate_cycle.mean(axis=0)
        cycle_weights = np.asarray([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        rmse = np.ma.sqrt(_time_mean((candidate_cycle - self.reference_cycle) ** 2, cycle_weights))
        centered_rmse = np.ma.sqrt(
            _time_mean((candidate_anomaly - reference_anomaly) ** 2, cycle_weights)
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            rmse_score = np.ma.exp(-np.ma.abs(centered_rmse / self.reference_temporal_std))
        rmse_score[rmse_score < 1e-16] = 0.0

        candidate_phase = MONTH_MIDPOINTS[np.argmax(candidate_cycle, axis=0)]
        phase_shift = candidate_phase - self.reference_phase
        phase_shift += (phase_shift < -182.5) * 365.0
        phase_shift -= (phase_shift > 182.5) * 365.0
        seasonal_score = 0.5 * (1.0 + np.cos(np.abs(phase_shift) / 365.0 * 2.0 * np.pi))

        scores: dict[str, dict[str, float]] = {}
        for region, region_mask in self.regions.items():
            area = np.ma.array(self.area, mask=region_mask)
            reference_weight = area * self.reference_mean
            values = {
                "benchmark_period_mean_percent": _spatial_mean(self.reference_mean, area),
                "model_period_mean_percent": _spatial_mean(candidate_mean, area),
                "spatial_distribution_score": _spatial_distribution(
                    self.reference_mean, candidate_mean, region_mask
                ),
                "phase_shift_months": _spatial_mean(np.ma.abs(phase_shift) / 30.0, area),
                "seasonal_cycle_score": _spatial_mean(seasonal_score, reference_weight),
                "bias_percent": _spatial_mean(bias, area),
                "bias_score": _spatial_mean(bias_score, reference_weight),
                "rmse_percent": _spatial_mean(rmse, area),
                "rmse_score": _spatial_mean(rmse_score, reference_weight),
            }
            values["overall_score"] = sum(
                values[name] * weight for name, weight in OVERALL_WEIGHTS.items()
            ) / sum(OVERALL_WEIGHTS.values())
            scores[region] = values
        return scores
