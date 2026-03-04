"""Plot/data validation helpers for GRIN examples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from backend.metrics import mean_pointwise_abs_relative_error
from matplotlib.image import imread

from .models import PlotArtifact, ValidationCheck, ValidationReport


@dataclass(frozen=True)
class WindowResult:
    left_index: int
    right_index: int
    mass_fraction: float


class WavelengthWindowSelector:
    """Selects an intensity-driven wavelength axis support."""

    def __init__(self, mass_fraction: float = 0.999):
        if mass_fraction <= 0.0 or mass_fraction > 1.0:
            raise ValueError("mass_fraction must be in (0, 1].")
        self.mass_fraction = float(mass_fraction)

    def select(
        self,
        axis_nm: np.ndarray,
        map_a: np.ndarray,
        map_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, WindowResult]:
        axis = np.asarray(axis_nm, dtype=np.float64).reshape(-1)
        data_a = np.asarray(map_a, dtype=np.float64)
        data_b = np.asarray(map_b, dtype=np.float64)
        if data_a.shape != data_b.shape:
            raise ValueError("map_a and map_b must share shape.")
        if data_a.ndim != 2:
            raise ValueError("spectral maps must have shape [record, wavelength].")
        if data_a.shape[1] != axis.size:
            raise ValueError("axis length must match spectral map width.")

        weights = np.clip(np.nan_to_num(data_a + data_b, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None).sum(axis=0)
        total = float(np.sum(weights))
        if total <= 0.0:
            return axis, data_a, data_b, WindowResult(0, int(axis.size - 1), 0.0)

        peak = int(np.argmax(weights))
        left = peak
        right = peak
        mass = float(weights[peak])
        target = self.mass_fraction * total

        while mass < target and (left > 0 or right < (axis.size - 1)):
            left_candidate = float(weights[left - 1]) if left > 0 else -1.0
            right_candidate = float(weights[right + 1]) if right < (axis.size - 1) else -1.0
            if right_candidate > left_candidate:
                right += 1
                mass += float(weights[right])
            else:
                left -= 1
                mass += float(weights[left])

        keep = slice(int(left), int(right + 1))
        selected_mass = mass / total if total > 0.0 else 0.0
        return axis[keep], data_a[:, keep], data_b[:, keep], WindowResult(int(left), int(right), float(selected_mass))


class PlotImageValidator:
    """Applies coarse sanity checks to saved PNG figures."""

    @staticmethod
    def _load_grayscale(path: Path) -> np.ndarray:
        image = np.asarray(imread(path), dtype=np.float64)
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] >= 3:
            gray = image[..., :3].mean(axis=2)
        else:
            raise ValueError(f"unsupported image shape for {path}: {image.shape}")
        return np.clip(np.nan_to_num(gray, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    def validate_artifacts(self, report: ValidationReport, artifacts: list[PlotArtifact]) -> None:
        for artifact in artifacts:
            path = Path(artifact.path)
            if not path.is_file():
                report.add(
                    ValidationCheck(
                        name=f"plot_exists:{artifact.key}",
                        level="fail",
                        passed=False,
                        detail=f"missing plot file: {path}",
                    )
                )
                continue
            report.add(
                ValidationCheck(
                    name=f"plot_exists:{artifact.key}",
                    level="fail",
                    passed=True,
                    detail=str(path),
                )
            )

            gray = self._load_grayscale(path)
            nonwhite = float(np.mean(gray < 0.98))
            std_value = float(np.std(gray))

            nonwhite_threshold = 0.02 if artifact.allow_uniform else 0.05
            std_threshold = 0.003 if artifact.allow_uniform else 0.01
            report.add_threshold(
                name=f"plot_nonwhite_ratio:{artifact.key}",
                value=nonwhite,
                threshold=nonwhite_threshold,
                comparator=">=",
                level="warn" if artifact.allow_uniform else "fail",
                detail=f"path={path}",
            )
            report.add_threshold(
                name=f"plot_gray_std:{artifact.key}",
                value=std_value,
                threshold=std_threshold,
                comparator=">=",
                level="warn" if artifact.allow_uniform else "fail",
                detail=f"path={path}",
            )


def relative_l2_error(numerical: np.ndarray, reference: np.ndarray) -> float:
    return mean_pointwise_abs_relative_error(
        numerical,
        reference,
        context="grin_validation:relative_error",
    )


def profile_correlation(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    a = np.asarray(curve_a, dtype=np.float64).reshape(-1)
    b = np.asarray(curve_b, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError("profile curves must share length for correlation.")
    a0 = a - float(np.mean(a))
    b0 = b - float(np.mean(b))
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a0, b0) / denom)


def write_report(report: ValidationReport, output_path: Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report.as_dict(), f, indent=2, sort_keys=True)
    return path
