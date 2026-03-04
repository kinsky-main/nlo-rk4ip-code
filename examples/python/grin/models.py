"""Shared data models for GRIN example apps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlotArtifact:
    """Describes a saved plot and how strict its image validation should be."""

    key: str
    path: Path
    allow_uniform: bool = False


@dataclass(frozen=True)
class ValidationCheck:
    """Single validation outcome."""

    name: str
    level: str
    passed: bool
    value: float | None = None
    threshold: float | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level,
            "passed": bool(self.passed),
            "value": None if self.value is None else float(self.value),
            "threshold": None if self.threshold is None else float(self.threshold),
            "detail": self.detail,
        }


@dataclass
class ValidationReport:
    """Aggregated validation output for run/replot workflows."""

    example_name: str
    run_group: str
    checks: list[ValidationCheck] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, check: ValidationCheck) -> None:
        self.checks.append(check)

    def add_threshold(
        self,
        *,
        name: str,
        value: float,
        threshold: float,
        level: str = "fail",
        comparator: str = "<=",
        detail: str = "",
    ) -> None:
        if comparator == "<=":
            passed = float(value) <= float(threshold)
        elif comparator == ">=":
            passed = float(value) >= float(threshold)
        else:
            raise ValueError("comparator must be '<=' or '>='.")
        self.add(
            ValidationCheck(
                name=name,
                level=level,
                passed=passed,
                value=float(value),
                threshold=float(threshold),
                detail=detail,
            )
        )

    def fail_count(self) -> int:
        return sum(1 for c in self.checks if c.level == "fail" and not c.passed)

    def warn_count(self) -> int:
        return sum(1 for c in self.checks if c.level == "warn" and not c.passed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "example_name": self.example_name,
            "run_group": self.run_group,
            "checks": [c.as_dict() for c in self.checks],
            "summary": {
                "total": len(self.checks),
                "failed": self.fail_count(),
                "warnings": self.warn_count(),
                "passed": sum(1 for c in self.checks if c.passed),
            },
            "metadata": self.metadata,
        }
