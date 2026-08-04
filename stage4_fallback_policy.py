"""Deterministic deployment policy for Stage 4C satellite missingness."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Stage4FallbackPolicy:
    """Route by observed satellite availability, without labels or model scores.

    Complete inputs use the learned quality gate, partially missing inputs use
    the frozen CoRA adapter, and fully missing inputs use the static mask gate.
    """

    complete_group: str = "quality_gate"
    partial_group: str = "cora"
    fully_missing_group: str = "static_mask_gate"
    endpoint_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 <= self.endpoint_tolerance < 0.5:
            raise ValueError("endpoint_tolerance must be within [0, 0.5)")

    def select_group(self, satellite_missing_ratio: float) -> str:
        ratio = float(satellite_missing_ratio)
        if not math.isfinite(ratio):
            raise ValueError("satellite_missing_ratio must be finite")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("satellite_missing_ratio must be within [0, 1]")
        if ratio <= self.endpoint_tolerance:
            return self.complete_group
        if ratio >= 1.0 - self.endpoint_tolerance:
            return self.fully_missing_group
        return self.partial_group

    def select_scenario_group(self, scenario: str) -> str:
        ratios = {"complete": 0.0, "missing50": 0.5, "missing100": 1.0}
        if scenario not in ratios:
            raise ValueError(f"unsupported Stage 4C scenario: {scenario}")
        return self.select_group(ratios[scenario])

    def as_dict(self) -> dict[str, str]:
        return {
            "complete": self.complete_group,
            "missing50": self.partial_group,
            "missing100": self.fully_missing_group,
        }
