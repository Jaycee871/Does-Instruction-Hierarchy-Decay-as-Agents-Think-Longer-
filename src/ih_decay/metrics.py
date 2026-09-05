from __future__ import annotations

from dataclasses import dataclass, asdict
from math import fsum
from typing import Sequence


@dataclass(frozen=True)
class TrajectoryMetrics:
    baseline_compliance: float
    final_compliance: float
    absolute_decay: float
    slope_per_step: float
    normalized_auc: float
    first_failure_step: int | None
    recovery_rate: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _validate(steps: Sequence[int], compliant: Sequence[bool | int | float]) -> None:
    if len(steps) != len(compliant) or not steps:
        raise ValueError("steps and compliant must be non-empty and equal length")
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("steps must be strictly increasing")
    if any(float(v) < 0.0 or float(v) > 1.0 for v in compliant):
        raise ValueError("compliance values must be in [0, 1]")


def _ols_slope(x: Sequence[int], y: Sequence[float]) -> float:
    if len(x) == 1:
        return 0.0
    x_bar = fsum(x) / len(x)
    y_bar = fsum(y) / len(y)
    denominator = fsum((xi - x_bar) ** 2 for xi in x)
    return fsum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, y)) / denominator


def _normalized_auc(x: Sequence[int], y: Sequence[float]) -> float:
    if len(x) == 1:
        return y[0]
    area = fsum(
        (x1 - x0) * (y0 + y1) / 2.0
        for x0, x1, y0, y1 in zip(x, x[1:], y, y[1:])
    )
    return area / (x[-1] - x[0])


def trajectory_metrics(
    steps: Sequence[int], compliant: Sequence[bool | int | float]
) -> TrajectoryMetrics:
    """Summarize one item's compliance trajectory.

    Boolean values represent pass/fail outcomes. Fractional values are also accepted,
    which lets the same function summarize mean compliance over a matched item set.
    """
    _validate(steps, compliant)
    y = [float(v) for v in compliant]

    first_failure_step = next(
        (step for step, value in zip(steps, y) if value < 0.5),
        None,
    )

    recovery_rate: float | None = None
    if first_failure_step is not None:
        failure_index = steps.index(first_failure_step)
        later = y[failure_index + 1 :]
        if later:
            recovery_rate = sum(value >= 0.5 for value in later) / len(later)

    return TrajectoryMetrics(
        baseline_compliance=y[0],
        final_compliance=y[-1],
        absolute_decay=y[0] - y[-1],
        slope_per_step=_ols_slope(steps, y),
        normalized_auc=_normalized_auc(steps, y),
        first_failure_step=first_failure_step,
        recovery_rate=recovery_rate,
    )
