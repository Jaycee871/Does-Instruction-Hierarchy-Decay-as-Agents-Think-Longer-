from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DEFAULT_CHECKPOINTS = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class TrajectoryPlan:
    checkpoints: tuple[int, ...] = DEFAULT_CHECKPOINTS
    preserve_original_messages: bool = True
    carry_forward_model_output: bool = True
    inject_new_lower_priority_content: bool = False

    def __post_init__(self) -> None:
        if not self.checkpoints:
            raise ValueError("At least one checkpoint is required")
        if any(step <= 0 for step in self.checkpoints):
            raise ValueError("Checkpoint steps must be positive")
        if any(b <= a for a, b in zip(self.checkpoints, self.checkpoints[1:])):
            raise ValueError("Checkpoint steps must be strictly increasing")


def checkpoint_events(max_step: int, checkpoints: Iterable[int] = DEFAULT_CHECKPOINTS):
    """Yield `(step, should_score)` for a trajectory up to `max_step`."""
    checkpoint_set = set(checkpoints)
    for step in range(1, max_step + 1):
        yield step, step in checkpoint_set
