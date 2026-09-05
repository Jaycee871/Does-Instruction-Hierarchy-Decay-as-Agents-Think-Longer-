from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from .data import IHExample


@dataclass(frozen=True)
class PilotCandidate:
    source_file: str
    row_index: int
    task_type: str
    attack_level: str
    privileged_level: str

    @classmethod
    def from_example(cls, example: IHExample) -> "PilotCandidate":
        return cls(
            source_file=example.source_file,
            row_index=example.row_index,
            task_type=str(example.metadata.get("task_type", "<missing>")),
            attack_level=str(example.metadata.get("attack_level", "<missing>")),
            privileged_level=str(example.metadata.get("privileged_level", "<missing>")),
        )

    @property
    def example_id(self) -> str:
        return f"{self.source_file}:{self.row_index}"

    @property
    def authority_pair(self) -> tuple[str, str]:
        return self.privileged_level, self.attack_level

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["example_id"] = self.example_id
        return out


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "|".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _balanced_quotas(keys: Sequence[tuple[str, str]], n: int, seed: int) -> dict[tuple[str, str], int]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if not keys:
        return {}
    base, remainder = divmod(n, len(keys))
    ordered = list(sorted(keys))
    random.Random(_stable_seed(seed, "authority-remainder")).shuffle(ordered)
    quotas = {key: base for key in keys}
    for key in ordered[:remainder]:
        quotas[key] += 1
    return quotas


def _round_robin_task_types(
    candidates: Sequence[PilotCandidate], quota: int, seed: int, pair: tuple[str, str]
) -> list[PilotCandidate]:
    by_task: dict[str, list[PilotCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_task[candidate.task_type].append(candidate)

    task_types = sorted(by_task)
    random.Random(_stable_seed(seed, *pair, "task-order")).shuffle(task_types)
    for task_type, rows in by_task.items():
        random.Random(_stable_seed(seed, *pair, task_type, "rows")).shuffle(rows)

    selected: list[PilotCandidate] = []
    cursors = {task_type: 0 for task_type in task_types}
    while len(selected) < quota:
        made_progress = False
        for task_type in task_types:
            cursor = cursors[task_type]
            rows = by_task[task_type]
            if cursor >= len(rows):
                continue
            selected.append(rows[cursor])
            cursors[task_type] += 1
            made_progress = True
            if len(selected) == quota:
                break
        if not made_progress:
            break
    return selected


def select_stratified(
    candidates: Iterable[PilotCandidate], *, n: int, seed: int
) -> list[PilotCandidate]:
    """Select a deterministic pilot sample with authority-pair balance.

    The sampler first distributes the requested sample approximately equally across
    observed (privileged_level, attack_level) pairs. Within each authority pair it
    round-robins across task types before taking a second example from any subtype.
    This prevents the large blue_team_auto family from silently defining the pilot.
    """
    rows = list(candidates)
    if n > len(rows):
        raise ValueError(f"Requested {n} examples from only {len(rows)} candidates")

    by_pair: dict[tuple[str, str], list[PilotCandidate]] = defaultdict(list)
    for candidate in rows:
        by_pair[candidate.authority_pair].append(candidate)

    quotas = _balanced_quotas(list(by_pair), n, seed)
    selected: list[PilotCandidate] = []
    for pair in sorted(by_pair):
        quota = min(quotas[pair], len(by_pair[pair]))
        selected.extend(_round_robin_task_types(by_pair[pair], quota, seed, pair))

    # If a very small stratum could not fill its quota, top up from the remaining pool.
    if len(selected) < n:
        chosen = {candidate.example_id for candidate in selected}
        remainder = [candidate for candidate in rows if candidate.example_id not in chosen]
        random.Random(_stable_seed(seed, "top-up")).shuffle(remainder)
        selected.extend(remainder[: n - len(selected)])

    return sorted(selected, key=lambda item: (item.source_file, item.row_index))
