from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

from huggingface_hub import HfFileSystem

DEFAULT_BUCKET = "hf://buckets/Jaycee766/ih-challenge-bucket"
DATA_FILES = (
    "single-constraint.jsonl",
    "multi-constraint.jsonl",
    "input-conditioned.jsonl",
    "anti-overrefusal.jsonl",
)


@dataclass(frozen=True)
class IHExample:
    source_file: str
    row_index: int
    attacker_meta_problem: str
    attacker_problem: str
    defender_problem_template: list[dict[str, str]]
    metadata: dict[str, Any]

    @property
    def example_id(self) -> str:
        return f"{self.source_file}:{self.row_index}"


def iter_examples(
    source_file: str,
    *,
    bucket: str = DEFAULT_BUCKET,
    limit: int | None = None,
    token: str | None = None,
) -> Iterator[IHExample]:
    """Stream IH-Challenge examples directly from a Hugging Face bucket.

    The function intentionally does not execute `grader_code_python`; dataset-provided
    grader code must be handled by a separate isolated evaluation layer.
    """
    if source_file not in DATA_FILES:
        raise ValueError(f"Unknown IH-Challenge file: {source_file}")

    fs = HfFileSystem(token=token)
    path = f"{bucket.rstrip('/')}/{source_file}"
    with fs.open(path, "r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if limit is not None and row_index >= limit:
                return
            row = json.loads(line)
            yield IHExample(
                source_file=source_file,
                row_index=row_index,
                attacker_meta_problem=row["attacker_meta_problem"],
                attacker_problem=row["attacker_problem"],
                defender_problem_template=row["defender_problem_template"],
                metadata=row["metadata"],
            )


def summarize_metadata(examples: Iterator[IHExample]) -> dict[str, dict[str, int]]:
    """Count key categorical fields without retaining full examples in memory."""
    fields = ("task_type", "attack_level", "privileged_level")
    out: dict[str, dict[str, int]] = {field: {} for field in fields}
    for example in examples:
        for field in fields:
            value = str(example.metadata.get(field, "<missing>"))
            out[field][value] = out[field].get(value, 0) + 1
    return out
