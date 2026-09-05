from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from ih_decay.data import DATA_FILES, iter_examples


def _value_shape(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def main() -> None:
    report: dict[str, Any] = {"files": {}}

    for source_file in DATA_FILES:
        role_sequences: Counter[str] = Counter()
        metadata_keys: Counter[str] = Counter()
        metadata_shapes: dict[str, Counter[str]] = {}
        template_lengths: Counter[int] = Counter()
        placeholder_declared = 0
        placeholder_found_in_template = 0
        grader_present = 0
        grader_lengths: list[int] = []
        rows = 0

        for example in iter_examples(source_file, token=os.getenv("HF_TOKEN")):
            rows += 1
            template = example.defender_problem_template
            roles = []
            rendered = []
            for message in template:
                roles.append(str(message.get("role", "<missing>")))
                content = message.get("content", "")
                if isinstance(content, str):
                    rendered.append(content)
            role_sequences[" -> ".join(roles)] += 1
            template_lengths[len(template)] += 1

            for key, value in example.metadata.items():
                metadata_keys[key] += 1
                metadata_shapes.setdefault(key, Counter())[_value_shape(value)] += 1

            attack_placeholder = example.metadata.get("attack_placeholder")
            if isinstance(attack_placeholder, str) and attack_placeholder:
                placeholder_declared += 1
                if attack_placeholder in "\n".join(rendered):
                    placeholder_found_in_template += 1

            grader = example.metadata.get("grader_code_python")
            if isinstance(grader, str) and grader:
                grader_present += 1
                grader_lengths.append(len(grader))

        report["files"][source_file] = {
            "rows": rows,
            "role_sequences": dict(role_sequences.most_common()),
            "template_lengths": {str(k): v for k, v in sorted(template_lengths.items())},
            "metadata_keys_present": dict(metadata_keys.most_common()),
            "metadata_value_shapes": {
                key: dict(counter.most_common())
                for key, counter in sorted(metadata_shapes.items())
            },
            "attack_placeholder": {
                "declared_rows": placeholder_declared,
                "found_verbatim_in_template_rows": placeholder_found_in_template,
            },
            "grader_code_python": {
                "present_rows": grader_present,
                "min_length": min(grader_lengths) if grader_lengths else None,
                "max_length": max(grader_lengths) if grader_lengths else None,
                "mean_length": (
                    round(sum(grader_lengths) / len(grader_lengths), 2)
                    if grader_lengths
                    else None
                ),
            },
        }

    Path("schema-audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
