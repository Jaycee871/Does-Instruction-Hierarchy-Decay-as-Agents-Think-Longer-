from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class MaterializedPrompt:
    messages: list[dict[str, str]]
    placeholder: str
    replacements: int


def materialize_attack(
    template: Sequence[Mapping[str, str]],
    *,
    placeholder: str,
    attack_text: str,
    require_exactly_one: bool = True,
) -> MaterializedPrompt:
    """Replace an IH-Challenge attack placeholder without mutating the source template.

    This function performs string substitution only. It never evaluates dataset-provided
    grader code and never interprets the generated attack as executable content.
    """
    if not placeholder:
        raise ValueError("placeholder must be non-empty")
    if not attack_text.strip():
        raise ValueError("attack_text must be non-empty")

    messages = deepcopy([dict(message) for message in template])
    replacements = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        count = content.count(placeholder)
        if count:
            message["content"] = content.replace(placeholder, attack_text)
            replacements += count

    if require_exactly_one and replacements != 1:
        raise ValueError(
            f"Expected exactly one placeholder occurrence, found {replacements}"
        )
    if not require_exactly_one and replacements == 0:
        raise ValueError("Placeholder was not found in the template")

    return MaterializedPrompt(
        messages=messages,
        placeholder=placeholder,
        replacements=replacements,
    )
