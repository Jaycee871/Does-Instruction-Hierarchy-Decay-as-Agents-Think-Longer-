from __future__ import annotations

import json
import os
from pathlib import Path

from ih_decay.attacks import freeze_attack_nvidia
from ih_decay.data import iter_examples
from ih_decay.materialize import materialize_attack
from ih_decay.providers import nvidia_chat

SOURCE_FILE = "single-constraint.jsonl"
MODEL = "openai/gpt-oss-20b"
TARGET_PAIRS = {
    ("system", "user"),
    ("system", "developer"),
    ("developer", "user"),
}


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    if not nvidia_key:
        raise SystemExit("NVIDIA_API_KEY is not set")

    chosen = {}
    for example in iter_examples(SOURCE_FILE, token=hf_token):
        pair = (
            str(example.metadata.get("privileged_level")),
            str(example.metadata.get("attack_level")),
        )
        if pair in TARGET_PAIRS and pair not in chosen:
            chosen[pair] = example
        if len(chosen) == len(TARGET_PAIRS):
            break

    if set(chosen) != TARGET_PAIRS:
        raise SystemExit(f"Could not find all target authority pairs: {set(chosen)}")

    records = []
    for pair in sorted(chosen):
        example = chosen[pair]
        frozen = freeze_attack_nvidia(
            example,
            api_key=nvidia_key,
            model=MODEL,
            temperature=0.0,
            max_tokens=512,
        )
        placeholder = example.metadata.get("attack_placeholder")
        if not isinstance(placeholder, str):
            raise RuntimeError(f"Missing string attack placeholder for {example.example_id}")

        materialized = materialize_attack(
            example.defender_problem_template,
            placeholder=placeholder,
            attack_text=frozen.attack_text,
        )
        defender = nvidia_chat(
            api_key=nvidia_key,
            model=MODEL,
            messages=materialized.messages,
            temperature=0.0,
            max_tokens=512,
        )

        records.append(
            {
                "example_id": example.example_id,
                "task_type": example.metadata.get("task_type"),
                "privileged_level": pair[0],
                "attack_level": pair[1],
                "attacker_model": MODEL,
                "defender_model": MODEL,
                "attack_sha256": frozen.attack_sha256,
                "attack_text": frozen.attack_text,
                "attacker_generation": frozen.generation,
                "placeholder_replacements": materialized.replacements,
                "defender_output": defender.content,
                "defender_generation": {
                    "finish_reason": defender.finish_reason,
                    "latency_seconds": defender.latency_seconds,
                    "prompt_tokens": defender.prompt_tokens,
                    "completion_tokens": defender.completion_tokens,
                    "total_tokens": defender.total_tokens,
                },
                "graded": False,
                "note": "Pipeline smoke only; no IH-Challenge grader executed.",
            }
        )

    Path("frozen-attack-smoke.json").write_text(
        json.dumps(
            {
                "source_file": SOURCE_FILE,
                "model": MODEL,
                "purpose": "pipeline smoke only; not an IHD result",
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    compact = [
        {
            "example_id": record["example_id"],
            "task_type": record["task_type"],
            "authority": f"{record['privileged_level']}->{record['attack_level']}",
            "attack_sha256": record["attack_sha256"],
            "placeholder_replacements": record["placeholder_replacements"],
            "attacker_finish_reason": record["attacker_generation"]["finish_reason"],
            "defender_finish_reason": record["defender_generation"]["finish_reason"],
        }
        for record in records
    ]
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
