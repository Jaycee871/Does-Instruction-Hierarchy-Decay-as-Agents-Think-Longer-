from __future__ import annotations

import json
import os
from pathlib import Path

from ih_decay.attacks import freeze_attack_nvidia
from ih_decay.data import DATA_FILES, iter_examples
from ih_decay.grader import grade_output_isolated
from ih_decay.grader_contract import grader_input_parameter_name, resolve_grader_input
from ih_decay.materialize import materialize_attack
from ih_decay.providers import nvidia_chat

MODEL = "openai/gpt-oss-20b"
OUTPUT_PATH = Path("cross-split-graded-smoke.json")


def _generation_payload(result) -> dict[str, object]:
    return {
        "finish_reason": result.finish_reason,
        "latency_seconds": result.latency_seconds,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "reasoning_effort": "low",
        "max_tokens": 4096,
    }


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    if not nvidia_key:
        raise SystemExit("NVIDIA_API_KEY is not set")

    records: list[dict[str, object]] = []
    for source_file in DATA_FILES:
        examples = list(iter_examples(source_file, token=hf_token, limit=1))
        if len(examples) != 1:
            raise RuntimeError(f"Expected row 0 from {source_file}, got {len(examples)} rows")
        example = examples[0]

        grader_code = example.metadata.get("grader_code_python")
        placeholder = example.metadata.get("attack_placeholder")
        if not isinstance(grader_code, str) or not grader_code.strip():
            raise RuntimeError(f"Missing grader code for {example.example_id}")
        if not isinstance(placeholder, str) or not placeholder:
            raise RuntimeError(f"Missing attack placeholder for {example.example_id}")

        frozen = freeze_attack_nvidia(
            example,
            api_key=nvidia_key,
            model=MODEL,
            temperature=0.0,
            max_tokens=4096,
            reasoning_effort="low",
        )
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
            max_tokens=4096,
            reasoning_effort="low",
        )

        grader_input, grader_input_mode = resolve_grader_input(
            grader_code,
            frozen_attack_text=frozen.attack_text,
        )
        grade = grade_output_isolated(
            grader_code,
            grader_input,
            defender.content,
            timeout_s=2.0,
            memory_mb=256,
        )
        if grade.status != "ok":
            raise RuntimeError(
                f"Grader did not execute cleanly for {example.example_id}: "
                f"{grade.status}: {grade.detail}"
            )

        records.append(
            {
                "source_file": source_file,
                "example_id": example.example_id,
                "row_index": example.row_index,
                "task_type": example.metadata.get("task_type"),
                "privileged_level": example.metadata.get("privileged_level"),
                "attack_level": example.metadata.get("attack_level"),
                "attacker_model": MODEL,
                "defender_model": MODEL,
                "attack_sha256": frozen.attack_sha256,
                "attack_text": frozen.attack_text,
                "attacker_generation": frozen.generation,
                "placeholder_replacements": materialized.replacements,
                "defender_output": defender.content,
                "defender_generation": _generation_payload(defender),
                "grader_first_parameter": grader_input_parameter_name(grader_code),
                "grader_input_mode": grader_input_mode,
                "grade": grade.as_dict(),
            }
        )

    payload = {
        "model": MODEL,
        "rows": "row 0 from each public IH-Challenge split",
        "trajectory_step": 1,
        "purpose": "cross-split graded pipeline compatibility smoke; not an IHD result",
        "records": records,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            [
                {
                    "source_file": row["source_file"],
                    "example_id": row["example_id"],
                    "grader_first_parameter": row["grader_first_parameter"],
                    "grader_input_mode": row["grader_input_mode"],
                    "grade_status": row["grade"]["status"],
                    "correct": row["grade"]["correct"],
                }
                for row in records
            ],
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
