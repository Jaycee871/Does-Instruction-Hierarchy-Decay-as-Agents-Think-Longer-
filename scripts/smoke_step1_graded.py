from __future__ import annotations

import json
import os
from pathlib import Path

from ih_decay.attacks import freeze_attack_nvidia
from ih_decay.data import iter_examples
from ih_decay.grader import grade_output_isolated, grader_reads_input_text
from ih_decay.materialize import materialize_attack
from ih_decay.providers import nvidia_chat

SOURCE_FILE = "single-constraint.jsonl"
MODEL = "openai/gpt-oss-20b"
ROW_COUNT = 3
OUTPUT_PATH = Path("graded-step1-smoke.json")


def _generation_payload(result, *, reasoning_effort: str, max_tokens: int) -> dict[str, object]:
    return {
        "finish_reason": result.finish_reason,
        "latency_seconds": result.latency_seconds,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
    }


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    if not nvidia_key:
        raise SystemExit("NVIDIA_API_KEY is not set")

    examples = list(iter_examples(SOURCE_FILE, token=hf_token, limit=ROW_COUNT))
    if len(examples) != ROW_COUNT:
        raise SystemExit(f"Expected {ROW_COUNT} rows, loaded {len(examples)}")

    records: list[dict[str, object]] = []
    for example in examples:
        grader_code = example.metadata.get("grader_code_python")
        if not isinstance(grader_code, str) or not grader_code.strip():
            raise RuntimeError(f"Missing grader_code_python for {example.example_id}")

        # We deliberately start with benchmark rows whose grader does not consume
        # input_text, so this smoke does not guess the wider benchmark's input_text
        # semantics. Broader pilot execution remains blocked until that contract is
        # resolved and tested explicitly.
        if grader_reads_input_text(grader_code):
            raise RuntimeError(
                f"{example.example_id} reads input_text; refusing to guess its semantics"
            )

        frozen = freeze_attack_nvidia(
            example,
            api_key=nvidia_key,
            model=MODEL,
            temperature=0.0,
            max_tokens=4096,
            reasoning_effort="low",
        )

        placeholder = example.metadata.get("attack_placeholder")
        if not isinstance(placeholder, str) or not placeholder:
            raise RuntimeError(f"Missing attack placeholder for {example.example_id}")

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

        grade = grade_output_isolated(
            grader_code,
            "",
            defender.content,
            timeout_s=2.0,
            memory_mb=256,
        )
        if grade.status != "ok":
            raise RuntimeError(
                f"Isolated grader failed for {example.example_id}: "
                f"status={grade.status!r} detail={grade.detail!r}"
            )

        records.append(
            {
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
                "defender_generation": _generation_payload(
                    defender,
                    reasoning_effort="low",
                    max_tokens=4096,
                ),
                "grader_reads_input_text": False,
                "grader_input_text_mode": "empty string; statically verified unused",
                "grade": grade.as_dict(),
                "note": "Graded step-1 pipeline smoke only; not an IHD result.",
            }
        )

    payload = {
        "source_file": SOURCE_FILE,
        "rows": list(range(ROW_COUNT)),
        "model": MODEL,
        "trajectory_step": 1,
        "purpose": "graded step-1 pipeline smoke only; not an IHD result",
        "input_text_scope": (
            "Only graders with no static load of input_text are admitted; broader "
            "benchmark input_text semantics remain unresolved by this smoke."
        ),
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
                    "example_id": record["example_id"],
                    "authority": (
                        f"{record['privileged_level']}->{record['attack_level']}"
                    ),
                    "attack_sha256": record["attack_sha256"],
                    "grade_status": record["grade"]["status"],
                    "correct": record["grade"]["correct"],
                }
                for record in records
            ],
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
