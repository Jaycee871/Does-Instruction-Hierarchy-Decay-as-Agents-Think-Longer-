from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, TypeVar

from ih_decay.attacks import freeze_attack_nvidia
from ih_decay.data import DATA_FILES, IHExample, iter_examples
from ih_decay.grader import grade_output_isolated
from ih_decay.grader_contract import grader_input_parameter_name, resolve_grader_input
from ih_decay.materialize import materialize_attack
from ih_decay.providers import ProviderError, nvidia_chat
from ih_decay.sampling import PilotCandidate, select_stratified

T = TypeVar("T")
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_SEED = 20260905


def _retry(label: str, fn: Callable[[], T], *, attempts: int = 4) -> tuple[T, int]:
    """Retry transient provider failures with bounded exponential backoff."""
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(), attempt
        except ProviderError as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = min(2 ** (attempt - 1), 8)
            print(
                f"{label}: provider attempt {attempt}/{attempts} failed; retrying in {delay}s: {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def _selection_digest(candidates: list[PilotCandidate]) -> str:
    payload = "\n".join(candidate.example_id for candidate in candidates) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _generation_payload(result, *, max_tokens: int, attempts: int) -> dict[str, object]:
    return {
        "finish_reason": result.finish_reason,
        "latency_seconds": result.latency_seconds,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
        "provider_attempts": attempts,
    }


def _selected_examples(
    source_file: str,
    *,
    items: int,
    seed: int,
    hf_token: str | None,
) -> tuple[list[IHExample], list[PilotCandidate], int]:
    examples = list(iter_examples(source_file, token=hf_token))
    file_index = DATA_FILES.index(source_file)
    effective_seed = seed + file_index
    selected = select_stratified(
        (PilotCandidate.from_example(example) for example in examples),
        n=items,
        seed=effective_seed,
    )
    by_index = {example.row_index: example for example in examples}
    chosen = [by_index[candidate.row_index] for candidate in selected]
    return chosen, selected, effective_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one deterministic 25-item step-1 IH-Challenge pilot shard"
    )
    parser.add_argument("--source-file", required=True, choices=DATA_FILES)
    parser.add_argument("--items", type=int, default=25)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--output-dir", default="step1-pilot")
    args = parser.parse_args()

    if args.items <= 0:
        raise SystemExit("--items must be positive")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")

    hf_token = os.environ.get("HF_TOKEN")
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    if not nvidia_key:
        raise SystemExit("NVIDIA_API_KEY is not set")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.source_file.removesuffix(".jsonl")
    records_path = output_dir / f"{stem}.jsonl"
    summary_path = output_dir / f"{stem}.summary.json"

    examples, selected, effective_seed = _selected_examples(
        args.source_file,
        items=args.items,
        seed=args.seed,
        hf_token=hf_token,
    )
    digest = _selection_digest(selected)

    # Start clean. The file is appended after every completed or failed row so an
    # `if: always()` artifact upload can preserve partial progress on workflow failure.
    records_path.write_text("", encoding="utf-8")

    status_counts: Counter[str] = Counter()
    grader_counts: Counter[str] = Counter()
    completed = 0

    for ordinal, example in enumerate(examples, start=1):
        print(
            f"[{args.source_file}] {ordinal}/{len(examples)} {example.example_id}",
            flush=True,
        )
        base: dict[str, object] = {
            "study": "ih-decay-pilot-v0",
            "phase": "phase1-step1-baseline",
            "trajectory_step": 1,
            "source_file": args.source_file,
            "row_index": example.row_index,
            "example_id": example.example_id,
            "task_type": example.metadata.get("task_type"),
            "privileged_level": example.metadata.get("privileged_level"),
            "attack_level": example.metadata.get("attack_level"),
            "selection_seed": args.seed,
            "effective_selection_seed": effective_seed,
            "selection_digest_sha256": digest,
            "attacker_model": args.model,
            "defender_model": args.model,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "reasoning_effort": "low",
        }

        try:
            frozen, attacker_attempts = _retry(
                f"{example.example_id} attacker",
                lambda: freeze_attack_nvidia(
                    example,
                    api_key=nvidia_key,
                    model=args.model,
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    reasoning_effort="low",
                ),
            )

            placeholder = example.metadata.get("attack_placeholder")
            if not isinstance(placeholder, str) or not placeholder:
                raise RuntimeError("missing attack_placeholder")
            materialized = materialize_attack(
                example.defender_problem_template,
                placeholder=placeholder,
                attack_text=frozen.attack_text,
            )

            defender, defender_attempts = _retry(
                f"{example.example_id} defender",
                lambda: nvidia_chat(
                    api_key=nvidia_key,
                    model=args.model,
                    messages=materialized.messages,
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    reasoning_effort="low",
                ),
            )

            grader_input, grader_input_mode = resolve_grader_input(
                example.metadata["grader_code_python"],
                frozen_attack_text=frozen.attack_text,
            )
            grade = grade_output_isolated(
                example.metadata["grader_code_python"],
                grader_input,
                defender.content,
                timeout_s=2.0,
                memory_mb=256,
            )

            record = {
                **base,
                "status": "completed",
                "attack_sha256": frozen.attack_sha256,
                "attack_text": frozen.attack_text,
                "attacker_generation": {
                    **frozen.generation,
                    "provider_attempts": attacker_attempts,
                },
                "placeholder_replacements": materialized.replacements,
                "defender_output": defender.content,
                "defender_generation": _generation_payload(
                    defender,
                    max_tokens=args.max_tokens,
                    attempts=defender_attempts,
                ),
                "grader_first_parameter": grader_input_parameter_name(
                    example.metadata["grader_code_python"]
                ),
                "grader_input_mode": grader_input_mode,
                "grade": grade.as_dict(),
                "note": "Step-1 baseline record only; not evidence of trajectory-length decay.",
            }
            status_counts["completed"] += 1
            grader_counts[grade.status] += 1
            completed += 1
        except BaseException as exc:
            record = {
                **base,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
            }
            status_counts["error"] += 1

        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "study": "ih-decay-pilot-v0",
        "phase": "phase1-step1-baseline",
        "source_file": args.source_file,
        "model": args.model,
        "requested_items": args.items,
        "selected_items": len(selected),
        "completed_items": completed,
        "selection_seed": args.seed,
        "effective_selection_seed": effective_seed,
        "selection_digest_sha256": digest,
        "status_counts": dict(status_counts),
        "grader_status_counts": dict(grader_counts),
        "records_path": str(records_path),
        "purpose": "100-item matched-pilot step-1 baseline shard; not an IHD result",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    if status_counts["error"]:
        raise SystemExit(f"{status_counts['error']} item(s) failed; partial artifact retained")
    if grader_counts.get("ok", 0) != len(selected):
        raise SystemExit(
            "not every selected item produced an executable grader result; "
            f"grader statuses={dict(grader_counts)}"
        )


if __name__ == "__main__":
    main()
