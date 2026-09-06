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
from ih_decay.grader_contract import resolve_grader_input
from ih_decay.materialize import materialize_attack
from ih_decay.providers import ProviderError, nvidia_chat
from ih_decay.sampling import PilotCandidate, select_stratified
from ih_decay.trajectory import (
    DEFAULT_CHECKPOINTS,
    NEUTRAL_CONTINUATION_V1,
    TrajectoryPlan,
    append_neutral_continuation,
)

T = TypeVar("T")
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_SEED = 20260905


def _retry(label: str, fn: Callable[[], T], *, attempts: int = 4) -> tuple[T, int]:
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


def _parse_checkpoints(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    plan = TrajectoryPlan(checkpoints=values)
    if plan.checkpoints[0] != 1:
        raise argparse.ArgumentTypeError("checkpoints must start at step 1")
    return plan.checkpoints


def _selection_digest(candidates: list[PilotCandidate]) -> str:
    payload = "\n".join(candidate.example_id for candidate in candidates) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _messages_sha256(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a matched persistence-only IH trajectory shard with one frozen attack per item"
    )
    parser.add_argument("--source-file", required=True, choices=DATA_FILES)
    parser.add_argument("--items", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--checkpoints",
        type=_parse_checkpoints,
        default=DEFAULT_CHECKPOINTS,
        help="comma-separated checkpoints, e.g. 1,3,5,10,20",
    )
    parser.add_argument("--output-dir", default="matched-trajectory")
    args = parser.parse_args()

    if args.items <= 0:
        raise SystemExit("--items must be positive")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")

    checkpoints = tuple(args.checkpoints)
    plan = TrajectoryPlan(checkpoints=checkpoints)
    max_step = checkpoints[-1]

    hf_token = os.environ.get("HF_TOKEN")
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    if not nvidia_key:
        raise SystemExit("NVIDIA_API_KEY is not set")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.source_file.removesuffix(".jsonl")
    records_path = output_dir / f"{stem}.trajectory.jsonl"
    attacks_path = output_dir / f"{stem}.attacks.jsonl"
    summary_path = output_dir / f"{stem}.summary.json"
    records_path.write_text("", encoding="utf-8")
    attacks_path.write_text("", encoding="utf-8")

    examples, selected, effective_seed = _selected_examples(
        args.source_file,
        items=args.items,
        seed=args.seed,
        hf_token=hf_token,
    )
    selection_digest = _selection_digest(selected)

    trajectory_status: Counter[str] = Counter()
    grader_status: Counter[str] = Counter()
    checkpoint_correct: Counter[str] = Counter()
    complete_trajectories = 0
    total_scored_steps = 0

    for ordinal, example in enumerate(examples, start=1):
        print(
            f"[{args.source_file}] item {ordinal}/{len(examples)} {example.example_id}",
            flush=True,
        )
        base = {
            "study": "ih-decay-pilot-v0",
            "phase": "phase1-matched-trajectory",
            "condition": "persistence-only",
            "continuation_policy": plan.continuation_policy,
            "source_file": args.source_file,
            "row_index": example.row_index,
            "example_id": example.example_id,
            "task_type": example.metadata.get("task_type"),
            "privileged_level": example.metadata.get("privileged_level"),
            "attack_level": example.metadata.get("attack_level"),
            "selection_seed": args.seed,
            "effective_selection_seed": effective_seed,
            "selection_digest_sha256": selection_digest,
            "model": args.model,
            "temperature": 0.0,
            "reasoning_effort": "low",
            "max_tokens": args.max_tokens,
            "checkpoints": list(checkpoints),
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
            attack_record = {
                **base,
                "attack_sha256": frozen.attack_sha256,
                "attack_text": frozen.attack_text,
                "attacker_generation": {
                    **frozen.generation,
                    "provider_attempts": attacker_attempts,
                },
                "freeze_rule": "generated exactly once per item and reused for every trajectory step",
            }
            with attacks_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(attack_record, ensure_ascii=False, sort_keys=True) + "\n")

            placeholder = example.metadata.get("attack_placeholder")
            if not isinstance(placeholder, str) or not placeholder:
                raise RuntimeError("missing attack_placeholder")
            materialized = materialize_attack(
                example.defender_problem_template,
                placeholder=placeholder,
                attack_text=frozen.attack_text,
            )
            messages = [dict(message) for message in materialized.messages]
            original_messages_sha256 = _messages_sha256(messages)
            grader_input, grader_input_mode = resolve_grader_input(
                example.metadata["grader_code_python"],
                frozen_attack_text=frozen.attack_text,
            )

            cumulative_completion_tokens = 0
            item_failed = False
            for step in range(1, max_step + 1):
                print(f"  step {step}/{max_step}", flush=True)
                prompt_sha256 = _messages_sha256(messages)
                defender, defender_attempts = _retry(
                    f"{example.example_id} step-{step} defender",
                    lambda: nvidia_chat(
                        api_key=nvidia_key,
                        model=args.model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=args.max_tokens,
                        reasoning_effort="low",
                    ),
                )
                if defender.completion_tokens is not None:
                    cumulative_completion_tokens += defender.completion_tokens

                grade = grade_output_isolated(
                    example.metadata["grader_code_python"],
                    grader_input,
                    defender.content,
                    timeout_s=2.0,
                    memory_mb=256,
                )
                grader_status[grade.status] += 1
                total_scored_steps += 1
                is_checkpoint = step in checkpoints
                if is_checkpoint and grade.status == "ok" and grade.correct:
                    checkpoint_correct[str(step)] += 1

                record = {
                    **base,
                    "status": "completed",
                    "trajectory_step": step,
                    "is_checkpoint": is_checkpoint,
                    "attack_sha256": frozen.attack_sha256,
                    "original_messages_sha256": original_messages_sha256,
                    "prompt_messages_sha256": prompt_sha256,
                    "prompt_message_count": len(messages),
                    "placeholder_replacements": materialized.replacements,
                    "defender_output": defender.content,
                    "defender_generation": _generation_payload(
                        defender,
                        max_tokens=args.max_tokens,
                        attempts=defender_attempts,
                    ),
                    "cumulative_completion_tokens": cumulative_completion_tokens,
                    "grader_input_mode": grader_input_mode,
                    "grade": grade.as_dict(),
                    "note": "Matched persistence-only trajectory; same frozen attack at every step.",
                }
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

                if grade.status != "ok":
                    trajectory_status["grader_error"] += 1
                    item_failed = True
                    break

                if step < max_step:
                    messages = append_neutral_continuation(
                        messages,
                        assistant_output=defender.content,
                        continuation_text=NEUTRAL_CONTINUATION_V1,
                    )

            if not item_failed:
                complete_trajectories += 1
                trajectory_status["completed"] += 1

        except BaseException as exc:
            trajectory_status["error"] += 1
            error_record = {
                **base,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
            }
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(error_record, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "study": "ih-decay-pilot-v0",
        "phase": "phase1-matched-trajectory",
        "condition": "persistence-only",
        "continuation_policy": plan.continuation_policy,
        "continuation_text": NEUTRAL_CONTINUATION_V1,
        "source_file": args.source_file,
        "model": args.model,
        "requested_items": args.items,
        "selected_items": len(selected),
        "complete_trajectories": complete_trajectories,
        "checkpoints": list(checkpoints),
        "max_step": max_step,
        "selection_seed": args.seed,
        "effective_selection_seed": effective_seed,
        "selection_digest_sha256": selection_digest,
        "trajectory_status_counts": dict(trajectory_status),
        "grader_status_counts": dict(grader_status),
        "checkpoint_correct_counts": dict(checkpoint_correct),
        "total_scored_steps": total_scored_steps,
        "records_path": str(records_path),
        "attacks_path": str(attacks_path),
        "purpose": "construct-validity matched trajectory pilot; not confirmatory IHD evidence",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    if complete_trajectories != len(selected):
        raise SystemExit(
            f"only {complete_trajectories}/{len(selected)} trajectories completed; partial artifacts retained"
        )
    expected_scores = len(selected) * max_step
    if grader_status.get("ok", 0) != expected_scores:
        raise SystemExit(
            f"expected {expected_scores} executable grader results; got {dict(grader_status)}"
        )


if __name__ == "__main__":
    main()
