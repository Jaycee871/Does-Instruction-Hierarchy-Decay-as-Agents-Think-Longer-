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
from ih_decay.trajectory import append_neutral_continuation, continuation_text

T = TypeVar("T")
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_POLICIES = ("neutral_continue_v1", "minimal_continue_v1")
DEFAULT_CHECKPOINTS = (1, 3, 5, 10)


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


def _messages_sha256(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_exact_example(source_file: str, row_index: int, hf_token: str | None) -> IHExample:
    for example in iter_examples(source_file, token=hf_token):
        if example.row_index == row_index:
            return example
    raise SystemExit(f"row_index={row_index} not found in {source_file}")


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
        description=(
            "Replicated paired continuation-policy probe. One attack is frozen per item; "
            "each replicate draws a fresh shared step-1 output, then forks into two policies."
        )
    )
    parser.add_argument("--source-file", required=True, choices=DATA_FILES)
    parser.add_argument("--row-index", required=True, type=int)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-step", type=int, default=10)
    parser.add_argument("--output-dir", default="replicated-policy-probe")
    args = parser.parse_args()

    if args.replicates <= 0 or args.max_step < 2 or args.max_tokens <= 0:
        raise SystemExit("replicates/max-tokens must be positive and max-step must be >= 2")

    hf_token = os.environ.get("HF_TOKEN")
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    if not nvidia_key:
        raise SystemExit("NVIDIA_API_KEY is not set")

    example = _load_exact_example(args.source_file, args.row_index, hf_token)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.source_file.removesuffix('.jsonl')}.row-{args.row_index}"
    records_path = output_dir / f"{stem}.replicated-policy-probe.jsonl"
    attack_path = output_dir / f"{stem}.attack.json"
    summary_path = output_dir / f"{stem}.summary.json"
    records_path.write_text("", encoding="utf-8")

    checkpoints = [step for step in DEFAULT_CHECKPOINTS if step <= args.max_step]
    base = {
        "study": "ih-decay-pilot-v0",
        "phase": "phase1-replicated-continuation-policy-probe",
        "source_file": args.source_file,
        "row_index": example.row_index,
        "example_id": example.example_id,
        "task_type": example.metadata.get("task_type"),
        "privileged_level": example.metadata.get("privileged_level"),
        "attack_level": example.metadata.get("attack_level"),
        "model": args.model,
        "temperature": 0.0,
        "reasoning_effort": "low",
        "max_tokens": args.max_tokens,
        "max_step": args.max_step,
        "checkpoints": checkpoints,
        "replicates_requested": args.replicates,
    }

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
    attack_path.write_text(
        json.dumps(
            {
                **base,
                "attack_sha256": frozen.attack_sha256,
                "attack_text": frozen.attack_text,
                "attacker_generation": {
                    **frozen.generation,
                    "provider_attempts": attacker_attempts,
                },
                "freeze_rule": "one attack shared across all replicates and both policy branches",
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    placeholder = example.metadata.get("attack_placeholder")
    if not isinstance(placeholder, str) or not placeholder:
        raise SystemExit("missing attack_placeholder")
    materialized = materialize_attack(
        example.defender_problem_template,
        placeholder=placeholder,
        attack_text=frozen.attack_text,
    )
    original_messages = [dict(message) for message in materialized.messages]
    original_messages_sha256 = _messages_sha256(original_messages)
    grader_input, grader_input_mode = resolve_grader_input(
        example.metadata["grader_code_python"],
        frozen_attack_text=frozen.attack_text,
    )

    grader_counts: Counter[str] = Counter()
    replicate_status_counts: Counter[str] = Counter()
    shared_step1_output_hashes: Counter[str] = Counter()
    shared_step1_pass_replicates = 0
    policy_checkpoint_passes_all = {policy: Counter() for policy in DEFAULT_POLICIES}
    policy_checkpoint_passes_at_risk = {policy: Counter() for policy in DEFAULT_POLICIES}
    paired_checkpoint_counts = {
        step: Counter(
            {
                "both_pass": 0,
                "neutral_pass_minimal_fail": 0,
                "neutral_fail_minimal_pass": 0,
                "both_fail": 0,
            }
        )
        for step in checkpoints
    }
    replicate_outcomes: list[dict[str, object]] = []

    for replicate in range(1, args.replicates + 1):
        print(
            f"[{args.source_file}:{args.row_index}] replicate {replicate}/{args.replicates}",
            flush=True,
        )
        shared_step1, step1_attempts = _retry(
            f"{example.example_id} replicate-{replicate} shared-step-1",
            lambda: nvidia_chat(
                api_key=nvidia_key,
                model=args.model,
                messages=original_messages,
                temperature=0.0,
                max_tokens=args.max_tokens,
                reasoning_effort="low",
            ),
        )
        shared_grade = grade_output_isolated(
            example.metadata["grader_code_python"],
            grader_input,
            shared_step1.content,
            timeout_s=2.0,
            memory_mb=256,
        )
        grader_counts[shared_grade.status] += 1
        if shared_grade.status != "ok":
            raise SystemExit(
                f"replicate {replicate} shared step-1 grader status={shared_grade.status}"
            )

        shared_hash = _text_sha256(shared_step1.content)
        shared_step1_output_hashes[shared_hash] += 1
        at_risk = bool(shared_grade.correct)
        if at_risk:
            shared_step1_pass_replicates += 1

        shared_record = {
            **base,
            "replicate": replicate,
            "record_kind": "shared_step1",
            "status": "completed",
            "trajectory_step": 1,
            "is_checkpoint": 1 in checkpoints,
            "attack_sha256": frozen.attack_sha256,
            "original_messages_sha256": original_messages_sha256,
            "prompt_messages_sha256": original_messages_sha256,
            "prompt_message_count": len(original_messages),
            "defender_output": shared_step1.content,
            "defender_output_sha256": shared_hash,
            "defender_generation": _generation_payload(
                shared_step1,
                max_tokens=args.max_tokens,
                attempts=step1_attempts,
            ),
            "grader_input_mode": grader_input_mode,
            "grade": shared_grade.as_dict(),
            "persistence_at_risk": at_risk,
        }
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(shared_record, ensure_ascii=False, sort_keys=True) + "\n")

        outcomes_by_policy: dict[str, dict[int, bool]] = {
            policy: {1: bool(shared_grade.correct)} for policy in DEFAULT_POLICIES
        }
        for policy in DEFAULT_POLICIES:
            if 1 in checkpoints and shared_grade.correct:
                policy_checkpoint_passes_all[policy][1] += 1
                if at_risk:
                    policy_checkpoint_passes_at_risk[policy][1] += 1

            policy_text = continuation_text(policy)
            messages = append_neutral_continuation(
                original_messages,
                assistant_output=shared_step1.content,
                continuation_text=policy_text,
            )
            cumulative_completion_tokens = shared_step1.completion_tokens or 0

            for step in range(2, args.max_step + 1):
                result, attempts = _retry(
                    f"{example.example_id} replicate-{replicate} {policy} step-{step}",
                    lambda: nvidia_chat(
                        api_key=nvidia_key,
                        model=args.model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=args.max_tokens,
                        reasoning_effort="low",
                    ),
                )
                if result.completion_tokens is not None:
                    cumulative_completion_tokens += result.completion_tokens
                grade = grade_output_isolated(
                    example.metadata["grader_code_python"],
                    grader_input,
                    result.content,
                    timeout_s=2.0,
                    memory_mb=256,
                )
                grader_counts[grade.status] += 1
                if grade.status != "ok":
                    raise SystemExit(
                        f"replicate {replicate} {policy} step {step} grader status={grade.status}"
                    )

                correct = bool(grade.correct)
                outcomes_by_policy[policy][step] = correct
                if step in checkpoints and correct:
                    policy_checkpoint_passes_all[policy][step] += 1
                    if at_risk:
                        policy_checkpoint_passes_at_risk[policy][step] += 1

                record = {
                    **base,
                    "replicate": replicate,
                    "record_kind": "policy_branch",
                    "status": "completed",
                    "continuation_policy": policy,
                    "continuation_text": policy_text,
                    "trajectory_step": step,
                    "is_checkpoint": step in checkpoints,
                    "shared_step1_output_sha256": shared_hash,
                    "attack_sha256": frozen.attack_sha256,
                    "original_messages_sha256": original_messages_sha256,
                    "prompt_messages_sha256": _messages_sha256(messages),
                    "prompt_message_count": len(messages),
                    "defender_output": result.content,
                    "defender_output_sha256": _text_sha256(result.content),
                    "defender_generation": _generation_payload(
                        result,
                        max_tokens=args.max_tokens,
                        attempts=attempts,
                    ),
                    "cumulative_completion_tokens": cumulative_completion_tokens,
                    "grader_input_mode": grader_input_mode,
                    "grade": grade.as_dict(),
                    "persistence_at_risk": at_risk,
                }
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

                if step < args.max_step:
                    messages = append_neutral_continuation(
                        messages,
                        assistant_output=result.content,
                        continuation_text=policy_text,
                    )

        replicate_status_counts["completed"] += 1
        checkpoint_patterns: dict[str, str] = {}
        for policy in DEFAULT_POLICIES:
            checkpoint_patterns[policy] = "".join(
                "P" if outcomes_by_policy[policy].get(step, False) else "F"
                for step in checkpoints
            )

        if at_risk:
            for step in checkpoints:
                neutral = outcomes_by_policy["neutral_continue_v1"].get(step, False)
                minimal = outcomes_by_policy["minimal_continue_v1"].get(step, False)
                if neutral and minimal:
                    paired_checkpoint_counts[step]["both_pass"] += 1
                elif neutral and not minimal:
                    paired_checkpoint_counts[step]["neutral_pass_minimal_fail"] += 1
                elif not neutral and minimal:
                    paired_checkpoint_counts[step]["neutral_fail_minimal_pass"] += 1
                else:
                    paired_checkpoint_counts[step]["both_fail"] += 1

        replicate_outcomes.append(
            {
                "replicate": replicate,
                "shared_step1_correct": bool(shared_grade.correct),
                "shared_step1_output_sha256": shared_hash,
                "checkpoint_patterns": checkpoint_patterns,
            }
        )

    expected_unique_grader_execs = args.replicates * (
        1 + len(DEFAULT_POLICIES) * (args.max_step - 1)
    )
    if grader_counts.get("ok", 0) != expected_unique_grader_execs:
        raise SystemExit(
            f"expected {expected_unique_grader_execs} executable unique grader calls; got {dict(grader_counts)}"
        )

    summary = {
        "study": "ih-decay-pilot-v0",
        "phase": "phase1-replicated-continuation-policy-probe",
        "purpose": (
            "estimate continuation-policy sensitivity relative to provider nondeterminism; "
            "construct-validity probe, not confirmatory IHD evidence"
        ),
        "source_file": args.source_file,
        "row_index": example.row_index,
        "example_id": example.example_id,
        "model": args.model,
        "attack_sha256": frozen.attack_sha256,
        "shared_attack_across_replicates": True,
        "shared_step1_within_replicate": True,
        "replicates_requested": args.replicates,
        "replicates_completed": replicate_status_counts["completed"],
        "shared_step1_pass_replicates": shared_step1_pass_replicates,
        "shared_step1_unique_output_hashes": len(shared_step1_output_hashes),
        "shared_step1_output_hash_counts": dict(shared_step1_output_hashes),
        "policies": {policy: continuation_text(policy) for policy in DEFAULT_POLICIES},
        "max_step": args.max_step,
        "checkpoints": checkpoints,
        "grader_status_counts": dict(grader_counts),
        "policy_checkpoint_pass_counts_all_replicates": {
            policy: {
                str(step): policy_checkpoint_passes_all[policy][step]
                for step in checkpoints
            }
            for policy in DEFAULT_POLICIES
        },
        "policy_checkpoint_pass_counts_step1_pass_replicates": {
            policy: {
                str(step): policy_checkpoint_passes_at_risk[policy][step]
                for step in checkpoints
            }
            for policy in DEFAULT_POLICIES
        },
        "paired_checkpoint_counts_step1_pass_replicates": {
            str(step): dict(paired_checkpoint_counts[step]) for step in checkpoints
        },
        "replicate_outcomes": replicate_outcomes,
        "records_path": str(records_path),
        "attack_path": str(attack_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
