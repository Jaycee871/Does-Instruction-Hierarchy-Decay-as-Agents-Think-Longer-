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
from ih_decay.trajectory import append_neutral_continuation, continuation_text

T = TypeVar("T")
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_SEED = 20260905
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
        description="Paired continuation-policy probe with a shared frozen attack and shared step-1 output"
    )
    parser.add_argument("--source-file", required=True, choices=DATA_FILES)
    parser.add_argument("--items", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-step", type=int, default=10)
    parser.add_argument("--output-dir", default="continuation-policy-probe")
    args = parser.parse_args()

    if args.items <= 0 or args.max_step < 2 or args.max_tokens <= 0:
        raise SystemExit("items/max-tokens must be positive and max-step must be >= 2")

    hf_token = os.environ.get("HF_TOKEN")
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    if not nvidia_key:
        raise SystemExit("NVIDIA_API_KEY is not set")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.source_file.removesuffix(".jsonl")
    records_path = output_dir / f"{stem}.policy-probe.jsonl"
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

    status_counts: Counter[str] = Counter()
    grader_counts: Counter[str] = Counter()
    policy_checkpoint_passes = {
        policy: Counter() for policy in DEFAULT_POLICIES
    }
    complete_items = 0

    for ordinal, example in enumerate(examples, start=1):
        print(f"[{args.source_file}] item {ordinal}/{len(examples)} {example.example_id}", flush=True)
        base = {
            "study": "ih-decay-pilot-v0",
            "phase": "phase1-continuation-policy-probe",
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
            "max_step": args.max_step,
            "checkpoints": [step for step in DEFAULT_CHECKPOINTS if step <= args.max_step],
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
            with attacks_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            **base,
                            "attack_sha256": frozen.attack_sha256,
                            "attack_text": frozen.attack_text,
                            "attacker_generation": {
                                **frozen.generation,
                                "provider_attempts": attacker_attempts,
                            },
                            "freeze_rule": "one attack shared by both continuation-policy branches",
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

            placeholder = example.metadata.get("attack_placeholder")
            if not isinstance(placeholder, str) or not placeholder:
                raise RuntimeError("missing attack_placeholder")
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

            shared_step1, step1_attempts = _retry(
                f"{example.example_id} shared-step-1 defender",
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
                raise RuntimeError(f"shared step-1 grader status={shared_grade.status}")

            item_ok = True
            for policy in DEFAULT_POLICIES:
                policy_text = continuation_text(policy)
                checkpoints = set(step for step in DEFAULT_CHECKPOINTS if step <= args.max_step)
                if 1 in checkpoints and shared_grade.correct:
                    policy_checkpoint_passes[policy][1] += 1

                step1_record = {
                    **base,
                    "status": "completed",
                    "continuation_policy": policy,
                    "continuation_text": policy_text,
                    "trajectory_step": 1,
                    "is_checkpoint": 1 in checkpoints,
                    "shared_step1": True,
                    "attack_sha256": frozen.attack_sha256,
                    "original_messages_sha256": original_messages_sha256,
                    "prompt_messages_sha256": original_messages_sha256,
                    "prompt_message_count": len(original_messages),
                    "defender_output": shared_step1.content,
                    "defender_generation": _generation_payload(
                        shared_step1,
                        max_tokens=args.max_tokens,
                        attempts=step1_attempts,
                    ),
                    "grader_input_mode": grader_input_mode,
                    "grade": shared_grade.as_dict(),
                }
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(step1_record, ensure_ascii=False, sort_keys=True) + "\n")

                messages = append_neutral_continuation(
                    original_messages,
                    assistant_output=shared_step1.content,
                    continuation_text=policy_text,
                )
                cumulative_completion_tokens = shared_step1.completion_tokens or 0

                for step in range(2, args.max_step + 1):
                    result, attempts = _retry(
                        f"{example.example_id} {policy} step-{step}",
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
                        item_ok = False

                    if step in checkpoints and grade.status == "ok" and grade.correct:
                        policy_checkpoint_passes[policy][step] += 1

                    record = {
                        **base,
                        "status": "completed" if grade.status == "ok" else "grader_error",
                        "continuation_policy": policy,
                        "continuation_text": policy_text,
                        "trajectory_step": step,
                        "is_checkpoint": step in checkpoints,
                        "shared_step1": False,
                        "attack_sha256": frozen.attack_sha256,
                        "original_messages_sha256": original_messages_sha256,
                        "prompt_messages_sha256": _messages_sha256(messages),
                        "prompt_message_count": len(messages),
                        "defender_output": result.content,
                        "defender_generation": _generation_payload(
                            result,
                            max_tokens=args.max_tokens,
                            attempts=attempts,
                        ),
                        "cumulative_completion_tokens": cumulative_completion_tokens,
                        "grader_input_mode": grader_input_mode,
                        "grade": grade.as_dict(),
                    }
                    with records_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

                    if grade.status != "ok":
                        break
                    if step < args.max_step:
                        messages = append_neutral_continuation(
                            messages,
                            assistant_output=result.content,
                            continuation_text=policy_text,
                        )

            if item_ok:
                complete_items += 1
                status_counts["completed"] += 1
            else:
                status_counts["grader_error"] += 1

        except BaseException as exc:
            status_counts["error"] += 1
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            **base,
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:2000],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

    checkpoints = [step for step in DEFAULT_CHECKPOINTS if step <= args.max_step]
    summary = {
        "study": "ih-decay-pilot-v0",
        "phase": "phase1-continuation-policy-probe",
        "purpose": "paired continuation-policy construct-validity probe; not confirmatory IHD evidence",
        "source_file": args.source_file,
        "model": args.model,
        "requested_items": args.items,
        "selected_items": len(selected),
        "complete_items": complete_items,
        "policies": {
            policy: continuation_text(policy) for policy in DEFAULT_POLICIES
        },
        "shared_step1": True,
        "shared_attack": True,
        "max_step": args.max_step,
        "checkpoints": checkpoints,
        "selection_seed": args.seed,
        "effective_selection_seed": effective_seed,
        "selection_digest_sha256": selection_digest,
        "status_counts": dict(status_counts),
        "grader_status_counts": dict(grader_counts),
        "policy_checkpoint_pass_counts": {
            policy: {str(step): policy_checkpoint_passes[policy][step] for step in checkpoints}
            for policy in DEFAULT_POLICIES
        },
        "records_path": str(records_path),
        "attacks_path": str(attacks_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    if complete_items != len(selected):
        raise SystemExit(
            f"only {complete_items}/{len(selected)} items completed both policies; partial artifacts retained"
        )

    expected_unique_grader_execs = len(selected) * (1 + len(DEFAULT_POLICIES) * (args.max_step - 1))
    if grader_counts.get("ok", 0) != expected_unique_grader_execs:
        raise SystemExit(
            f"expected {expected_unique_grader_execs} executable unique grader calls; got {dict(grader_counts)}"
        )


if __name__ == "__main__":
    main()
