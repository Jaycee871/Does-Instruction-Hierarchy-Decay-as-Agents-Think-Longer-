from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

GradeStatus = Literal["ok", "rejected", "timeout", "error"]


@dataclass(frozen=True)
class GradeResult:
    status: GradeStatus
    correct: bool | None
    detail: str | None
    duration_ms: float

    def as_dict(self) -> dict[str, bool | float | str | None]:
        return asdict(self)


def grade_output_isolated(
    grader_code: str,
    input_text: str,
    assistant_response: str,
    *,
    timeout_s: float = 2.0,
    memory_mb: int = 256,
) -> GradeResult:
    """Evaluate an IH-Challenge grader in a separate restricted Python process.

    The orchestration process never executes dataset-provided code. The worker runs
    with isolated Python flags, a scrubbed environment, an AST/import allowlist and,
    on POSIX, CPU/address-space/file-descriptor limits. This is defense in depth for
    benchmark graders, not a general-purpose security boundary for hostile code.
    """
    if not isinstance(grader_code, str) or not grader_code.strip():
        raise ValueError("grader_code must be a non-empty string")
    if not isinstance(input_text, str) or not isinstance(assistant_response, str):
        raise TypeError("input_text and assistant_response must be strings")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if memory_mb < 64:
        raise ValueError("memory_mb must be at least 64")

    worker = Path(__file__).with_name("_grader_worker.py")
    payload = json.dumps(
        {
            "grader_code": grader_code,
            "input_text": input_text,
            "assistant_response": assistant_response,
            "cpu_seconds": max(1, int(timeout_s)),
            "memory_mb": memory_mb,
        },
        ensure_ascii=False,
    )
    env = {
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
    }

    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="ih-grader-") as temp_dir:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(worker)],
                input=payload,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=temp_dir,
                env=env,
                timeout=timeout_s,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return GradeResult(
            status="timeout",
            correct=None,
            detail=f"grader exceeded {timeout_s:.3g}s wall-clock limit",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    duration_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"worker exited with code {completed.returncode}"
        return GradeResult("error", None, detail[:1000], duration_ms)

    try:
        raw = json.loads(completed.stdout)
        status = raw["status"]
        if status not in {"ok", "rejected", "error"}:
            raise ValueError(f"unexpected worker status: {status!r}")
        correct = raw.get("correct")
        if status == "ok" and type(correct) is not bool:
            raise ValueError("successful worker result lacks a boolean grade")
        detail = raw.get("detail")
        if detail is not None and not isinstance(detail, str):
            raise ValueError("worker detail is not a string")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return GradeResult(
            "error",
            None,
            f"invalid worker response: {exc}",
            duration_ms,
        )

    return GradeResult(
        status=status,
        correct=correct if status == "ok" else None,
        detail=detail,
        duration_ms=duration_ms,
    )
