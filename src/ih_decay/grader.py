from __future__ import annotations

import ast
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


def _parse_grader(grader_code: str) -> ast.Module:
    if not isinstance(grader_code, str) or not grader_code.strip():
        raise ValueError("grader_code must be a non-empty string")
    return ast.parse(grader_code, filename="<dataset-grader>", mode="exec")


def grader_reads_input_text(grader_code: str) -> bool:
    """Return whether a grader contains any static load of `input_text`.

    This deliberately over-approximates semantic use: forwarding `input_text` into a
    helper also counts as a read. Use `grader_consumes_input_text` when the distinction
    between forwarding and actually inspecting the value matters.
    """
    tree = _parse_grader(grader_code)
    return any(
        isinstance(node, ast.Name)
        and node.id == "input_text"
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(tree)
    )


def grader_consumes_input_text(grader_code: str) -> bool:
    """Conservatively trace whether the public grader can inspect `input_text`.

    The IH-Challenge grader signature always accepts `input_text`, but many composite
    graders merely forward that parameter to local helper graders that themselves ignore
    it. This lightweight inter-procedural taint analysis follows direct calls between
    top-level local functions and treats every other use as semantic consumption.

    Unknown calls, attributes, starred arguments, closures, and other ambiguous cases are
    intentionally classified as consuming the value rather than guessed away.
    """
    tree = _parse_grader(grader_code)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    root = functions.get("grade_output_correct")
    if root is None:
        raise ValueError("grade_output_correct is missing")

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    def parameters(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        return [
            arg.arg
            for arg in (
                list(function.args.posonlyargs)
                + list(function.args.args)
                + list(function.args.kwonlyargs)
            )
        ]

    def owner_function(node: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        current = parents.get(node)
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current
            current = parents.get(current)
        return None

    def forwarded_target(
        name_node: ast.Name,
    ) -> tuple[str, str] | None:
        parent = parents.get(name_node)
        call: ast.Call | None = None
        target_parameter: str | None = None

        if isinstance(parent, ast.Call) and name_node in parent.args:
            call = parent
            position = parent.args.index(name_node)
            if isinstance(call.func, ast.Name) and call.func.id in functions:
                callee_params = parameters(functions[call.func.id])
                if position < len(callee_params):
                    target_parameter = callee_params[position]
        elif isinstance(parent, ast.keyword) and parent.value is name_node:
            maybe_call = parents.get(parent)
            if isinstance(maybe_call, ast.Call):
                call = maybe_call
                target_parameter = parent.arg

        if (
            call is None
            or target_parameter is None
            or not isinstance(call.func, ast.Name)
            or call.func.id not in functions
        ):
            return None
        return call.func.id, target_parameter

    visiting: set[tuple[str, str]] = set()
    resolved: dict[tuple[str, str], bool] = {}

    def consumes(function_name: str, parameter_name: str) -> bool:
        key = (function_name, parameter_name)
        if key in resolved:
            return resolved[key]
        if key in visiting:
            return False
        function = functions[function_name]
        if parameter_name not in parameters(function):
            return True

        visiting.add(key)
        try:
            for node in ast.walk(function):
                if not (
                    isinstance(node, ast.Name)
                    and node.id == parameter_name
                    and isinstance(node.ctx, ast.Load)
                    and owner_function(node) is function
                ):
                    continue
                target = forwarded_target(node)
                if target is None:
                    resolved[key] = True
                    return True
                if consumes(*target):
                    resolved[key] = True
                    return True
            resolved[key] = False
            return False
        finally:
            visiting.discard(key)

    return consumes("grade_output_correct", "input_text")


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
