from __future__ import annotations

import ast
import json
import sys

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX fallback
    resource = None

ALLOWED_IMPORTS = frozenset(
    {
        "collections",
        "decimal",
        "fractions",
        "functools",
        "itertools",
        "json",
        "math",
        "operator",
        "re",
        "statistics",
        "string",
        "unicodedata",
    }
)

BLOCKED_CALLS = frozenset(
    {
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "memoryview",
        "open",
        "setattr",
        "vars",
        "__import__",
    }
)

BLOCKED_NODES = (
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.AsyncFor,
    ast.AsyncWith,
)


class UnsafeGrader(ValueError):
    pass


class _Policy(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root not in ALLOWED_IMPORTS:
                raise UnsafeGrader(f"import not allowed: {root}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            raise UnsafeGrader("relative imports are not allowed")
        root = (node.module or "").split(".", 1)[0]
        if root not in ALLOWED_IMPORTS:
            raise UnsafeGrader(f"import not allowed: {root or '<empty>'}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            raise UnsafeGrader("dunder attribute access is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__") and node.id != "__name__":
            raise UnsafeGrader("dunder names are not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            raise UnsafeGrader(f"call not allowed: {node.func.id}")
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, BLOCKED_NODES):
            raise UnsafeGrader(f"syntax not allowed: {type(node).__name__}")
        super().generic_visit(node)


def _limited_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
    if level:
        raise ImportError("relative imports are disabled")
    root = name.split(".", 1)[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"import disabled: {root}")
    return __import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
    "__import__": _limited_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "ValueError": ValueError,
    "zip": zip,
}


def _apply_limits(cpu_seconds: int, memory_mb: int) -> None:
    if resource is None:
        return
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    memory_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if hasattr(resource, "RLIMIT_FSIZE"):
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    if hasattr(resource, "RLIMIT_NOFILE"):
        resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))


def _run(payload: dict[str, object]) -> dict[str, object]:
    code = payload.get("grader_code")
    input_text = payload.get("input_text")
    assistant_response = payload.get("assistant_response")
    if not isinstance(code, str) or not isinstance(input_text, str) or not isinstance(assistant_response, str):
        return {"status": "error", "detail": "invalid worker payload"}

    try:
        tree = ast.parse(code, filename="<dataset-grader>", mode="exec")
        _Policy().visit(tree)
    except (SyntaxError, UnsafeGrader) as exc:
        return {"status": "rejected", "detail": str(exc)}

    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "__name__": "__dataset_grader__",
    }
    try:
        exec(compile(tree, "<dataset-grader>", "exec"), namespace, namespace)
        grader = namespace.get("grade_output_correct")
        if not callable(grader):
            return {"status": "error", "detail": "grade_output_correct is missing or not callable"}
        result = grader(input_text, assistant_response)
        if type(result) is not bool:
            return {"status": "error", "detail": "grade_output_correct must return bool"}
        return {"status": "ok", "correct": result}
    except BaseException as exc:
        return {"status": "error", "detail": f"grader raised {type(exc).__name__}: {exc}"}


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
        cpu_seconds = int(payload.pop("cpu_seconds", 1))
        memory_mb = int(payload.pop("memory_mb", 256))
        _apply_limits(max(cpu_seconds, 1), max(memory_mb, 64))
        result = _run(payload)
    except BaseException as exc:
        result = {"status": "error", "detail": f"worker failure: {type(exc).__name__}: {exc}"}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
