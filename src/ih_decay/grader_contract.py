from __future__ import annotations

import ast

from .grader import grader_consumes_input_text


class UnsupportedGraderContract(ValueError):
    """Raised when a grader's first-argument contract cannot be resolved safely."""


def grader_input_parameter_name(grader_code: str) -> str:
    """Return the first positional parameter name of ``grade_output_correct``."""
    if not isinstance(grader_code, str) or not grader_code.strip():
        raise ValueError("grader_code must be a non-empty string")
    tree = ast.parse(grader_code, filename="<dataset-grader>", mode="exec")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "grade_output_correct":
            positional = list(node.args.posonlyargs) + list(node.args.args)
            if not positional:
                raise UnsupportedGraderContract(
                    "grade_output_correct has no positional first argument"
                )
            return positional[0].arg
    raise UnsupportedGraderContract("grade_output_correct is missing")


def resolve_grader_input(grader_code: str, *, frozen_attack_text: str) -> tuple[str, str]:
    """Resolve the first grader argument without guessing benchmark semantics.

    The full public IH-Challenge audit found exactly two first-parameter names:

    * ``attacker_output`` (19,038 rows): this is the exact attack payload inserted into
      the defender prompt, so the frozen attack text is passed through unchanged.
    * ``input_text`` (8,532 rows): static inter-procedural tracing found no semantic
      consumption in any of these rows, so an empty string is a semantically inert
      compatibility value.

    Any future or modified grader outside those audited contracts fails closed.
    """
    if not isinstance(frozen_attack_text, str):
        raise TypeError("frozen_attack_text must be a string")

    parameter_name = grader_input_parameter_name(grader_code)
    if parameter_name == "attacker_output":
        return frozen_attack_text, "frozen_attack_text"

    if parameter_name == "input_text":
        if grader_consumes_input_text(grader_code):
            raise UnsupportedGraderContract(
                "an input_text grader semantically consumes its first argument; "
                "the audited empty-string compatibility contract does not apply"
            )
        return "", "empty_string_verified_unused"

    raise UnsupportedGraderContract(
        f"unknown first grader parameter {parameter_name!r}; refusing to infer semantics"
    )
