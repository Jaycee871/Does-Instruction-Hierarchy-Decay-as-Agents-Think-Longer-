from __future__ import annotations

import ast
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from ih_decay.data import DATA_FILES, iter_examples
from ih_decay.grader import grader_consumes_input_text, grader_reads_input_text

OUTPUT_PATH = Path("grader-input-contract-audit.json")
REPRESENTATIVES_PER_SPLIT = 5


def _input_parameter_name(tree: ast.Module) -> str:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "grade_output_correct":
            positional = list(node.args.posonlyargs) + list(node.args.args)
            if not positional:
                raise ValueError("grade_output_correct has no positional input argument")
            return positional[0].arg
    raise ValueError("grade_output_correct is missing")


def _input_text_contexts(grader_code: str) -> tuple[str, list[str]]:
    tree = ast.parse(grader_code, filename="<dataset-grader>", mode="exec")
    parameter_name = _input_parameter_name(tree)
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    contexts: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Name)
            and node.id == parameter_name
            and isinstance(node.ctx, ast.Load)
        ):
            continue
        parent = parents.get(node)
        grandparent = parents.get(parent) if parent is not None else None
        if isinstance(parent, ast.Call):
            contexts.append("direct_call_argument")
        elif isinstance(parent, ast.Compare):
            contexts.append("comparison")
        elif isinstance(parent, ast.Subscript):
            contexts.append("subscript")
        elif isinstance(parent, ast.Attribute):
            contexts.append("attribute_base")
        elif isinstance(parent, ast.BoolOp):
            contexts.append("boolean_expression")
        elif isinstance(parent, ast.BinOp):
            contexts.append("binary_expression")
        elif isinstance(parent, ast.UnaryOp):
            contexts.append("unary_expression")
        elif isinstance(parent, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
            contexts.append("container_literal")
        elif isinstance(grandparent, ast.Call):
            contexts.append("nested_call_argument")
        else:
            contexts.append(type(parent).__name__ if parent is not None else "unknown")
    return parameter_name, contexts


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    overall = Counter()
    by_split: dict[str, Counter] = {}
    by_task: dict[str, Counter] = defaultdict(Counter)
    usage_contexts = Counter()
    parameter_names = Counter()
    forwarding_representatives: dict[str, list[dict[str, object]]] = defaultdict(list)
    consuming_representatives: dict[str, list[dict[str, object]]] = defaultdict(list)

    for source_file in DATA_FILES:
        split_counts = Counter()
        for example in iter_examples(source_file, token=hf_token):
            grader_code = example.metadata.get("grader_code_python")
            if not isinstance(grader_code, str) or not grader_code.strip():
                split_counts["missing_grader"] += 1
                overall["missing_grader"] += 1
                continue

            task_type = str(example.metadata.get("task_type", "<missing>"))
            reads = grader_reads_input_text(grader_code)
            consumes = grader_consumes_input_text(grader_code)
            if consumes and not reads:
                raise RuntimeError(
                    f"Semantic tracer reported consumption without a static load: {example.example_id}"
                )

            if consumes:
                bucket = "consumes_input_text"
            elif reads:
                bucket = "forwards_input_text_only"
            else:
                bucket = "no_input_text_load"

            split_counts[bucket] += 1
            overall[bucket] += 1
            by_task[task_type][bucket] += 1

            parameter_name, contexts = _input_text_contexts(grader_code)
            parameter_names[parameter_name] += 1
            if reads:
                usage_contexts.update(contexts)
                record = {
                    "example_id": example.example_id,
                    "row_index": example.row_index,
                    "task_type": task_type,
                    "privileged_level": example.metadata.get("privileged_level"),
                    "attack_level": example.metadata.get("attack_level"),
                    "attacker_problem": example.attacker_problem,
                    "grader_code_python": grader_code,
                    "input_parameter_name": parameter_name,
                    "input_text_contexts": contexts,
                    "semantic_consumption": consumes,
                }
                destination = (
                    consuming_representatives if consumes else forwarding_representatives
                )
                if len(destination[source_file]) < REPRESENTATIVES_PER_SPLIT:
                    destination[source_file].append(record)
        by_split[source_file] = split_counts

    payload = {
        "purpose": "determine whether IH-Challenge graders inspect their first input argument or merely forward it",
        "python_version_note": "Audit runs under Python 3.12 because some public grader source uses PEP 701 f-string syntax.",
        "overall": dict(overall),
        "by_split": {name: dict(counts) for name, counts in by_split.items()},
        "by_task_type": {
            name: dict(counts) for name, counts in sorted(by_task.items())
        },
        "input_parameter_names": dict(parameter_names),
        "input_text_ast_contexts": dict(usage_contexts),
        "forwarding_representatives": dict(forwarding_representatives),
        "consuming_representatives": dict(consuming_representatives),
        "note": (
            "Representative grader source is included only in this workflow artifact for contract inspection; "
            "the dataset is public. Counts are static AST/dataflow observations, not grading results."
        ),
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "overall": payload["overall"],
                "by_split": payload["by_split"],
                "by_task_type": payload["by_task_type"],
                "input_parameter_names": payload["input_parameter_names"],
                "input_text_ast_contexts": payload["input_text_ast_contexts"],
                "consuming_representative_counts": {
                    name: len(rows)
                    for name, rows in payload["consuming_representatives"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
