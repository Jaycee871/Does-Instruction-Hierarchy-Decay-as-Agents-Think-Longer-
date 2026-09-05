from __future__ import annotations

import ast
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from ih_decay.data import DATA_FILES, iter_examples
from ih_decay.grader import grader_reads_input_text

OUTPUT_PATH = Path("grader-input-contract-audit.json")
REPRESENTATIVES_PER_SPLIT = 5


def _input_text_contexts(grader_code: str) -> list[str]:
    tree = ast.parse(grader_code, filename="<dataset-grader>", mode="exec")
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    contexts: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Name)
            and node.id == "input_text"
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
    return contexts


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    overall = Counter()
    by_split: dict[str, Counter] = {}
    by_task: dict[str, Counter] = defaultdict(Counter)
    usage_contexts = Counter()
    representatives: dict[str, list[dict[str, object]]] = defaultdict(list)

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
            bucket = "reads_input_text" if reads else "ignores_input_text"
            split_counts[bucket] += 1
            overall[bucket] += 1
            by_task[task_type][bucket] += 1

            if reads:
                contexts = _input_text_contexts(grader_code)
                usage_contexts.update(contexts)
                if len(representatives[source_file]) < REPRESENTATIVES_PER_SPLIT:
                    representatives[source_file].append(
                        {
                            "example_id": example.example_id,
                            "row_index": example.row_index,
                            "task_type": task_type,
                            "privileged_level": example.metadata.get("privileged_level"),
                            "attack_level": example.metadata.get("attack_level"),
                            "attacker_problem": example.attacker_problem,
                            "grader_code_python": grader_code,
                            "input_text_contexts": contexts,
                        }
                    )
        by_split[source_file] = split_counts

    payload = {
        "purpose": "determine where IH-Challenge graders consume input_text before expanding graded pilots",
        "overall": dict(overall),
        "by_split": {name: dict(counts) for name, counts in by_split.items()},
        "by_task_type": {
            name: dict(counts) for name, counts in sorted(by_task.items())
        },
        "input_text_ast_contexts": dict(usage_contexts),
        "representatives": dict(representatives),
        "note": (
            "Representative grader source is included only in this workflow artifact for contract inspection; "
            "the dataset is public. Counts are static AST observations, not grading results."
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
                "input_text_ast_contexts": payload["input_text_ast_contexts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
