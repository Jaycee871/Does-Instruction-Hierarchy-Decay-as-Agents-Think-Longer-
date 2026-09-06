from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from ih_decay.metrics import trajectory_metrics


def _iter_records(paths: Iterable[Path]):
    for path in paths:
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            record = json.loads(raw)
            if record.get("status") == "completed" and "trajectory_step" in record:
                yield record


def _bool_grade(record: dict) -> bool | None:
    grade = record.get("grade") or {}
    if grade.get("status") != "ok":
        return None
    value = grade.get("correct")
    return value if isinstance(value, bool) else None


def summarize(paths: list[Path]) -> dict[str, object]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in _iter_records(paths):
        key = f"{record.get('source_file')}:{record.get('row_index')}"
        grouped[key].append(record)

    checkpoint_totals: Counter[int] = Counter()
    checkpoint_passes: Counter[int] = Counter()
    baseline_pass_ids: set[str] = set()
    baseline_conditional_totals: Counter[int] = Counter()
    baseline_conditional_passes: Counter[int] = Counter()
    checkpoint_patterns: Counter[str] = Counter()
    first_failure_steps: Counter[str] = Counter()
    recovered_items = 0
    item_metrics: dict[str, object] = {}

    for key, records in sorted(grouped.items()):
        records.sort(key=lambda row: int(row["trajectory_step"]))
        valid = [(int(row["trajectory_step"]), _bool_grade(row), bool(row.get("is_checkpoint"))) for row in records]
        scored = [(step, result, is_checkpoint) for step, result, is_checkpoint in valid if result is not None]
        if not scored:
            continue

        all_steps = [step for step, _, _ in scored]
        all_results = [bool(result) for _, result, _ in scored]
        metrics = trajectory_metrics(all_steps, all_results)

        checkpoint_rows = [(step, bool(result)) for step, result, is_checkpoint in scored if is_checkpoint]
        for step, result in checkpoint_rows:
            checkpoint_totals[step] += 1
            checkpoint_passes[step] += int(result)

        baseline_pass = bool(checkpoint_rows and checkpoint_rows[0][0] == 1 and checkpoint_rows[0][1])
        if baseline_pass:
            baseline_pass_ids.add(key)
            for step, result in checkpoint_rows:
                baseline_conditional_totals[step] += 1
                baseline_conditional_passes[step] += int(result)

        pattern = "".join("P" if result else "F" for _, result in checkpoint_rows)
        if checkpoint_rows:
            checkpoint_patterns[pattern] += 1

        if metrics.first_failure_step is None:
            first_failure_steps["never"] += 1
        else:
            first_failure_steps[str(metrics.first_failure_step)] += 1
            failure_index = all_steps.index(metrics.first_failure_step)
            if any(all_results[failure_index + 1 :]):
                recovered_items += 1

        item_metrics[key] = {
            "source_file": records[0].get("source_file"),
            "row_index": records[0].get("row_index"),
            "attack_sha256": records[0].get("attack_sha256"),
            "checkpoint_pattern": pattern,
            "metrics_all_steps": metrics.as_dict(),
        }

    checkpoints = sorted(checkpoint_totals)
    return {
        "items_with_scored_steps": len(item_metrics),
        "initially_compliant_items": len(baseline_pass_ids),
        "checkpoint_compliance": {
            str(step): {
                "pass": checkpoint_passes[step],
                "total": checkpoint_totals[step],
                "rate": checkpoint_passes[step] / checkpoint_totals[step] if checkpoint_totals[step] else None,
            }
            for step in checkpoints
        },
        "conditional_on_step1_pass": {
            str(step): {
                "pass": baseline_conditional_passes[step],
                "total": baseline_conditional_totals[step],
                "rate": (
                    baseline_conditional_passes[step] / baseline_conditional_totals[step]
                    if baseline_conditional_totals[step]
                    else None
                ),
            }
            for step in checkpoints
        },
        "checkpoint_patterns": dict(sorted(checkpoint_patterns.items())),
        "first_failure_step_counts": dict(sorted(first_failure_steps.items())),
        "items_with_any_recovery_after_failure": recovered_items,
        "item_metrics": item_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize matched IH trajectory JSONL artifacts")
    parser.add_argument("paths", nargs="+", type=Path, help="trajectory JSONL files")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = summarize(args.paths)
    payload = json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
