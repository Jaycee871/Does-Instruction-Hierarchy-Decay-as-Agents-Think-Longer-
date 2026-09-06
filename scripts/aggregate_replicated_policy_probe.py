from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def _two_sided_exact_sign_p(neutral_only: int, minimal_only: int) -> float | None:
    discordant = neutral_only + minimal_only
    if discordant == 0:
        return None
    k = min(neutral_only, minimal_only)
    lower_tail = sum(math.comb(discordant, i) for i in range(k + 1)) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate replicated continuation-policy probe summary JSON files"
    )
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    if not summaries:
        raise SystemExit("no summaries supplied")

    checkpoint_sets = {tuple(summary["checkpoints"]) for summary in summaries}
    if len(checkpoint_sets) != 1:
        raise SystemExit(f"checkpoint mismatch across summaries: {checkpoint_sets}")
    checkpoints = list(next(iter(checkpoint_sets)))

    aggregate_counts: dict[int, Counter[str]] = {step: Counter() for step in checkpoints}
    at_risk_total = 0
    replicates_total = 0
    item_rows: list[dict[str, object]] = []

    for summary in summaries:
        at_risk = int(summary["shared_step1_pass_replicates"])
        completed = int(summary["replicates_completed"])
        at_risk_total += at_risk
        replicates_total += completed
        item_rows.append(
            {
                "source_file": summary["source_file"],
                "row_index": summary["row_index"],
                "example_id": summary["example_id"],
                "replicates_completed": completed,
                "shared_step1_pass_replicates": at_risk,
                "shared_step1_unique_output_hashes": summary[
                    "shared_step1_unique_output_hashes"
                ],
            }
        )
        paired = summary["paired_checkpoint_counts_step1_pass_replicates"]
        for step in checkpoints:
            aggregate_counts[step].update(paired[str(step)])

    checkpoint_results: dict[str, dict[str, object]] = {}
    for step in checkpoints:
        counts = aggregate_counts[step]
        both_pass = counts["both_pass"]
        neutral_only = counts["neutral_pass_minimal_fail"]
        minimal_only = counts["neutral_fail_minimal_pass"]
        both_fail = counts["both_fail"]
        total = both_pass + neutral_only + minimal_only + both_fail
        discordant = neutral_only + minimal_only
        checkpoint_results[str(step)] = {
            "at_risk_paired_replicates": total,
            "both_pass": both_pass,
            "neutral_pass_minimal_fail": neutral_only,
            "neutral_fail_minimal_pass": minimal_only,
            "both_fail": both_fail,
            "discordant": discordant,
            "discordance_rate": (discordant / total) if total else None,
            "paired_pass_rate_difference_neutral_minus_minimal": (
                (neutral_only - minimal_only) / total if total else None
            ),
            "two_sided_exact_sign_p_exploratory": _two_sided_exact_sign_p(
                neutral_only, minimal_only
            ),
        }

    output = {
        "phase": "phase1-replicated-continuation-policy-probe-aggregate",
        "interpretation": (
            "exploratory construct-validity summary; exact sign p-values are descriptive "
            "and are not a confirmatory causal test"
        ),
        "items": item_rows,
        "replicates_completed_total": replicates_total,
        "shared_step1_pass_replicates_total": at_risk_total,
        "checkpoints": checkpoints,
        "paired_checkpoint_results": checkpoint_results,
    }

    text = json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
