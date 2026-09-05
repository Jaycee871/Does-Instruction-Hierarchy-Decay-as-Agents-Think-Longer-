from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from ih_decay.data import DATA_FILES, iter_examples
from ih_decay.sampling import PilotCandidate, select_stratified


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic stratified IH-Challenge pilot manifest")
    parser.add_argument("--items-per-file", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--output", default="pilot-manifest.jsonl")
    args = parser.parse_args()

    output = Path(args.output)
    selected_all: list[PilotCandidate] = []

    for file_index, source_file in enumerate(DATA_FILES):
        candidates = (
            PilotCandidate.from_example(example)
            for example in iter_examples(source_file, token=os.getenv("HF_TOKEN"))
        )
        selected = select_stratified(
            candidates,
            n=args.items_per_file,
            seed=args.seed + file_index,
        )
        selected_all.extend(selected)

    with output.open("w", encoding="utf-8") as handle:
        for candidate in selected_all:
            record = candidate.as_dict()
            record["selection_seed"] = args.seed
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    authority = Counter((x.privileged_level, x.attack_level) for x in selected_all)
    task_types = Counter(x.task_type for x in selected_all)
    summary = {
        "output": str(output),
        "rows": len(selected_all),
        "items_per_file": args.items_per_file,
        "seed": args.seed,
        "authority_pairs": {f"{p}->{a}": n for (p, a), n in sorted(authority.items())},
        "task_types": dict(task_types.most_common()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
