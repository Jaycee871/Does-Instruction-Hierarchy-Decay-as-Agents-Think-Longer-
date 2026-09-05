from __future__ import annotations

import argparse
import json
import os

from ih_decay.data import DATA_FILES, iter_examples, summarize_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect IH-Challenge metadata from the HF bucket")
    parser.add_argument("--file", choices=DATA_FILES, default="single-constraint.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    summary = summarize_metadata(
        iter_examples(
            args.file,
            limit=args.limit,
            token=os.getenv("HF_TOKEN"),
        )
    )
    print(json.dumps({"file": args.file, "limit": args.limit, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
