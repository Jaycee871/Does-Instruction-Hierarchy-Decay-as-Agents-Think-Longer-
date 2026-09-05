from __future__ import annotations

import json
import os
from collections import Counter

from ih_decay.data import DATA_FILES, iter_examples


def main() -> None:
    report: dict[str, object] = {"files": {}, "total_rows": 0}
    grand_task_type: Counter[str] = Counter()
    grand_attack_level: Counter[str] = Counter()
    grand_privileged_level: Counter[str] = Counter()

    for source_file in DATA_FILES:
        rows = 0
        task_type: Counter[str] = Counter()
        attack_level: Counter[str] = Counter()
        privileged_level: Counter[str] = Counter()

        for example in iter_examples(source_file, token=os.getenv("HF_TOKEN")):
            rows += 1
            task_type[str(example.metadata.get("task_type", "<missing>"))] += 1
            attack_level[str(example.metadata.get("attack_level", "<missing>"))] += 1
            privileged_level[str(example.metadata.get("privileged_level", "<missing>"))] += 1

        report["files"][source_file] = {
            "rows": rows,
            "task_type": dict(task_type.most_common()),
            "attack_level": dict(attack_level.most_common()),
            "privileged_level": dict(privileged_level.most_common()),
        }
        report["total_rows"] = int(report["total_rows"]) + rows
        grand_task_type.update(task_type)
        grand_attack_level.update(attack_level)
        grand_privileged_level.update(privileged_level)

    report["overall"] = {
        "task_type": dict(grand_task_type.most_common()),
        "attack_level": dict(grand_attack_level.most_common()),
        "privileged_level": dict(grand_privileged_level.most_common()),
    }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
