from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

API_URL = "https://integrate.api.nvidia.com/v1/models"


def main() -> None:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY is not set")

    request = urllib.request.Request(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "ih-decay-model-discovery/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)

    model_ids = sorted(
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    )
    preferred = [
        model_id
        for model_id in model_ids
        if any(
            token in model_id.lower()
            for token in ("llama", "qwen", "nemotron", "mistral", "gemma")
        )
    ]

    report = {
        "endpoint": API_URL,
        "total_models": len(model_ids),
        "preferred_text_model_candidates": preferred,
        "all_model_ids": model_ids,
    }
    Path("nvidia-model-inventory.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "total_models": len(model_ids),
                "preferred_text_model_candidates": preferred[:30],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
