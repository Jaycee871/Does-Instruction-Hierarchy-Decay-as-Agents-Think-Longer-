from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


class ProviderError(RuntimeError):
    """Raised when a model provider returns an unusable response."""


@dataclass(frozen=True)
class ChatResult:
    model: str
    content: str
    finish_reason: str | None
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_int(mapping: Mapping[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    return int(value) if isinstance(value, int) else None


def nvidia_chat(
    *,
    api_key: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout_seconds: int = 120,
    reasoning_effort: str | None = None,
) -> ChatResult:
    """Call NVIDIA's OpenAI-compatible chat endpoint and retain final content only.

    Deliberately ignores provider-specific reasoning/analysis fields. The research
    pipeline records observable answers and usage metadata, not hidden reasoning.
    """
    if not api_key:
        raise ValueError("api_key must be provided")
    if not model:
        raise ValueError("model must be provided")
    if not messages:
        raise ValueError("messages must be non-empty")
    if reasoning_effort not in {None, "low", "medium", "high"}:
        raise ValueError("reasoning_effort must be one of: low, medium, high, or None")

    payload: dict[str, Any] = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort

    request = urllib.request.Request(
        NVIDIA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ih-decay/0.2",
        },
        method="POST",
    )

    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        preview = exc.read().decode("utf-8", errors="replace")[:500]
        raise ProviderError(f"NVIDIA HTTP {exc.code}: {preview}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"NVIDIA request failed: {exc.reason}") from exc

    try:
        choice = body["choices"][0]
        message = choice.get("message") or {}
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("NVIDIA response did not contain choices[0].message") from exc

    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    finish_reason = choice.get("finish_reason")
    content_value = message.get("content")
    content = content_value if isinstance(content_value, str) else ""
    if not content.strip():
        raise ProviderError(
            "NVIDIA response contained no final content; hidden reasoning fields are not used "
            f"(finish_reason={finish_reason!r}, "
            f"completion_tokens={_optional_int(usage, 'completion_tokens')!r})"
        )

    return ChatResult(
        model=model,
        content=content,
        finish_reason=finish_reason,
        latency_seconds=round(time.monotonic() - started, 3),
        prompt_tokens=_optional_int(usage, "prompt_tokens"),
        completion_tokens=_optional_int(usage, "completion_tokens"),
        total_tokens=_optional_int(usage, "total_tokens"),
    )
