from __future__ import annotations

import time

from .base import Response


def _usage_from_response(resp) -> tuple[int, int, int]:
    usage = (resp.get("usage") if isinstance(resp, dict) else getattr(resp, "usage", None)) or {}
    get = usage.get if isinstance(usage, dict) else (lambda k, d=0: getattr(usage, k, d))
    return (int(get("prompt_tokens", 0) or 0),
            int(get("completion_tokens", 0) or 0),
            int(get("total_tokens", 0) or 0))


class LiteLLMProvider:
    """Real models (DeepSeek/Qwen/OpenAI/Claude/...) via LiteLLM."""

    def __init__(self, model: str):
        self.model = model

    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0) -> Response:
        import litellm

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        start = time.perf_counter()
        resp = litellm.completion(model=self.model, messages=messages, temperature=temperature)
        latency_ms = (time.perf_counter() - start) * 1000
        pt, ct, tt = _usage_from_response(resp)
        try:
            cost = float(litellm.completion_cost(completion_response=resp))
        except Exception:
            cost = 0.0
        return Response(text=resp["choices"][0]["message"]["content"], latency_ms=latency_ms,
                        prompt_tokens=pt, completion_tokens=ct, total_tokens=tt, cost_usd=cost)
