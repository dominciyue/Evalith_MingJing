from __future__ import annotations

import time

from .base import Response


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
        return Response(text=resp["choices"][0]["message"]["content"], latency_ms=latency_ms)
