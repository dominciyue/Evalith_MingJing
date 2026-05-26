from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Response:
    text: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class Provider(Protocol):
    model: str

    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0) -> Response: ...


class FakeProvider:
    """Test helper: returns canned responses keyed by prompt, else `default`."""

    def __init__(self, responses: dict[str, str] | None = None,
                 default: str = "", model: str = "fake"):
        self.responses = responses or {}
        self.default = default
        self.model = model

    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0) -> Response:
        return Response(text=self.responses.get(prompt, self.default))


class EchoProvider:
    """Offline provider: echoes the prompt, or returns `fixed` if given."""

    def __init__(self, fixed: str | None = None, model: str = "echo"):
        self.fixed = fixed
        self.model = model

    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0) -> Response:
        return Response(text=self.fixed if self.fixed is not None else prompt)
