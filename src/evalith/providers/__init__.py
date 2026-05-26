from __future__ import annotations

from .base import EchoProvider, FakeProvider, Provider, Response


def get_provider(model: str) -> Provider:
    from ..presets import resolve_model
    model = resolve_model(model)
    if model == "echo" or model.startswith("echo:"):
        fixed = model.split(":", 1)[1] if ":" in model else None
        return EchoProvider(fixed=fixed)
    from .litellm_provider import LiteLLMProvider

    return LiteLLMProvider(model=model)


__all__ = ["Provider", "Response", "FakeProvider", "EchoProvider", "get_provider"]
