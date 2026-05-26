from __future__ import annotations

# Curated aliases for first-class 国产 models, verified against litellm's native
# providers. Any litellm model id (e.g. "gpt-4o-mini", "claude-3-5-sonnet") also
# works directly without an alias.
# value: {"litellm": <litellm model id>, "env": <required API key env var>, "note": <str>}
CHINA_MODELS: dict[str, dict] = {
    "deepseek-chat":     {"litellm": "deepseek/deepseek-chat",     "env": "DEEPSEEK_API_KEY",  "note": "DeepSeek V3 chat"},
    "deepseek-reasoner": {"litellm": "deepseek/deepseek-reasoner", "env": "DEEPSEEK_API_KEY",  "note": "DeepSeek R1 reasoner"},
    "qwen-max":          {"litellm": "dashscope/qwen-max",         "env": "DASHSCOPE_API_KEY", "note": "Alibaba Qwen-Max"},
    "qwen-plus":         {"litellm": "dashscope/qwen-plus",        "env": "DASHSCOPE_API_KEY", "note": "Alibaba Qwen-Plus"},
}

_ALIASES = {name: info["litellm"] for name, info in CHINA_MODELS.items()}


def resolve_model(name: str) -> str:
    """Map a friendly alias to its litellm id; pass anything else through unchanged."""
    return _ALIASES.get(name, name)
