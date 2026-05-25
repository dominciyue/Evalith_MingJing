from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ScorerConfig(BaseModel):
    type: str
    params: dict = Field(default_factory=dict)


class EvalConfig(BaseModel):
    name: str
    dataset: str
    model: str
    prompt_template: str = "{{input}}"
    system: str | None = None
    temperature: float = 0.0
    scorers: list[ScorerConfig] = Field(default_factory=list)


def load_config(path: str | Path) -> EvalConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return EvalConfig.model_validate(data)
