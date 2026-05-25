from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    id: str
    input: str
    expected: str | None = None
    metadata: dict = Field(default_factory=dict)


class Dataset(BaseModel):
    name: str
    cases: list[TestCase]


class Score(BaseModel):
    scorer: str
    value: float
    passed: bool
    detail: str = ""


class CaseResult(BaseModel):
    case_id: str
    input: str
    output: str
    scores: list[Score] = Field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def mean_score(self) -> float:
        return sum(s.value for s in self.scores) / len(self.scores) if self.scores else 0.0


class Run(BaseModel):
    id: str
    name: str
    created_at: datetime
    model: str
    results: list[CaseResult] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)
