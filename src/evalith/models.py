from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    __test__ = False  # tell pytest this pydantic model is not a test class
    id: str
    input: str
    expected: str | None = None
    expected_concepts: list[str] | None = None
    domain: str | None = None
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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    pass_rate_samples: list[float] = Field(default_factory=list)  # one per trial when samples>1
    domain: str | None = None
    # Judge consensus panel (v0.7) — diagnostics only, never part of the gate
    panel_samples: dict[str, list[float]] = Field(default_factory=dict)  # judge -> per-trial pass (−1.0 = judge call failed)
    panel_details: dict[str, str] = Field(default_factory=dict)          # judge -> representative reason
    judge_tokens: int = 0       # primary + panel judging calls combined
    judge_cost_usd: float = 0.0

    @property
    def mean_score(self) -> float:
        if self.pass_rate_samples:
            return sum(self.pass_rate_samples) / len(self.pass_rate_samples)
        return sum(s.value for s in self.scores) / len(self.scores) if self.scores else 0.0


class Run(BaseModel):
    id: str
    name: str
    created_at: datetime
    model: str
    results: list[CaseResult] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        checks = [s for r in self.results for s in r.scores]
        return sum(1 for s in checks if s.passed) / len(checks) if checks else 1.0

    @property
    def mean_score(self) -> float:
        return sum(r.mean_score for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.results)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def total_judge_tokens(self) -> int:
        return sum(r.judge_tokens for r in self.results)

    @property
    def total_judge_cost_usd(self) -> float:
        return sum(r.judge_cost_usd for r in self.results)

    @property
    def mean_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.results) / len(self.results) if self.results else 0.0
