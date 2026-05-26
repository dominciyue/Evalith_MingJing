from __future__ import annotations

import uuid
from datetime import datetime, timezone

from .config import EvalConfig
from .dataset import load_dataset
from .models import CaseResult, Run
from .providers.base import Provider
from .scorers.rules import build_scorer


def _render(template: str, case_input: str) -> str:
    return template.replace("{{input}}", case_input)


def run_eval(config: EvalConfig, provider: Provider,
             judge_provider: Provider | None = None) -> Run:
    dataset = load_dataset(config.dataset)
    scorers = [build_scorer(s, judge_provider=judge_provider or provider)
               for s in config.scorers]
    results: list[CaseResult] = []
    for case in dataset.cases:
        prompt = _render(config.prompt_template, case.input)
        resp = provider.complete(prompt, system=config.system, temperature=config.temperature)
        scores = [scorer.score(case, resp.text) for scorer in scorers]
        results.append(
            CaseResult(case_id=case.id, input=case.input, output=resp.text,
                       scores=scores, latency_ms=resp.latency_ms,
                       prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                       total_tokens=resp.total_tokens, cost_usd=resp.cost_usd)
        )
    return Run(
        id=uuid.uuid4().hex[:12],
        name=config.name,
        created_at=datetime.now(timezone.utc),
        model=config.model,
        results=results,
        config=config.model_dump(),
    )
