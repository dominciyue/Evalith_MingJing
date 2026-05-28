from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .config import EvalConfig
from .dataset import load_dataset
from .models import CaseResult, Run, Score
from .providers.base import Provider
from .scorers.rules import build_scorer


def _render(template: str, case_input: str) -> str:
    return template.replace("{{input}}", case_input)


def run_eval(config: EvalConfig, provider: Provider,
             judge_provider: Provider | None = None,
             concurrency: int | None = None) -> Run:
    dataset = load_dataset(config.dataset)
    scorers = [build_scorer(s, judge_provider=judge_provider or provider)
               for s in config.scorers]
    workers = max(1, concurrency if concurrency is not None else config.concurrency)

    samples = max(1, config.samples)

    def _eval_once(case) -> CaseResult:
        try:
            prompt = _render(config.prompt_template, case.input)
            resp = provider.complete(prompt, system=config.system, temperature=config.temperature)
            scores = [scorer.score(case, resp.text) for scorer in scorers]
            return CaseResult(case_id=case.id, input=case.input, output=resp.text,
                              scores=scores, latency_ms=resp.latency_ms,
                              prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                              total_tokens=resp.total_tokens, cost_usd=resp.cost_usd)
        except Exception as e:  # one failed case must not kill the whole run
            return CaseResult(case_id=case.id, input=case.input, output="",
                              scores=[Score(scorer="error", value=0.0, passed=False,
                                            detail=f"{type(e).__name__}: {e}")])

    def _trial_pass_rate(cr: CaseResult) -> float:
        return sum(1 for s in cr.scores if s.passed) / len(cr.scores) if cr.scores else 0.0

    def _eval_case(case) -> CaseResult:
        if samples == 1:
            return _eval_once(case)
        trials = [_eval_once(case) for _ in range(samples)]
        rep = trials[0]  # representative trial for output/scores; aggregate the rest
        return CaseResult(
            case_id=rep.case_id, input=rep.input, output=rep.output, scores=rep.scores,
            latency_ms=sum(t.latency_ms for t in trials) / samples,
            prompt_tokens=sum(t.prompt_tokens for t in trials),
            completion_tokens=sum(t.completion_tokens for t in trials),
            total_tokens=sum(t.total_tokens for t in trials),
            cost_usd=sum(t.cost_usd for t in trials),
            pass_rate_samples=[_trial_pass_rate(t) for t in trials],
        )

    if workers == 1:
        results = [_eval_case(c) for c in dataset.cases]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_eval_case, dataset.cases))  # map preserves input order

    return Run(id=uuid.uuid4().hex[:12], name=config.name,
               created_at=datetime.now(timezone.utc), model=config.model,
               results=results, config=config.model_dump())
