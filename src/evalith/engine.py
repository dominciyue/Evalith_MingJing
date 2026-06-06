from __future__ import annotations

import random as _random
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .config import EvalConfig
from .consensus import MISSING
from .dataset import load_dataset
from .models import CaseResult, Run, Score
from .providers.base import Provider
from .scorers.rules import build_scorer


def _bootstrap_mean_ci(
    values: list[float],
    n_resamples: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a percentile bootstrap CI on the mean of `values`.

    Returns the widest possible interval (0.0, 1.0) when there is only one
    sample — not enough data to estimate variance.
    """
    if len(values) < 2:
        return (0.0, 1.0)
    rng = _random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int(n_resamples * alpha / 2)
    hi_idx = min(n_resamples - 1, int(n_resamples * (1 - alpha / 2)))
    return means[lo_idx], means[hi_idx]


def _render(template: str, case_input: str) -> str:
    return template.replace("{{input}}", case_input)


def _merge_panel(trials: list[CaseResult],
                 prs: list[float]) -> tuple[dict[str, list[float]], dict[str, str]]:
    """Concatenate per-trial panel values; pick each judge's reason from the
    trial where it disagrees most with the primary pass rate."""
    samples: dict[str, list[float]] = {}
    trial_idx: dict[str, list[int]] = {}
    for ti, t in enumerate(trials):
        for j, vals in t.panel_samples.items():
            samples.setdefault(j, []).extend(vals)
            trial_idx.setdefault(j, []).extend([ti] * len(vals))
    details: dict[str, str] = {}
    for j, vals in samples.items():
        best_k, best_gap = None, -1.0
        for k, v in enumerate(vals):
            if v == MISSING:
                continue
            ti = trial_idx[j][k]
            ref = prs[ti] if ti < len(prs) else 0.0
            gap = abs(v - ref)
            if gap > best_gap:
                best_gap, best_k = gap, k
        if best_k is not None:
            d = trials[trial_idx[j][best_k]].panel_details.get(j, "")
            if d:
                details[j] = d
    return samples, details


def run_eval(config: EvalConfig, provider: Provider,
             judge_provider: Provider | None = None,
             concurrency: int | None = None) -> Run:
    dataset = load_dataset(config.dataset)
    scorers = [build_scorer(s, judge_provider=judge_provider or provider)
               for s in config.scorers]
    workers = max(1, concurrency if concurrency is not None else config.concurrency)

    samples = max(1, config.samples)

    def _eval_once(case, panel_cache) -> CaseResult:
        domain = case.domain or case.metadata.get("domain")
        try:
            prompt = _render(config.prompt_template, case.input)
            resp = provider.complete(prompt, system=config.system, temperature=config.temperature)
            scores: list[Score] = []
            panel_scores: dict = {}
            judge_tokens, judge_cost = 0, 0.0
            for scorer in scorers:
                if hasattr(scorer, "score_with_usage"):
                    s, tok, c = scorer.score_with_usage(case, resp.text)
                    judge_tokens += tok
                    judge_cost += c
                else:
                    s = scorer.score(case, resp.text)
                scores.append(s)
                if getattr(scorer, "panel", None):
                    pres, tok, c = scorer.panel_score(case, resp.text, panel_cache)
                    judge_tokens += tok
                    judge_cost += c
                    panel_scores.update(pres)
            return CaseResult(
                case_id=case.id, input=case.input, output=resp.text, scores=scores,
                latency_ms=resp.latency_ms, prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens, total_tokens=resp.total_tokens,
                cost_usd=resp.cost_usd, domain=domain,
                judge_tokens=judge_tokens, judge_cost_usd=judge_cost,
                panel_samples={j: [(1.0 if s.passed else 0.0) if s is not None else MISSING]
                               for j, s in panel_scores.items()},
                panel_details={j: s.detail for j, s in panel_scores.items()
                               if s is not None and s.detail},
            )
        except Exception as e:  # one failed case must not kill the whole run
            panel_judges = [j for sc in scorers for j in (getattr(sc, "panel", None) or {})]
            return CaseResult(case_id=case.id, input=case.input, output="", domain=domain,
                              scores=[Score(scorer="error", value=0.0, passed=False,
                                            detail=f"{type(e).__name__}: {e}")],
                              panel_samples={j: [MISSING] for j in panel_judges})

    def _trial_pass_rate(cr: CaseResult) -> float:
        return sum(1 for s in cr.scores if s.passed) / len(cr.scores) if cr.scores else 0.0

    def _aggregate(trials: list[CaseResult], prs: list[float]) -> CaseResult:
        rep = trials[0]  # representative trial for output/scores; aggregate the rest
        n = len(trials)
        panel_samples, panel_details = _merge_panel(trials, prs)
        return CaseResult(
            case_id=rep.case_id, input=rep.input, output=rep.output, scores=rep.scores,
            latency_ms=sum(t.latency_ms for t in trials) / n,
            prompt_tokens=sum(t.prompt_tokens for t in trials),
            completion_tokens=sum(t.completion_tokens for t in trials),
            total_tokens=sum(t.total_tokens for t in trials),
            cost_usd=sum(t.cost_usd for t in trials),
            domain=rep.domain,
            judge_tokens=sum(t.judge_tokens for t in trials),
            judge_cost_usd=sum(t.judge_cost_usd for t in trials),
            pass_rate_samples=prs,
            panel_samples=panel_samples,
            panel_details=panel_details,
        )

    def _eval_case(case) -> CaseResult:
        panel_cache: dict = {}
        if config.adaptive:
            trials: list[CaseResult] = []
            prs: list[float] = []
            for i in range(config.max_samples):
                trial = _eval_once(case, panel_cache)
                trials.append(trial)
                prs.append(_trial_pass_rate(trial))
                if (i + 1) >= config.min_samples:
                    lo, hi = _bootstrap_mean_ci(prs)
                    if (hi - lo) < config.ci_tolerance:
                        break
            return _aggregate(trials, prs)
        if samples == 1:
            return _eval_once(case, panel_cache)
        trials = [_eval_once(case, panel_cache) for _ in range(samples)]
        return _aggregate(trials, [_trial_pass_rate(t) for t in trials])

    if workers == 1:
        results = [_eval_case(c) for c in dataset.cases]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_eval_case, dataset.cases))  # map preserves input order

    return Run(id=uuid.uuid4().hex[:12], name=config.name,
               created_at=datetime.now(timezone.utc), model=config.model,
               results=results, config=config.model_dump())
