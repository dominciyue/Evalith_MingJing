"""Judge consensus statistics (v0.7).

Panel judges are diagnostics only: they never affect scores, pass/fail or
exit codes. Statistics ported from docs/blog/article4/multi_compare.py
(validated against the article's published numbers).
"""
from __future__ import annotations

from itertools import combinations

from .models import CaseResult, Run

MISSING = -1.0   # sentinel: panel judge call failed for that trial
PRIMARY = "primary"


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two equal-length binary label vectors."""
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return 1.0   # both raters constant and equal — degenerate perfect agreement
    return (po - pe) / (1 - pe)


def judge_means(case: CaseResult) -> dict[str, float]:
    """Per-judge mean pass rate for one case; MISSING trials are skipped."""
    means: dict[str, float] = {}
    if case.pass_rate_samples:
        means[PRIMARY] = sum(case.pass_rate_samples) / len(case.pass_rate_samples)
    for name, vals in case.panel_samples.items():
        ok = [v for v in vals if v != MISSING]
        if ok:
            means[name] = sum(ok) / len(ok)
    return means


def case_spread(case: CaseResult) -> float:
    """Max − min of per-judge means; 0.0 when fewer than two judges scored."""
    means = judge_means(case)
    if len(means) < 2:
        return 0.0
    return max(means.values()) - min(means.values())


def _trial_series(run: Run) -> dict[str, list[float]]:
    """Flat per-(case, trial) value series per judge, MISSING-padded to align."""
    judges = [PRIMARY] + sorted({j for r in run.results for j in r.panel_samples})
    series: dict[str, list[float]] = {j: [] for j in judges}
    for r in run.results:
        n = len(r.pass_rate_samples)
        for j in judges:
            vals = list(r.pass_rate_samples if j == PRIMARY
                        else r.panel_samples.get(j, []))
            vals = (vals + [MISSING] * n)[:n]   # pad/truncate to trial count
            series[j].extend(vals)
    return series


def pairwise_kappa(run: Run) -> dict[tuple[str, str], float]:
    """Pairwise Cohen's kappa over per-(case, trial) binary labels.

    Trials where either judge is MISSING are dropped pairwise.
    """
    series = _trial_series(run)
    out: dict[tuple[str, str], float] = {}
    for j1, j2 in combinations(series.keys(), 2):
        pairs = [(x, y) for x, y in zip(series[j1], series[j2])
                 if x != MISSING and y != MISSING]
        a = [1 if x >= 0.5 else 0 for x, _ in pairs]
        b = [1 if y >= 0.5 else 0 for _, y in pairs]
        out[(j1, j2)] = cohen_kappa(a, b)
    return out


def domain_agreement(run: Run, threshold: float = 0.5) -> dict[str, dict]:
    """Per-domain judge means and low-consensus counts ("?" for untagged)."""
    by_dom: dict[str, list[CaseResult]] = {}
    for r in run.results:
        by_dom.setdefault(r.domain or "?", []).append(r)
    out: dict[str, dict] = {}
    for dom, cases in sorted(by_dom.items()):
        sums: dict[str, list[float]] = {}
        low = 0
        for c in cases:
            for j, m in judge_means(c).items():
                sums.setdefault(j, []).append(m)
            if case_spread(c) >= threshold:
                low += 1
        out[dom] = {"n": len(cases),
                    "means": {j: sum(v) / len(v) for j, v in sums.items()},
                    "low_consensus": low}
    return out


def consensus_summary(run: Run, threshold: float = 0.5) -> dict | None:
    """Inputs for the one-line CLI summary; None when the run has no panel."""
    panel_judges = sorted({j for r in run.results for j in r.panel_samples})
    if not panel_judges:
        return None
    low = [r.case_id for r in run.results if case_spread(r) >= threshold]
    kappas = pairwise_kappa(run)
    valid = [v for v in kappas.values() if v == v]   # drop NaN
    return {"judges": [PRIMARY] + panel_judges,
            "n_cases": len(run.results),
            "low_consensus_cases": low,
            "min_kappa": min(valid) if valid else float("nan"),
            "kappa": kappas,
            "threshold": threshold}


def threshold_from_config(config: dict) -> float:
    """Read consensus_threshold from a Run.config dict; default 0.5."""
    for s in config.get("scorers", []):
        if s.get("type") == "llm_judge":
            return float(s.get("params", {}).get("consensus_threshold", 0.5))
    return 0.5
