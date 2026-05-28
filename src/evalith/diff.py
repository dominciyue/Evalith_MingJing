from __future__ import annotations

import random
from dataclasses import dataclass

from .models import CaseResult, Run


def case_score(result: CaseResult) -> float:
    return result.mean_score


def _case_samples(r: CaseResult) -> list[float]:
    """Per-trial pass rates if samples>1, else a single-element list with the case's mean score."""
    return list(r.pass_rate_samples) if r.pass_rate_samples else [case_score(r)]


def bootstrap_diff_ci(before: list[float], after: list[float], *,
                      n_resamples: int = 1000, alpha: float = 0.05,
                      seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI on (mean(after) - mean(before)). Deterministic via seed."""
    rng = random.Random(seed)
    n_b, n_a = len(before), len(after)
    diffs: list[float] = []
    for _ in range(n_resamples):
        b_mean = sum(rng.choice(before) for _ in range(n_b)) / n_b
        a_mean = sum(rng.choice(after) for _ in range(n_a)) / n_a
        diffs.append(a_mean - b_mean)
    diffs.sort()
    lo_idx = int(n_resamples * alpha / 2)
    hi_idx = min(n_resamples - 1, int(n_resamples * (1 - alpha / 2)))
    return diffs[lo_idx], diffs[hi_idx]


@dataclass
class CaseDiff:
    case_id: str
    status: str  # improved | regressed | unchanged | new | removed
    before: float | None
    after: float | None
    before_output: str | None = None
    after_output: str | None = None
    ci: tuple[float, float] | None = None  # 95% CI on (after - before) when bootstrapped


@dataclass
class DiffReport:
    cases: list[CaseDiff]

    @property
    def improved(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.status == "improved"]

    @property
    def regressed(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.status == "regressed"]

    @property
    def unchanged(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.status == "unchanged"]

    @property
    def new(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.status == "new"]

    @property
    def removed(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.status == "removed"]

    def summary(self) -> dict[str, int]:
        return {
            "improved": len(self.improved),
            "regressed": len(self.regressed),
            "unchanged": len(self.unchanged),
            "new": len(self.new),
            "removed": len(self.removed),
        }


def diff_runs(before: Run, after: Run, tol: float = 1e-9) -> DiffReport:
    before_r = {r.case_id: r for r in before.results}
    after_r = {r.case_id: r for r in after.results}
    cases: list[CaseDiff] = []
    for cid, ar in after_r.items():
        a = case_score(ar)
        if cid not in before_r:
            cases.append(CaseDiff(cid, "new", None, a, None, ar.output))
            continue
        br = before_r[cid]
        b = case_score(br)
        b_samples, a_samples = _case_samples(br), _case_samples(ar)
        # Bootstrap CI only when at least one side has >=2 trials — else fall back to point compare
        if max(len(b_samples), len(a_samples)) >= 2:
            lo, hi = bootstrap_diff_ci(b_samples, a_samples)
            if hi < -tol:
                status = "regressed"
            elif lo > tol:
                status = "improved"
            else:
                status = "unchanged"
            cases.append(CaseDiff(cid, status, b, a, br.output, ar.output, ci=(lo, hi)))
        else:
            if a > b + tol:
                status = "improved"
            elif a < b - tol:
                status = "regressed"
            else:
                status = "unchanged"
            cases.append(CaseDiff(cid, status, b, a, br.output, ar.output))
    for cid, br in before_r.items():
        if cid not in after_r:
            cases.append(CaseDiff(cid, "removed", case_score(br), None, br.output, None))
    cases.sort(key=lambda c: c.case_id)
    return DiffReport(cases=cases)
