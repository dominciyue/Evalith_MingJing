from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist

from .models import CaseResult, Run


def case_score(result: CaseResult) -> float:
    return result.mean_score


def _case_samples(r: CaseResult) -> list[float]:
    """Per-trial pass rates if samples>1, else a single-element list with the case's mean score."""
    return list(r.pass_rate_samples) if r.pass_rate_samples else [case_score(r)]


def bootstrap_diff_ci(before: list[float], after: list[float], *,
                      method: str = "percentile",
                      n_resamples: int = 1000, alpha: float = 0.05,
                      seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI on (mean(after) - mean(before)).

    method: "percentile" (v0.4 default), "bca", or "paired".
    Deterministic via seed."""
    if method not in {"percentile", "bca", "paired"}:
        raise ValueError(f"unknown method: {method!r}; expected percentile|bca|paired")
    if method == "percentile":
        return _bootstrap_percentile(before, after, n_resamples=n_resamples, alpha=alpha, seed=seed)
    if method == "bca":
        return _bootstrap_bca(before, after, n_resamples=n_resamples, alpha=alpha, seed=seed)
    # paired
    return _bootstrap_paired(before, after, n_resamples=n_resamples, alpha=alpha, seed=seed)


def _bootstrap_percentile(before: list[float], after: list[float], *,
                          n_resamples: int = 1000, alpha: float = 0.05,
                          seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap — v0.4 behavior, preserved exactly."""
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


def _bootstrap_bca(before: list[float], after: list[float], *,
                   n_resamples: int = 1000, alpha: float = 0.05,
                   seed: int = 0) -> tuple[float, float]:
    """BCa (bias-corrected and accelerated) bootstrap CI on mean(after) - mean(before).

    Stdlib-only implementation following scipy.stats.bootstrap method='BCa'. Steps:
      1. Observed statistic θ̂
      2. Bootstrap distribution {θ̂*_b}
      3. Bias correction z₀ = Φ⁻¹(mean-kind percentile of θ̂ in {θ̂*_b})
         where mean-kind = (#{< θ̂} + #{≤ θ̂}) / (2B)  — matches scipy's _percentile_of_score
      4. Per-sample jackknife with U_ji = (n_j-1)*(θ̂_j. - θ̂_ji) influence values
         to compute acceleration a_hat  — matches scipy's independent-sample BCa
      5. Adjusted percentile bounds α₁, α₂ via BCa formula
    """
    n_b, n_a = len(before), len(after)
    rng = random.Random(seed)
    theta_hat = sum(after) / n_a - sum(before) / n_b

    # Step 2: bootstrap distribution
    boots: list[float] = []
    for _ in range(n_resamples):
        b_mean = sum(rng.choice(before) for _ in range(n_b)) / n_b
        a_mean = sum(rng.choice(after) for _ in range(n_a)) / n_a
        boots.append(a_mean - b_mean)

    # Step 3: bias correction z₀ — use "mean" percentile kind to match scipy
    n_below = sum(1 for t in boots if t < theta_hat)
    n_below_eq = sum(1 for t in boots if t <= theta_hat)
    proportion = (n_below + n_below_eq) / (2 * n_resamples)
    # Clip to avoid Φ⁻¹(0) or Φ⁻¹(1)
    proportion = max(1e-6, min(1 - 1e-6, proportion))
    nd = NormalDist()
    z0 = nd.inv_cdf(proportion)

    # Step 4: per-sample jackknife to compute acceleration a_hat
    # For each sample (before, after) independently, compute leave-one-out influence values
    # U_ji = (n_j - 1) * (theta_j_dot - theta_ji)  where theta_ji = stat with obs i removed
    full_sum_b = sum(before)
    full_sum_a = sum(after)

    # Jackknife on before: removing before[i], keep all of after
    jk_b = [(full_sum_a / n_a) - ((full_sum_b - before[i]) / (n_b - 1))
            for i in range(n_b)] if n_b > 1 else [0.0]
    jk_b_dot = sum(jk_b) / n_b
    U_b = [(n_b - 1) * (jk_b_dot - t) for t in jk_b]

    # Jackknife on after: keep all of before, removing after[j]
    jk_a = [((full_sum_a - after[j]) / (n_a - 1)) - (full_sum_b / n_b)
            for j in range(n_a)] if n_a > 1 else [0.0]
    jk_a_dot = sum(jk_a) / n_a
    U_a = [(n_a - 1) * (jk_a_dot - t) for t in jk_a]

    num_b = sum(u ** 3 for u in U_b) / n_b ** 3
    num_a = sum(u ** 3 for u in U_a) / n_a ** 3
    den_b = sum(u ** 2 for u in U_b) / n_b ** 2
    den_a = sum(u ** 2 for u in U_a) / n_a ** 2
    den_total = den_b + den_a
    accel = 0.0 if den_total == 0 else (1 / 6) * (num_b + num_a) / den_total ** 1.5

    # Step 5: BCa-adjusted percentiles
    z_lo = nd.inv_cdf(alpha / 2)
    z_hi = nd.inv_cdf(1 - alpha / 2)

    def adjusted_p(z_target: float) -> float:
        num_z = z0 + z_target
        den_p = 1 - accel * num_z
        if den_p == 0:
            return 0.5
        return nd.cdf(z0 + num_z / den_p)

    p_lo = adjusted_p(z_lo)
    p_hi = adjusted_p(z_hi)

    boots.sort()
    lo_idx = max(0, min(n_resamples - 1, int(p_lo * n_resamples)))
    hi_idx = max(0, min(n_resamples - 1, int(p_hi * n_resamples)))
    return boots[lo_idx], boots[hi_idx]


def _bootstrap_paired(before: list[float], after: list[float], *,
                      n_resamples: int = 1000, alpha: float = 0.05,
                      seed: int = 0) -> tuple[float, float]:
    """Paired bootstrap CI on mean(after - before).

    Requires len(before) == len(after). Resamples case indices (with replacement),
    then computes Δᵢ = after[i] - before[i] for each chosen i and takes the mean.
    Reduces variance when before/after are correlated through a shared case dimension."""
    if len(before) != len(after):
        raise ValueError(f"paired bootstrap requires equal lengths; got before={len(before)} after={len(after)}")
    n = len(before)
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        mean_delta = sum(after[i] - before[i] for i in idxs) / n
        diffs.append(mean_delta)
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
