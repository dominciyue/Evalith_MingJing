from datetime import datetime, timezone

from evalith.diff import diff_runs
from evalith.models import CaseResult, Run, Score


def _run(rid: str, scores_by_case: dict[str, float]) -> Run:
    results = [
        CaseResult(case_id=cid, input="q", output="o",
                   scores=[Score(scorer="s", value=v, passed=v >= 0.5)])
        for cid, v in scores_by_case.items()
    ]
    return Run(id=rid, name="t", created_at=datetime(2026, 5, 25, tzinfo=timezone.utc),
               model="echo", results=results)


def test_diff_detects_all_statuses():
    before = _run("a", {"1": 0.5, "2": 1.0, "3": 0.0})
    after = _run("b", {"1": 1.0, "2": 1.0, "4": 0.7})
    # case 1 improved, case 2 unchanged, case 3 removed, case 4 new
    report = diff_runs(before, after)
    assert report.summary() == {
        "improved": 1, "regressed": 0, "unchanged": 1, "new": 1, "removed": 1,
    }


def test_diff_detects_regression():
    before = _run("a", {"1": 1.0})
    after = _run("b", {"1": 0.0})
    report = diff_runs(before, after)
    assert [c.case_id for c in report.regressed] == ["1"]
    assert report.regressed[0].before == 1.0
    assert report.regressed[0].after == 0.0


def _sampled_run(rid: str, samples_by_case: dict[str, list[float]]) -> Run:
    """Build a Run where each case carries explicit pass_rate_samples."""
    results = []
    for cid, samples in samples_by_case.items():
        # representative score reflects the mean — keeps `mean_score` consistent
        mean = sum(samples) / len(samples)
        cr = CaseResult(case_id=cid, input="q", output="o",
                        scores=[Score(scorer="s", value=mean, passed=mean >= 0.5)],
                        pass_rate_samples=samples)
        results.append(cr)
    return Run(id=rid, name="t", created_at=datetime(2026, 5, 28, tzinfo=timezone.utc),
               model="echo", results=results)


def test_diff_bootstrap_does_not_flag_noise_as_regression():
    """A 0.6 -> 0.4 drop with high variance should be `unchanged` (CI crosses zero)."""
    before = _sampled_run("a", {"c1": [1.0, 0.0, 1.0, 0.0, 1.0]})  # mean 0.6, very noisy
    after  = _sampled_run("b", {"c1": [0.0, 1.0, 0.0, 1.0, 0.0]})  # mean 0.4, same noise
    report = diff_runs(before, after)
    assert [c.case_id for c in report.regressed] == []          # noise must not gate CI
    assert report.cases[0].status == "unchanged"
    # CI on Δ should be present and should straddle 0
    assert report.cases[0].ci is not None
    lo, hi = report.cases[0].ci
    assert lo < 0 < hi


def test_diff_bootstrap_flags_real_regression():
    """A clean 1.0 -> 0.0 across many trials must still be flagged as regressed."""
    before = _sampled_run("a", {"c1": [1.0] * 8})
    after  = _sampled_run("b", {"c1": [0.0] * 8})
    report = diff_runs(before, after)
    assert [c.case_id for c in report.regressed] == ["c1"]
    lo, hi = report.cases[0].ci
    assert hi < 0                                               # CI fully below zero


def test_diff_carries_before_after_output():
    def mk(rid, out, val):
        return Run(id=rid, name="t", created_at=datetime(2026, 5, 27, tzinfo=timezone.utc),
                   model="m", results=[CaseResult(case_id="1", input="q", output=out,
                                                   scores=[Score(scorer="s", value=val,
                                                                 passed=val >= 0.5)])])
    report = diff_runs(mk("a", "four", 1.0), mk("b", "five", 0.0))
    c = report.regressed[0]
    assert c.before_output == "four"
    assert c.after_output == "five"


def test_bootstrap_diff_ci_method_percentile_matches_v04_default():
    """Passing method='percentile' explicitly must be byte-identical to v0.4 default."""
    from evalith.diff import bootstrap_diff_ci
    before = [1.0, 1.0, 0.0, 1.0, 0.0]
    after = [0.0, 1.0, 0.0, 0.0, 0.0]
    lo_default, hi_default = bootstrap_diff_ci(before, after, seed=42)
    lo_explicit, hi_explicit = bootstrap_diff_ci(before, after, method="percentile", seed=42)
    assert lo_default == lo_explicit
    assert hi_default == hi_explicit


def test_bootstrap_bca_matches_scipy_within_tolerance():
    """BCa CI bounds should be close (~0.10) to scipy.stats.bootstrap with method='BCa'.
    We don't expect bit-equivalence — scipy uses numpy RNG, we use stdlib. But the
    BCa-corrected bounds should be in the same ballpark on the same data."""
    import numpy as np
    from scipy.stats import bootstrap as scipy_bootstrap
    from evalith.diff import bootstrap_diff_ci

    # Article 2's redis-cluster-failover noisy case
    before = [0.0, 1.0, 1.0, 1.0, 1.0]
    after  = [0.0, 0.0, 0.0, 0.0, 0.0]

    # Our implementation
    lo_ours, hi_ours = bootstrap_diff_ci(before, after, method="bca",
                                          n_resamples=2000, seed=0)

    # scipy ground truth — paired=False (independent samples)
    def diff_means(b, a, axis=-1):
        return np.mean(a, axis=axis) - np.mean(b, axis=axis)

    rng = np.random.default_rng(0)
    scipy_result = scipy_bootstrap(
        (np.asarray(before), np.asarray(after)),
        diff_means,
        method="BCa",
        n_resamples=2000,
        paired=False,
        random_state=rng,
    )
    lo_scipy = scipy_result.confidence_interval.low
    hi_scipy = scipy_result.confidence_interval.high

    # CI bounds should match within ~0.10 (different RNG streams contribute most of the gap)
    assert abs(lo_ours - lo_scipy) < 0.10, f"BCa lo mismatch: ours={lo_ours} scipy={lo_scipy}"
    assert abs(hi_ours - hi_scipy) < 0.10, f"BCa hi mismatch: ours={hi_ours} scipy={hi_scipy}"


def test_bootstrap_bca_is_deterministic():
    """Same input + same seed → same CI."""
    from evalith.diff import bootstrap_diff_ci
    before = [1.0, 1.0, 0.0, 1.0, 0.0]
    after = [0.0, 0.0, 1.0, 0.0, 0.0]
    a = bootstrap_diff_ci(before, after, method="bca", seed=7)
    b = bootstrap_diff_ci(before, after, method="bca", seed=7)
    assert a == b


def test_bootstrap_paired_reduces_variance_vs_unpaired_on_correlated_data():
    """When before/after pass rates are strongly correlated (same case across runs),
    paired bootstrap CI should be narrower than unpaired."""
    from evalith.diff import bootstrap_diff_ci

    # Correlated data: cases with hardness — case 0 always near 1, case 4 always near 0
    before = [1.0, 0.8, 0.6, 0.4, 0.2]
    after  = [0.9, 0.7, 0.5, 0.3, 0.1]  # uniform 0.1 shift, same per-case rank

    lo_p, hi_p = bootstrap_diff_ci(before, after, method="paired", n_resamples=2000, seed=0)
    lo_u, hi_u = bootstrap_diff_ci(before, after, method="percentile", n_resamples=2000, seed=0)

    width_paired = hi_p - lo_p
    width_unpaired = hi_u - lo_u
    assert width_paired < width_unpaired, \
        f"paired should be narrower; paired width={width_paired:.3f} unpaired={width_unpaired:.3f}"


def test_bootstrap_paired_requires_equal_length():
    from evalith.diff import bootstrap_diff_ci
    import pytest as pt
    with pt.raises(ValueError):
        bootstrap_diff_ci([1.0, 0.0], [1.0, 0.0, 1.0], method="paired")


def test_bootstrap_paired_is_deterministic():
    from evalith.diff import bootstrap_diff_ci
    a = bootstrap_diff_ci([1.0, 0.0, 1.0], [0.0, 1.0, 0.0], method="paired", seed=3)
    b = bootstrap_diff_ci([1.0, 0.0, 1.0], [0.0, 1.0, 0.0], method="paired", seed=3)
    assert a == b


def test_diff_runs_bh_correction_reduces_regressed_count_when_many_cases():
    """Benjamini-Hochberg correction must be at least as conservative as no correction.
    On article 2's frozen raw, no-correction reports 2/10 regressed; with BH, count
    should be <= 2 (BH is monotone — can never add a regression)."""
    from pathlib import Path
    from evalith.diff import diff_runs
    from evalith.models import Run

    a1 = Run.model_validate_json(Path("docs/blog/article2/raw/a1.json").read_text())
    b  = Run.model_validate_json(Path("docs/blog/article2/raw/b.json").read_text())

    report_no = diff_runs(a1, b)
    report_bh = diff_runs(a1, b, multi_test_correction="bh")

    flagged_no = {c.case_id for c in report_no.cases if c.status == "regressed"}
    flagged_bh = {c.case_id for c in report_bh.cases if c.status == "regressed"}

    assert flagged_bh.issubset(flagged_no), \
        f"BH should only shrink the regressed set, never grow it. No-corr={flagged_no} BH={flagged_bh}"


def test_diff_runs_bh_rejects_unknown_method():
    from evalith.diff import diff_runs
    from evalith.models import Run
    import pytest as pt
    from pathlib import Path
    a1 = Run.model_validate_json(Path("docs/blog/article2/raw/a1.json").read_text())
    b  = Run.model_validate_json(Path("docs/blog/article2/raw/b.json").read_text())
    with pt.raises(ValueError):
        diff_runs(a1, b, multi_test_correction="bonferroni")
