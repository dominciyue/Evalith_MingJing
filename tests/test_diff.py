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
