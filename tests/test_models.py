from datetime import datetime, timezone

from evalith.models import CaseResult, Run, Score


def test_case_result_mean_score():
    cr = CaseResult(
        case_id="1", input="q", output="a",
        scores=[Score(scorer="x", value=1.0, passed=True),
                Score(scorer="y", value=0.0, passed=False)],
    )
    assert cr.mean_score == 0.5


def test_case_result_mean_score_empty_is_zero():
    assert CaseResult(case_id="1", input="q", output="a").mean_score == 0.0


def test_run_round_trip():
    run = Run(
        id="abc", name="t", created_at=datetime(2026, 5, 25, tzinfo=timezone.utc),
        model="echo", results=[CaseResult(case_id="1", input="q", output="a")],
    )
    back = Run.model_validate_json(run.model_dump_json())
    assert back.id == "abc"
    assert back.results[0].case_id == "1"


def test_run_pass_rate():
    r = Run(
        id="x", name="t", created_at=datetime(2026, 5, 26, tzinfo=timezone.utc), model="echo",
        results=[
            CaseResult(case_id="1", input="i", output="o",
                       scores=[Score(scorer="s", value=1.0, passed=True),
                               Score(scorer="s", value=0.0, passed=False)]),
            CaseResult(case_id="2", input="i", output="o",
                       scores=[Score(scorer="s", value=1.0, passed=True)]),
        ],
    )
    assert r.pass_rate == 2 / 3            # 2 of 3 checks passed
    assert Run(id="y", name="t", created_at=datetime(2026, 5, 26, tzinfo=timezone.utc),
               model="echo", results=[]).pass_rate == 1.0   # no checks => nothing failed


def test_run_cost_and_latency_aggregates():
    r = Run(
        id="x", name="t", created_at=datetime.now(timezone.utc), model="m",
        results=[
            CaseResult(case_id="1", input="i", output="o", latency_ms=100.0,
                       total_tokens=10, cost_usd=0.001),
            CaseResult(case_id="2", input="i", output="o", latency_ms=300.0,
                       total_tokens=20, cost_usd=0.002),
        ],
    )
    assert r.total_tokens == 30
    assert abs(r.total_cost_usd - 0.003) < 1e-9
    assert r.mean_latency_ms == 200.0
