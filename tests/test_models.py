from datetime import datetime, timezone

from evalith.models import CaseResult, Run, Score, TestCase


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


def test_case_result_panel_fields_default_empty():
    cr = CaseResult(case_id="c", input="i", output="o")
    assert cr.panel_samples == {}
    assert cr.panel_details == {}
    assert cr.judge_tokens == 0
    assert cr.judge_cost_usd == 0.0
    assert cr.domain is None


def test_test_case_accepts_domain():
    tc = TestCase(id="1", input="q", domain="code")
    assert tc.domain == "code"


def test_old_run_json_loads_without_panel_fields():
    """v0.6 的 run JSON(无 panel 字段)必须能原样加载。"""
    old = {
        "id": "abc", "name": "n", "created_at": "2026-06-01T00:00:00Z",
        "model": "m",
        "results": [{"case_id": "c", "input": "i", "output": "o"}],
    }
    run = Run.model_validate(old)
    assert run.results[0].panel_samples == {}
    assert run.total_judge_cost_usd == 0.0
    assert run.total_judge_tokens == 0


def test_run_judge_cost_aggregates():
    run = Run(id="r", name="n", created_at=datetime(2026, 6, 6, tzinfo=timezone.utc),
              model="m", results=[
                  CaseResult(case_id="a", input="i", output="o",
                             judge_tokens=10, judge_cost_usd=0.01),
                  CaseResult(case_id="b", input="i", output="o",
                             judge_tokens=5, judge_cost_usd=0.02),
              ])
    assert run.total_judge_tokens == 15
    assert abs(run.total_judge_cost_usd - 0.03) < 1e-9
