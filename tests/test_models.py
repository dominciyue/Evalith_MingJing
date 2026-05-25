from datetime import datetime, timezone

from mingjing.models import CaseResult, Run, Score


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
