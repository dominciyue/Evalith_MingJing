from datetime import datetime, timezone

from mingjing.models import CaseResult, Run, Score


def _run():
    return Run(id="abc123", name="demo", created_at=datetime(2026, 5, 26, tzinfo=timezone.utc),
               model="deepseek/deepseek-chat",
               results=[CaseResult(case_id="1", input="2+2?", output="4", latency_ms=120.0,
                                   total_tokens=8, cost_usd=0.0001,
                                   scores=[Score(scorer="contains", value=1.0, passed=True)])])


def test_run_to_markdown_has_summary_and_row():
    from mingjing.report import run_to_markdown
    md = run_to_markdown(_run())
    assert "# Run abc123" in md
    assert "deepseek/deepseek-chat" in md
    assert "100" in md.replace(",", "")  # pass rate 100%
    assert "| 1 |" in md                 # the case row


def test_cli_report_markdown(tmp_path):
    from typer.testing import CliRunner

    from mingjing.cli import app
    from mingjing.store import RunStore
    runner = CliRunner()
    store = str(tmp_path / "d")
    RunStore(store).save(_run())
    res = runner.invoke(app, ["report", "abc123", "--store", store])
    assert res.exit_code == 0
    assert "# Run abc123" in res.stdout


def test_diff_to_markdown():
    from mingjing.diff import diff_runs
    from mingjing.report import diff_to_markdown
    a = _run()
    b = Run(id="def456", name="demo", created_at=a.created_at, model=a.model,
            results=[CaseResult(case_id="1", input="2+2?", output="5",
                                scores=[Score(scorer="contains", value=0.0, passed=False)])])
    md = diff_to_markdown(diff_runs(a, b), "abc123", "def456")
    assert "abc123" in md and "def456" in md
    assert "regressed" in md
    assert "| 1 |" in md
