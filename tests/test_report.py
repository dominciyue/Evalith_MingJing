from datetime import datetime, timezone

from evalith.models import CaseResult, Run, Score


def _run():
    return Run(id="abc123", name="demo", created_at=datetime(2026, 5, 26, tzinfo=timezone.utc),
               model="deepseek/deepseek-chat",
               results=[CaseResult(case_id="1", input="2+2?", output="4", latency_ms=120.0,
                                   total_tokens=8, cost_usd=0.0001,
                                   scores=[Score(scorer="contains", value=1.0, passed=True)])])


def test_run_to_markdown_has_summary_and_row():
    from evalith.report import run_to_markdown
    md = run_to_markdown(_run())
    assert "# Run abc123" in md
    assert "deepseek/deepseek-chat" in md
    assert "100" in md.replace(",", "")  # pass rate 100%
    assert "| 1 |" in md                 # the case row


def test_cli_report_markdown(tmp_path):
    from typer.testing import CliRunner

    from evalith.cli import app
    from evalith.store import RunStore
    runner = CliRunner()
    store = str(tmp_path / "d")
    RunStore(store).save(_run())
    res = runner.invoke(app, ["report", "abc123", "--store", store])
    assert res.exit_code == 0
    assert "# Run abc123" in res.stdout


def test_diff_to_markdown():
    from evalith.diff import diff_runs
    from evalith.report import diff_to_markdown
    a = _run()
    b = Run(id="def456", name="demo", created_at=a.created_at, model=a.model,
            results=[CaseResult(case_id="1", input="2+2?", output="5",
                                scores=[Score(scorer="contains", value=0.0, passed=False)])])
    md = diff_to_markdown(diff_runs(a, b), "abc123", "def456")
    assert "abc123" in md and "def456" in md
    assert "regressed" in md
    assert "| 1 |" in md


def test_markdown_escapes_pipes_in_output():
    from evalith.report import run_to_markdown
    run = Run(id="z", name="n", created_at=datetime(2026, 5, 26, tzinfo=timezone.utc), model="m",
              results=[CaseResult(case_id="a|b", input="i", output="yes | no",
                                  scores=[Score(scorer="contains", value=1.0, passed=True)])])
    md = run_to_markdown(run)
    assert "yes \\| no" in md   # pipes in output escaped so the table doesn't break
    assert "a\\|b" in md        # pipes in case_id escaped


def test_cli_report_rejects_bad_format(tmp_path):
    from typer.testing import CliRunner

    from evalith.cli import app
    from evalith.store import RunStore
    runner = CliRunner()
    store = str(tmp_path / "d")
    RunStore(store).save(_run())
    res = runner.invoke(app, ["report", "abc123", "--store", store, "--format", "pdf"])
    assert res.exit_code != 0


def test_run_to_html_is_self_contained():
    from evalith.report import run_to_html
    html = run_to_html(_run())
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert "<table" in html
    assert "abc123" in html
    assert "<style" in html  # inline styling, no external deps


def test_cli_report_html_to_file(tmp_path):
    from typer.testing import CliRunner

    from evalith.cli import app
    from evalith.store import RunStore
    runner = CliRunner()
    store = str(tmp_path / "d")
    RunStore(store).save(_run())
    out = tmp_path / "r.html"
    res = runner.invoke(app, ["report", "abc123", "--store", store,
                              "--format", "html", "--output", str(out)])
    assert res.exit_code == 0
    assert out.exists()
    assert "<table" in out.read_text(encoding="utf-8")
