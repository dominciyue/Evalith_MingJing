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


def _diff_pair(before_out, after_out):
    from evalith.diff import diff_runs

    def mk(rid, out, val):
        return Run(id=rid, name="d", created_at=datetime(2026, 5, 27, tzinfo=timezone.utc),
                   model="m", results=[CaseResult(case_id="c1", input="q", output=out,
                                                  scores=[Score(scorer="s", value=val,
                                                                passed=val >= 0.5)])])
    return diff_runs(mk("a", before_out, 1.0), mk("b", after_out, 0.0))


def test_diff_markdown_shows_regressed_outputs():
    from evalith.report import diff_to_markdown
    md = diff_to_markdown(_diff_pair("GOODOUT", "BADOUT"), "a", "b")
    assert "GOODOUT" in md and "BADOUT" in md   # before & after outputs shown for the regression


def test_diff_html_shows_regressed_outputs():
    from evalith.report import diff_to_html
    html = diff_to_html(_diff_pair("GOODOUT", "BADOUT"), "a", "b")
    assert "GOODOUT" in html and "BADOUT" in html


def _sampled_diff(before_samples, after_samples):
    """Build a diff between two runs whose case carries explicit pass_rate_samples (triggers bootstrap)."""
    from evalith.diff import diff_runs

    def mk(rid, samples):
        mean = sum(samples) / len(samples)
        return Run(id=rid, name="d", created_at=datetime(2026, 5, 28, tzinfo=timezone.utc),
                   model="m", results=[CaseResult(
                       case_id="c1", input="q", output="o",
                       scores=[Score(scorer="s", value=mean, passed=mean >= 0.5)],
                       pass_rate_samples=samples)])
    return diff_runs(mk("a", before_samples), mk("b", after_samples))


def test_diff_markdown_shows_ci_column_when_bootstrapped():
    from evalith.report import diff_to_markdown
    report = _sampled_diff([1.0] * 6, [0.0] * 6)   # clean regression — CI fully below 0
    md = diff_to_markdown(report, "a", "b")
    assert "Δ 95% CI" in md                         # new column header surfaces the CI
    # CI numbers themselves appear (Δ should be near -1.0)
    assert "-1.00" in md or "-0.99" in md or "-1.0" in md


def test_diff_html_shows_ci_column_when_bootstrapped():
    from evalith.report import diff_to_html
    report = _sampled_diff([1.0] * 6, [0.0] * 6)
    html = diff_to_html(report, "a", "b")
    assert "95% CI" in html


def test_diff_markdown_no_ci_column_for_single_sample_runs():
    """Backward-compat: legacy single-sample diffs must NOT add a CI column."""
    from evalith.report import diff_to_markdown
    md = diff_to_markdown(_diff_pair("x", "y"), "a", "b")
    assert "Δ 95% CI" not in md                     # quiet for samples=1 runs


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


def _panel_run():
    s = Score(scorer="llm_judge", value=1.0, passed=True, detail="ok")
    return Run(id="rid", name="panel-run",
               created_at=datetime(2026, 6, 6, tzinfo=timezone.utc), model="m",
               config={"scorers": [{"type": "llm_judge",
                                    "params": {"consensus_threshold": 0.5}}]},
               results=[
                   CaseResult(case_id="agree", input="i", output="o", scores=[s],
                              pass_rate_samples=[1.0, 1.0], domain="safety",
                              panel_samples={"qw": [1.0, 1.0]},
                              judge_tokens=10, judge_cost_usd=0.001),
                   CaseResult(case_id="fight", input="i", output="o", scores=[s],
                              pass_rate_samples=[1.0, 1.0], domain="code",
                              panel_samples={"qw": [0.0, 0.0]},
                              panel_details={"qw": "code does not compile"},
                              judge_tokens=10, judge_cost_usd=0.001),
               ])


def test_markdown_has_consensus_section():
    from evalith.report import run_to_markdown
    md = run_to_markdown(_panel_run())
    assert "## Judge Consensus" in md
    assert "fight" in md                       # low-consensus case listed
    assert "code does not compile" in md       # judge reason shown
    assert "⚠" in md
    assert "$0.0020" in md                     # judge cost split out


def test_markdown_no_consensus_section_without_panel():
    from evalith.report import run_to_markdown
    run = _panel_run()
    for r in run.results:
        r.panel_samples = {}
        r.panel_details = {}
    md = run_to_markdown(run)
    assert "Judge Consensus" not in md


def test_html_has_consensus_section():
    from evalith.report import run_to_html
    html = run_to_html(_panel_run())
    assert "Judge Consensus" in html
    assert "code does not compile" in html
