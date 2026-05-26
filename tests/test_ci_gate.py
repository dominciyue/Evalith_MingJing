from typer.testing import CliRunner

from mingjing.cli import app
from mingjing.store import RunStore

runner = CliRunner()


def _cfg(tmp_path, expected, scorer_text):
    ds = tmp_path / "ds.yaml"
    ds.write_text(f"name: d\ncases:\n  - id: '1'\n    input: hello\n    expected: {expected}\n",
                  encoding="utf-8")
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(
        f"name: t\ndataset: {ds}\nmodel: echo\nprompt_template: '{{{{input}}}}'\n"
        f"scorers:\n  - type: contains\n    params: {{text: {scorer_text}}}\n",
        encoding="utf-8")
    return cfg


def test_run_fail_under_fails_when_below(tmp_path):
    # echo returns the prompt 'hello'; needle 'zzz' never matches -> pass_rate 0.0
    cfg = _cfg(tmp_path, "hello", "zzz")
    res = runner.invoke(app, ["run", str(cfg), "--store", str(tmp_path / "d"),
                              "--fail-under", "0.9"])
    assert res.exit_code == 1


def test_run_fail_under_passes_when_met(tmp_path):
    cfg = _cfg(tmp_path, "hello", "hello")
    res = runner.invoke(app, ["run", str(cfg), "--store", str(tmp_path / "d"),
                              "--fail-under", "0.9"])
    assert res.exit_code == 0


def test_diff_fail_on_regression(tmp_path):
    from datetime import datetime, timezone

    from mingjing.models import CaseResult, Run, Score

    s = str(tmp_path / "d")
    store = RunStore(s)

    def mk(rid, val):
        return Run(id=rid, name="t", created_at=datetime(2026, 5, 26, tzinfo=timezone.utc),
                   model="echo",
                   results=[CaseResult(case_id="1", input="i", output="o",
                                       scores=[Score(scorer="s", value=val, passed=val >= 0.5)])])

    store.save(mk("base", 1.0))   # before: passing
    store.save(mk("bad", 0.0))    # after: case '1' regressed 1.0 -> 0.0

    ok = runner.invoke(app, ["diff", "base", "bad", "--store", s])
    assert ok.exit_code == 0  # no flag -> never fails
    gated = runner.invoke(app, ["diff", "base", "bad", "--store", s, "--fail-on-regression"])
    assert gated.exit_code == 1
