from typer.testing import CliRunner

from evalith.cli import app
from evalith.store import RunStore

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


def test_run_fail_under_fails_when_no_checks(tmp_path):
    # a config with no scorers => 0 checks => must NOT be a false green under a gate
    ds = tmp_path / "ds.yaml"
    ds.write_text("name: d\ncases:\n  - id: '1'\n    input: hello\n", encoding="utf-8")
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(f"name: t\ndataset: {ds}\nmodel: echo\n"
                   f"prompt_template: '{{{{input}}}}'\nscorers: []\n", encoding="utf-8")
    res = runner.invoke(app, ["run", str(cfg), "--store", str(tmp_path / "d"),
                              "--fail-under", "0.9"])
    assert res.exit_code == 1


def test_diff_fail_on_regression(tmp_path):
    from datetime import datetime, timezone

    from evalith.models import CaseResult, Run, Score

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


def test_run_out_writes_run_file(tmp_path):
    from evalith.models import Run
    cfg = _cfg(tmp_path, "hello", "hello")
    out = tmp_path / "run.json"
    res = runner.invoke(app, ["run", str(cfg), "--store", str(tmp_path / "d"), "--out", str(out)])
    assert res.exit_code == 0
    assert out.exists()
    run = Run.model_validate_json(out.read_text(encoding="utf-8"))
    assert run.results and run.results[0].case_id == "1"


def test_diff_accepts_file_paths(tmp_path):
    # CI baseline pattern: diff a committed baseline file against a fresh run file, no store needed
    from datetime import datetime, timezone

    from evalith.models import CaseResult, Run, Score

    def mk(rid, val):
        return Run(id=rid, name="t", created_at=datetime(2026, 5, 27, tzinfo=timezone.utc),
                   model="echo",
                   results=[CaseResult(case_id="1", input="i", output="o",
                                       scores=[Score(scorer="s", value=val, passed=val >= 0.5)])])
    base = tmp_path / "base.json"
    base.write_text(mk("base", 1.0).model_dump_json(), encoding="utf-8")
    new = tmp_path / "new.json"
    new.write_text(mk("new", 0.0).model_dump_json(), encoding="utf-8")
    res = runner.invoke(app, ["diff", str(base), str(new), "--fail-on-regression"])
    assert res.exit_code == 1
    assert "regressed" in res.stdout.lower()   # gated for the right reason, not a load crash
