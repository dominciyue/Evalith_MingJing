from typer.testing import CliRunner

from evalith.cli import app
from evalith.store import RunStore

runner = CliRunner()


def _write_cfg(tmp_path):
    ds = tmp_path / "ds.yaml"
    ds.write_text(
        "name: d\ncases:\n  - id: '1'\n    input: hello\n    expected: hello\n",
        encoding="utf-8",
    )
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(
        f"name: t\ndataset: {ds}\nmodel: echo\n"
        f"prompt_template: '{{{{input}}}}'\n"
        f"scorers:\n  - type: contains\n    params: {{text: hello}}\n",
        encoding="utf-8",
    )
    return cfg


def test_cli_run_then_list_and_diff(tmp_path):
    cfg = _write_cfg(tmp_path)
    store = str(tmp_path / "data")

    assert runner.invoke(app, ["run", str(cfg), "--store", store]).exit_code == 0
    assert runner.invoke(app, ["run", str(cfg), "--store", store]).exit_code == 0

    listed = runner.invoke(app, ["list", "--store", store])
    assert listed.exit_code == 0

    ids = [r.id for r in RunStore(store).list_runs()]
    assert len(ids) == 2
    diffed = runner.invoke(app, ["diff", ids[1], ids[0], "--store", store])
    assert diffed.exit_code == 0


def test_run_prints_panel_summary(tmp_path):
    PASS_JSON = '{"score": 1.0, "pass": true, "reason": "ok"}'
    FAIL_JSON = '{"score": 0.0, "pass": false, "reason": "no"}'
    ds = tmp_path / "ds.yaml"
    ds.write_text("name: d\ncases:\n  - id: c1\n    input: hi\n", encoding="utf-8")
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(f"""
name: panel-t
dataset: {ds}
model: "echo:answer"
samples: 2
scorers:
  - type: llm_judge
    params:
      judge_model: 'echo:{PASS_JSON}'
      panel: ['echo:{FAIL_JSON}']
""", encoding="utf-8")
    result = runner.invoke(app, ["run", str(cfg), "--store", str(tmp_path / "s")])
    assert result.exit_code == 0
    assert "panel: 2 judges" in result.output
    assert "1/1 low-consensus cases" in result.output
    assert "min pairwise" in result.output


def test_run_no_panel_no_summary_line(tmp_path):
    ds = tmp_path / "ds.yaml"
    ds.write_text("name: d\ncases:\n  - id: c1\n    input: hi\n", encoding="utf-8")
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(f"""
name: t
dataset: {ds}
model: "echo:hi"
scorers:
  - type: contains
    params: {{text: hi}}
""", encoding="utf-8")
    result = runner.invoke(app, ["run", str(cfg), "--store", str(tmp_path / "s")])
    assert result.exit_code == 0
    assert "panel:" not in result.output
