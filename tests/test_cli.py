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
