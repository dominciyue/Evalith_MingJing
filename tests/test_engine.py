from mingjing.config import EvalConfig, ScorerConfig
from mingjing.engine import run_eval
from mingjing.providers.base import FakeProvider


def test_run_eval_with_fake_provider(tmp_path):
    ds = tmp_path / "ds.yaml"
    ds.write_text(
        "name: d\ncases:\n  - id: '1'\n    input: hello\n    expected: HELLO\n",
        encoding="utf-8",
    )
    cfg = EvalConfig(
        name="t", dataset=str(ds), model="fake", prompt_template="{{input}}",
        scorers=[ScorerConfig(type="contains", params={"text": "HELLO"})],
    )
    provider = FakeProvider(responses={"hello": "HELLO world"})
    run = run_eval(cfg, provider)
    assert len(run.results) == 1
    assert run.results[0].output == "HELLO world"
    assert run.results[0].scores[0].passed is True
    assert run.model == "fake"
    assert run.config["name"] == "t"


def test_run_eval_renders_template(tmp_path):
    ds = tmp_path / "ds.yaml"
    ds.write_text("name: d\ncases:\n  - id: '1'\n    input: world\n", encoding="utf-8")
    cfg = EvalConfig(name="t", dataset=str(ds), model="echo",
                     prompt_template="Hello {{input}}")
    from mingjing.providers.base import EchoProvider

    run = run_eval(cfg, EchoProvider())
    assert run.results[0].output == "Hello world"
