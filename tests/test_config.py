from evalith.config import load_config


def test_load_config(tmp_path):
    p = tmp_path / "eval.yaml"
    p.write_text(
        "name: demo\n"
        "dataset: ds.yaml\n"
        "model: echo\n"
        "prompt_template: 'Q: {{input}}'\n"
        "scorers:\n"
        "  - type: contains\n"
        "    params: {text: hi}\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.name == "demo"
    assert cfg.model == "echo"
    assert cfg.prompt_template == "Q: {{input}}"
    assert cfg.scorers[0].type == "contains"
    assert cfg.scorers[0].params["text"] == "hi"
