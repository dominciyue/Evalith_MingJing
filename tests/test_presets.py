def test_resolve_known_alias_and_passthrough():
    from evalith.presets import CHINA_MODELS, resolve_model
    assert resolve_model("deepseek-chat") == "deepseek/deepseek-chat"
    assert resolve_model("deepseek/deepseek-chat") == "deepseek/deepseek-chat"  # already-qualified
    assert resolve_model("echo") == "echo"                                      # passthrough
    assert resolve_model("gpt-4o-mini") == "gpt-4o-mini"
    for name, info in CHINA_MODELS.items():
        assert "litellm" in info and "env" in info


def test_get_provider_resolves_alias():
    from evalith.providers import get_provider
    p = get_provider("deepseek-chat")          # alias -> litellm id, no network
    assert p.model == "deepseek/deepseek-chat"


def test_cli_models_lists_deepseek():
    from typer.testing import CliRunner

    from evalith.cli import app
    res = CliRunner().invoke(app, ["models"])
    assert res.exit_code == 0
    assert "deepseek-chat" in res.stdout
