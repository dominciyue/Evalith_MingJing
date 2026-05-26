from mingjing.providers import get_provider
from mingjing.providers.base import EchoProvider, FakeProvider


def test_echo_provider_echoes_prompt():
    assert EchoProvider().complete("hello").text == "hello"


def test_echo_provider_fixed_text():
    assert EchoProvider(fixed="ok").complete("anything").text == "ok"


def test_fake_provider_canned_and_default():
    p = FakeProvider(responses={"q": "a"}, default="d")
    assert p.complete("q").text == "a"
    assert p.complete("z").text == "d"


def test_get_provider_echo():
    assert isinstance(get_provider("echo"), EchoProvider)
    assert get_provider("echo:ok").complete("x").text == "ok"


def test_usage_from_response_pure():
    from mingjing.providers.litellm_provider import _usage_from_response
    fake = {"usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}}
    assert _usage_from_response(fake) == (5, 7, 12)
    assert _usage_from_response({}) == (0, 0, 0)   # missing usage -> zeros
