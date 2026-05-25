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
