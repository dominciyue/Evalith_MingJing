from mingjing.config import ScorerConfig
from mingjing.models import TestCase
from mingjing.scorers.rules import Contains, ExactMatch, Regex, build_scorer


def test_exact_match():
    c = TestCase(id="1", input="q", expected="hello")
    assert ExactMatch().score(c, "hello").passed is True
    assert ExactMatch().score(c, "world").passed is False


def test_contains_uses_expected_when_no_text():
    c = TestCase(id="1", input="q", expected="lo")
    assert Contains().score(c, "hello").passed is True


def test_contains_with_text_param():
    c = TestCase(id="1", input="q")
    assert Contains(text="ell").score(c, "hello").passed is True


def test_regex():
    c = TestCase(id="1", input="q")
    assert Regex(pattern=r"\d+").score(c, "abc123").passed is True
    assert Regex(pattern=r"\d+").score(c, "abc").passed is False


def test_build_scorer():
    s = build_scorer(ScorerConfig(type="contains", params={"text": "x"}))
    assert isinstance(s, Contains)
    assert s.text == "x"


def test_build_scorer_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        build_scorer(ScorerConfig(type="nope"))
