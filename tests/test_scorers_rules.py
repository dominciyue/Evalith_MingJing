from evalith.config import ScorerConfig
from evalith.models import TestCase
from evalith.scorers.rules import Contains, ExactMatch, Regex, build_scorer


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


def test_build_llm_judge_with_judge_model_and_panel():
    cfg = ScorerConfig(type="llm_judge", params={
        "criteria": "quality",
        "judge_model": "echo:primary-judge",
        "panel": ["echo:panel-a", "echo:panel-b"],
    })
    judge = build_scorer(cfg, judge_provider=None)
    assert judge.provider.fixed == "primary-judge"        # judge_model resolved
    assert set(judge.panel) == {"echo:panel-a", "echo:panel-b"}


def test_build_llm_judge_panel_dedupes_primary():
    cfg = ScorerConfig(type="llm_judge", params={
        "judge_model": "echo:j",
        "panel": ["echo:j", "echo:other", "echo:other"],
    })
    judge = build_scorer(cfg, judge_provider=None)
    assert list(judge.panel) == ["echo:other"]            # primary + dup removed


def test_build_llm_judge_defaults_unchanged():
    from evalith.providers.base import FakeProvider
    fp = FakeProvider()
    cfg = ScorerConfig(type="llm_judge", params={"criteria": "q"})
    judge = build_scorer(cfg, judge_provider=fp)
    assert judge.provider is fp                           # falls back, v0.6 behavior
    assert judge.panel == {}
