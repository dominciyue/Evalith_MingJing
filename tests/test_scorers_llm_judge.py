from mingjing.models import TestCase
from mingjing.providers.base import FakeProvider
from mingjing.scorers.llm_judge import LLMJudge


def test_llm_judge_parses_json():
    judge = LLMJudge(
        provider=FakeProvider(default='{"score": 0.8, "pass": true, "reason": "good"}'),
        criteria="quality",
    )
    score = judge.score(TestCase(id="1", input="q"), "answer")
    assert score.value == 0.8
    assert score.passed is True
    assert "good" in score.detail


def test_llm_judge_extracts_json_from_noise():
    judge = LLMJudge(
        provider=FakeProvider(default='Sure!\n{"score": 1.0, "pass": true, "reason": "ok"}\nThanks'),
    )
    score = judge.score(TestCase(id="1", input="q"), "answer")
    assert score.value == 1.0


def test_llm_judge_handles_garbage():
    judge = LLMJudge(provider=FakeProvider(default="not json at all"))
    score = judge.score(TestCase(id="1", input="q"), "answer")
    assert score.passed is False
    assert score.value == 0.0
