from evalith.models import TestCase
from evalith.providers.base import FakeProvider
from evalith.scorers.llm_judge import LLMJudge


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


def test_llm_judge_chinese_prompt_and_parse():
    from evalith.scorers.llm_judge import JUDGE_PROMPTS
    assert "zh" in JUDGE_PROMPTS and "请" in JUDGE_PROMPTS["zh"]
    judge = LLMJudge(
        provider=FakeProvider(default='{"score": 1.0, "pass": true, "reason": "对"}'),
        criteria="准确性", language="zh",
    )
    score = judge.score(TestCase(id="1", input="x"), "y")
    assert score.passed is True and score.value == 1.0 and score.detail == "对"


def test_llm_judge_defaults_to_english():
    assert LLMJudge(provider=None).language == "en"
