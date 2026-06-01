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


# ---------------------------------------------------------------------------
# v0.6 — per-case expected_concepts
# ---------------------------------------------------------------------------

class _CapturingProvider:
    """Minimal provider that records the last prompt seen."""
    def __init__(self, response_text: str = '{"score": 1.0, "pass": true, "reason": "ok"}'):
        self.last_prompt: str | None = None
        self._response_text = response_text

    def complete(self, prompt: str, **kwargs):
        self.last_prompt = prompt
        from evalith.providers.base import Response
        return Response(text=self._response_text)


def test_llm_judge_uses_expected_concepts_when_present():
    """When the case has expected_concepts, the judge prompt should include them as a checklist."""
    fp = _CapturingProvider()
    judge = LLMJudge(provider=fp, criteria="评判标准", language="zh")
    case = TestCase(id="x", input="解释 TCP 三次握手",
                    expected_concepts=["SYN", "SYN-ACK", "ACK", "建立连接"])
    judge.score(case, output="模型回答了...")
    assert fp.last_prompt is not None
    assert "核心概念清单" in fp.last_prompt or "expected concepts" in fp.last_prompt.lower()
    assert "SYN-ACK" in fp.last_prompt
    assert "建立连接" in fp.last_prompt


def test_llm_judge_backward_compat_no_concepts():
    """If case has no expected_concepts, judge prompt is identical to v0.5 (static criteria only)."""
    fp = _CapturingProvider()
    judge = LLMJudge(provider=fp, criteria="评判标准", language="zh")
    case = TestCase(id="y", input="解释什么", expected_concepts=None)
    judge.score(case, output="...")
    assert fp.last_prompt is not None
    # Backward compatible: no concept block, criteria string preserved
    assert "核心概念清单" not in fp.last_prompt
    assert "评判标准" in fp.last_prompt


def test_dataset_loads_expected_concepts_field(tmp_path):
    """Dataset loader should read expected_concepts list[str] from YAML when present."""
    import yaml
    from evalith.dataset import load_dataset

    data = {
        "name": "tmp",
        "cases": [
            {"id": "a", "input": "q1", "expected_concepts": ["c1", "c2", "c3"]},
            {"id": "b", "input": "q2"},  # no concepts field — should still load
        ],
    }
    p = tmp_path / "ds.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    ds = load_dataset(p)
    assert len(ds.cases) == 2
    assert ds.cases[0].expected_concepts == ["c1", "c2", "c3"]
    assert ds.cases[1].expected_concepts is None or ds.cases[1].expected_concepts == []
