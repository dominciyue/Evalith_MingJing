from __future__ import annotations

import json

from ..models import Score, TestCase

JUDGE_PROMPTS = {
    "en": """You are grading an AI answer.

Question/Input:
{input}

AI Answer:
{output}

Criteria: {criteria}

Respond with ONLY a JSON object:
{{"score": <float 0..1>, "pass": <true|false>, "reason": "<short reason>"}}
""",
    "zh": """你是一名严格的 AI 答案评审。

问题/输入:
{input}

AI 的回答:
{output}

评判标准: {criteria}

请只输出一个 JSON 对象,不要任何其它文字:
{{"score": <0到1之间的小数>, "pass": <true 或 false>, "reason": "<简短理由>"}}
""",
}


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in judge output")
    return text[start : end + 1]


class LLMJudge:
    name = "llm_judge"

    def __init__(self, provider, criteria: str = "", language: str = "en"):
        self.provider = provider
        self.criteria = criteria
        self.language = language if language in JUDGE_PROMPTS else "en"

    def score(self, case: TestCase, output: str) -> Score:
        prompt = JUDGE_PROMPTS[self.language].format(
            input=case.input, output=output, criteria=self.criteria or "overall quality"
        )
        resp = self.provider.complete(prompt, temperature=0.0)
        try:
            data = json.loads(_extract_json(resp.text))
            value = float(data.get("score", 0.0))
            passed = bool(data.get("pass", value >= 0.5))
            reason = str(data.get("reason", ""))
        except Exception as e:  # noqa: BLE001 - any parse failure means judge failed
            return Score(scorer=self.name, value=0.0, passed=False,
                         detail=f"judge parse error: {e}")
        return Score(scorer=self.name, value=value, passed=passed, detail=reason)
