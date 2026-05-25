from __future__ import annotations

import json

from ..models import Score, TestCase

JUDGE_PROMPT = """You are grading an AI answer.

Question/Input:
{input}

AI Answer:
{output}

Criteria: {criteria}

Respond with ONLY a JSON object:
{{"score": <float 0..1>, "pass": <true|false>, "reason": "<short reason>"}}
"""


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in judge output")
    return text[start : end + 1]


class LLMJudge:
    name = "llm_judge"

    def __init__(self, provider, criteria: str = ""):
        self.provider = provider
        self.criteria = criteria

    def score(self, case: TestCase, output: str) -> Score:
        prompt = JUDGE_PROMPT.format(
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
