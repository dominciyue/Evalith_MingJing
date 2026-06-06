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

    def __init__(self, provider, criteria: str = "", language: str = "en",
                 panel: dict | None = None, consensus_threshold: float = 0.5):
        self.provider = provider
        self.criteria = criteria
        self.language = language if language in JUDGE_PROMPTS else "en"
        self.panel = panel or {}                      # judge name -> Provider
        self.consensus_threshold = consensus_threshold

    def _build_prompt(self, case: TestCase, output: str) -> str:
        # Build effective criteria — append expected_concepts checklist if present
        criteria_eff = self.criteria or "overall quality"
        if case.expected_concepts:
            concept_lines = "\n".join(f"- {c}" for c in case.expected_concepts)
            if self.language == "zh":
                criteria_eff = (
                    f"{criteria_eff}\n\n核心概念清单(回答须覆盖):\n{concept_lines}"
                )
            else:
                criteria_eff = (
                    f"{criteria_eff}\n\nExpected concepts (response must cover):\n{concept_lines}"
                )
        return JUDGE_PROMPTS[self.language].format(
            input=case.input, output=output, criteria=criteria_eff
        )

    def _judge(self, provider, case: TestCase, output: str) -> tuple[Score, int, float]:
        """One judging call. Parse errors -> fail Score; provider errors propagate."""
        resp = provider.complete(self._build_prompt(case, output), temperature=0.0)
        try:
            data = json.loads(_extract_json(resp.text))
            value = float(data.get("score", 0.0))
            passed = bool(data.get("pass", value >= 0.5))
            score = Score(scorer=self.name, value=value, passed=passed,
                          detail=str(data.get("reason", "")))
        except Exception as e:  # noqa: BLE001 - any parse failure means judge failed
            score = Score(scorer=self.name, value=0.0, passed=False,
                          detail=f"judge parse error: {e}")
        return score, resp.total_tokens, resp.cost_usd

    def score(self, case: TestCase, output: str) -> Score:
        return self._judge(self.provider, case, output)[0]

    def score_with_usage(self, case: TestCase, output: str) -> tuple[Score, int, float]:
        """Primary judge scoring + (tokens, cost) of the judging call."""
        return self._judge(self.provider, case, output)

    def panel_score(self, case: TestCase, output: str,
                    cache: dict) -> tuple[dict[str, Score | None], int, float]:
        """Score `output` with every panel judge (diagnostics only).

        Provider failures record None (missing); parse failures record a fail
        Score, same as the primary judge. `cache` maps (judge, output) -> Score
        so identical trial outputs within a case are judged once per judge.
        """
        results: dict[str, Score | None] = {}
        tokens, cost = 0, 0.0
        for name, prov in self.panel.items():
            key = (name, output)
            if key in cache:
                results[name] = cache[key]
                continue
            try:
                s, tok, c = self._judge(prov, case, output)
            except Exception:  # noqa: BLE001 - one judge down must not kill the case
                results[name] = None
                continue
            tokens += tok
            cost += c
            cache[key] = s
            results[name] = s
        return results, tokens, cost
