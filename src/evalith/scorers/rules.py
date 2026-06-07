from __future__ import annotations

import re

from ..config import ScorerConfig
from ..models import Score, TestCase
from .base import Scorer


class ExactMatch:
    name = "exact_match"

    def score(self, case: TestCase, output: str) -> Score:
        target = (case.expected or "").strip()
        ok = output.strip() == target
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok,
                     detail=f"expected={target!r}")


class Contains:
    name = "contains"

    def __init__(self, text: str | None = None):
        self.text = text

    def score(self, case: TestCase, output: str) -> Score:
        target = self.text if self.text is not None else (case.expected or "")
        ok = target != "" and target in output
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok,
                     detail=f"needle={target!r}")


class Regex:
    name = "regex"

    def __init__(self, pattern: str):
        self.pattern = pattern

    def score(self, case: TestCase, output: str) -> Score:
        ok = re.search(self.pattern, output) is not None
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok,
                     detail=f"pattern={self.pattern!r}")


def build_scorer(cfg: ScorerConfig, judge_provider=None) -> Scorer:
    if cfg.type == "exact_match":
        return ExactMatch()
    if cfg.type == "contains":
        return Contains(text=cfg.params.get("text"))
    if cfg.type == "regex":
        return Regex(pattern=cfg.params["pattern"])
    if cfg.type == "llm_judge":
        from ..providers import get_provider
        from .llm_judge import LLMJudge

        judge_model = cfg.params.get("judge_model")
        primary = get_provider(judge_model) if judge_model else judge_provider
        panel_models = [m for m in dict.fromkeys(cfg.params.get("panel") or [])
                        if m and m != judge_model]
        panel = {m: get_provider(m) for m in panel_models}
        return LLMJudge(provider=primary,
                        criteria=cfg.params.get("criteria", ""),
                        language=cfg.params.get("language", "en"),
                        panel=panel)
    if cfg.type == "code_exec":
        import os

        from .hard import CodeExec

        if os.environ.get("EVALITH_ALLOW_CODE_EXEC") != "1":
            raise ValueError(
                "code_exec runs untrusted model code; "
                "set EVALITH_ALLOW_CODE_EXEC=1 to enable")
        return CodeExec(timeout=cfg.params.get("timeout", 5),
                        memory_mb=cfg.params.get("memory_mb", 256))
    if cfg.type == "numeric_match":
        from .hard import NumericMatch

        return NumericMatch(rel_tol=cfg.params.get("rel_tol", 1e-3),
                            abs_tol=cfg.params.get("abs_tol", 0.0))
    raise ValueError(f"Unknown scorer type: {cfg.type}")
