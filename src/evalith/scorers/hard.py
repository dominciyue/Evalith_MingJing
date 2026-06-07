from __future__ import annotations

import math
import re

from ..models import Score, TestCase
from .sandbox import run_program

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(output: str) -> str | None:
    """Pull code out of a model reply.

    Returns the first fenced block's contents if present, else the whole
    stripped reply. Returns None only when nothing non-blank remains.
    """
    m = _FENCE.search(output)
    if m:
        return m.group(1).strip() or None
    stripped = output.strip()
    return stripped or None


class CodeExec:
    name = "code_exec"

    def __init__(self, timeout: float = 5, memory_mb: int = 256):
        self.timeout = timeout
        self.memory_mb = memory_mb

    def score(self, case: TestCase, output: str) -> Score:
        entry = case.metadata.get("entry_point")
        test = case.metadata.get("test")
        if not entry or not test:
            return Score(scorer=self.name, value=0.0, passed=False,
                         detail="missing entry_point or test in metadata")
        code = extract_code(output)
        if not code:
            return Score(scorer=self.name, value=0.0, passed=False,
                         detail="no code in output")
        program = f"{code}\n{test}\ncheck({entry})\n"
        ok, detail = run_program(program, timeout=self.timeout, memory_mb=self.memory_mb)
        return Score(scorer=self.name, value=1.0 if ok else 0.0,
                     passed=ok, detail=detail)


_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _last_number(text: str) -> float | None:
    for tok in reversed(_NUM.findall(text)):
        try:
            return float(tok)
        except ValueError:
            continue
    return None


class NumericMatch:
    name = "numeric_match"

    def __init__(self, rel_tol: float = 1e-3, abs_tol: float = 0.0):
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def score(self, case: TestCase, output: str) -> Score:
        raw = (case.expected or "").strip()
        try:
            target = float(raw)
        except (ValueError, TypeError):
            return Score(scorer=self.name, value=0.0, passed=False,
                         detail=f"expected not numeric: {raw!r}")
        got = _last_number(output)
        if got is None:
            return Score(scorer=self.name, value=0.0, passed=False,
                         detail="no number in output")
        ok = math.isclose(got, target, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok,
                     detail=f"got={got} expected={target}")
