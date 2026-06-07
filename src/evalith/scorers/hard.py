from __future__ import annotations

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
