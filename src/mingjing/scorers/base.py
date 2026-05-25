from __future__ import annotations

from typing import Protocol

from ..models import Score, TestCase


class Scorer(Protocol):
    name: str

    def score(self, case: TestCase, output: str) -> Score: ...
