from __future__ import annotations

from dataclasses import dataclass

from .models import CaseResult, Run


def case_score(result: CaseResult) -> float:
    return result.mean_score


@dataclass
class CaseDiff:
    case_id: str
    status: str  # improved | regressed | unchanged | new | removed
    before: float | None
    after: float | None


@dataclass
class DiffReport:
    cases: list[CaseDiff]

    @property
    def improved(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.status == "improved"]

    @property
    def regressed(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.status == "regressed"]

    @property
    def unchanged(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.status == "unchanged"]

    @property
    def new(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.status == "new"]

    @property
    def removed(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.status == "removed"]

    def summary(self) -> dict[str, int]:
        return {
            "improved": len(self.improved),
            "regressed": len(self.regressed),
            "unchanged": len(self.unchanged),
            "new": len(self.new),
            "removed": len(self.removed),
        }


def diff_runs(before: Run, after: Run, tol: float = 1e-9) -> DiffReport:
    before_map = {r.case_id: case_score(r) for r in before.results}
    after_map = {r.case_id: case_score(r) for r in after.results}
    cases: list[CaseDiff] = []
    for cid, a in after_map.items():
        if cid not in before_map:
            cases.append(CaseDiff(cid, "new", None, a))
            continue
        b = before_map[cid]
        if a > b + tol:
            status = "improved"
        elif a < b - tol:
            status = "regressed"
        else:
            status = "unchanged"
        cases.append(CaseDiff(cid, status, b, a))
    for cid, b in before_map.items():
        if cid not in after_map:
            cases.append(CaseDiff(cid, "removed", b, None))
    cases.sort(key=lambda c: c.case_id)
    return DiffReport(cases=cases)
