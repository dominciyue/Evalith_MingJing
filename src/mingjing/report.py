from __future__ import annotations

from .diff import DiffReport
from .models import Run


def _truncate(s: str, n: int = 80) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def run_to_markdown(run: Run) -> str:
    lines = [
        f"# Run {run.id} — {run.name}",
        "",
        f"- **Model:** {run.model}",
        f"- **Created:** {run.created_at:%Y-%m-%d %H:%M UTC}",
        f"- **Pass rate:** {run.pass_rate:.0%}  ({run.mean_score:.2f} mean score)",
        f"- **Cost:** ${run.total_cost_usd:.4f}  ·  {run.total_tokens} tokens  ·  "
        f"{run.mean_latency_ms:.0f} ms/case avg",
        "",
        "| case | output | scores | pass |",
        "| --- | --- | --- | --- |",
    ]
    for r in run.results:
        scores = ", ".join(f"{s.scorer}={s.value:.2f}" for s in r.scores)
        ok = "✅" if all(s.passed for s in r.scores) and r.scores else "❌"
        lines.append(f"| {r.case_id} | {_truncate(r.output)} | {scores} | {ok} |")
    return "\n".join(lines) + "\n"


def run_to_html(run: Run) -> str:  # fleshed out in Task 7
    return (f"<!doctype html><meta charset='utf-8'><title>Run {run.id}</title>"
            f"<pre>{run_to_markdown(run)}</pre>")


def diff_to_markdown(report: DiffReport, before_id: str, after_id: str) -> str:
    s = report.summary()
    lines = [
        f"# Diff {before_id} → {after_id}",
        "",
        "  ·  ".join(f"**{k}:** {v}" for k, v in s.items()),
        "",
        "| case | status | before | after |",
        "| --- | --- | --- | --- |",
    ]
    for c in report.cases:
        b = "—" if c.before is None else f"{c.before:.2f}"
        a = "—" if c.after is None else f"{c.after:.2f}"
        lines.append(f"| {c.case_id} | {c.status} | {b} | {a} |")
    return "\n".join(lines) + "\n"


def diff_to_html(report: DiffReport, before_id: str, after_id: str) -> str:  # fleshed out in Task 7
    return (f"<!doctype html><meta charset='utf-8'><title>Diff {before_id}→{after_id}</title>"
            f"<pre>{diff_to_markdown(report, before_id, after_id)}</pre>")
