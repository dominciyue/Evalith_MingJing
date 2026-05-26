from __future__ import annotations

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
