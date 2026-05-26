from __future__ import annotations

from html import escape

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


_HTML_SHELL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:2rem;color:#222}}
 h1{{font-size:1.3rem}} .meta{{color:#555}} table{{border-collapse:collapse;margin-top:1rem;width:100%}}
 th,td{{border:1px solid #ddd;padding:.4rem .6rem;text-align:left;vertical-align:top}}
 th{{background:#f6f8fa}} .pass{{color:#137333}} .fail{{color:#c5221f}}
 .regressed{{background:#fce8e6}} .improved{{background:#e6f4ea}}
</style></head><body>
{body}
</body></html>
"""


def run_to_html(run: Run) -> str:
    rows = ""
    for r in run.results:
        scores = ", ".join(f"{escape(s.scorer)}={s.value:.2f}" for s in r.scores)
        ok = all(s.passed for s in r.scores) and bool(r.scores)
        cls = "pass" if ok else "fail"
        rows += (f"<tr><td>{escape(r.case_id)}</td><td>{escape(_truncate(r.output))}</td>"
                 f"<td>{scores}</td><td class='{cls}'>{'PASS' if ok else 'FAIL'}</td></tr>")
    body = (f"<h1>Run {escape(run.id)} — {escape(run.name)}</h1>"
            f"<p class='meta'>{escape(run.model)} · pass rate {run.pass_rate:.0%} · "
            f"${run.total_cost_usd:.4f} · {run.total_tokens} tokens · "
            f"{run.mean_latency_ms:.0f} ms/case</p>"
            f"<table><tr><th>case</th><th>output</th><th>scores</th><th>pass</th></tr>"
            f"{rows}</table>")
    return _HTML_SHELL.format(title=f"Run {escape(run.id)}", body=body)


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


def diff_to_html(report: DiffReport, before_id: str, after_id: str) -> str:
    rows = ""
    for c in report.cases:
        b = "—" if c.before is None else f"{c.before:.2f}"
        a = "—" if c.after is None else f"{c.after:.2f}"
        cls = c.status if c.status in {"regressed", "improved"} else ""
        rows += (f"<tr class='{cls}'><td>{escape(c.case_id)}</td><td>{escape(c.status)}</td>"
                 f"<td>{b}</td><td>{a}</td></tr>")
    summary = " · ".join(f"{k}: {v}" for k, v in report.summary().items())
    body = (f"<h1>Diff {escape(before_id)} → {escape(after_id)}</h1>"
            f"<p class='meta'>{escape(summary)}</p>"
            f"<table><tr><th>case</th><th>status</th><th>before</th><th>after</th></tr>"
            f"{rows}</table>")
    return _HTML_SHELL.format(title=f"Diff {escape(before_id)}→{escape(after_id)}", body=body)
