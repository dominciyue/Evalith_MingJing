from __future__ import annotations

from html import escape

from .consensus import consensus_summary, domain_agreement, judge_means, threshold_from_config
from .diff import DiffReport
from .models import Run


def _truncate(s: str, n: int = 80) -> str:
    s = " ".join(s.split())   # collapse all whitespace (newlines, tabs) to single spaces
    return s if len(s) <= n else s[: n - 1] + "…"


def _md(s: str) -> str:
    """Escape a value for a Markdown table cell (truncate + escape pipes)."""
    return _truncate(s).replace("|", "\\|")


_KAPPA_NOTE = ("*κ collapses toward 0 when one judge's labels are nearly constant "
               "(expected-agreement inflation) — read it together with the means.*")


def _fmt_kappa(v: float) -> str:
    return "n/a" if v != v else f"{v:+.3f}"


def _consensus_md(run: Run) -> list[str]:
    """Markdown lines for the Judge Consensus section; [] when no panel ran."""
    cs = consensus_summary(run, threshold_from_config(run.config))
    if not cs:
        return []
    th = cs["threshold"]
    judges = cs["judges"]
    lines = ["", "## Judge Consensus", "",
             f"- **Judges:** {', '.join(judges)}",
             f"- **Low-consensus cases (spread ≥ {th:.2f}):** "
             f"{len(cs['low_consensus_cases'])}/{cs['n_cases']}",
             f"- **Judging cost:** ${run.total_judge_cost_usd:.4f}  ·  "
             f"{run.total_judge_tokens} tokens",
             "", "### Pairwise Cohen's κ", "",
             "| judge A | judge B | κ |", "| --- | --- | --- |"]
    lines += [f"| {_md(a)} | {_md(b)} | {_fmt_kappa(v)} |" for (a, b), v in cs["kappa"].items()]
    lines += ["", _KAPPA_NOTE]
    doms = domain_agreement(run, th)
    if any(d != "?" for d in doms):
        lines += ["", "### By domain", "",
                  "| domain | n | " + " | ".join(_md(j) for j in judges) + " | low-consensus |",
                  "| --- | --- | " + " | ".join("---" for _ in judges) + " | --- |"]
        for dom, info in doms.items():
            cells = " | ".join(f"{info['means'][j]:.2f}" if j in info["means"] else "n/a"
                               for j in judges)
            lines.append(f"| {_md(dom)} | {info['n']} | {cells} | {info['low_consensus']} |")
    low_ids = set(cs["low_consensus_cases"])
    low = [r for r in run.results if r.case_id in low_ids]
    if low:
        lines += ["", "### ⚠ Low-consensus cases", ""]
        for r in low:
            means = ", ".join(f"{j}={v:.2f}" for j, v in judge_means(r).items())
            lines.append(f"- **{_md(r.case_id)}** ({means})")
            lines += [f"  - {_md(j)}: {_md(reason)}"
                      for j, reason in r.panel_details.items()]
    return lines


def run_to_markdown(run: Run) -> str:
    cost_line = (f"- **Cost:** ${run.total_cost_usd:.4f}  ·  {run.total_tokens} tokens  ·  "
                 f"{run.mean_latency_ms:.0f} ms/case avg")
    if run.total_judge_cost_usd > 0:
        cost_line += f"  ·  judging ${run.total_judge_cost_usd:.4f}"
    lines = [
        f"# Run {run.id} — {run.name}",
        "",
        f"- **Model:** {run.model}",
        f"- **Created:** {run.created_at:%Y-%m-%d %H:%M UTC}",
        f"- **Pass rate:** {run.pass_rate:.0%}  ({run.mean_score:.2f} mean score)",
        cost_line,
        "",
        "| case | output | scores | pass |",
        "| --- | --- | --- | --- |",
    ]
    for r in run.results:
        scores = ", ".join(f"{_md(s.scorer)}={s.value:.2f}" for s in r.scores)
        ok = "✅" if all(s.passed for s in r.scores) and r.scores else "❌"
        lines.append(f"| {_md(r.case_id)} | {_md(r.output)} | {scores} | {ok} |")
    lines += _consensus_md(run)
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
    cmd = _consensus_md(run)
    if cmd:
        items = "".join(f"<p>{escape(line)}</p>"
                        for line in cmd if line and not line.startswith("|"))
        body += f"<hr>{items}"
    return _HTML_SHELL.format(title=f"Run {escape(run.id)}", body=body)


def diff_to_markdown(report: DiffReport, before_id: str, after_id: str) -> str:
    s = report.summary()
    has_ci = any(c.ci is not None for c in report.cases)
    lines = [
        f"# Diff {before_id} → {after_id}",
        "",
        "  ·  ".join(f"**{k}:** {v}" for k, v in s.items()),
        "",
    ]
    if has_ci:
        lines += [
            "| case | status | before | after | Δ 95% CI |",
            "| --- | --- | --- | --- | --- |",
        ]
    else:
        lines += [
            "| case | status | before | after |",
            "| --- | --- | --- | --- |",
        ]
    for c in report.cases:
        b = "—" if c.before is None else f"{c.before:.2f}"
        a = "—" if c.after is None else f"{c.after:.2f}"
        if has_ci:
            ci = "—" if c.ci is None else f"[{c.ci[0]:+.2f}, {c.ci[1]:+.2f}]"
            lines.append(f"| {_md(c.case_id)} | {_md(c.status)} | {b} | {a} | {ci} |")
        else:
            lines.append(f"| {_md(c.case_id)} | {_md(c.status)} | {b} | {a} |")
    if report.regressed:
        lines += ["", "## Regressed case outputs", ""]
        for c in report.regressed:
            lines += [
                f"### `{c.case_id}`  ({c.before:.2f} → {c.after:.2f})",
                f"- **before:** {_truncate(c.before_output or '', 200)}",
                f"- **after:** {_truncate(c.after_output or '', 200)}",
                "",
            ]
    return "\n".join(lines) + "\n"


def diff_to_html(report: DiffReport, before_id: str, after_id: str) -> str:
    has_ci = any(c.ci is not None for c in report.cases)
    rows = ""
    for c in report.cases:
        b = "—" if c.before is None else f"{c.before:.2f}"
        a = "—" if c.after is None else f"{c.after:.2f}"
        cls = c.status if c.status in {"regressed", "improved"} else ""
        ci_cell = ""
        if has_ci:
            ci = "—" if c.ci is None else f"[{c.ci[0]:+.2f}, {c.ci[1]:+.2f}]"
            ci_cell = f"<td>{ci}</td>"
        rows += (f"<tr class='{cls}'><td>{escape(c.case_id)}</td><td>{escape(c.status)}</td>"
                 f"<td>{b}</td><td>{a}</td>{ci_cell}</tr>")
    summary = " · ".join(f"{k}: {v}" for k, v in report.summary().items())
    detail = ""
    if report.regressed:
        items = ""
        for c in report.regressed:
            items += (f"<h3>{escape(c.case_id)} <small>({c.before:.2f} → {c.after:.2f})</small></h3>"
                      f"<p><b>before:</b> {escape(_truncate(c.before_output or '', 300))}</p>"
                      f"<p><b>after:</b> {escape(_truncate(c.after_output or '', 300))}</p>")
        detail = f"<h2>Regressed case outputs</h2>{items}"
    ci_header = "<th>Δ 95% CI</th>" if has_ci else ""
    body = (f"<h1>Diff {escape(before_id)} → {escape(after_id)}</h1>"
            f"<p class='meta'>{escape(summary)}</p>"
            f"<table><tr><th>case</th><th>status</th><th>before</th><th>after</th>{ci_header}</tr>"
            f"{rows}</table>{detail}")
    return _HTML_SHELL.format(title=f"Diff {escape(before_id)}→{escape(after_id)}", body=body)
