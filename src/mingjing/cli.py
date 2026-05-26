from __future__ import annotations

from pathlib import Path

import typer

from .config import load_config
from .diff import diff_runs
from .engine import run_eval
from .providers import get_provider
from .store import RunStore

app = typer.Typer(help="明镜 / Evalith — AI regression testing")


@app.command()
def run(config: str, store: str = ".mingjing",
        concurrency: int = typer.Option(None, "--concurrency",
            help="Parallel provider calls (default: from config, else 1)."),
        fail_under: float = typer.Option(None, "--fail-under",
            help="Exit 1 if the pass rate is below this threshold (0..1).")) -> None:
    """Run an eval defined by CONFIG and save the resulting run."""
    cfg = load_config(config)
    result = run_eval(cfg, get_provider(cfg.model), concurrency=concurrency)
    path = RunStore(store).save(result)
    passed = sum(1 for r in result.results for s in r.scores if s.passed)
    total = sum(len(r.scores) for r in result.results)
    typer.echo(f"Run {result.id} saved to {path} — {passed}/{total} checks passed")
    if fail_under is not None and (total == 0 or result.pass_rate < fail_under):
        detail = ("no checks ran" if total == 0
                  else f"pass rate {result.pass_rate:.2%} < threshold {fail_under:.2%}")
        typer.echo(f"FAIL: {detail}")
        raise typer.Exit(code=1)


@app.command()
def diff(before: str, after: str, store: str = ".mingjing",
         fail_on_regression: bool = typer.Option(False, "--fail-on-regression",
             help="Exit 1 if any case regressed."),
         fmt: str = typer.Option("text", "--format", help="text, md, or html"),
         output: str = typer.Option(None, "--output", help="write report to file")) -> None:
    """Compare two runs and show which cases improved or regressed."""
    if fmt not in {"text", "md", "html"}:
        raise typer.BadParameter("format must be 'text', 'md', or 'html'")
    s = RunStore(store)
    report = diff_runs(s.load(before), s.load(after))
    if fmt in {"md", "html"}:
        from .report import diff_to_html, diff_to_markdown
        text = (diff_to_html(report, before, after) if fmt == "html"
                else diff_to_markdown(report, before, after))
        if output:
            Path(output).write_text(text, encoding="utf-8")
            typer.echo(f"Wrote {fmt} diff to {output}")
        else:
            typer.echo(text)
    else:
        typer.echo(f"Diff {before} -> {after}")
        for c in report.cases:
            before_s = "-" if c.before is None else f"{c.before:.2f}"
            after_s = "-" if c.after is None else f"{c.after:.2f}"
            typer.echo(f"  {c.case_id:<16} {c.status:<10} {before_s:>6} -> {after_s:>6}")
        typer.echo(str(report.summary()))
    if fail_on_regression and report.regressed:
        typer.echo(f"FAIL: {len(report.regressed)} case(s) regressed")
        raise typer.Exit(code=1)


@app.command("list")
def list_runs(store: str = ".mingjing") -> None:
    """List stored runs, newest first."""
    for r in RunStore(store).list_runs():
        typer.echo(f"{r.id}  {r.created_at:%Y-%m-%d %H:%M}  {r.name}  ({r.model})")


@app.command()
def report(run_id: str, store: str = ".mingjing",
           fmt: str = typer.Option("md", "--format", help="md or html"),
           output: str = typer.Option(None, "--output",
               help="write to file instead of stdout")) -> None:
    """Render a saved run as a shareable Markdown/HTML report."""
    if fmt not in {"md", "html"}:
        raise typer.BadParameter("format must be 'md' or 'html'")
    from .report import run_to_html, run_to_markdown
    run = RunStore(store).load(run_id)
    text = run_to_html(run) if fmt == "html" else run_to_markdown(run)
    if output:
        Path(output).write_text(text, encoding="utf-8")
        typer.echo(f"Wrote {fmt} report to {output}")
    else:
        typer.echo(text)


@app.command("models")
def list_models() -> None:
    """List first-class 国产 model aliases and their API key env vars."""
    from .presets import CHINA_MODELS
    for alias, info in CHINA_MODELS.items():
        typer.echo(f"{alias:<20} -> {info['litellm']:<28} (env: {info['env']})  {info['note']}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
