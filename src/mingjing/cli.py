from __future__ import annotations

import typer

from .config import load_config
from .diff import diff_runs
from .engine import run_eval
from .providers import get_provider
from .store import RunStore

app = typer.Typer(help="明镜 / Evalith — AI regression testing")


@app.command()
def run(config: str, store: str = ".mingjing",
        fail_under: float = typer.Option(None, "--fail-under",
            help="Exit 1 if the pass rate is below this threshold (0..1).")) -> None:
    """Run an eval defined by CONFIG and save the resulting run."""
    cfg = load_config(config)
    result = run_eval(cfg, get_provider(cfg.model))
    path = RunStore(store).save(result)
    passed = sum(1 for r in result.results for s in r.scores if s.passed)
    total = sum(len(r.scores) for r in result.results)
    typer.echo(f"Run {result.id} saved to {path} — {passed}/{total} checks passed")
    if fail_under is not None and result.pass_rate < fail_under:
        typer.echo(f"FAIL: pass rate {result.pass_rate:.2%} < threshold {fail_under:.2%}")
        raise typer.Exit(code=1)


@app.command()
def diff(before: str, after: str, store: str = ".mingjing",
         fail_on_regression: bool = typer.Option(False, "--fail-on-regression",
             help="Exit 1 if any case regressed.")) -> None:
    """Compare two runs and show which cases improved or regressed."""
    s = RunStore(store)
    report = diff_runs(s.load(before), s.load(after))
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
