# Evalith v0.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn v0.1 (a working eval→store→diff loop) into a tool a team can *depend on in CI* — gate PRs on regressions, produce shareable reports, scale to real datasets, track cost, and make 国产模型 first-class.

**Architecture:** Pure functions for new logic (report rendering, usage extraction) kept dependency-free and network-free so the test suite still runs with no install and no API key. CLI commands wire pure functions to I/O and exit codes. Concurrency via `ThreadPoolExecutor` (provider calls are blocking I/O), order-preserving. litellm stays an optional, lazily-imported extra.

**Tech Stack:** Python ≥3.10, pydantic v2, PyYAML, Typer, pytest, `concurrent.futures`. litellm optional. No new hard dependencies.

---

## File Structure

**New files:**
- `src/mingjing/report.py` — pure render functions: `run_to_markdown`, `diff_to_markdown`, `run_to_html`, `diff_to_html`.
- `src/mingjing/presets.py` — curated 国产/global model aliases → litellm id + env var; `resolve_model`, `CHINA_MODELS`.
- `action.yml` — composite GitHub Action: install Evalith + run an eval gate.
- `.github/workflows/eval-example.yml` — example PR workflow using the composite action with the offline echo eval (no key needed).
- `examples/qa.jsonl` — JSONL dataset example.
- `tests/test_report.py`, `tests/test_presets.py`, `tests/test_ci_gate.py` (CLI exit codes), `tests/test_github_action.py` (YAML validity).

**Modified files:**
- `src/mingjing/models.py` — token/cost fields on `CaseResult`; computed props on `Run`.
- `src/mingjing/providers/base.py` — token/cost fields on `Response`.
- `src/mingjing/providers/litellm_provider.py` — extract usage + best-effort cost; `_usage_from_response` pure helper.
- `src/mingjing/providers/__init__.py` — resolve preset aliases in `get_provider`.
- `src/mingjing/engine.py` — populate usage onto `CaseResult`; concurrent per-case eval (order-preserving).
- `src/mingjing/config.py` — `EvalConfig.concurrency: int = 1`.
- `src/mingjing/dataset.py` — `.jsonl` support.
- `src/mingjing/scorers/llm_judge.py` — `language` param + Chinese judge template.
- `src/mingjing/scorers/rules.py` — pass `language` into `LLMJudge` from config.
- `src/mingjing/cli.py` — `run --fail-under/--concurrency`; `diff --fail-on-regression/--format/--output`; new `report` and `models` commands.
- `README.md`, `README.zh-CN.md` — document all of the above.

**Type/signature contract (used consistently across tasks):**
- `Response(text, latency_ms=0.0, prompt_tokens=0, completion_tokens=0, total_tokens=0, cost_usd=0.0)`
- `CaseResult(..., latency_ms=0.0, prompt_tokens=0, completion_tokens=0, total_tokens=0, cost_usd=0.0)`
- `Run.pass_rate -> float`, `Run.mean_score -> float`, `Run.total_tokens -> int`, `Run.total_cost_usd -> float`, `Run.mean_latency_ms -> float`
- `run_eval(config, provider, judge_provider=None, concurrency: int | None = None) -> Run`
- `report.run_to_markdown(run: Run) -> str`, `report.diff_to_markdown(report: DiffReport, before_id: str, after_id: str) -> str`, `run_to_html(run: Run) -> str`, `diff_to_html(report: DiffReport, before_id: str, after_id: str) -> str`
- `presets.resolve_model(name: str) -> str`, `presets.CHINA_MODELS: dict[str, dict]`
- `LLMJudge(provider, criteria="", language="en")`

---

## Part A — CI gating (Feature 1)

### Task 1: `mingjing run --fail-under` + `Run.pass_rate`

**Files:**
- Modify: `src/mingjing/models.py` (add `Run.pass_rate`)
- Modify: `src/mingjing/cli.py` (`run` gains `--fail-under`)
- Test: `tests/test_models.py`, `tests/test_ci_gate.py` (new)

- [ ] **Step 1: Failing test for `Run.pass_rate`** — append to `tests/test_models.py`:

```python
def test_run_pass_rate():
    from mingjing.models import CaseResult, Run, Score
    from datetime import datetime, timezone
    r = Run(id="x", name="t", created_at=datetime.now(timezone.utc), model="echo",
            results=[
                CaseResult(case_id="1", input="i", output="o",
                           scores=[Score(scorer="s", value=1.0, passed=True),
                                   Score(scorer="s", value=0.0, passed=False)]),
                CaseResult(case_id="2", input="i", output="o",
                           scores=[Score(scorer="s", value=1.0, passed=True)]),
            ])
    assert r.pass_rate == 2 / 3            # 2 of 3 checks passed
    assert Run(id="y", name="t", created_at=datetime.now(timezone.utc),
               model="echo", results=[]).pass_rate == 1.0   # no checks => nothing failed
```

- [ ] **Step 2: Run, expect fail** — `python3 -m pytest tests/test_models.py::test_run_pass_rate -v` → AttributeError/fail.

- [ ] **Step 3: Implement `Run.pass_rate`** in `src/mingjing/models.py` (add inside `Run`):

```python
    @property
    def pass_rate(self) -> float:
        checks = [s for r in self.results for s in r.scores]
        return sum(1 for s in checks if s.passed) / len(checks) if checks else 1.0
```

- [ ] **Step 4: Run, expect pass.**

- [ ] **Step 5: Failing CLI test** — create `tests/test_ci_gate.py`:

```python
from typer.testing import CliRunner
from mingjing.cli import app

runner = CliRunner()


def _cfg(tmp_path, expected, scorer_text):
    ds = tmp_path / "ds.yaml"
    ds.write_text(f"name: d\ncases:\n  - id: '1'\n    input: hello\n    expected: {expected}\n",
                  encoding="utf-8")
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(
        f"name: t\ndataset: {ds}\nmodel: echo\nprompt_template: '{{{{input}}}}'\n"
        f"scorers:\n  - type: contains\n    params: {{text: {scorer_text}}}\n",
        encoding="utf-8")
    return cfg


def test_run_fail_under_fails_when_below(tmp_path):
    # echo returns the prompt 'hello'; needle 'zzz' never matches -> pass_rate 0.0
    cfg = _cfg(tmp_path, "hello", "zzz")
    res = runner.invoke(app, ["run", str(cfg), "--store", str(tmp_path / "d"),
                              "--fail-under", "0.9"])
    assert res.exit_code == 1


def test_run_fail_under_passes_when_met(tmp_path):
    cfg = _cfg(tmp_path, "hello", "hello")
    res = runner.invoke(app, ["run", str(cfg), "--store", str(tmp_path / "d"),
                              "--fail-under", "0.9"])
    assert res.exit_code == 0
```

- [ ] **Step 6: Run, expect fail** (unknown option `--fail-under`).

- [ ] **Step 7: Implement** — in `src/mingjing/cli.py`, update `run`:

```python
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
```

- [ ] **Step 8: Run, expect pass.** Full suite: `python3 -m pytest -q`.

- [ ] **Step 9: Commit** — `git add -A && git commit -m "feat(ci): mingjing run --fail-under gate + Run.pass_rate"`

### Task 2: `mingjing diff --fail-on-regression`

**Files:** Modify `src/mingjing/cli.py`; Test `tests/test_ci_gate.py`.

- [ ] **Step 1: Failing test** — append to `tests/test_ci_gate.py`:

```python
from mingjing.store import RunStore


def test_diff_fail_on_regression(tmp_path):
    store = str(tmp_path / "d")
    base = _cfg(tmp_path, "hello", "hello")   # passes
    bad = _cfg(tmp_path, "hello", "zzz")      # fails
    runner.invoke(app, ["run", str(base), "--store", store])
    runner.invoke(app, ["run", str(bad), "--store", store])
    ids = [r.id for r in RunStore(store).list_runs()]  # newest first
    older, newer = ids[1], ids[0]
    ok = runner.invoke(app, ["diff", older, newer, "--store", store])
    assert ok.exit_code == 0  # no flag -> never fails
    gated = runner.invoke(app, ["diff", older, newer, "--store", store,
                                "--fail-on-regression"])
    assert gated.exit_code == 1  # case '1' regressed 1.0 -> 0.0
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement** — update `diff` in `src/mingjing/cli.py` to add the option and, after printing, exit 1 when regressions exist and the flag is set:

```python
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
```

- [ ] **Step 4: Run, expect pass.** `python3 -m pytest tests/test_ci_gate.py -q`.

- [ ] **Step 5: Commit** — `git commit -am "feat(ci): mingjing diff --fail-on-regression"`

### Task 3: Composite GitHub Action + example workflow

**Files:** Create `action.yml`, `.github/workflows/eval-example.yml`; Test `tests/test_github_action.py`.

- [ ] **Step 1: Failing test** — create `tests/test_github_action.py`:

```python
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_action_yml_is_valid_composite():
    data = yaml.safe_load((ROOT / "action.yml").read_text(encoding="utf-8"))
    assert data["runs"]["using"] == "composite"
    assert "config" in data["inputs"]
    assert "fail-under" in data["inputs"]


def test_example_workflow_valid():
    data = yaml.safe_load((ROOT / ".github/workflows/eval-example.yml").read_text(encoding="utf-8"))
    assert "jobs" in data
    # 'on:' parses to True (YAML truthy key); just assert the file has jobs + a step
    job = next(iter(data["jobs"].values()))
    assert any("uses" in step or "run" in step for step in job["steps"])
```

- [ ] **Step 2: Run, expect fail** (files missing).

- [ ] **Step 3: Create `action.yml`** (repo root):

```yaml
name: "Evalith eval gate"
description: "Run an Evalith AI eval and fail the build if the pass rate drops below a threshold."
branding:
  icon: "check-circle"
  color: "blue"
inputs:
  config:
    description: "Path to the eval config YAML"
    required: true
  fail-under:
    description: "Minimum pass rate (0..1) required to pass"
    required: false
    default: "1.0"
  python-version:
    description: "Python version"
    required: false
    default: "3.11"
runs:
  using: "composite"
  steps:
    - uses: actions/setup-python@v5
      with:
        python-version: ${{ inputs.python-version }}
    - name: Install Evalith
      shell: bash
      run: pip install -e "${{ github.action_path }}[litellm]"
    - name: Run eval gate
      shell: bash
      run: mingjing run "${{ inputs.config }}" --fail-under "${{ inputs.fail-under }}"
```

- [ ] **Step 4: Create `.github/workflows/eval-example.yml`:**

```yaml
name: AI eval gate (example)
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Evalith gate (offline echo demo — no API key)
        uses: ./
        with:
          config: examples/eval.yaml
          fail-under: "1.0"
```

- [ ] **Step 5: Run, expect pass.** `python3 -m pytest tests/test_github_action.py -q`.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(ci): composite GitHub Action + example PR workflow"`

---

## Part B — Cost / latency / token tracking (Feature 4)

### Task 4: token/cost on Response, CaseResult, Run + engine population

**Files:** Modify `models.py`, `providers/base.py`, `providers/litellm_provider.py`, `engine.py`; Test `tests/test_models.py`, `tests/test_providers.py`, `tests/test_engine.py`.

- [ ] **Step 1: Failing tests** — append to `tests/test_models.py`:

```python
def test_run_cost_and_latency_aggregates():
    from mingjing.models import CaseResult, Run
    from datetime import datetime, timezone
    r = Run(id="x", name="t", created_at=datetime.now(timezone.utc), model="m",
            results=[
                CaseResult(case_id="1", input="i", output="o", latency_ms=100.0,
                           total_tokens=10, cost_usd=0.001),
                CaseResult(case_id="2", input="i", output="o", latency_ms=300.0,
                           total_tokens=20, cost_usd=0.002),
            ])
    assert r.total_tokens == 30
    assert abs(r.total_cost_usd - 0.003) < 1e-9
    assert r.mean_latency_ms == 200.0
```

And to `tests/test_providers.py`:

```python
def test_usage_from_response_pure():
    from mingjing.providers.litellm_provider import _usage_from_response
    fake = {"usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}}
    assert _usage_from_response(fake) == (5, 7, 12)
    assert _usage_from_response({}) == (0, 0, 0)   # missing usage -> zeros
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement model fields.** In `providers/base.py` `Response`:

```python
@dataclass
class Response:
    text: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
```

In `models.py` `CaseResult`, add fields after `latency_ms`:

```python
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
```

In `models.py` `Run`, add properties:

```python
    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.results)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def mean_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mean_score(self) -> float:
        return sum(r.mean_score for r in self.results) / len(self.results) if self.results else 0.0
```

- [ ] **Step 4: Implement provider usage extraction** in `litellm_provider.py`:

```python
def _usage_from_response(resp) -> tuple[int, int, int]:
    usage = (resp.get("usage") if isinstance(resp, dict) else getattr(resp, "usage", None)) or {}
    get = usage.get if isinstance(usage, dict) else (lambda k, d=0: getattr(usage, k, d))
    return (int(get("prompt_tokens", 0) or 0),
            int(get("completion_tokens", 0) or 0),
            int(get("total_tokens", 0) or 0))
```

Update `LiteLLMProvider.complete` to populate usage + best-effort cost:

```python
        resp = litellm.completion(model=self.model, messages=messages, temperature=temperature)
        latency_ms = (time.perf_counter() - start) * 1000
        pt, ct, tt = _usage_from_response(resp)
        try:
            cost = float(litellm.completion_cost(completion_response=resp))
        except Exception:
            cost = 0.0
        return Response(text=resp["choices"][0]["message"]["content"], latency_ms=latency_ms,
                        prompt_tokens=pt, completion_tokens=ct, total_tokens=tt, cost_usd=cost)
```

- [ ] **Step 5: Engine populates usage.** In `engine.py`, when building each `CaseResult`, copy usage from the response (the per-case build is refactored in Task 9; for now in the serial loop):

```python
        results.append(
            CaseResult(case_id=case.id, input=case.input, output=resp.text,
                       scores=scores, latency_ms=resp.latency_ms,
                       prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                       total_tokens=resp.total_tokens, cost_usd=resp.cost_usd)
        )
```

- [ ] **Step 6: Run, expect pass.** `python3 -m pytest -q`.

- [ ] **Step 7: Commit** — `git commit -am "feat: track tokens/cost/latency on responses, results, runs"`

---

## Part C — Readable reports (Feature 2)

### Task 5: `run_to_markdown` + `mingjing report`

**Files:** Create `src/mingjing/report.py`, `tests/test_report.py`; Modify `cli.py`.

- [ ] **Step 1: Failing test** — create `tests/test_report.py`:

```python
from datetime import datetime, timezone
from mingjing.models import CaseResult, Run, Score


def _run():
    return Run(id="abc123", name="demo", created_at=datetime(2026, 5, 26, tzinfo=timezone.utc),
               model="deepseek/deepseek-chat",
               results=[CaseResult(case_id="1", input="2+2?", output="4", latency_ms=120.0,
                                   total_tokens=8, cost_usd=0.0001,
                                   scores=[Score(scorer="contains", value=1.0, passed=True)])])


def test_run_to_markdown_has_summary_and_row():
    from mingjing.report import run_to_markdown
    md = run_to_markdown(_run())
    assert "# Run abc123" in md
    assert "deepseek/deepseek-chat" in md
    assert "100" in md.replace(",", "")  # pass rate 100%
    assert "| 1 |" in md                 # the case row
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement `run_to_markdown`** in `src/mingjing/report.py`:

```python
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
```

- [ ] **Step 4: Run, expect pass.**

- [ ] **Step 5: Failing CLI test** — append to `tests/test_report.py`:

```python
def test_cli_report_markdown(tmp_path):
    from typer.testing import CliRunner
    from mingjing.cli import app
    from mingjing.store import RunStore
    runner = CliRunner()
    store = str(tmp_path / "d")
    RunStore(store).save(_run())
    res = runner.invoke(app, ["report", "abc123", "--store", store])
    assert res.exit_code == 0
    assert "# Run abc123" in res.stdout
```

- [ ] **Step 6: Run, expect fail** (no `report` command).

- [ ] **Step 7: Implement `report` command** in `cli.py` (import `from .report import run_to_markdown, run_to_html, diff_to_markdown, diff_to_html` — the html/diff names are added in Tasks 6–7; add them to report.py stubs only when those tasks run, OR import lazily inside each command to avoid ImportError). Use lazy import:

```python
@app.command()
def report(run_id: str, store: str = ".mingjing",
           fmt: str = typer.Option("md", "--format", help="md or html"),
           output: str = typer.Option(None, "--output", help="write to file instead of stdout")) -> None:
    """Render a saved run as a shareable Markdown/HTML report."""
    from .report import run_to_markdown, run_to_html
    run = RunStore(store).load(run_id)
    text = run_to_html(run) if fmt == "html" else run_to_markdown(run)
    if output:
        from pathlib import Path
        Path(output).write_text(text, encoding="utf-8")
        typer.echo(f"Wrote {fmt} report to {output}")
    else:
        typer.echo(text)
```

(Note: `run_to_html` is implemented in Task 7. To keep tests green between tasks, implement a minimal `run_to_html` stub in Task 5's report.py that returns `run_to_markdown(run)` wrapped in `<pre>`, then flesh it out in Task 7. Include the stub now:)

```python
def run_to_html(run: Run) -> str:  # fleshed out in Task 7
    return f"<!doctype html><meta charset='utf-8'><title>Run {run.id}</title><pre>{run_to_markdown(run)}</pre>"
```

- [ ] **Step 8: Run, expect pass.** `python3 -m pytest tests/test_report.py -q`.

- [ ] **Step 9: Commit** — `git add -A && git commit -m "feat(report): run_to_markdown + mingjing report command"`

### Task 6: `diff_to_markdown` + `mingjing diff --format md`

**Files:** Modify `report.py`, `cli.py`; Test `tests/test_report.py`.

- [ ] **Step 1: Failing test** — append to `tests/test_report.py`:

```python
def test_diff_to_markdown():
    from mingjing.diff import diff_runs
    from mingjing.report import diff_to_markdown
    a = _run()
    b = Run(id="def456", name="demo", created_at=a.created_at, model=a.model,
            results=[CaseResult(case_id="1", input="2+2?", output="5",
                                scores=[Score(scorer="contains", value=0.0, passed=False)])])
    md = diff_to_markdown(diff_runs(a, b), "abc123", "def456")
    assert "abc123" in md and "def456" in md
    assert "regressed" in md
    assert "| 1 |" in md
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement `diff_to_markdown`** in `report.py`:

```python
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
```

- [ ] **Step 4: Run, expect pass.**

- [ ] **Step 5: Wire `--format`/`--output` into `diff`** in `cli.py` (extend the Task 2 version):

```python
@app.command()
def diff(before: str, after: str, store: str = ".mingjing",
         fail_on_regression: bool = typer.Option(False, "--fail-on-regression",
             help="Exit 1 if any case regressed."),
         fmt: str = typer.Option("text", "--format", help="text, md, or html"),
         output: str = typer.Option(None, "--output", help="write report to file")) -> None:
    """Compare two runs and show which cases improved or regressed."""
    s = RunStore(store)
    report = diff_runs(s.load(before), s.load(after))
    if fmt in {"md", "html"}:
        from .report import diff_to_markdown, diff_to_html
        text = diff_to_html(report, before, after) if fmt == "html" else diff_to_markdown(report, before, after)
        if output:
            from pathlib import Path
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
```

Add a `diff_to_html` stub to `report.py` now (fleshed out in Task 7):

```python
def diff_to_html(report: DiffReport, before_id: str, after_id: str) -> str:  # fleshed out in Task 7
    return (f"<!doctype html><meta charset='utf-8'><title>Diff {before_id}→{after_id}</title>"
            f"<pre>{diff_to_markdown(report, before_id, after_id)}</pre>")
```

- [ ] **Step 6: Run, expect pass.** Full suite `python3 -m pytest -q`.

- [ ] **Step 7: Commit** — `git commit -am "feat(report): diff_to_markdown + diff --format/--output"`

### Task 7: Real HTML reports

**Files:** Modify `report.py`; Test `tests/test_report.py`.

- [ ] **Step 1: Failing test** — append:

```python
def test_run_to_html_is_self_contained():
    from mingjing.report import run_to_html
    html = run_to_html(_run())
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert "<table" in html
    assert "abc123" in html
    assert "<style" in html  # inline styling, no external deps
```

- [ ] **Step 2: Run, expect fail** (stub has no `<table>`).

- [ ] **Step 3: Implement real `run_to_html` and `diff_to_html`** in `report.py` (replace stubs). Use a shared HTML shell + escape:

```python
from html import escape

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
            f"${run.total_cost_usd:.4f} · {run.total_tokens} tokens · {run.mean_latency_ms:.0f} ms/case</p>"
            f"<table><tr><th>case</th><th>output</th><th>scores</th><th>pass</th></tr>{rows}</table>")
    return _HTML_SHELL.format(title=f"Run {escape(run.id)}", body=body)


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
            f"<table><tr><th>case</th><th>status</th><th>before</th><th>after</th></tr>{rows}</table>")
    return _HTML_SHELL.format(title=f"Diff {escape(before_id)}→{escape(after_id)}", body=body)
```

- [ ] **Step 4: Run, expect pass.** `python3 -m pytest tests/test_report.py -q`.

- [ ] **Step 5: Commit** — `git commit -am "feat(report): self-contained HTML run/diff reports"`

---

## Part D — Scale: JSONL + concurrency (Feature 3)

### Task 8: JSONL dataset support

**Files:** Modify `dataset.py`; Create `examples/qa.jsonl`; Test `tests/test_dataset.py`.

- [ ] **Step 1: Failing test** — append to `tests/test_dataset.py`:

```python
def test_load_jsonl(tmp_path):
    from mingjing.dataset import load_dataset
    p = tmp_path / "d.jsonl"
    p.write_text('{"id": "a", "input": "hi", "expected": "yo"}\n'
                 '{"input": "bye"}\n', encoding="utf-8")
    ds = load_dataset(p)
    assert [c.id for c in ds.cases] == ["a", "1"]   # missing id -> index
    assert ds.cases[0].expected == "yo"
    assert ds.cases[1].expected is None
```

- [ ] **Step 2: Run, expect fail** (unsupported `.jsonl`).

- [ ] **Step 3: Implement** — in `dataset.py`, add before the final `raise`:

```python
    if suffix == ".jsonl":
        cases = []
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cases.append(TestCase(
                id=str(row.get("id") or i),
                input=row["input"],
                expected=row.get("expected"),
                metadata=row.get("metadata") or {},
            ))
        return Dataset(name=p.stem, cases=cases)
```

- [ ] **Step 4: Run, expect pass.**

- [ ] **Step 5: Create `examples/qa.jsonl`** (mirror of qa.yaml, JSONL form):

```
{"id": "capital-cn", "input": "中国的首都是哪座城市?只回答城市名称,不要其它内容。", "expected": "北京"}
{"id": "arithmetic", "input": "请计算 25 + 17,只输出最终的数字结果。", "expected": "42"}
{"id": "sunrise", "input": "太阳从哪个方向升起?用一个字回答:东、南、西、北。", "expected": "东"}
```

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(dataset): JSONL support + example"`

### Task 9: Concurrent, order-preserving `run_eval`

**Files:** Modify `engine.py`, `config.py`, `cli.py`; Test `tests/test_engine.py`.

- [ ] **Step 1: Failing test** — append to `tests/test_engine.py`:

```python
def test_run_eval_concurrent_preserves_order(tmp_path):
    from mingjing.config import EvalConfig, ScorerConfig
    from mingjing.engine import run_eval
    from mingjing.providers.base import FakeProvider
    ds = tmp_path / "ds.yaml"
    cases = "".join(f"  - id: '{i}'\n    input: q{i}\n" for i in range(20))
    ds.write_text(f"name: d\ncases:\n{cases}", encoding="utf-8")
    cfg = EvalConfig(name="t", dataset=str(ds), model="echo", prompt_template="{{input}}",
                     scorers=[ScorerConfig(type="contains", params={"text": "q"})])
    # echo returns the prompt, which contains 'q'
    run = run_eval(cfg, FakeProvider(default="q-default"), concurrency=8)
    assert [r.case_id for r in run.results] == [str(i) for i in range(20)]
    assert len(run.results) == 20
```

(Note: FakeProvider returns `default` for unknown prompts; needle "q" is in "q-default" so every check passes — but the test asserts *order/count*, which is the concurrency contract.)

- [ ] **Step 2: Run, expect fail** (`run_eval` has no `concurrency` kwarg).

- [ ] **Step 3: Add `EvalConfig.concurrency`** in `config.py` (after `temperature`):

```python
    concurrency: int = 1
```

- [ ] **Step 4: Refactor `engine.py`** to evaluate cases concurrently while preserving order:

```python
from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .config import EvalConfig
from .dataset import load_dataset
from .models import CaseResult, Run
from .providers.base import Provider
from .scorers.rules import build_scorer


def _render(template: str, case_input: str) -> str:
    return template.replace("{{input}}", case_input)


def run_eval(config: EvalConfig, provider: Provider,
             judge_provider: Provider | None = None,
             concurrency: int | None = None) -> Run:
    dataset = load_dataset(config.dataset)
    scorers = [build_scorer(s, judge_provider=judge_provider or provider)
               for s in config.scorers]
    workers = max(1, concurrency if concurrency is not None else config.concurrency)

    def _eval_case(case) -> CaseResult:
        prompt = _render(config.prompt_template, case.input)
        resp = provider.complete(prompt, system=config.system, temperature=config.temperature)
        scores = [scorer.score(case, resp.text) for scorer in scorers]
        return CaseResult(case_id=case.id, input=case.input, output=resp.text,
                          scores=scores, latency_ms=resp.latency_ms,
                          prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                          total_tokens=resp.total_tokens, cost_usd=resp.cost_usd)

    if workers == 1:
        results = [_eval_case(c) for c in dataset.cases]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_eval_case, dataset.cases))  # ex.map preserves input order

    return Run(id=uuid.uuid4().hex[:12], name=config.name,
               created_at=datetime.now(timezone.utc), model=config.model,
               results=results, config=config.model_dump())
```

- [ ] **Step 5: Run, expect pass.** `python3 -m pytest tests/test_engine.py -q` (all engine tests, incl. existing serial ones).

- [ ] **Step 6: Wire CLI `run --concurrency`** in `cli.py`:

```python
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
    if fail_under is not None and result.pass_rate < fail_under:
        typer.echo(f"FAIL: pass rate {result.pass_rate:.2%} < threshold {fail_under:.2%}")
        raise typer.Exit(code=1)
```

- [ ] **Step 7: Run full suite, expect pass.** `python3 -m pytest -q`.

- [ ] **Step 8: Commit** — `git add -A && git commit -m "feat(engine): order-preserving concurrent run_eval + run --concurrency"`

---

## Part E — China-first hooks (Feature 5)

### Task 10: Chinese llm-judge

**Files:** Modify `scorers/llm_judge.py`, `scorers/rules.py`; Test `tests/test_scorers_llm_judge.py`.

- [ ] **Step 1: Failing test** — append to `tests/test_scorers_llm_judge.py`:

```python
def test_llm_judge_chinese_prompt_and_parse():
    from mingjing.providers.base import FakeProvider
    from mingjing.scorers.llm_judge import LLMJudge, JUDGE_PROMPTS
    from mingjing.models import TestCase
    assert "zh" in JUDGE_PROMPTS and "请" in JUDGE_PROMPTS["zh"]
    judge = LLMJudge(provider=FakeProvider(default='{"score": 1.0, "pass": true, "reason": "对"}'),
                     criteria="准确性", language="zh")
    score = judge.score(TestCase(id="1", input="x"), "y")
    assert score.passed is True and score.value == 1.0 and score.detail == "对"


def test_llm_judge_defaults_to_english():
    from mingjing.scorers.llm_judge import LLMJudge
    assert LLMJudge(provider=None).language == "en"
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement** — replace the single `JUDGE_PROMPT` in `llm_judge.py` with a dict and add `language`:

```python
JUDGE_PROMPTS = {
    "en": """You are grading an AI answer.

Question/Input:
{input}

AI Answer:
{output}

Criteria: {criteria}

Respond with ONLY a JSON object:
{{"score": <float 0..1>, "pass": <true|false>, "reason": "<short reason>"}}
""",
    "zh": """你是一名严格的 AI 答案评审。

问题/输入:
{input}

AI 的回答:
{output}

评判标准: {criteria}

只输出一个 JSON 对象,不要其它任何文字:
{{"score": <0到1之间的小数>, "pass": <true 或 false>, "reason": "<简短理由>"}}
""",
}
```

Update `LLMJudge`:

```python
class LLMJudge:
    name = "llm_judge"

    def __init__(self, provider, criteria: str = "", language: str = "en"):
        self.provider = provider
        self.criteria = criteria
        self.language = language if language in JUDGE_PROMPTS else "en"

    def score(self, case: TestCase, output: str) -> Score:
        prompt = JUDGE_PROMPTS[self.language].format(
            input=case.input, output=output, criteria=self.criteria or "overall quality")
        resp = self.provider.complete(prompt, temperature=0.0)
        ...  # unchanged parse logic
```

(Keep the existing `_extract_json` and try/except parse body unchanged.)

- [ ] **Step 4: Wire language through `build_scorer`** in `scorers/rules.py`:

```python
    if cfg.type == "llm_judge":
        from .llm_judge import LLMJudge
        return LLMJudge(provider=judge_provider, criteria=cfg.params.get("criteria", ""),
                        language=cfg.params.get("language", "en"))
```

- [ ] **Step 5: Run, expect pass.** `python3 -m pytest tests/test_scorers_llm_judge.py -q`.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(judge): Chinese llm_judge template via language param"`

### Task 11: 国产模型 presets + `mingjing models`

**Files:** Create `src/mingjing/presets.py`, `tests/test_presets.py`; Modify `providers/__init__.py`, `cli.py`.

- [ ] **Step 1: Verify litellm model IDs** (impl-time check, no commit): run
  `python3 -c "import litellm; print([m for m in litellm.model_cost if any(k in m for k in ['deepseek','qwen','glm','moonshot','doubao','volcengine','dashscope'])][:40])"`
  Only ship aliases whose litellm ids appear (or are documented current). DeepSeek is verified (`deepseek/deepseek-chat`).

- [ ] **Step 2: Failing test** — create `tests/test_presets.py`:

```python
def test_resolve_known_alias_and_passthrough():
    from mingjing.presets import resolve_model, CHINA_MODELS
    assert resolve_model("deepseek-chat") == "deepseek/deepseek-chat"
    assert resolve_model("deepseek/deepseek-chat") == "deepseek/deepseek-chat"  # already-qualified
    assert resolve_model("echo") == "echo"                                      # passthrough
    assert resolve_model("gpt-4o-mini") == "gpt-4o-mini"
    for name, info in CHINA_MODELS.items():
        assert "litellm" in info and "env" in info


def test_get_provider_resolves_alias(monkeypatch):
    # alias resolves to a litellm id -> LiteLLMProvider with the resolved model (no network)
    from mingjing.providers import get_provider
    p = get_provider("deepseek-chat")
    assert p.model == "deepseek/deepseek-chat"
```

- [ ] **Step 3: Run, expect fail.**

- [ ] **Step 4: Implement `presets.py`** (ship only verified ids; structure allows easy extension):

```python
from __future__ import annotations

# Curated aliases for first-class 国产 + common global models.
# value: {"litellm": <litellm model id>, "env": <required API key env var>, "note": <str>}
CHINA_MODELS: dict[str, dict] = {
    "deepseek-chat":     {"litellm": "deepseek/deepseek-chat",     "env": "DEEPSEEK_API_KEY", "note": "DeepSeek V3 chat"},
    "deepseek-reasoner": {"litellm": "deepseek/deepseek-reasoner", "env": "DEEPSEEK_API_KEY", "note": "DeepSeek R1 reasoner"},
}

_ALIASES = {name: info["litellm"] for name, info in CHINA_MODELS.items()}


def resolve_model(name: str) -> str:
    """Map a friendly alias to its litellm id; pass through anything else unchanged."""
    return _ALIASES.get(name, name)
```

(At Step 1's verification, append any confirmed entries — e.g. moonshot/qwen — to `CHINA_MODELS`.)

- [ ] **Step 5: Resolve aliases in `get_provider`** — `providers/__init__.py`:

```python
def get_provider(model: str) -> Provider:
    from .presets import resolve_model
    model = resolve_model(model)
    if model == "echo" or model.startswith("echo:"):
        fixed = model.split(":", 1)[1] if ":" in model else None
        return EchoProvider(fixed=fixed)
    from .litellm_provider import LiteLLMProvider
    return LiteLLMProvider(model=model)
```

- [ ] **Step 6: Add `mingjing models` command** in `cli.py`:

```python
@app.command("models")
def list_models() -> None:
    """List first-class model aliases (国产 + global) and their API key env vars."""
    from .presets import CHINA_MODELS
    for alias, info in CHINA_MODELS.items():
        typer.echo(f"{alias:<20} -> {info['litellm']:<28} (env: {info['env']})  {info['note']}")
```

- [ ] **Step 7: Failing CLI test** — append to `tests/test_presets.py`:

```python
def test_cli_models_lists_deepseek():
    from typer.testing import CliRunner
    from mingjing.cli import app
    res = CliRunner().invoke(app, ["models"])
    assert res.exit_code == 0
    assert "deepseek-chat" in res.stdout
```

- [ ] **Step 8: Run full suite, expect pass.** `python3 -m pytest -q`.

- [ ] **Step 9: Commit** — `git add -A && git commit -m "feat(models): 国产模型 aliases + mingjing models command"`

---

## Part F — Docs, examples, final verification

### Task 12: Documentation + examples + full verification

**Files:** Modify `README.md`, `README.zh-CN.md`; add a concurrency-aware DeepSeek example note; verify everything.

- [ ] **Step 1: Update `README.md`** — add sections:
  - **CI gating:** `run --fail-under`, `diff --fail-on-regression`, the composite GitHub Action (`uses: dominciyue/Evalith_MingJing@main` with `config`/`fail-under`), and a copy-paste workflow snippet.
  - **Reports:** `mingjing report <id> --format md|html --output report.html`; `mingjing diff ... --format md`.
  - **Scale:** `--concurrency`, JSONL datasets (`examples/qa.jsonl`).
  - **Cost:** note that reports show cost/tokens/latency (real numbers when using litellm models).
  - **国产模型:** `mingjing models`, alias usage (`model: deepseek-chat`), Chinese judge (`params: {language: zh}`).
  - Bump **Status** to v0.2.

- [ ] **Step 2: Mirror all changes into `README.zh-CN.md`.**

- [ ] **Step 3: Add llm_judge `language: zh` to `examples/eval.deepseek.yaml`** and bump its comment to mention `--concurrency`.

- [ ] **Step 4: Full offline verification (no key, no network):**
  - `python3 -m pytest -q` → all pass, 0 warnings.
  - `mingjing run examples/eval.yaml` → 2/2 checks passed.
  - `mingjing report <that run id> --format html --output /tmp/r.html` → file written; open-check it has `<table>`.
  - `mingjing models` → lists deepseek aliases.
  - `mingjing run examples/eval.yaml --fail-under 1.0` → exit 0; with a deliberately-failing needle → exit 1.

- [ ] **Step 5: Optional real-model re-smoke** (only if `DEEPSEEK_API_KEY` is present in env):
  - `mingjing run examples/eval.deepseek.yaml --concurrency 3` → 6/6; `mingjing report <id> --format md` shows non-zero cost/tokens.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "docs: document v0.2 (CI gating, reports, scale, cost, 国产模型)"`

---

## Self-Review

**Spec coverage:**
1. CI + GitHub Action + `--fail-on-regression` → Tasks 1, 2, 3. ✅
2. Readable md/HTML run & diff reports → Tasks 5, 6, 7. ✅
3. Concurrency + dataset scale (CSV existed; JSONL added) → Tasks 8, 9. ✅
4. Cost/latency/token tracking → Task 4 (+ surfaced in reports Tasks 5/7, README Task 12). ✅
5. Chinese llm_judge + 国产模型 presets → Tasks 10, 11. ✅

**Network/dependency safety:** All tests use Echo/Fake providers and pure functions; no test imports litellm at module load or hits the network. litellm stays lazy/optional.

**Type consistency:** `Response`/`CaseResult` token/cost field names identical; `run_eval(..., concurrency=None)` matches CLI call; `report` function names match CLI lazy imports; `diff_to_html`/`run_to_html` stubbed in Tasks 5/6 then fleshed in Task 7 so the suite stays green between tasks.

**Ordering note:** Features built foundation-first (CI → cost → reports → scale → 国产模型) rather than literal 1–5, because reports (Feature 2) surface cost (Feature 4). All five ship.
