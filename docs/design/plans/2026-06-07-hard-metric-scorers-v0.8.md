# Hard-Metric Scorers (v0.8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two deterministic, non-LLM scorers — `code_exec` (HumanEval-style unit-test execution) and `numeric_match` (tolerance numeric compare) — giving Evalith ground-truth scoring for the code/math domains where LLM judges disagree most.

**Architecture:** Two new modules under `src/evalith/scorers/`: `sandbox.py` runs untrusted Python in an isolated subprocess (resource limits injected in-process — thread-safe, no `preexec_fn`), `hard.py` holds `extract_code`, `CodeExec`, `NumericMatch`. Both scorers satisfy the existing `Scorer` protocol (`score(case, output) -> Score`), so the engine, report and diff need **zero changes**. `code_exec` is gated behind `EVALITH_ALLOW_CODE_EXEC=1`.

**Tech Stack:** Python 3, pydantic v2 models, `subprocess` + `resource` (Linux RLIMIT), pytest. HumanEval data via the `datasets` library (already installed).

**Spec:** `docs/design/specs/2026-06-07-hard-metric-scorers-design.md`

---

## File Structure

- Create `src/evalith/scorers/sandbox.py` — `run_program(source, *, timeout, memory_mb) -> (bool, str)`. Single responsibility: execute a Python source string in a locked-down subprocess, return pass/fail + reason.
- Create `src/evalith/scorers/hard.py` — `extract_code`, `CodeExec`, `NumericMatch`. The two hard-metric scorers + the code-extraction helper.
- Modify `src/evalith/scorers/rules.py` — add `code_exec` / `numeric_match` branches to `build_scorer` (with the opt-in gate). Keep the simple string scorers where they are.
- Create `tests/test_scorers_sandbox.py`, `tests/test_scorers_hard.py`.
- Modify `tests/test_scorers_rules.py` (gate), `tests/test_engine.py` (end-to-end).
- Create `docs/blog/article4/build_code_exec_dataset.py` (builder) → generates `examples/code.humaneval.yaml`.
- Create `docs/blog/article4/configs/eval.code-exec-accept.yaml`.
- Modify `pyproject.toml`, `README.md`, `README.zh-CN.md`, `examples/eval.code-exec.yaml`.

**Why limits are injected in-process, not via `preexec_fn`:** the engine scores cases on a `ThreadPoolExecutor`. CPython's docs warn that `preexec_fn` can deadlock in multithreaded programs. Setting `resource.setrlimit` at the top of the injected program (it runs *inside* the child, before user code) is thread-safe and achieves the same bound.

---

### Task 1: `extract_code` helper

**Files:**
- Create: `src/evalith/scorers/hard.py`
- Test: `tests/test_scorers_hard.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scorers_hard.py
from evalith.scorers.hard import extract_code


def test_extract_fenced_python():
    out = "Here you go:\n```python\ndef f():\n    return 1\n```\nDone"
    assert extract_code(out) == "def f():\n    return 1"


def test_extract_bare_fence():
    out = "```\nx = 2\n```"
    assert extract_code(out) == "x = 2"


def test_extract_no_fence_returns_whole():
    assert extract_code("def f(): return 1") == "def f(): return 1"


def test_extract_first_block_when_multiple():
    out = "```python\na = 1\n```\nthen\n```python\nb = 2\n```"
    assert extract_code(out) == "a = 1"


def test_extract_empty_returns_none():
    assert extract_code("   \n  ") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_scorers_hard.py -v`
Expected: FAIL — `ModuleNotFoundError` / `cannot import name 'extract_code'`.

- [ ] **Step 3: Implement `extract_code`**

```python
# src/evalith/scorers/hard.py
from __future__ import annotations

import re

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
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_scorers_hard.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evalith/scorers/hard.py tests/test_scorers_hard.py
git commit -m "feat(scorers): extract_code helper for hard-metric scorers"
```

---

### Task 2: `sandbox.run_program`

**Files:**
- Create: `src/evalith/scorers/sandbox.py`
- Test: `tests/test_scorers_sandbox.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scorers_sandbox.py
from evalith.scorers.sandbox import run_program


def test_passing_program():
    ok, detail = run_program("assert 1 + 1 == 2")
    assert ok is True
    assert detail == "ok"


def test_failing_assert():
    ok, detail = run_program("assert 1 == 2")
    assert ok is False
    assert "AssertionError" in detail or "assert" in detail.lower()


def test_timeout():
    ok, detail = run_program("while True:\n    pass", timeout=1)
    assert ok is False
    assert "timeout" in detail.lower()


def test_memory_limit():
    ok, detail = run_program("x = [0] * (10 ** 9)", memory_mb=128)
    assert ok is False


def test_dangerous_call_blocked():
    ok, detail = run_program("import os\nos.system('echo hi')")
    assert ok is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_scorers_sandbox.py -v`
Expected: FAIL — `cannot import name 'run_program'`.

- [ ] **Step 3: Implement the sandbox**

```python
# src/evalith/scorers/sandbox.py
from __future__ import annotations

import subprocess
import sys
import tempfile

# Injected at the top of every program. Runs INSIDE the child before user
# code: sets resource limits (thread-safe, unlike preexec_fn) and neuters the
# most dangerous calls (defense-in-depth on top of the subprocess boundary).
_PREAMBLE = """\
import resource as _r
for _name, _lim in [("RLIMIT_AS", {mem}), ("RLIMIT_CPU", {cpu}), ("RLIMIT_FSIZE", {fsize})]:
    try:
        _r.setrlimit(getattr(_r, _name), (_lim, _lim))
    except (ValueError, OSError):
        pass
import os, shutil
for _m, _a in [(os, "system"), (os, "remove"), (os, "unlink"), (os, "rmdir"),
               (os, "removedirs"), (os, "kill"), (os, "killpg"),
               (shutil, "rmtree"), (shutil, "move")]:
    try:
        setattr(_m, _a, None)
    except Exception:
        pass
try:
    import subprocess as _sp
    _sp.Popen = _sp.run = _sp.call = None
except Exception:
    pass
"""


def run_program(source: str, *, timeout: float = 5, memory_mb: int = 256) -> tuple[bool, str]:
    """Run `source` in an isolated subprocess.

    Returns (passed, detail). Exit 0 -> (True, "ok"); timeout / nonzero exit /
    spawn failure -> (False, reason). Never raises.
    """
    preamble = _PREAMBLE.format(
        mem=memory_mb * 1024 * 1024, cpu=int(timeout) + 1, fsize=1024 * 1024
    )
    program = preamble + "\n" + source
    try:
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", program],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env={"PATH": "/usr/bin:/bin"},
            )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout:g}s"
    except Exception as e:  # spawn failure, etc. — stay resilient
        return False, f"sandbox error: {e}"
    if proc.returncode == 0:
        return True, "ok"
    lines = (proc.stderr or proc.stdout or "").strip().splitlines()
    tail = " ".join(lines[-3:]) if lines else f"exit {proc.returncode}"
    return False, tail[:300]
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_scorers_sandbox.py -v`
Expected: 5 passed. (The timeout test takes ~1s; the rest are fast.)

- [ ] **Step 5: Commit**

```bash
git add src/evalith/scorers/sandbox.py tests/test_scorers_sandbox.py
git commit -m "feat(scorers): isolated subprocess sandbox for code execution"
```

---

### Task 3: `CodeExec` scorer

**Files:**
- Modify: `src/evalith/scorers/hard.py`
- Test: `tests/test_scorers_hard.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_scorers_hard.py
from evalith.models import TestCase
from evalith.scorers.hard import CodeExec

_TEST = "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n"


def _case(**meta):
    return TestCase(id="c1", input="add a and b", metadata=meta)


def test_code_exec_passes_correct_solution():
    case = _case(entry_point="add", test=_TEST)
    out = "```python\ndef add(a, b):\n    return a + b\n```"
    score = CodeExec().score(case, out)
    assert score.passed is True
    assert score.value == 1.0


def test_code_exec_fails_wrong_solution():
    case = _case(entry_point="add", test=_TEST)
    out = "```python\ndef add(a, b):\n    return a - b\n```"
    score = CodeExec().score(case, out)
    assert score.passed is False
    assert "assert" in score.detail.lower() or "AssertionError" in score.detail


def test_code_exec_missing_metadata():
    score = CodeExec().score(_case(), "def add(a, b): return a + b")
    assert score.passed is False
    assert "missing" in score.detail


def test_code_exec_no_code_in_output():
    case = _case(entry_point="add", test=_TEST)
    score = CodeExec().score(case, "   ")
    assert score.passed is False
    assert "no code" in score.detail
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_scorers_hard.py -k code_exec -v`
Expected: FAIL — `cannot import name 'CodeExec'`.

- [ ] **Step 3: Implement `CodeExec`**

```python
# append to src/evalith/scorers/hard.py
from ..models import Score, TestCase
from .sandbox import run_program


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
```

Note: keep the `from __future__ import annotations` and the `import re` / `extract_code` from Task 1 at the top of the file; add these imports below them.

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_scorers_hard.py -k code_exec -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evalith/scorers/hard.py tests/test_scorers_hard.py
git commit -m "feat(scorers): code_exec HumanEval-style execution scorer"
```

---

### Task 4: `NumericMatch` scorer

**Files:**
- Modify: `src/evalith/scorers/hard.py`
- Test: `tests/test_scorers_hard.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_scorers_hard.py
from evalith.scorers.hard import NumericMatch


def _ncase(expected):
    return TestCase(id="n1", input="compute", expected=expected)


def test_numeric_exact():
    s = NumericMatch().score(_ncase("42"), "the answer is 42")
    assert s.passed is True


def test_numeric_within_tol():
    s = NumericMatch(rel_tol=1e-3).score(_ncase("3.14159"), "result is 3.1416")
    assert s.passed is True


def test_numeric_outside_tol():
    s = NumericMatch(rel_tol=1e-3).score(_ncase("3.14159"), "about 3.0")
    assert s.passed is False


def test_numeric_no_number():
    s = NumericMatch().score(_ncase("42"), "no digits here")
    assert s.passed is False
    assert "no number" in s.detail


def test_numeric_expected_not_numeric():
    s = NumericMatch().score(_ncase("cat"), "42")
    assert s.passed is False
    assert "not numeric" in s.detail
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_scorers_hard.py -k numeric -v`
Expected: FAIL — `cannot import name 'NumericMatch'`.

- [ ] **Step 3: Implement `NumericMatch`**

```python
# append to src/evalith/scorers/hard.py
import math

_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _last_number(text: str) -> float | None:
    for tok in reversed(_NUM.findall(text)):
        try:
            return float(tok)
        except ValueError:
            continue
    return None


class NumericMatch:
    name = "numeric_match"

    def __init__(self, rel_tol: float = 1e-3, abs_tol: float = 0.0):
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def score(self, case: TestCase, output: str) -> Score:
        raw = (case.expected or "").strip()
        try:
            target = float(raw)
        except (ValueError, TypeError):
            return Score(scorer=self.name, value=0.0, passed=False,
                         detail=f"expected not numeric: {raw!r}")
        got = _last_number(output)
        if got is None:
            return Score(scorer=self.name, value=0.0, passed=False,
                         detail="no number in output")
        ok = math.isclose(got, target, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok,
                     detail=f"got={got} expected={target}")
```

Note: `import re` already exists at the top from Task 1 — do not duplicate it. Put `import math` with it.

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_scorers_hard.py -k numeric -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evalith/scorers/hard.py tests/test_scorers_hard.py
git commit -m "feat(scorers): numeric_match tolerance scorer"
```

---

### Task 5: `build_scorer` wiring + opt-in gate

**Files:**
- Modify: `src/evalith/scorers/rules.py:45-65` (the `build_scorer` function)
- Test: `tests/test_scorers_rules.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_scorers_rules.py
import pytest

from evalith.config import ScorerConfig
from evalith.scorers.rules import build_scorer


def test_code_exec_gate_blocks_without_env(monkeypatch):
    monkeypatch.delenv("EVALITH_ALLOW_CODE_EXEC", raising=False)
    with pytest.raises(ValueError, match="EVALITH_ALLOW_CODE_EXEC"):
        build_scorer(ScorerConfig(type="code_exec", params={}))


def test_code_exec_builds_with_env(monkeypatch):
    monkeypatch.setenv("EVALITH_ALLOW_CODE_EXEC", "1")
    scorer = build_scorer(ScorerConfig(type="code_exec", params={"timeout": 3}))
    assert scorer.name == "code_exec"
    assert scorer.timeout == 3


def test_numeric_match_builds():
    scorer = build_scorer(ScorerConfig(type="numeric_match", params={"rel_tol": 1e-2}))
    assert scorer.name == "numeric_match"
    assert scorer.rel_tol == 1e-2
```

Note: confirm `ScorerConfig`'s import path and that it accepts `type=` / `params=` — check the top of `src/evalith/scorers/rules.py` (it already imports `from ..config import ScorerConfig`).

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_scorers_rules.py -k "code_exec or numeric" -v`
Expected: FAIL — `Unknown scorer type: code_exec`.

- [ ] **Step 3: Add the branches to `build_scorer`**

Insert these two branches immediately before the final `raise ValueError(...)` line in `build_scorer` (after the `llm_judge` branch):

```python
    if cfg.type == "code_exec":
        import os

        from .hard import CodeExec

        if os.environ.get("EVALITH_ALLOW_CODE_EXEC") != "1":
            raise ValueError(
                "code_exec runs untrusted model code; "
                "set EVALITH_ALLOW_CODE_EXEC=1 to enable")
        return CodeExec(timeout=cfg.params.get("timeout", 5),
                        memory_mb=cfg.params.get("memory_mb", 256))
    if cfg.type == "numeric_match":
        from .hard import NumericMatch

        return NumericMatch(rel_tol=cfg.params.get("rel_tol", 1e-3),
                            abs_tol=cfg.params.get("abs_tol", 0.0))
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_scorers_rules.py -v`
Expected: all pass (new 3 + existing).

- [ ] **Step 5: Commit**

```bash
git add src/evalith/scorers/rules.py tests/test_scorers_rules.py
git commit -m "feat(scorers): wire code_exec/numeric_match into build_scorer with opt-in gate"
```

---

### Task 6: Engine end-to-end + full regression

No engine change — this task proves the new scorers flow through `run_eval` untouched and that nothing regressed.

**Files:**
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write the failing test**

`run_eval` signature is `run_eval(config, provider, judge_provider=None, concurrency=None)`
and it loads the dataset from `config.dataset` (a path). So we write a dataset YAML to
`tmp_path`, point `config.dataset` at it, and pass an `EchoProvider` whose fixed reply is a
correct solution. Mirror the existing tests at the top of `tests/test_engine.py`.

```python
# append to tests/test_engine.py
def test_code_exec_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALITH_ALLOW_CODE_EXEC", "1")
    from evalith.config import EvalConfig, ScorerConfig
    from evalith.engine import run_eval
    from evalith.providers.base import EchoProvider

    ds = tmp_path / "ds.yaml"
    ds.write_text(
        "name: d\n"
        "cases:\n"
        "  - id: add\n"
        "    input: write add\n"
        "    metadata:\n"
        "      entry_point: add\n"
        "      test: |\n"
        "        def check(candidate):\n"
        "            assert candidate(2, 3) == 5\n",
        encoding="utf-8",
    )
    cfg = EvalConfig(
        name="e2e", dataset=str(ds), model="echo", prompt_template="{{input}}",
        scorers=[ScorerConfig(type="code_exec", params={})],
    )
    run = run_eval(cfg, EchoProvider(fixed="def add(a, b): return a + b"))
    assert run.results[0].scores[0].passed is True
```

- [ ] **Step 2: Run to verify it passes (no production change needed)**

Run: `python3 -m pytest tests/test_engine.py::test_code_exec_end_to_end -v`
Expected: PASS. If it errors, fix the **test** to match the real API — do **not** change the engine.

- [ ] **Step 3: Full regression**

Run: `python3 -m pytest -q`
Expected: all previous tests (112) + the new hard-metric/sandbox/rules/engine tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_engine.py
git commit -m "test(engine): code_exec end-to-end via echo provider"
```

---

### Task 7: Acceptance dataset (real HumanEval) + config

**Files:**
- Create: `docs/blog/article4/build_code_exec_dataset.py`
- Create (generated): `examples/code.humaneval.yaml`
- Create: `docs/blog/article4/configs/eval.code-exec-accept.yaml`

- [ ] **Step 1: Write the dataset builder**

```python
# docs/blog/article4/build_code_exec_dataset.py
"""Build examples/code.humaneval.yaml from the canonical HumanEval set.

Reuses the same problem ids as article 4's he-* cases so the execution
ground truth can be contrasted directly against the judge panel.

Run: python3 docs/blog/article4/build_code_exec_dataset.py
"""
from __future__ import annotations

from pathlib import Path

import yaml
from datasets import load_dataset

IDS = [151, 28, 163, 108, 62, 70]
PROMPT = ("补全下面的 Python 函数。只输出完整的函数体(从 def 开始,可读、能跑通)。"
          "\n\n```python\n{sig}```")

OUT = Path(__file__).resolve().parents[2] / "examples" / "code.humaneval.yaml"


def main() -> None:
    rows = {r["task_id"]: r for r in load_dataset("openai_humaneval", split="test")}
    cases = []
    for n in IDS:
        r = rows[f"HumanEval/{n}"]
        cases.append({
            "id": f"he-humaneval-{n}",
            "source": "HumanEval",
            "domain": "code",
            "input": PROMPT.format(sig=r["prompt"]),
            "metadata": {"entry_point": r["entry_point"], "test": r["test"]},
        })
    data = {"name": "code-humaneval-exec-v1", "cases": cases}
    OUT.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    print(f"wrote {OUT} with {len(cases)} cases")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the dataset and sanity-check it loads**

```bash
python3 docs/blog/article4/build_code_exec_dataset.py
python3 -c "from evalith.dataset import load_dataset as L; d=L('examples/code.humaneval.yaml'); \
print(len(d.cases), d.cases[0].metadata['entry_point'], bool(d.cases[0].metadata['test']))"
```
Expected: `wrote .../examples/code.humaneval.yaml with 6 cases`, then `6 double_the_difference True`.

- [ ] **Step 3: Write the acceptance config**

```yaml
# docs/blog/article4/configs/eval.code-exec-accept.yaml
# v0.8 acceptance: execution ground truth on article-4's HumanEval problems.
# Run (executes untrusted model code — opt in):
#   EVALITH_ALLOW_CODE_EXEC=1 DEEPSEEK_API_KEY=... \
#     evalith run docs/blog/article4/configs/eval.code-exec-accept.yaml
name: code-exec-accept-v08
dataset: examples/code.humaneval.yaml
model: deepseek-chat
prompt_template: "{{input}}"
temperature: 0.0
samples: 1
concurrency: 4
scorers:
  - type: code_exec
    params:
      timeout: 5
      memory_mb: 256
```

- [ ] **Step 4: Offline smoke (echo canonical solution through one case)**

```bash
python3 - <<'PY'
import os
os.environ["EVALITH_ALLOW_CODE_EXEC"] = "1"
from evalith.dataset import load_dataset
from evalith.scorers.hard import CodeExec
from datasets import load_dataset as hf
rows = {r["task_id"]: r for r in hf("openai_humaneval", split="test")}
d = load_dataset("examples/code.humaneval.yaml")
c = d.cases[0]
sol = rows["HumanEval/151"]
full = sol["prompt"] + sol["canonical_solution"]
print("canonical passes:", CodeExec().score(c, full).passed)
PY
```
Expected: `canonical passes: True`.

- [ ] **Step 5: Commit**

```bash
git add docs/blog/article4/build_code_exec_dataset.py examples/code.humaneval.yaml \
        docs/blog/article4/configs/eval.code-exec-accept.yaml
git commit -m "test(accept): v0.8 code_exec acceptance dataset + config from HumanEval"
```

---

### Task 8: Version bump, example config, README, final regression

**Files:**
- Modify: `pyproject.toml`
- Create: `examples/eval.code-exec.yaml`
- Modify: `README.md`, `README.zh-CN.md`

- [ ] **Step 1: Bump version**

In `pyproject.toml`, change `version = "0.7.0"` to `version = "0.8.0"`.

- [ ] **Step 2: Add a runnable example config**

```yaml
# examples/eval.code-exec.yaml
# Deterministic code grading via unit-test execution (HumanEval-style).
# Executes untrusted model code in a locked-down subprocess — opt in with
# EVALITH_ALLOW_CODE_EXEC=1. Also shows numeric_match for math answers.
name: code-exec-demo
dataset: examples/code.humaneval.yaml
model: deepseek-chat
prompt_template: "{{input}}"
temperature: 0.0
samples: 1
scorers:
  - type: code_exec
    params:
      timeout: 5
      memory_mb: 256
```

- [ ] **Step 3: Add README sections**

In `README.md`, under the existing "What's new" area, add:

```markdown
### What's new in v0.8

- **Hard-metric scorers** — deterministic, non-LLM grading where judges disagree most:
  - `code_exec` runs the model's code against HumanEval-style unit tests in a
    locked-down subprocess (resource limits + dangerous-call guard); opt in with
    `EVALITH_ALLOW_CODE_EXEC=1`.
  - `numeric_match` compares an extracted number to `expected` with `rel_tol` / `abs_tol`.
- Closes the loop on v0.7's consensus panel: the panel *detects* judge
  disagreement, `code_exec` gives the *ground truth* for code.
```

In `README.zh-CN.md`, add the mirror:

```markdown
### v0.8 新功能

- **硬指标 scorer** —— 在 judge 最易分歧处给确定性、非 LLM 判分:
  - `code_exec` 把模型代码放进受限子进程(资源限制 + 危险调用拦截)跑 HumanEval
    式单测;需 `EVALITH_ALLOW_CODE_EXEC=1` 显式开启。
  - `numeric_match` 用 `rel_tol` / `abs_tol` 容差比对输出里的数值与 `expected`。
- 与 v0.7 共识面板形成闭环:面板*检测*分歧,`code_exec` 给出代码的*ground truth*。
```

- [ ] **Step 4: Final full regression + version check**

Run:
```bash
python3 -m pytest -q
python3 -c "import evalith; print(evalith.__version__)" 2>/dev/null || grep '^version' pyproject.toml
```
Expected: all tests pass; version reads 0.8.0.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml examples/eval.code-exec.yaml README.md README.zh-CN.md
git commit -m "chore: v0.8.0 — hard-metric scorers example config + README"
```

---

## Self-Review

**Spec coverage:**
- code_exec contract (extract → assemble → exec → judge) → Tasks 1, 2, 3 ✓
- subprocess + timeout + RLIMIT + reliability_guard → Task 2 (limits in-process, not preexec_fn) ✓
- opt-in `EVALITH_ALLOW_CODE_EXEC=1` gate → Task 5 ✓
- test/entry_point in metadata → Tasks 3, 7 ✓
- numeric_match with tolerance → Task 4 ✓
- engine/report/diff zero-change → Task 6 (no production edit) ✓
- acceptance dataset from same HumanEval ids → Task 7 ✓
- version/example/README → Task 8 ✓
- All 5 acceptance criteria in the spec are exercised: gate (Task 5), canonical pass / wrong fail (Tasks 3, 7), timeout+memory (Task 2), numeric fail/pass (Task 4), regression (Tasks 6, 8). ✓

**Placeholder scan:** No TBD/TODO; every code step has full code. Tasks 6 & 7 note "verify the real signature" — these are explicit instructions to match existing APIs, with the fallback (mirror existing tests / don't change the engine) stated, not vague hand-waving.

**Type consistency:** `extract_code(output) -> str | None`, `run_program(source, *, timeout, memory_mb) -> (bool, str)`, `CodeExec(timeout, memory_mb)`, `NumericMatch(rel_tol, abs_tol)` — names/signatures used identically across Tasks 1–8. `Score(scorer, value, passed, detail)` matches the existing model. Metadata keys `entry_point` / `test` consistent in Tasks 3, 6, 7.
