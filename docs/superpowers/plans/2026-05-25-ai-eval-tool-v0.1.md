# AI Eval/Regression Tool v0.1 (明镜 / Evalith) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless, local-first Python tool that runs an eval over a dataset against any LLM, scores each case, stores the run, and diffs two runs to surface regressions ("did this change make it better or worse?").

**Architecture:** A small library + CLI. A pluggable **Provider** layer (国产 + 海外 models via LiteLLM, plus offline `echo`/`fake` providers for tests) feeds an **Engine** that runs each `TestCase` through configurable **Scorers** (rule-based + LLM-as-judge) into a `Run`. A file-based **RunStore** persists runs as JSON; a pure-function **Diff** compares two runs. A Typer **CLI** wires `run` / `diff` / `list`. Every unit has one responsibility and is tested in isolation with no network.

**Tech Stack:** Python ≥3.11, pydantic v2 (models), PyYAML (config/datasets), LiteLLM (model access), Typer + Rich (CLI), pytest (tests), Ruff (lint). Build via Hatchling, src layout.

---

## File Structure

```
pyproject.toml                         # project metadata, deps, script entry, pytest config
src/mingjing/__init__.py               # package version
src/mingjing/models.py                 # pydantic models: TestCase, Dataset, Score, CaseResult, Run
src/mingjing/config.py                 # ScorerConfig, EvalConfig, load_config()
src/mingjing/dataset.py                # load_dataset(): yaml/json/csv -> Dataset
src/mingjing/providers/__init__.py     # get_provider() factory + re-exports
src/mingjing/providers/base.py         # Provider protocol, Response, FakeProvider, EchoProvider
src/mingjing/providers/litellm_provider.py  # LiteLLMProvider (real models)
src/mingjing/scorers/__init__.py       # (empty package marker)
src/mingjing/scorers/base.py           # Scorer protocol
src/mingjing/scorers/rules.py          # ExactMatch, Contains, Regex + build_scorer()
src/mingjing/scorers/llm_judge.py      # LLMJudge scorer
src/mingjing/engine.py                 # run_eval(config, provider) -> Run
src/mingjing/store.py                  # RunStore: save/load/list runs (JSON files)
src/mingjing/diff.py                   # CaseDiff, DiffReport, diff_runs()
src/mingjing/cli.py                    # Typer app: run / diff / list + main()
examples/dataset.yaml                  # sample dataset
examples/eval.yaml                     # sample eval config
tests/test_smoke.py
tests/test_models.py
tests/test_config.py
tests/test_dataset.py
tests/test_providers.py
tests/test_scorers_rules.py
tests/test_scorers_llm_judge.py
tests/test_engine.py
tests/test_store.py
tests/test_diff.py
tests/test_cli.py
README.md
```

Each `src/mingjing/*` file owns exactly one concern; files that change together (a unit + its tests) are added in the same task.

---

## Task 1: Project scaffold + smoke test

**Files:**
- Create: `pyproject.toml`
- Create: `src/mingjing/__init__.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
import mingjing


def test_version():
    assert mingjing.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mingjing'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[project]
name = "mingjing"
version = "0.1.0"
description = "Neutral AI regression-testing tool (明镜 / Evalith)"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.6",
    "pyyaml>=6.0",
    "typer>=0.12",
    "rich>=13.7",
    "litellm>=1.40",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.5"]

[project.scripts]
mingjing = "mingjing.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mingjing"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```python
# src/mingjing/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 4: Install and run the test to verify it passes**

Run:
```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest tests/test_smoke.py -q
```
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/mingjing/__init__.py tests/test_smoke.py
git commit -m "chore: scaffold mingjing package with smoke test"
```

---

## Task 2: Core data models

**Files:**
- Create: `src/mingjing/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from datetime import datetime, timezone

from mingjing.models import CaseResult, Run, Score


def test_case_result_mean_score():
    cr = CaseResult(
        case_id="1", input="q", output="a",
        scores=[Score(scorer="x", value=1.0, passed=True),
                Score(scorer="y", value=0.0, passed=False)],
    )
    assert cr.mean_score == 0.5


def test_case_result_mean_score_empty_is_zero():
    assert CaseResult(case_id="1", input="q", output="a").mean_score == 0.0


def test_run_round_trip():
    run = Run(
        id="abc", name="t", created_at=datetime(2026, 5, 25, tzinfo=timezone.utc),
        model="echo", results=[CaseResult(case_id="1", input="q", output="a")],
    )
    back = Run.model_validate_json(run.model_dump_json())
    assert back.id == "abc"
    assert back.results[0].case_id == "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mingjing.models'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mingjing/models.py
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    id: str
    input: str
    expected: str | None = None
    metadata: dict = Field(default_factory=dict)


class Dataset(BaseModel):
    name: str
    cases: list[TestCase]


class Score(BaseModel):
    scorer: str
    value: float
    passed: bool
    detail: str = ""


class CaseResult(BaseModel):
    case_id: str
    input: str
    output: str
    scores: list[Score] = Field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def mean_score(self) -> float:
        return sum(s.value for s in self.scores) / len(self.scores) if self.scores else 0.0


class Run(BaseModel):
    id: str
    name: str
    created_at: datetime
    model: str
    results: list[CaseResult] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/mingjing/models.py tests/test_models.py
git commit -m "feat: add core pydantic models"
```

---

## Task 3: Eval config

**Files:**
- Create: `src/mingjing/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from mingjing.config import load_config


def test_load_config(tmp_path):
    p = tmp_path / "eval.yaml"
    p.write_text(
        "name: demo\n"
        "dataset: ds.yaml\n"
        "model: echo\n"
        "prompt_template: 'Q: {{input}}'\n"
        "scorers:\n"
        "  - type: contains\n"
        "    params: {text: hi}\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.name == "demo"
    assert cfg.model == "echo"
    assert cfg.prompt_template == "Q: {{input}}"
    assert cfg.scorers[0].type == "contains"
    assert cfg.scorers[0].params["text"] == "hi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mingjing.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mingjing/config.py
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ScorerConfig(BaseModel):
    type: str
    params: dict = Field(default_factory=dict)


class EvalConfig(BaseModel):
    name: str
    dataset: str
    model: str
    prompt_template: str = "{{input}}"
    system: str | None = None
    temperature: float = 0.0
    scorers: list[ScorerConfig] = Field(default_factory=list)


def load_config(path: str | Path) -> EvalConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return EvalConfig.model_validate(data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/mingjing/config.py tests/test_config.py
git commit -m "feat: add eval config model and loader"
```

---

## Task 4: Dataset loader

**Files:**
- Create: `src/mingjing/dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
import pytest

from mingjing.dataset import load_dataset


def test_load_yaml_dataset(tmp_path):
    p = tmp_path / "ds.yaml"
    p.write_text(
        "name: d\ncases:\n  - id: '1'\n    input: hello\n    expected: hi\n",
        encoding="utf-8",
    )
    ds = load_dataset(p)
    assert ds.name == "d"
    assert ds.cases[0].input == "hello"
    assert ds.cases[0].expected == "hi"


def test_load_csv_dataset(tmp_path):
    p = tmp_path / "ds.csv"
    p.write_text("id,input,expected\n1,hello,hi\n", encoding="utf-8")
    ds = load_dataset(p)
    assert ds.cases[0].id == "1"
    assert ds.cases[0].expected == "hi"


def test_unsupported_format_raises(tmp_path):
    p = tmp_path / "ds.txt"
    p.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mingjing.dataset'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mingjing/dataset.py
from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from .models import Dataset, TestCase


def load_dataset(path: str | Path) -> Dataset:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return Dataset.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
    if suffix == ".json":
        return Dataset.model_validate(json.loads(p.read_text(encoding="utf-8")))
    if suffix == ".csv":
        with p.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        cases = [
            TestCase(
                id=row.get("id") or str(i),
                input=row["input"],
                expected=row.get("expected") or None,
            )
            for i, row in enumerate(rows)
        ]
        return Dataset(name=p.stem, cases=cases)
    raise ValueError(f"Unsupported dataset format: {suffix}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/mingjing/dataset.py tests/test_dataset.py
git commit -m "feat: add dataset loader (yaml/json/csv)"
```

---

## Task 5: Provider layer

**Files:**
- Create: `src/mingjing/providers/__init__.py`
- Create: `src/mingjing/providers/base.py`
- Create: `src/mingjing/providers/litellm_provider.py`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers.py
from mingjing.providers import get_provider
from mingjing.providers.base import EchoProvider, FakeProvider


def test_echo_provider_echoes_prompt():
    assert EchoProvider().complete("hello").text == "hello"


def test_echo_provider_fixed_text():
    assert EchoProvider(fixed="ok").complete("anything").text == "ok"


def test_fake_provider_canned_and_default():
    p = FakeProvider(responses={"q": "a"}, default="d")
    assert p.complete("q").text == "a"
    assert p.complete("z").text == "d"


def test_get_provider_echo():
    assert isinstance(get_provider("echo"), EchoProvider)
    assert get_provider("echo:ok").complete("x").text == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_providers.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mingjing.providers'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mingjing/providers/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Response:
    text: str
    latency_ms: float = 0.0


class Provider(Protocol):
    model: str

    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0) -> Response: ...


class FakeProvider:
    """Test helper: returns canned responses keyed by prompt, else `default`."""

    def __init__(self, responses: dict[str, str] | None = None,
                 default: str = "", model: str = "fake"):
        self.responses = responses or {}
        self.default = default
        self.model = model

    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0) -> Response:
        return Response(text=self.responses.get(prompt, self.default))


class EchoProvider:
    """Offline provider: echoes the prompt, or returns `fixed` if given."""

    def __init__(self, fixed: str | None = None, model: str = "echo"):
        self.fixed = fixed
        self.model = model

    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0) -> Response:
        return Response(text=self.fixed if self.fixed is not None else prompt)
```

```python
# src/mingjing/providers/litellm_provider.py
from __future__ import annotations

import time

from .base import Response


class LiteLLMProvider:
    """Real models (DeepSeek/Qwen/OpenAI/Claude/...) via LiteLLM."""

    def __init__(self, model: str):
        self.model = model

    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0) -> Response:
        import litellm

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        start = time.perf_counter()
        resp = litellm.completion(model=self.model, messages=messages, temperature=temperature)
        latency_ms = (time.perf_counter() - start) * 1000
        return Response(text=resp["choices"][0]["message"]["content"], latency_ms=latency_ms)
```

```python
# src/mingjing/providers/__init__.py
from __future__ import annotations

from .base import EchoProvider, FakeProvider, Provider, Response


def get_provider(model: str) -> Provider:
    if model == "echo" or model.startswith("echo:"):
        fixed = model.split(":", 1)[1] if ":" in model else None
        return EchoProvider(fixed=fixed)
    from .litellm_provider import LiteLLMProvider

    return LiteLLMProvider(model=model)


__all__ = ["Provider", "Response", "FakeProvider", "EchoProvider", "get_provider"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_providers.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/mingjing/providers/ tests/test_providers.py
git commit -m "feat: add provider layer (echo/fake/litellm)"
```

---

## Task 6: Rule-based scorers

**Files:**
- Create: `src/mingjing/scorers/__init__.py` (empty)
- Create: `src/mingjing/scorers/base.py`
- Create: `src/mingjing/scorers/rules.py`
- Test: `tests/test_scorers_rules.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scorers_rules.py
from mingjing.config import ScorerConfig
from mingjing.models import TestCase
from mingjing.scorers.rules import Contains, ExactMatch, Regex, build_scorer


def test_exact_match():
    c = TestCase(id="1", input="q", expected="hello")
    assert ExactMatch().score(c, "hello").passed is True
    assert ExactMatch().score(c, "world").passed is False


def test_contains_uses_expected_when_no_text():
    c = TestCase(id="1", input="q", expected="lo")
    assert Contains().score(c, "hello").passed is True


def test_contains_with_text_param():
    c = TestCase(id="1", input="q")
    assert Contains(text="ell").score(c, "hello").passed is True


def test_regex():
    c = TestCase(id="1", input="q")
    assert Regex(pattern=r"\d+").score(c, "abc123").passed is True
    assert Regex(pattern=r"\d+").score(c, "abc").passed is False


def test_build_scorer():
    s = build_scorer(ScorerConfig(type="contains", params={"text": "x"}))
    assert isinstance(s, Contains)
    assert s.text == "x"


def test_build_scorer_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        build_scorer(ScorerConfig(type="nope"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scorers_rules.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mingjing.scorers'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mingjing/scorers/__init__.py
```

```python
# src/mingjing/scorers/base.py
from __future__ import annotations

from typing import Protocol

from ..models import Score, TestCase


class Scorer(Protocol):
    name: str

    def score(self, case: TestCase, output: str) -> Score: ...
```

```python
# src/mingjing/scorers/rules.py
from __future__ import annotations

import re

from ..config import ScorerConfig
from ..models import Score, TestCase
from .base import Scorer


class ExactMatch:
    name = "exact_match"

    def score(self, case: TestCase, output: str) -> Score:
        target = (case.expected or "").strip()
        ok = output.strip() == target
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok,
                     detail=f"expected={target!r}")


class Contains:
    name = "contains"

    def __init__(self, text: str | None = None):
        self.text = text

    def score(self, case: TestCase, output: str) -> Score:
        target = self.text if self.text is not None else (case.expected or "")
        ok = target != "" and target in output
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok,
                     detail=f"needle={target!r}")


class Regex:
    name = "regex"

    def __init__(self, pattern: str):
        self.pattern = pattern

    def score(self, case: TestCase, output: str) -> Score:
        ok = re.search(self.pattern, output) is not None
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok,
                     detail=f"pattern={self.pattern!r}")


def build_scorer(cfg: ScorerConfig, judge_provider=None) -> Scorer:
    if cfg.type == "exact_match":
        return ExactMatch()
    if cfg.type == "contains":
        return Contains(text=cfg.params.get("text"))
    if cfg.type == "regex":
        return Regex(pattern=cfg.params["pattern"])
    if cfg.type == "llm_judge":
        from .llm_judge import LLMJudge

        return LLMJudge(provider=judge_provider, criteria=cfg.params.get("criteria", ""))
    raise ValueError(f"Unknown scorer type: {cfg.type}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scorers_rules.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/mingjing/scorers/__init__.py src/mingjing/scorers/base.py src/mingjing/scorers/rules.py tests/test_scorers_rules.py
git commit -m "feat: add rule-based scorers and build_scorer factory"
```

---

## Task 7: LLM-as-judge scorer

**Files:**
- Create: `src/mingjing/scorers/llm_judge.py`
- Test: `tests/test_scorers_llm_judge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scorers_llm_judge.py
from mingjing.models import TestCase
from mingjing.providers.base import FakeProvider
from mingjing.scorers.llm_judge import LLMJudge


def test_llm_judge_parses_json():
    judge = LLMJudge(
        provider=FakeProvider(default='{"score": 0.8, "pass": true, "reason": "good"}'),
        criteria="quality",
    )
    score = judge.score(TestCase(id="1", input="q"), "answer")
    assert score.value == 0.8
    assert score.passed is True
    assert "good" in score.detail


def test_llm_judge_extracts_json_from_noise():
    judge = LLMJudge(
        provider=FakeProvider(default='Sure!\n{"score": 1.0, "pass": true, "reason": "ok"}\nThanks'),
    )
    score = judge.score(TestCase(id="1", input="q"), "answer")
    assert score.value == 1.0


def test_llm_judge_handles_garbage():
    judge = LLMJudge(provider=FakeProvider(default="not json at all"))
    score = judge.score(TestCase(id="1", input="q"), "answer")
    assert score.passed is False
    assert score.value == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scorers_llm_judge.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mingjing.scorers.llm_judge'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mingjing/scorers/llm_judge.py
from __future__ import annotations

import json

from ..models import Score, TestCase

JUDGE_PROMPT = """You are grading an AI answer.

Question/Input:
{input}

AI Answer:
{output}

Criteria: {criteria}

Respond with ONLY a JSON object:
{{"score": <float 0..1>, "pass": <true|false>, "reason": "<short reason>"}}
"""


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in judge output")
    return text[start : end + 1]


class LLMJudge:
    name = "llm_judge"

    def __init__(self, provider, criteria: str = ""):
        self.provider = provider
        self.criteria = criteria

    def score(self, case: TestCase, output: str) -> Score:
        prompt = JUDGE_PROMPT.format(
            input=case.input, output=output, criteria=self.criteria or "overall quality"
        )
        resp = self.provider.complete(prompt, temperature=0.0)
        try:
            data = json.loads(_extract_json(resp.text))
            value = float(data.get("score", 0.0))
            passed = bool(data.get("pass", value >= 0.5))
            reason = str(data.get("reason", ""))
        except Exception as e:  # noqa: BLE001 - any parse failure means judge failed
            return Score(scorer=self.name, value=0.0, passed=False,
                         detail=f"judge parse error: {e}")
        return Score(scorer=self.name, value=value, passed=passed, detail=reason)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scorers_llm_judge.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/mingjing/scorers/llm_judge.py tests/test_scorers_llm_judge.py
git commit -m "feat: add LLM-as-judge scorer with robust JSON parsing"
```

---

## Task 8: Eval engine

**Files:**
- Create: `src/mingjing/engine.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine.py
from mingjing.config import EvalConfig, ScorerConfig
from mingjing.engine import run_eval
from mingjing.providers.base import FakeProvider


def test_run_eval_with_fake_provider(tmp_path):
    ds = tmp_path / "ds.yaml"
    ds.write_text(
        "name: d\ncases:\n  - id: '1'\n    input: hello\n    expected: HELLO\n",
        encoding="utf-8",
    )
    cfg = EvalConfig(
        name="t", dataset=str(ds), model="fake", prompt_template="{{input}}",
        scorers=[ScorerConfig(type="contains", params={"text": "HELLO"})],
    )
    provider = FakeProvider(responses={"hello": "HELLO world"})
    run = run_eval(cfg, provider)
    assert len(run.results) == 1
    assert run.results[0].output == "HELLO world"
    assert run.results[0].scores[0].passed is True
    assert run.model == "fake"
    assert run.config["name"] == "t"


def test_run_eval_renders_template(tmp_path):
    ds = tmp_path / "ds.yaml"
    ds.write_text("name: d\ncases:\n  - id: '1'\n    input: world\n", encoding="utf-8")
    cfg = EvalConfig(name="t", dataset=str(ds), model="echo",
                     prompt_template="Hello {{input}}")
    # echo provider returns the rendered prompt verbatim
    from mingjing.providers.base import EchoProvider

    run = run_eval(cfg, EchoProvider())
    assert run.results[0].output == "Hello world"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mingjing.engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mingjing/engine.py
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from .config import EvalConfig
from .dataset import load_dataset
from .models import CaseResult, Run
from .providers.base import Provider
from .scorers.rules import build_scorer


def _render(template: str, case_input: str) -> str:
    return template.replace("{{input}}", case_input)


def run_eval(config: EvalConfig, provider: Provider,
             judge_provider: Provider | None = None) -> Run:
    dataset = load_dataset(config.dataset)
    scorers = [build_scorer(s, judge_provider=judge_provider or provider)
               for s in config.scorers]
    results: list[CaseResult] = []
    for case in dataset.cases:
        prompt = _render(config.prompt_template, case.input)
        resp = provider.complete(prompt, system=config.system, temperature=config.temperature)
        scores = [scorer.score(case, resp.text) for scorer in scorers]
        results.append(
            CaseResult(case_id=case.id, input=case.input, output=resp.text,
                       scores=scores, latency_ms=resp.latency_ms)
        )
    return Run(
        id=uuid.uuid4().hex[:12],
        name=config.name,
        created_at=datetime.now(timezone.utc),
        model=config.model,
        results=results,
        config=config.model_dump(),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/mingjing/engine.py tests/test_engine.py
git commit -m "feat: add eval engine that runs a dataset into a Run"
```

---

## Task 9: Run store

**Files:**
- Create: `src/mingjing/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py
from datetime import datetime, timezone

from mingjing.models import CaseResult, Run
from mingjing.store import RunStore


def _run(rid: str) -> Run:
    return Run(id=rid, name="t", created_at=datetime(2026, 5, 25, tzinfo=timezone.utc),
               model="echo", results=[CaseResult(case_id="1", input="q", output="a")])


def test_store_round_trip(tmp_path):
    store = RunStore(tmp_path / "data")
    store.save(_run("r1"))
    loaded = store.load("r1")
    assert loaded.id == "r1"
    assert loaded.results[0].output == "a"


def test_store_list_runs(tmp_path):
    store = RunStore(tmp_path / "data")
    store.save(_run("r1"))
    store.save(_run("r2"))
    ids = {r.id for r in store.list_runs()}
    assert ids == {"r1", "r2"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mingjing.store'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mingjing/store.py
from __future__ import annotations

from pathlib import Path

from .models import Run


class RunStore:
    """Persists Runs as JSON files under <root>/runs/."""

    def __init__(self, root: str | Path = ".mingjing"):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def save(self, run: Run) -> Path:
        path = self.runs_dir / f"{run.id}.json"
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, run_id: str) -> Run:
        path = self.runs_dir / f"{run_id}.json"
        return Run.model_validate_json(path.read_text(encoding="utf-8"))

    def list_runs(self) -> list[Run]:
        runs = [Run.model_validate_json(p.read_text(encoding="utf-8"))
                for p in self.runs_dir.glob("*.json")]
        return sorted(runs, key=lambda r: r.created_at, reverse=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_store.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/mingjing/store.py tests/test_store.py
git commit -m "feat: add file-based run store"
```

---

## Task 10: Diff service (killer feature)

**Files:**
- Create: `src/mingjing/diff.py`
- Test: `tests/test_diff.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_diff.py
from datetime import datetime, timezone

from mingjing.diff import diff_runs
from mingjing.models import CaseResult, Run, Score


def _run(rid: str, scores_by_case: dict[str, float]) -> Run:
    results = [
        CaseResult(case_id=cid, input="q", output="o",
                   scores=[Score(scorer="s", value=v, passed=v >= 0.5)])
        for cid, v in scores_by_case.items()
    ]
    return Run(id=rid, name="t", created_at=datetime(2026, 5, 25, tzinfo=timezone.utc),
               model="echo", results=results)


def test_diff_detects_all_statuses():
    before = _run("a", {"1": 0.5, "2": 1.0, "3": 0.0})
    after = _run("b", {"1": 1.0, "2": 1.0, "4": 0.7})
    # case 1 improved, case 2 unchanged, case 3 removed, case 4 new
    report = diff_runs(before, after)
    assert report.summary() == {
        "improved": 1, "regressed": 0, "unchanged": 1, "new": 1, "removed": 1,
    }


def test_diff_detects_regression():
    before = _run("a", {"1": 1.0})
    after = _run("b", {"1": 0.0})
    report = diff_runs(before, after)
    assert [c.case_id for c in report.regressed] == ["1"]
    assert report.regressed[0].before == 1.0
    assert report.regressed[0].after == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_diff.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mingjing.diff'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mingjing/diff.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_diff.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/mingjing/diff.py tests/test_diff.py
git commit -m "feat: add run diff/regression detection"
```

---

## Task 11: CLI + examples

**Files:**
- Create: `src/mingjing/cli.py`
- Create: `examples/dataset.yaml`
- Create: `examples/eval.yaml`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from typer.testing import CliRunner

from mingjing.cli import app
from mingjing.store import RunStore

runner = CliRunner()


def _write_cfg(tmp_path):
    ds = tmp_path / "ds.yaml"
    ds.write_text(
        "name: d\ncases:\n  - id: '1'\n    input: hello\n    expected: hello\n",
        encoding="utf-8",
    )
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(
        f"name: t\ndataset: {ds}\nmodel: echo\n"
        f"prompt_template: '{{{{input}}}}'\n"
        f"scorers:\n  - type: contains\n    params: {{text: hello}}\n",
        encoding="utf-8",
    )
    return cfg


def test_cli_run_then_list_and_diff(tmp_path):
    cfg = _write_cfg(tmp_path)
    store = str(tmp_path / "data")

    assert runner.invoke(app, ["run", str(cfg), "--store", store]).exit_code == 0
    assert runner.invoke(app, ["run", str(cfg), "--store", store]).exit_code == 0

    listed = runner.invoke(app, ["list", "--store", store])
    assert listed.exit_code == 0

    ids = [r.id for r in RunStore(store).list_runs()]
    assert len(ids) == 2
    diffed = runner.invoke(app, ["diff", ids[1], ids[0], "--store", store])
    assert diffed.exit_code == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mingjing.cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mingjing/cli.py
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .diff import diff_runs
from .engine import run_eval
from .providers import get_provider
from .store import RunStore

app = typer.Typer(help="明镜 / Evalith — AI regression testing")
console = Console()

_STATUS_COLOR = {
    "improved": "green",
    "regressed": "red",
    "unchanged": "dim",
    "new": "cyan",
    "removed": "yellow",
}


@app.command()
def run(config: str, store: str = ".mingjing") -> None:
    """Run an eval defined by CONFIG and save the resulting run."""
    cfg = load_config(config)
    result = run_eval(cfg, get_provider(cfg.model))
    path = RunStore(store).save(result)
    passed = sum(1 for r in result.results for s in r.scores if s.passed)
    total = sum(len(r.scores) for r in result.results)
    console.print(f"[green]Run {result.id}[/] saved to {path} — {passed}/{total} checks passed")


@app.command()
def diff(before: str, after: str, store: str = ".mingjing") -> None:
    """Compare two runs and show which cases improved or regressed."""
    s = RunStore(store)
    report = diff_runs(s.load(before), s.load(after))
    table = Table(title=f"Diff {before} → {after}")
    for col in ("case", "status", "before", "after"):
        table.add_column(col)
    for c in report.cases:
        color = _STATUS_COLOR[c.status]
        table.add_row(
            c.case_id,
            f"[{color}]{c.status}[/]",
            "-" if c.before is None else f"{c.before:.2f}",
            "-" if c.after is None else f"{c.after:.2f}",
        )
    console.print(table)
    console.print(report.summary())


@app.command("list")
def list_runs(store: str = ".mingjing") -> None:
    """List stored runs, newest first."""
    for r in RunStore(store).list_runs():
        console.print(f"{r.id}  {r.created_at:%Y-%m-%d %H:%M}  {r.name}  ({r.model})")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
```

```yaml
# examples/dataset.yaml
name: faq-demo
cases:
  - id: "1"
    input: "What is your refund window?"
    expected: "30 days"
  - id: "2"
    input: "Do you ship internationally?"
    expected: "yes"
```

```yaml
# examples/eval.yaml
name: faq-demo-eval
dataset: examples/dataset.yaml
model: echo            # swap for e.g. deepseek/deepseek-chat or gpt-4o-mini
prompt_template: |
  Answer the customer question concisely.
  Question: {{input}}
scorers:
  - type: contains     # checks the model output contains each case's `expected`
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full test suite**

Run: `pytest -q`
Expected: PASS (all tests across all files green)

- [ ] **Step 6: Commit**

```bash
git add src/mingjing/cli.py examples/dataset.yaml examples/eval.yaml tests/test_cli.py
git commit -m "feat: add CLI (run/diff/list) and example config"
```

---

## Task 12: README quickstart

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

```markdown
# 明镜 / Evalith

A neutral, local-first **AI regression-testing** tool. Define a test set, run it
against any model (DeepSeek / Qwen / OpenAI / Claude / …), score each case, and
**diff two runs to see exactly what got better or worse** — instead of shipping on vibes.

## Install

```bash
pip install -e ".[dev]"   # from source (v0.1)
```

## Quickstart

```bash
# 1. Run the example eval (offline `echo` model, no API key needed)
mingjing run examples/eval.yaml

# 2. Change your prompt/model in examples/eval.yaml, run again
mingjing run examples/eval.yaml

# 3. List runs, then diff the two newest to spot regressions
mingjing list
mingjing diff <OLDER_RUN_ID> <NEWER_RUN_ID>
```

## Using a real model

Set the provider API key (see LiteLLM docs) and set `model:` in `examples/eval.yaml`,
e.g. `deepseek/deepseek-chat`, `qwen/qwen-max`, `gpt-4o-mini`, or `claude-3-5-sonnet`.

```bash
export DEEPSEEK_API_KEY=sk-...
# model: deepseek/deepseek-chat  in eval.yaml
mingjing run examples/eval.yaml
```

## Scorers

- `exact_match` — output equals the case's `expected`
- `contains` — output contains `params.text` (or the case's `expected`)
- `regex` — output matches `params.pattern`
- `llm_judge` — an LLM grades the output against `params.criteria`
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README quickstart"
```

---

## Self-Review (completed)

**1. Spec coverage** (against `docs/superpowers/specs/2026-05-25-ai-eval-regression-tool-design.md` §5.1):
- ✅ Define test set (incl. import from file): Task 4 (`load_dataset` yaml/json/csv).
- ✅ Run eval; rule assertions + LLM-judge + model access (国产+海外): Tasks 5–8.
- ✅ Killer feature — version compare / regression: Task 10 (`diff_runs`) + Task 11 (`mingjing diff`).
- ⏭️ Local Web UI (§5.1 #4): **intentionally deferred to a follow-on plan** (CLI delivers the core value; noted in scope). Not a gap — a scoping decision.
- Built-in metrics (relevance/faithfulness/hallucination) beyond `llm_judge` are deferred to a follow-on; `llm_judge` covers the v0.1 "quality" need.

**2. Placeholder scan:** No "TBD/TODO/handle edge cases" steps; every code step contains complete code; every run step has an exact command + expected result.

**3. Type consistency:** `Score(scorer,value,passed,detail)`, `CaseResult(case_id,input,output,scores,latency_ms)` + `.mean_score`, `Run(id,name,created_at,model,results,config)`, `Provider.complete(prompt,*,system,temperature)->Response(text,latency_ms)`, `build_scorer(cfg,judge_provider=None)`, `run_eval(config,provider,judge_provider=None)`, `RunStore(root).save/load/list_runs`, `diff_runs(before,after,tol)->DiffReport` with `.improved/.regressed/.unchanged/.new/.removed/.summary()`. Names/signatures are consistent across all tasks and tests.
