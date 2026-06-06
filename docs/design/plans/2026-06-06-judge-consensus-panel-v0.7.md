# Judge Consensus Panel (v0.7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 一次 eval 挂 N 个 judge:主 judge 评分与 gate 语义不变,panel judge 产出 per-case 分歧、pairwise Cohen's κ、领域分组一致性与 ⚠ 低共识标记;judge 成本独立统计。

**Architecture:** `LLMJudge` 增加 `panel`(judge名→Provider)与 usage 返回;engine 在每个 trial 打分时让全部 panel judge 各判一次(同 case 内相同输出走缓存),聚合进 `CaseResult.panel_samples/panel_details`;新模块 `consensus.py` 承载统计(kappa/spread/domain);report/CLI 只读这些字段渲染。spec 见 `docs/design/specs/2026-06-06-judge-consensus-panel-design.md`。

**Tech Stack:** Python 3.10+、pydantic v2、typer、pytest。无新依赖。

**与 spec 的两处已确认偏差(实现侦察后的修正):**
1. spec 写 `metadata.domain`;实际数据集(`docs/blog/article4/qa.small.yaml`)用顶层 `domain:` 字段且现有 `TestCase` 会静默丢弃它 → `TestCase`/`CaseResult` 各加 `domain: str | None` 字段,consensus 取 `case.domain`,缺省回退 `metadata["domain"]`。
2. 相同输出判分缓存只作用于 panel judge,不作用于主 judge——主 judge 的 per-trial 结果是 judge 噪声测量的对象(文章 2 的立身之本),缓存会人为抹平噪声。

---

### Task 1: 数据模型新字段(models.py)

**Files:**
- Modify: `src/evalith/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_models.py` 追加:

```python
def test_case_result_panel_fields_default_empty():
    cr = CaseResult(case_id="c", input="i", output="o")
    assert cr.panel_samples == {}
    assert cr.panel_details == {}
    assert cr.judge_tokens == 0
    assert cr.judge_cost_usd == 0.0
    assert cr.domain is None


def test_test_case_accepts_domain():
    tc = TestCase(id="1", input="q", domain="code")
    assert tc.domain == "code"


def test_old_run_json_loads_without_panel_fields():
    """v0.6 的 run JSON(无 panel 字段)必须能原样加载。"""
    old = {
        "id": "abc", "name": "n", "created_at": "2026-06-01T00:00:00Z",
        "model": "m",
        "results": [{"case_id": "c", "input": "i", "output": "o"}],
    }
    run = Run.model_validate(old)
    assert run.results[0].panel_samples == {}
    assert run.total_judge_cost_usd == 0.0
    assert run.total_judge_tokens == 0


def test_run_judge_cost_aggregates():
    run = Run(id="r", name="n", created_at=datetime(2026, 6, 6, tzinfo=timezone.utc),
              model="m", results=[
                  CaseResult(case_id="a", input="i", output="o",
                             judge_tokens=10, judge_cost_usd=0.01),
                  CaseResult(case_id="b", input="i", output="o",
                             judge_tokens=5, judge_cost_usd=0.02),
              ])
    assert run.total_judge_tokens == 15
    assert abs(run.total_judge_cost_usd - 0.03) < 1e-9
```

文件顶部 import 需含 `from datetime import datetime, timezone` 与 `from evalith.models import CaseResult, Run, TestCase`(看现有 import,缺啥补啥)。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_models.py -v`
Expected: 新增 4 个测试 FAIL(`panel_samples` 等字段不存在)

- [ ] **Step 3: 最小实现**

`src/evalith/models.py`:

`TestCase` 加一行(`expected_concepts` 之后):

```python
    domain: str | None = None
```

`CaseResult` 在 `pass_rate_samples` 之后加:

```python
    domain: str | None = None
    # Judge consensus panel (v0.7) — diagnostics only, never part of the gate
    panel_samples: dict[str, list[float]] = Field(default_factory=dict)  # judge -> per-trial pass (−1.0 = judge call failed)
    panel_details: dict[str, str] = Field(default_factory=dict)          # judge -> representative reason
    judge_tokens: int = 0       # primary + panel judging calls combined
    judge_cost_usd: float = 0.0
```

`Run` 在 `total_cost_usd` 之后加:

```python
    @property
    def total_judge_tokens(self) -> int:
        return sum(r.judge_tokens for r in self.results)

    @property
    def total_judge_cost_usd(self) -> float:
        return sum(r.judge_cost_usd for r in self.results)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_models.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/evalith/models.py tests/test_models.py
git commit -m "feat(models): panel_samples/panel_details/judge cost fields + domain (v0.7)"
```

---

### Task 2: 共识统计模块(consensus.py)

**Files:**
- Create: `src/evalith/consensus.py`
- Test: `tests/test_consensus.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_consensus.py`:

```python
import json
import math
from pathlib import Path

from evalith.consensus import (
    MISSING,
    case_spread,
    cohen_kappa,
    consensus_summary,
    domain_agreement,
    judge_means,
    pairwise_kappa,
    threshold_from_config,
)
from evalith.models import CaseResult, Run


def _case(cid, prs, panel=None, domain=None):
    return CaseResult(case_id=cid, input="i", output="o",
                      pass_rate_samples=prs, panel_samples=panel or {},
                      domain=domain)


def _run(cases):
    from datetime import datetime, timezone
    return Run(id="r", name="n", created_at=datetime(2026, 6, 6, tzinfo=timezone.utc),
               model="m", results=cases)


# ---------------------------------------------------------------- kappa

def test_cohen_kappa_hand_computed():
    # po=0.5, pa1=pb1=0.5 -> pe=0.5 -> kappa=0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 1]) == 0.0
    # perfect agreement, mixed labels
    assert cohen_kappa([1, 0, 1], [1, 0, 1]) == 1.0
    # degenerate: both constant-and-equal -> pe=1 -> return 1.0
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0
    # empty / mismatched -> nan
    assert math.isnan(cohen_kappa([], []))
    assert math.isnan(cohen_kappa([1], [1, 0]))


def test_cohen_kappa_golden_article4():
    """Golden 回归:必须复现 multi_compare.py 在文章 4 raw 数据上的 κ。"""
    raw = Path("docs/blog/article4/raw")
    if not raw.exists():
        import pytest
        pytest.skip("article4 raw data not present")

    def labels(name):
        run = json.loads((raw / name).read_text(encoding="utf-8"))
        out = []
        for c in run["results"]:
            for v in (c.get("pass_rate_samples") or []):
                out.append(1 if v >= 0.5 else 0)
        return out

    assert abs(cohen_kappa(labels("j_ds_by_ds_a1.json"),
                           labels("j_ds_by_qw_a1.json")) - 0.253920) < 1e-4
    assert abs(cohen_kappa(labels("j_ds_by_ds_a1.json"),
                           labels("j_ds_by_glm_a1.json")) - 0.568138) < 1e-4
    assert abs(cohen_kappa(labels("j_ds_by_qw_a1.json"),
                           labels("j_ds_by_glm_a1.json")) - 0.252408) < 1e-4


# ---------------------------------------------------------------- spread

def test_judge_means_and_spread():
    c = _case("c1", [1.0, 1.0], panel={"qw": [0.0, 0.0], "glm": [1.0, MISSING]})
    m = judge_means(c)
    assert m["primary"] == 1.0
    assert m["qw"] == 0.0
    assert m["glm"] == 1.0          # MISSING trial skipped
    assert case_spread(c) == 1.0


def test_spread_zero_without_panel():
    assert case_spread(_case("c", [1.0, 0.0])) == 0.0


# ---------------------------------------------------------------- pairwise

def test_pairwise_kappa_pairwise_deletion():
    run = _run([
        _case("c1", [1.0, 1.0], panel={"qw": [1.0, MISSING]}),
        _case("c2", [0.0, 1.0], panel={"qw": [0.0, 0.0]}),
    ])
    k = pairwise_kappa(run)
    # labels after dropping the MISSING trial: primary=[1,0,1], qw=[1,0,0]
    # po=2/3, pa1=2/3, pb1=1/3 -> pe=2/9+2/9=4/9 -> kappa=(6/9-4/9)/(5/9)=0.4
    assert abs(k[("primary", "qw")] - 0.4) < 1e-9


# ---------------------------------------------------------------- domain

def test_domain_agreement_groups_and_flags():
    run = _run([
        _case("a", [1.0], panel={"qw": [0.0]}, domain="code"),     # spread 1.0 -> low
        _case("b", [1.0], panel={"qw": [1.0]}, domain="safety"),   # spread 0
        _case("c", [1.0], panel={"qw": [1.0]}),                    # untagged -> "?"
    ])
    d = domain_agreement(run, threshold=0.5)
    assert d["code"]["n"] == 1 and d["code"]["low_consensus"] == 1
    assert d["safety"]["low_consensus"] == 0
    assert "?" in d


def test_domain_falls_back_to_metadata():
    c = _case("a", [1.0], panel={"qw": [1.0]})
    c.metadata = {"domain": "math"}
    d = domain_agreement(_run([c]), threshold=0.5)
    assert "math" in d
```

注意:`CaseResult` 没有 `metadata` 字段——上面最后一个测试会暴露这一点。修正:domain 回退逻辑放在 **engine 写入 CaseResult.domain 时**(Task 5),consensus 只读 `case.domain`。**删掉 `test_domain_falls_back_to_metadata`**,在 Task 5 的 engine 测试里验证回退。

```python
# ---------------------------------------------------------------- summary

def test_consensus_summary_none_without_panel():
    assert consensus_summary(_run([_case("c", [1.0])])) is None


def test_consensus_summary_counts():
    run = _run([
        _case("a", [1.0, 1.0], panel={"qw": [0.0, 0.0]}),   # spread 1.0
        _case("b", [1.0, 1.0], panel={"qw": [1.0, 1.0]}),   # spread 0
    ])
    s = consensus_summary(run, threshold=0.5)
    assert s["judges"] == ["primary", "qw"]
    assert s["low_consensus_cases"] == ["a"]
    assert s["n_cases"] == 2
    assert ("primary", "qw") in s["kappa"]


def test_threshold_from_config():
    cfg = {"scorers": [{"type": "llm_judge",
                        "params": {"consensus_threshold": 0.3}}]}
    assert threshold_from_config(cfg) == 0.3
    assert threshold_from_config({}) == 0.5
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_consensus.py -v`
Expected: FAIL — `ModuleNotFoundError: evalith.consensus`

- [ ] **Step 3: 实现 `src/evalith/consensus.py`**

```python
"""Judge consensus statistics (v0.7).

Panel judges are diagnostics only: they never affect scores, pass/fail or
exit codes. Statistics ported from docs/blog/article4/multi_compare.py
(validated against the article's published numbers).
"""
from __future__ import annotations

from itertools import combinations

from .models import CaseResult, Run

MISSING = -1.0   # sentinel: panel judge call failed for that trial
PRIMARY = "primary"


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two equal-length binary label vectors."""
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return 1.0   # both raters constant and equal — degenerate perfect agreement
    return (po - pe) / (1 - pe)


def judge_means(case: CaseResult) -> dict[str, float]:
    """Per-judge mean pass rate for one case; MISSING trials are skipped."""
    means: dict[str, float] = {}
    if case.pass_rate_samples:
        means[PRIMARY] = sum(case.pass_rate_samples) / len(case.pass_rate_samples)
    for name, vals in case.panel_samples.items():
        ok = [v for v in vals if v != MISSING]
        if ok:
            means[name] = sum(ok) / len(ok)
    return means


def case_spread(case: CaseResult) -> float:
    """Max − min of per-judge means; 0.0 when fewer than two judges scored."""
    means = judge_means(case)
    if len(means) < 2:
        return 0.0
    return max(means.values()) - min(means.values())


def _trial_series(run: Run) -> dict[str, list[float]]:
    """Flat per-(case, trial) value series per judge, MISSING-padded to align."""
    judges = [PRIMARY] + sorted({j for r in run.results for j in r.panel_samples})
    series: dict[str, list[float]] = {j: [] for j in judges}
    for r in run.results:
        n = len(r.pass_rate_samples)
        for j in judges:
            vals = list(r.pass_rate_samples if j == PRIMARY
                        else r.panel_samples.get(j, []))
            vals = (vals + [MISSING] * n)[:n]   # pad/truncate to trial count
            series[j].extend(vals)
    return series


def pairwise_kappa(run: Run) -> dict[tuple[str, str], float]:
    """Pairwise Cohen's kappa over per-(case, trial) binary labels.

    Trials where either judge is MISSING are dropped pairwise.
    """
    series = _trial_series(run)
    out: dict[tuple[str, str], float] = {}
    for j1, j2 in combinations(series.keys(), 2):
        pairs = [(x, y) for x, y in zip(series[j1], series[j2])
                 if x != MISSING and y != MISSING]
        a = [1 if x >= 0.5 else 0 for x, _ in pairs]
        b = [1 if y >= 0.5 else 0 for _, y in pairs]
        out[(j1, j2)] = cohen_kappa(a, b)
    return out


def domain_agreement(run: Run, threshold: float = 0.5) -> dict[str, dict]:
    """Per-domain judge means and low-consensus counts ("?" for untagged)."""
    by_dom: dict[str, list[CaseResult]] = {}
    for r in run.results:
        by_dom.setdefault(r.domain or "?", []).append(r)
    out: dict[str, dict] = {}
    for dom, cases in sorted(by_dom.items()):
        sums: dict[str, list[float]] = {}
        low = 0
        for c in cases:
            for j, m in judge_means(c).items():
                sums.setdefault(j, []).append(m)
            if case_spread(c) >= threshold:
                low += 1
        out[dom] = {"n": len(cases),
                    "means": {j: sum(v) / len(v) for j, v in sums.items()},
                    "low_consensus": low}
    return out


def consensus_summary(run: Run, threshold: float = 0.5) -> dict | None:
    """Inputs for the one-line CLI summary; None when the run has no panel."""
    panel_judges = sorted({j for r in run.results for j in r.panel_samples})
    if not panel_judges:
        return None
    low = [r.case_id for r in run.results if case_spread(r) >= threshold]
    kappas = pairwise_kappa(run)
    valid = [v for v in kappas.values() if v == v]   # drop NaN
    return {"judges": [PRIMARY] + panel_judges,
            "n_cases": len(run.results),
            "low_consensus_cases": low,
            "min_kappa": min(valid) if valid else float("nan"),
            "kappa": kappas,
            "threshold": threshold}


def threshold_from_config(config: dict) -> float:
    """Read consensus_threshold from a Run.config dict; default 0.5."""
    for s in config.get("scorers", []):
        if s.get("type") == "llm_judge":
            return float(s.get("params", {}).get("consensus_threshold", 0.5))
    return 0.5
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_consensus.py -v`
Expected: 全部 PASS(含 golden 三连)

- [ ] **Step 5: 提交**

```bash
git add src/evalith/consensus.py tests/test_consensus.py
git commit -m "feat(consensus): kappa/spread/domain agreement stats module (v0.7)"
```

---

### Task 3: LLMJudge panel 支持与 usage 返回(llm_judge.py)

**Files:**
- Modify: `src/evalith/scorers/llm_judge.py`
- Test: `tests/test_scorers_llm_judge.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_scorers_llm_judge.py` 追加(沿用文件里已有的 fake provider 风格;`FakeProvider` 来自 `evalith.providers.base`,judge 的 prompt 不可预知所以用 `default=` 注入):

```python
from evalith.models import TestCase
from evalith.providers.base import FakeProvider, Response
from evalith.scorers.llm_judge import LLMJudge


def _case():
    return TestCase(id="c", input="q")


def test_score_with_usage_returns_tokens_and_cost():
    class CostedProvider:
        model = "fake"

        def complete(self, prompt, *, system=None, temperature=0.0):
            return Response(text='{"score": 1.0, "pass": true, "reason": "ok"}',
                            total_tokens=42, cost_usd=0.001)

    judge = LLMJudge(provider=CostedProvider())
    score, tokens, cost = judge.score_with_usage(_case(), "ans")
    assert score.passed is True
    assert tokens == 42
    assert abs(cost - 0.001) < 1e-9


def test_panel_score_judges_with_every_panel_member():
    good = FakeProvider(default='{"score": 1.0, "pass": true, "reason": "fine"}')
    harsh = FakeProvider(default='{"score": 0.0, "pass": false, "reason": "bad code"}')
    judge = LLMJudge(provider=good, panel={"harsh": harsh, "kind": good})
    results, tokens, cost = judge.panel_score(_case(), "ans", cache={})
    assert results["harsh"].passed is False
    assert results["harsh"].detail == "bad code"
    assert results["kind"].passed is True


def test_panel_score_provider_failure_yields_none():
    class BoomProvider:
        model = "boom"

        def complete(self, prompt, *, system=None, temperature=0.0):
            raise RuntimeError("rate limited")

    ok = FakeProvider(default='{"score": 1.0, "pass": true, "reason": "ok"}')
    judge = LLMJudge(provider=ok, panel={"boom": BoomProvider(), "ok": ok})
    results, _, _ = judge.panel_score(_case(), "ans", cache={})
    assert results["boom"] is None          # failed judge recorded as missing
    assert results["ok"].passed is True     # others unaffected


def test_panel_score_caches_identical_output():
    class CountingProvider:
        model = "count"

        def __init__(self):
            self.calls = 0

        def complete(self, prompt, *, system=None, temperature=0.0):
            self.calls += 1
            return Response(text='{"score": 1.0, "pass": true, "reason": "ok"}')

    counting = CountingProvider()
    judge = LLMJudge(provider=FakeProvider(default='{"score": 1.0, "pass": true}'),
                     panel={"j": counting})
    cache = {}
    judge.panel_score(_case(), "same output", cache)
    judge.panel_score(_case(), "same output", cache)   # second trial, same text
    assert counting.calls == 1


def test_panel_parse_error_is_fail_score_not_missing():
    garbage = FakeProvider(default="not json at all")
    judge = LLMJudge(provider=FakeProvider(default='{"score": 1.0, "pass": true}'),
                     panel={"g": garbage})
    results, _, _ = judge.panel_score(_case(), "ans", cache={})
    assert results["g"] is not None
    assert results["g"].passed is False
    assert "judge parse error" in results["g"].detail
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_scorers_llm_judge.py -v`
Expected: 新增测试 FAIL(`score_with_usage`/`panel_score` 不存在)

- [ ] **Step 3: 实现**

重构 `src/evalith/scorers/llm_judge.py` 的 `LLMJudge`(保留 `JUDGE_PROMPTS`/`_extract_json` 不动):

```python
class LLMJudge:
    name = "llm_judge"

    def __init__(self, provider, criteria: str = "", language: str = "en",
                 panel: dict | None = None, consensus_threshold: float = 0.5):
        self.provider = provider
        self.criteria = criteria
        self.language = language if language in JUDGE_PROMPTS else "en"
        self.panel = panel or {}                      # judge name -> Provider
        self.consensus_threshold = consensus_threshold

    def _build_prompt(self, case: TestCase, output: str) -> str:
        # Build effective criteria — append expected_concepts checklist if present
        criteria_eff = self.criteria or "overall quality"
        if case.expected_concepts:
            concept_lines = "\n".join(f"- {c}" for c in case.expected_concepts)
            if self.language == "zh":
                criteria_eff = (
                    f"{criteria_eff}\n\n核心概念清单(回答须覆盖):\n{concept_lines}"
                )
            else:
                criteria_eff = (
                    f"{criteria_eff}\n\nExpected concepts (response must cover):\n{concept_lines}"
                )
        return JUDGE_PROMPTS[self.language].format(
            input=case.input, output=output, criteria=criteria_eff
        )

    def _judge(self, provider, case: TestCase, output: str) -> tuple[Score, int, float]:
        """One judging call. Parse errors -> fail Score; provider errors propagate."""
        resp = provider.complete(self._build_prompt(case, output), temperature=0.0)
        try:
            data = json.loads(_extract_json(resp.text))
            value = float(data.get("score", 0.0))
            passed = bool(data.get("pass", value >= 0.5))
            score = Score(scorer=self.name, value=value, passed=passed,
                          detail=str(data.get("reason", "")))
        except Exception as e:  # noqa: BLE001 - any parse failure means judge failed
            score = Score(scorer=self.name, value=0.0, passed=False,
                          detail=f"judge parse error: {e}")
        return score, resp.total_tokens, resp.cost_usd

    def score(self, case: TestCase, output: str) -> Score:
        return self._judge(self.provider, case, output)[0]

    def score_with_usage(self, case: TestCase, output: str) -> tuple[Score, int, float]:
        """Primary judge scoring + (tokens, cost) of the judging call."""
        return self._judge(self.provider, case, output)

    def panel_score(self, case: TestCase, output: str,
                    cache: dict) -> tuple[dict[str, Score | None], int, float]:
        """Score `output` with every panel judge (diagnostics only).

        Provider failures record None (missing); parse failures record a fail
        Score, same as the primary judge. `cache` maps (judge, output) -> Score
        so identical trial outputs within a case are judged once per judge.
        """
        results: dict[str, Score | None] = {}
        tokens, cost = 0, 0.0
        for name, prov in self.panel.items():
            key = (name, output)
            if key in cache:
                results[name] = cache[key]
                continue
            try:
                s, tok, c = self._judge(prov, case, output)
            except Exception:  # noqa: BLE001 - one judge down must not kill the case
                results[name] = None
                continue
            tokens += tok
            cost += c
            cache[key] = s
            results[name] = s
        return results, tokens, cost
```

旧 `score()` 的逻辑全部移入 `_build_prompt` + `_judge`,行为不变(同样的 prompt、同样的 parse-error 语义)。

- [ ] **Step 4: 跑测试确认通过(含旧测试)**

Run: `python -m pytest tests/test_scorers_llm_judge.py -v`
Expected: 全部 PASS(旧测试验证重构未破坏行为)

- [ ] **Step 5: 提交**

```bash
git add src/evalith/scorers/llm_judge.py tests/test_scorers_llm_judge.py
git commit -m "feat(judge): panel scoring + usage tracking in LLMJudge (v0.7)"
```

---

### Task 4: 配置接线(rules.py build_scorer)

**Files:**
- Modify: `src/evalith/scorers/rules.py`
- Test: `tests/test_scorers_rules.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_scorers_rules.py` 追加:

```python
from evalith.config import ScorerConfig
from evalith.scorers.rules import build_scorer


def test_build_llm_judge_with_judge_model_and_panel():
    cfg = ScorerConfig(type="llm_judge", params={
        "criteria": "quality",
        "judge_model": "echo:primary-judge",
        "panel": ["echo:panel-a", "echo:panel-b"],
        "consensus_threshold": 0.3,
    })
    judge = build_scorer(cfg, judge_provider=None)
    assert judge.provider.fixed == "primary-judge"        # judge_model resolved
    assert set(judge.panel) == {"echo:panel-a", "echo:panel-b"}
    assert judge.consensus_threshold == 0.3


def test_build_llm_judge_panel_dedupes_primary():
    cfg = ScorerConfig(type="llm_judge", params={
        "judge_model": "echo:j",
        "panel": ["echo:j", "echo:other", "echo:other"],
    })
    judge = build_scorer(cfg, judge_provider=None)
    assert list(judge.panel) == ["echo:other"]            # primary + dup removed


def test_build_llm_judge_defaults_unchanged():
    from evalith.providers.base import FakeProvider
    fp = FakeProvider()
    cfg = ScorerConfig(type="llm_judge", params={"criteria": "q"})
    judge = build_scorer(cfg, judge_provider=fp)
    assert judge.provider is fp                           # falls back, v0.6 behavior
    assert judge.panel == {}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_scorers_rules.py -v`
Expected: 新增测试 FAIL

- [ ] **Step 3: 实现**

`src/evalith/scorers/rules.py` 的 `build_scorer` 中 `llm_judge` 分支替换为:

```python
    if cfg.type == "llm_judge":
        from ..providers import get_provider
        from .llm_judge import LLMJudge

        judge_model = cfg.params.get("judge_model")
        primary = get_provider(judge_model) if judge_model else judge_provider
        panel_models = [m for m in dict.fromkeys(cfg.params.get("panel") or [])
                        if m and m != judge_model]
        panel = {m: get_provider(m) for m in panel_models}
        return LLMJudge(provider=primary,
                        criteria=cfg.params.get("criteria", ""),
                        language=cfg.params.get("language", "en"),
                        panel=panel,
                        consensus_threshold=float(
                            cfg.params.get("consensus_threshold", 0.5)))
```

(`dict.fromkeys` 去重保序;`get_provider` 在函数体内 import 避免循环依赖。)

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_scorers_rules.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/evalith/scorers/rules.py tests/test_scorers_rules.py
git commit -m "feat(config): judge_model/panel/consensus_threshold params for llm_judge (v0.7)"
```

---

### Task 5: Engine 集成(engine.py)

**Files:**
- Modify: `src/evalith/engine.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_engine.py` 追加(panel 用 `echo:<judge JSON>` provider,零网络):

```python
# ---------------------------------------------------------------------------
# Judge consensus panel tests (v0.7)
# ---------------------------------------------------------------------------

PASS_JSON = '{"score": 1.0, "pass": true, "reason": "fine"}'
FAIL_JSON = '{"score": 0.0, "pass": false, "reason": "harsh take"}'


def _panel_cfg(tmp_path, samples=2, panel=None):
    ds = tmp_path / "ds.yaml"
    ds.write_text(
        "name: d\ncases:\n"
        "  - id: c1\n    input: hi\n    domain: code\n"
        "  - id: c2\n    input: yo\n    metadata: {domain: math}\n",
        encoding="utf-8",
    )
    return EvalConfig(
        name="t", dataset=str(ds), model="fake", prompt_template="{{input}}",
        samples=samples,
        scorers=[ScorerConfig(type="llm_judge", params={
            "criteria": "q",
            "judge_model": f"echo:{PASS_JSON}",
            "panel": panel if panel is not None else [f"echo:{FAIL_JSON}"],
        })],
    )


def test_panel_samples_recorded_per_trial(tmp_path):
    cfg = _panel_cfg(tmp_path, samples=2)
    run = run_eval(cfg, FakeProvider(default="answer"))
    cr = run.results[0]
    assert cr.pass_rate_samples == [1.0, 1.0]              # primary still gates
    assert cr.panel_samples[f"echo:{FAIL_JSON}"] == [0.0, 0.0]
    assert cr.panel_details[f"echo:{FAIL_JSON}"] == "harsh take"
    assert run.pass_rate == 1.0                            # panel never gates


def test_domain_copied_from_case_with_metadata_fallback(tmp_path):
    cfg = _panel_cfg(tmp_path)
    run = run_eval(cfg, FakeProvider(default="answer"))
    by_id = {r.case_id: r for r in run.results}
    assert by_id["c1"].domain == "code"                    # top-level field
    assert by_id["c2"].domain == "math"                    # metadata fallback


def test_panel_judge_failure_recorded_as_missing(tmp_path):
    import evalith.scorers.rules as rules_mod
    from evalith.providers.base import EchoProvider

    class BoomProvider:
        model = "boom"

        def complete(self, prompt, *, system=None, temperature=0.0):
            raise RuntimeError("down")

    real_get_provider = rules_mod.__dict__.get("get_provider")  # not present; patch providers
    import evalith.providers as providers_mod
    orig = providers_mod.get_provider

    def fake_get_provider(model):
        if model == "boom":
            return BoomProvider()
        return orig(model)

    providers_mod.get_provider = fake_get_provider
    try:
        cfg = _panel_cfg(tmp_path, samples=2, panel=["boom"])
        run = run_eval(cfg, FakeProvider(default="answer"))
    finally:
        providers_mod.get_provider = orig
    cr = run.results[0]
    assert cr.panel_samples["boom"] == [-1.0, -1.0]        # missing, not zero
    assert cr.pass_rate_samples == [1.0, 1.0]              # primary unaffected


def test_judge_cost_accumulated(tmp_path):
    """Primary + panel judge usage lands in judge_tokens/judge_cost_usd."""
    from evalith.providers.base import Response

    class CostedJudge:
        model = "costed"

        def complete(self, prompt, *, system=None, temperature=0.0):
            return Response(text=PASS_JSON, total_tokens=10, cost_usd=0.001)

    import evalith.providers as providers_mod
    orig = providers_mod.get_provider

    def fake_get_provider(model):
        if model == "costed":
            return CostedJudge()
        return orig(model)

    providers_mod.get_provider = fake_get_provider
    try:
        ds = tmp_path / "ds.yaml"
        ds.write_text("name: d\ncases:\n  - id: c1\n    input: hi\n", encoding="utf-8")
        cfg = EvalConfig(name="t", dataset=str(ds), model="fake",
                         prompt_template="{{input}}", samples=2,
                         scorers=[ScorerConfig(type="llm_judge", params={
                             "judge_model": "costed", "panel": []})])
        run = run_eval(cfg, FakeProvider(default="answer"))
    finally:
        providers_mod.get_provider = orig
    cr = run.results[0]
    assert cr.judge_tokens == 20                # 2 trials × 10 tokens
    assert abs(cr.judge_cost_usd - 0.002) < 1e-9
    assert run.total_judge_tokens == 20


def test_no_panel_runs_byte_identical_to_v06(tmp_path):
    """Without panel config the new fields stay at defaults."""
    ds = tmp_path / "ds.yaml"
    ds.write_text("name: d\ncases:\n  - id: '1'\n    input: hello\n", encoding="utf-8")
    cfg = EvalConfig(name="t", dataset=str(ds), model="fake", prompt_template="{{input}}",
                     scorers=[ScorerConfig(type="contains", params={"text": "x"})])
    run = run_eval(cfg, FakeProvider(default="y"))
    cr = run.results[0]
    assert cr.panel_samples == {} and cr.panel_details == {}
    assert cr.judge_tokens == 0 and cr.judge_cost_usd == 0.0
```

注意 `test_panel_judge_failure_recorded_as_missing` 里 monkeypatch 的对象:`build_scorer` 用 `from ..providers import get_provider` 在**函数体内** import,所以 patch `evalith.providers.get_provider` 生效。测试里前两行残留变量 `real_get_provider` 删掉(写计划时的笔误,执行时直接不要写)。

另外 dataset 里 `metadata: {domain: math}` 的回退要求 engine 写入 `CaseResult.domain` 时做:`case.domain or case.metadata.get("domain")`。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_engine.py -v -k "panel or domain_copied or judge_cost or byte_identical"`
Expected: 新增测试 FAIL

- [ ] **Step 3: 实现**

`src/evalith/engine.py` 改动:

顶部 import 加:

```python
from .consensus import MISSING
```

`run_eval` 内,替换 `_eval_once` 为(签名加 `panel_cache`):

```python
    def _eval_once(case, panel_cache) -> CaseResult:
        domain = case.domain or case.metadata.get("domain")
        try:
            prompt = _render(config.prompt_template, case.input)
            resp = provider.complete(prompt, system=config.system, temperature=config.temperature)
            scores: list[Score] = []
            panel_scores: dict = {}
            judge_tokens, judge_cost = 0, 0.0
            for scorer in scorers:
                if hasattr(scorer, "score_with_usage"):
                    s, tok, c = scorer.score_with_usage(case, resp.text)
                    judge_tokens += tok
                    judge_cost += c
                else:
                    s = scorer.score(case, resp.text)
                scores.append(s)
                if getattr(scorer, "panel", None):
                    pres, tok, c = scorer.panel_score(case, resp.text, panel_cache)
                    judge_tokens += tok
                    judge_cost += c
                    panel_scores.update(pres)
            return CaseResult(
                case_id=case.id, input=case.input, output=resp.text, scores=scores,
                latency_ms=resp.latency_ms, prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens, total_tokens=resp.total_tokens,
                cost_usd=resp.cost_usd, domain=domain,
                judge_tokens=judge_tokens, judge_cost_usd=judge_cost,
                panel_samples={j: [(1.0 if s.passed else 0.0) if s is not None else MISSING]
                               for j, s in panel_scores.items()},
                panel_details={j: s.detail for j, s in panel_scores.items()
                               if s is not None and s.detail},
            )
        except Exception as e:  # one failed case must not kill the whole run
            return CaseResult(case_id=case.id, input=case.input, output="", domain=domain,
                              scores=[Score(scorer="error", value=0.0, passed=False,
                                            detail=f"{type(e).__name__}: {e}")])
```

新增模块级辅助函数(放 `_render` 旁):

```python
def _merge_panel(trials: list[CaseResult],
                 prs: list[float]) -> tuple[dict[str, list[float]], dict[str, str]]:
    """Concatenate per-trial panel values; pick each judge's reason from the
    trial where it disagrees most with the primary pass rate."""
    from .consensus import MISSING
    samples: dict[str, list[float]] = {}
    trial_idx: dict[str, list[int]] = {}
    for ti, t in enumerate(trials):
        for j, vals in t.panel_samples.items():
            samples.setdefault(j, []).extend(vals)
            trial_idx.setdefault(j, []).extend([ti] * len(vals))
    details: dict[str, str] = {}
    for j, vals in samples.items():
        best_k, best_gap = None, -1.0
        for k, v in enumerate(vals):
            if v == MISSING:
                continue
            ti = trial_idx[j][k]
            ref = prs[ti] if ti < len(prs) else 0.0
            gap = abs(v - ref)
            if gap > best_gap:
                best_gap, best_k = gap, k
        if best_k is not None:
            d = trials[trial_idx[j][best_k]].panel_details.get(j, "")
            if d:
                details[j] = d
    return samples, details
```

`_eval_case` 三条路径都接上:

```python
    def _eval_case(case) -> CaseResult:
        panel_cache: dict = {}
        if config.adaptive:
            trials: list[CaseResult] = []
            prs: list[float] = []
            for i in range(config.max_samples):
                trial = _eval_once(case, panel_cache)
                trials.append(trial)
                prs.append(_trial_pass_rate(trial))
                if (i + 1) >= config.min_samples:
                    lo, hi = _bootstrap_mean_ci(prs)
                    if (hi - lo) < config.ci_tolerance:
                        break
            return _aggregate(trials, prs)
        if samples == 1:
            return _eval_once(case, panel_cache)
        trials = [_eval_once(case, panel_cache) for _ in range(samples)]
        return _aggregate(trials, [_trial_pass_rate(t) for t in trials])
```

新增 `_aggregate`(在 `run_eval` 内、`_eval_case` 之前;把 fixed/adaptive 两份重复的聚合代码合并 — DRY):

```python
    def _aggregate(trials: list[CaseResult], prs: list[float]) -> CaseResult:
        rep = trials[0]  # representative trial for output/scores; aggregate the rest
        n = len(trials)
        panel_samples, panel_details = _merge_panel(trials, prs)
        return CaseResult(
            case_id=rep.case_id, input=rep.input, output=rep.output, scores=rep.scores,
            latency_ms=sum(t.latency_ms for t in trials) / n,
            prompt_tokens=sum(t.prompt_tokens for t in trials),
            completion_tokens=sum(t.completion_tokens for t in trials),
            total_tokens=sum(t.total_tokens for t in trials),
            cost_usd=sum(t.cost_usd for t in trials),
            domain=rep.domain,
            judge_tokens=sum(t.judge_tokens for t in trials),
            judge_cost_usd=sum(t.judge_cost_usd for t in trials),
            pass_rate_samples=prs,
            panel_samples=panel_samples,
            panel_details=panel_details,
        )
```

(原 adaptive/fixed 两段手写聚合删除,改调 `_aggregate`。)

- [ ] **Step 4: 跑全部 engine 测试(新旧都过)**

Run: `python -m pytest tests/test_engine.py -v`
Expected: 全部 PASS(旧的 adaptive/samples/error 韧性测试验证聚合重构无回归)

- [ ] **Step 5: 提交**

```bash
git add src/evalith/engine.py tests/test_engine.py
git commit -m "feat(engine): panel judging per trial + judge cost + domain passthrough (v0.7)"
```

---

### Task 6: CLI 摘要行(cli.py)

**Files:**
- Modify: `src/evalith/cli.py:63-67`(run 命令的结果输出处)
- Test: `tests/test_cli.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_cli.py` 追加(沿用该文件现有的 `CliRunner` 风格;先读一眼现有测试的 fixture 写法再落笔,保持一致):

```python
def test_run_prints_panel_summary(tmp_path):
    PASS_JSON = '{"score": 1.0, "pass": true, "reason": "ok"}'
    FAIL_JSON = '{"score": 0.0, "pass": false, "reason": "no"}'
    ds = tmp_path / "ds.yaml"
    ds.write_text("name: d\ncases:\n  - id: c1\n    input: hi\n", encoding="utf-8")
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(f"""
name: panel-t
dataset: {ds}
model: "echo:answer"
samples: 2
scorers:
  - type: llm_judge
    params:
      judge_model: "echo:{PASS_JSON}"
      panel: ["echo:{FAIL_JSON}"]
""", encoding="utf-8")
    result = runner.invoke(app, ["run", str(cfg), "--store", str(tmp_path / "s")])
    assert result.exit_code == 0
    assert "panel: 2 judges" in result.output
    assert "1/1 low-consensus cases" in result.output
    assert "min pairwise" in result.output


def test_run_no_panel_no_summary_line(tmp_path):
    ds = tmp_path / "ds.yaml"
    ds.write_text("name: d\ncases:\n  - id: c1\n    input: hi\n", encoding="utf-8")
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(f"""
name: t
dataset: {ds}
model: "echo:hi"
scorers:
  - type: contains
    params: {{text: hi}}
""", encoding="utf-8")
    result = runner.invoke(app, ["run", str(cfg), "--store", str(tmp_path / "s")])
    assert result.exit_code == 0
    assert "panel:" not in result.output
```

注意 YAML 里 judge JSON 含 `{`/`}` 与冒号,所以 model 值必须带引号(如上)。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_cli.py -v -k panel`
Expected: FAIL(无 panel 摘要行)

- [ ] **Step 3: 实现**

`src/evalith/cli.py` 的 `run` 命令,在 `typer.echo(f"Run {result.id} saved ...")` 之后、`fail_under` 判定之前插入:

```python
    from .consensus import consensus_summary, threshold_from_config
    cs = consensus_summary(result, threshold_from_config(result.config))
    if cs:
        k = cs["min_kappa"]
        k_s = "n/a" if k != k else f"{k:+.2f}"   # NaN check
        typer.echo(f"panel: {len(cs['judges'])} judges, "
                   f"{len(cs['low_consensus_cases'])}/{cs['n_cases']} low-consensus cases, "
                   f"min pairwise κ={k_s}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_cli.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/evalith/cli.py tests/test_cli.py
git commit -m "feat(cli): panel consensus summary line after run (v0.7)"
```

---

### Task 7: 报告渲染(report.py)

**Files:**
- Modify: `src/evalith/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_report.py` 追加:

```python
from datetime import datetime, timezone

from evalith.models import CaseResult, Run, Score
from evalith.report import run_to_html, run_to_markdown


def _panel_run():
    s = Score(scorer="llm_judge", value=1.0, passed=True, detail="ok")
    return Run(id="rid", name="panel-run",
               created_at=datetime(2026, 6, 6, tzinfo=timezone.utc), model="m",
               config={"scorers": [{"type": "llm_judge",
                                    "params": {"consensus_threshold": 0.5}}]},
               results=[
                   CaseResult(case_id="agree", input="i", output="o", scores=[s],
                              pass_rate_samples=[1.0, 1.0], domain="safety",
                              panel_samples={"qw": [1.0, 1.0]},
                              judge_tokens=10, judge_cost_usd=0.001),
                   CaseResult(case_id="fight", input="i", output="o", scores=[s],
                              pass_rate_samples=[1.0, 1.0], domain="code",
                              panel_samples={"qw": [0.0, 0.0]},
                              panel_details={"qw": "code does not compile"},
                              judge_tokens=10, judge_cost_usd=0.001),
               ])


def test_markdown_has_consensus_section():
    md = run_to_markdown(_panel_run())
    assert "## Judge Consensus" in md
    assert "fight" in md                       # low-consensus case listed
    assert "code does not compile" in md       # judge reason shown
    assert "⚠" in md
    assert "$0.0020" in md                     # judge cost split out


def test_markdown_no_consensus_section_without_panel():
    run = _panel_run()
    for r in run.results:
        r.panel_samples = {}
        r.panel_details = {}
    md = run_to_markdown(run)
    assert "Judge Consensus" not in md


def test_html_has_consensus_section():
    html = run_to_html(_panel_run())
    assert "Judge Consensus" in html
    assert "code does not compile" in html
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_report.py -v`
Expected: 新增 3 个 FAIL

- [ ] **Step 3: 实现**

`src/evalith/report.py`:

顶部 import 加:

```python
from .consensus import (case_spread, consensus_summary, domain_agreement,
                        judge_means, threshold_from_config)
```

新增辅助(放 `run_to_markdown` 之前):

```python
_KAPPA_NOTE = ("*κ collapses toward 0 when one judge's labels are nearly constant "
               "(expected-agreement inflation) — read it together with the means.*")


def _fmt_kappa(v: float) -> str:
    return "n/a" if v != v else f"{v:+.3f}"


def _consensus_md(run: Run) -> list[str]:
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
    lines += [f"| {a} | {b} | {_fmt_kappa(v)} |" for (a, b), v in cs["kappa"].items()]
    lines += ["", _KAPPA_NOTE]
    doms = domain_agreement(run, th)
    if any(d != "?" for d in doms):
        lines += ["", "### By domain", "",
                  "| domain | n | " + " | ".join(judges) + " | low-consensus |",
                  "| --- | --- | " + " | ".join("---" for _ in judges) + " | --- |"]
        for dom, info in doms.items():
            cells = " | ".join(f"{info['means'][j]:.2f}" if j in info["means"] else "n/a"
                               for j in judges)
            lines.append(f"| {dom} | {info['n']} | {cells} | {info['low_consensus']} |")
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
```

`run_to_markdown` 末尾 `return` 前接上:

```python
    lines += _consensus_md(run)
    return "\n".join(lines) + "\n"
```

并把 Cost 行改为(仅有 judge 成本时追加,旧输出不变):

```python
    cost_line = (f"- **Cost:** ${run.total_cost_usd:.4f}  ·  {run.total_tokens} tokens  ·  "
                 f"{run.mean_latency_ms:.0f} ms/case avg")
    if run.total_judge_cost_usd > 0:
        cost_line += f"  ·  judging ${run.total_judge_cost_usd:.4f}"
```

(`lines` 列表里原来的 Cost 行换成 `cost_line` 变量。)

`run_to_html` 在 `body` 拼接 `</table>` 之后追加:

```python
    cmd = _consensus_md(run)
    if cmd:
        import re as _re
        items = "".join(f"<p>{escape(line)}</p>" for line in cmd if line and not line.startswith("|"))
        body += f"<hr>{items}"
```

HTML 版做成段落级简版即可(md 是主要消费格式,HTML 表格渲染留待有需求时再做)——测试只断言 section 标题与 reason 出现。注意 `escape` 已在文件头 import。`import re as _re` 不需要,删掉(笔误,执行时不要写)。

- [ ] **Step 4: 跑测试确认通过(新旧都过)**

Run: `python -m pytest tests/test_report.py -v`
Expected: 全部 PASS(旧 12 个快照式测试不受影响——无 panel 时输出逐字节不变)

- [ ] **Step 5: 提交**

```bash
git add src/evalith/report.py tests/test_report.py
git commit -m "feat(report): Judge Consensus section in md/html + judging cost split (v0.7)"
```

---

### Task 8: 版本号、示例配置、文档、全量回归

**Files:**
- Modify: `pyproject.toml:3`(0.6.0 → 0.7.0)
- Create: `examples/eval.panel.yaml`
- Modify: `README.md`、`README.zh-CN.md`(功能列表加一条)

- [ ] **Step 1: 版本号**

`pyproject.toml`: `version = "0.7.0"`

- [ ] **Step 2: 示例配置**

创建 `examples/eval.panel.yaml`(照抄 `examples/eval.yaml` 的结构改造;先读它确认字段后落笔):

```yaml
# Judge consensus panel: primary judge gates, panel judges diagnose.
# Run: DEEPSEEK_API_KEY=... DASHSCOPE_API_KEY=... evalith run examples/eval.panel.yaml
name: panel-demo
dataset: examples/qa.yaml
model: deepseek-chat
samples: 3
scorers:
  - type: llm_judge
    params:
      criteria: 回答应当准确、覆盖核心概念
      language: zh
      judge_model: deepseek-chat      # primary judge — the only one that gates
      panel: [qwen-plus]              # diagnostic judges — agreement stats only
      consensus_threshold: 0.5        # spread >= 0.5 flags ⚠ low consensus
```

(`dataset:` 指向 examples 下已有的数据集文件,执行时 `ls examples/` 确认实际文件名。)

- [ ] **Step 3: README 双语各加一条功能 bullet**

在 README.md 功能列表(找 "bootstrap CI" 那条附近)加:

```markdown
- **Judge consensus panel** — attach extra judges to one eval; get per-case disagreement, pairwise Cohen's κ, per-domain agreement and ⚠ low-consensus flags. Primary judge still gates; the panel never blocks CI.
```

README.zh-CN.md 对应位置加:

```markdown
- **多 judge 共识面板** —— 一次 eval 挂多个 judge:per-case 分歧、pairwise Cohen's κ、分领域一致性、⚠ 低共识标记。主 judge 照常 gate,panel 只诊断不拦截 CI。
```

- [ ] **Step 4: 全量测试 + 冒烟**

```bash
python -m pytest -q
evalith run examples/eval.yaml --store /tmp/ev-smoke   # 旧路径冒烟(echo,零成本)
```

Expected: 全部 PASS;冒烟输出无 `panel:` 行。

- [ ] **Step 5: 提交**

```bash
git add pyproject.toml examples/eval.panel.yaml README.md README.zh-CN.md
git commit -m "chore: v0.7.0 — panel example config + README feature bullets"
```

---

### Task 9: 验收实验(真实 API,spec 验收标准 2)

**Files:**
- Create: `docs/blog/article4/configs/eval.panel-accept.yaml`(验收用,跑完保留)

- [ ] **Step 1: 写验收配置**

GLM(SiliconFlow)需要自定义 api_base,presets 不支持 → 验收 panel 用 `qwen-plus` 单 panel judge(文章 4 的 code 领域撕裂正是 ds vs qw,足以复现领域结构)。

```yaml
name: panel-accept-v07
dataset: docs/blog/article4/qa.small.yaml
model: deepseek-chat
temperature: 1.0
samples: 3
concurrency: 4
scorers:
  - type: llm_judge
    params:
      criteria: 回答应当准确、相关、覆盖问题的核心概念,无事实错误
      language: zh
      judge_model: deepseek-chat
      panel: [qwen-plus]
```

(criteria 与 `docs/blog/article2/configs/eval.high-temp.yaml` 对齐,执行时核对原文。)

- [ ] **Step 2: 跑验收(约 90 gen + 180 judge 调用,几分钟)**

```bash
DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY DASHSCOPE_API_KEY=$DASHSCOPE_API_KEY \
  evalith run docs/blog/article4/configs/eval.panel-accept.yaml \
  --store /tmp/ev-accept --out /tmp/panel-accept.json
evalith report $(ls /tmp/ev-accept | head -1) --store /tmp/ev-accept --format md
```

- [ ] **Step 3: 核对验收标准**

- CLI 出现 `panel: 2 judges, N/30 low-consensus cases, min pairwise κ=...`
- md 报告 By domain 表:code 行 primary 与 qw 差值显著大于 safety/concept 行(复现文章 4 领域结构)
- 报告 Cost 行带 `judging $...`,⚠ 列表带 qw 的 reason

- [ ] **Step 4: 提交验收配置(不提交 /tmp 产物)**

```bash
git add docs/blog/article4/configs/eval.panel-accept.yaml
git commit -m "test(accept): real-API panel acceptance config for v0.7"
```

---

## Self-Review 记录

- **Spec 覆盖**:配置(T4)、数据模型(T1)、engine 含缓存/韧性/成本(T5)、consensus 统计(T2)、报告+CLI(T6/T7)、兼容测试(T1/T5/T7)、验收(T9)。spec 全部 8 节有对应任务。
- **偏差**:①domain 字段化(spec 写 metadata.domain,见文件头偏差说明);②缓存仅 panel;③验收 panel 降为 [qwen-plus](GLM 需自定义 api_base,presets 不支持);④HTML 共识节为段落级简版。
- **类型一致性**:`panel_score(case, output, cache) -> (dict[str, Score|None], int, float)`、`score_with_usage -> (Score, int, float)`、`MISSING=-1.0` 全计划统一;`judge_means`/`consensus_summary`/`threshold_from_config` 名称在 T2/T6/T7 一致。
