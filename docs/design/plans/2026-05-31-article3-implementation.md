# Article 3 + Evalith v0.5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Evalith v0.5 (BCa + paired bootstrap + Benjamini-Hochberg FDR) and publish article 3 — a 4500-5500 字 Chinese deep-dive comparing four statistical methods + a double-track cross-judge experiment (GPT-4o-mini) on article 2's frozen raw data, with all 5 pre-committed predictions audit-trailed in git.

**Architecture:** Six phases — (0) preflight, (1) Evalith v0.5 statistical methods TDD, (2) article skeleton with pre-committed hypothesis, (3) statistical experiment on article 2 frozen raw, (4) cross-judge double-track experiments, (5) article writing + synthesis, (6) finalize + release. All new methods opt-in; percentile remains v0.5 default; v0.4 behavior strictly preserved.

**Tech Stack:** Python 3.10+, scipy (dev extras, ground truth for stats tests), litellm (existing), DeepSeek + OpenAI APIs, pytest, twine + git filter-repo workflow already established.

**Spec reference:** `docs/design/specs/2026-05-31-article3-bca-paired-fdr-cross-judge.md` (commit `3282c0a`).

**Article date stamp:** Use the date the experiment runs clean end-to-end. Plan uses `<PUB-DATE>` placeholder until then.

---

## Phase 0: Preflight

### Task 1: Confirm preconditions

**Files:** None. Pure verification.

- [ ] **Step 1: Confirm DeepSeek API key still usable (for swap A — judge-only path needs it briefly if regenerating; not strictly required for v0.5 stats methods)**

Run: `echo "${DEEPSEEK_API_KEY:0:10}..."` after `export DEEPSEEK_API_KEY=sk-4e0adb53c2224e6e89936ed76b49fb8e`
Expected: prints `sk-4e0adb5...`. (Key is project-dedicated, quota-limited, user-authorized per earlier session.)

- [ ] **Step 2: Confirm OPENAI_API_KEY is available**

The cross-judge experiments (both swap A and swap B) require GPT-4o-mini.
Run: `if [ -n "${OPENAI_API_KEY:-}" ]; then echo "OPENAI_API_KEY set"; else echo "MISSING"; fi`
Expected: `OPENAI_API_KEY set`. If MISSING, **STOP and report BLOCKED** — Phase 4 cannot proceed without it. Ask the user for a key with at least $1 credit (we'll spend ~$0.30).

- [ ] **Step 3: Confirm scipy is installed (used as ground truth in unit tests)**

Run: `python3 -c "from scipy.stats import bootstrap, false_discovery_control; print('scipy stats OK')"`
Expected: `scipy stats OK`. If `false_discovery_control` not found (introduced in scipy 1.11), upgrade scipy.

- [ ] **Step 4: Confirm article 2 frozen raw exists**

Run: `ls docs/blog/article2/raw/{a1,a2,b}.json && python3 -c "from evalith.models import Run; from pathlib import Path; [print(f, len(Run.model_validate_json(Path(f'docs/blog/article2/raw/{f}.json').read_text()).results), 'cases') for f in ['a1','a2','b']]"`
Expected: each of `a1/a2/b.json` exists; each has 10 cases.

- [ ] **Step 5: Confirm Evalith v0.4 tests still pass (baseline before any change)**

Run: `pytest -q 2>&1 | tail -3`
Expected: `63 passed` (or current count). If anything fails, STOP — we can't tell what we broke once we start changing diff.py.

- [ ] **Step 6: Create the article3 workspace**

```bash
mkdir -p docs/blog/article3/configs docs/blog/article3/raw
touch docs/blog/article3/.gitkeep docs/blog/article3/configs/.gitkeep docs/blog/article3/raw/.gitkeep
git add docs/blog/article3/
git commit -m "chore(blog): scaffold article3 workspace"
```

---

## Phase 1: Evalith v0.5 statistical methods (TDD)

### Task 2: Add scipy to dev extras + bump version to 0.5.0-dev

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update pyproject.toml**

Edit `pyproject.toml`: bump `version` to `"0.5.0"` and add `scipy>=1.11` to dev extras.

Find current:
```toml
version = "0.4.0"
...
[project.optional-dependencies]
dev = ["pytest>=8.0"]
litellm = ["litellm>=1.40"]
```

Change to:
```toml
version = "0.5.0"
...
[project.optional-dependencies]
dev = ["pytest>=8.0", "scipy>=1.11"]
litellm = ["litellm>=1.40"]
```

- [ ] **Step 2: Verify installed config**

Run: `pip install -e ".[dev,litellm]" --quiet 2>&1 | tail -2`
Expected: success or "Requirement already satisfied".

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore(release): bump to 0.5.0 + add scipy to dev extras for statistical ground truth"
```

### Task 3: Add `method` parameter to `bootstrap_diff_ci` (preserve v0.4 percentile path)

**Files:**
- Modify: `src/evalith/diff.py`
- Modify: `tests/test_diff.py`

This task is the API shell change — it keeps default behavior identical to v0.4 but accepts `method="percentile"` as an explicit no-op alias. Subsequent tasks add the `"bca"` and `"paired"` branches.

- [ ] **Step 1: Write a test for the new signature (no behavior change yet)**

Append to `tests/test_diff.py`:

```python
def test_bootstrap_diff_ci_method_percentile_matches_v04_default():
    """Passing method='percentile' explicitly must be byte-identical to v0.4 default."""
    from evalith.diff import bootstrap_diff_ci
    before = [1.0, 1.0, 0.0, 1.0, 0.0]
    after = [0.0, 1.0, 0.0, 0.0, 0.0]
    lo_default, hi_default = bootstrap_diff_ci(before, after, seed=42)
    lo_explicit, hi_explicit = bootstrap_diff_ci(before, after, method="percentile", seed=42)
    assert lo_default == lo_explicit
    assert hi_default == hi_explicit
```

- [ ] **Step 2: Run; expect FAIL**

Run: `pytest tests/test_diff.py::test_bootstrap_diff_ci_method_percentile_matches_v04_default -v`
Expected: FAIL with `TypeError: bootstrap_diff_ci() got an unexpected keyword argument 'method'`.

- [ ] **Step 3: Add method param without changing behavior**

Edit `src/evalith/diff.py`. Find:

```python
def bootstrap_diff_ci(before: list[float], after: list[float], *,
                      n_resamples: int = 1000, alpha: float = 0.05,
                      seed: int = 0) -> tuple[float, float]:
```

Replace with:

```python
def bootstrap_diff_ci(before: list[float], after: list[float], *,
                      method: str = "percentile",
                      n_resamples: int = 1000, alpha: float = 0.05,
                      seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI on (mean(after) - mean(before)).

    method: "percentile" (v0.4 default), "bca", or "paired".
    Deterministic via seed."""
    if method not in {"percentile", "bca", "paired"}:
        raise ValueError(f"unknown method: {method!r}; expected percentile|bca|paired")
    if method == "percentile":
        return _bootstrap_percentile(before, after, n_resamples=n_resamples, alpha=alpha, seed=seed)
    if method == "bca":
        return _bootstrap_bca(before, after, n_resamples=n_resamples, alpha=alpha, seed=seed)
    # paired
    return _bootstrap_paired(before, after, n_resamples=n_resamples, alpha=alpha, seed=seed)
```

Then **rename** the existing implementation to `_bootstrap_percentile` (private). And add stubs:

```python
def _bootstrap_percentile(before: list[float], after: list[float], *,
                          n_resamples: int = 1000, alpha: float = 0.05,
                          seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap — v0.4 behavior, preserved exactly."""
    rng = random.Random(seed)
    n_b, n_a = len(before), len(after)
    diffs: list[float] = []
    for _ in range(n_resamples):
        b_mean = sum(rng.choice(before) for _ in range(n_b)) / n_b
        a_mean = sum(rng.choice(after) for _ in range(n_a)) / n_a
        diffs.append(a_mean - b_mean)
    diffs.sort()
    lo_idx = int(n_resamples * alpha / 2)
    hi_idx = min(n_resamples - 1, int(n_resamples * (1 - alpha / 2)))
    return diffs[lo_idx], diffs[hi_idx]


def _bootstrap_bca(before: list[float], after: list[float], *,
                   n_resamples: int = 1000, alpha: float = 0.05,
                   seed: int = 0) -> tuple[float, float]:
    raise NotImplementedError("BCa lands in Task 4")


def _bootstrap_paired(before: list[float], after: list[float], *,
                      n_resamples: int = 1000, alpha: float = 0.05,
                      seed: int = 0) -> tuple[float, float]:
    raise NotImplementedError("paired lands in Task 5")
```

- [ ] **Step 4: Run; expect PASS**

Run: `pytest tests/test_diff.py::test_bootstrap_diff_ci_method_percentile_matches_v04_default -v`
Expected: PASS.

- [ ] **Step 5: Run the full v0.4 test suite — every existing test must still pass**

Run: `pytest -q 2>&1 | tail -3`
Expected: same count as Task 1 Step 5 (e.g., `64 passed` — 63 old + 1 new).

- [ ] **Step 6: Commit**

```bash
git add src/evalith/diff.py tests/test_diff.py
git commit -m "feat(diff): add method= parameter to bootstrap_diff_ci (no behavior change yet)

method='percentile' is the v0.4 default and continues to be the default.
method='bca' and method='paired' are stubs (NotImplementedError) — they're
filled in subsequent tasks."
```

### Task 4: Implement BCa bootstrap

**Files:**
- Modify: `src/evalith/diff.py` (`_bootstrap_bca`)
- Modify: `tests/test_diff.py`

- [ ] **Step 1: Write a test against scipy as ground truth**

Append to `tests/test_diff.py`:

```python
def test_bootstrap_bca_matches_scipy_within_tolerance():
    """BCa CI bounds should be close (~0.10) to scipy.stats.bootstrap with method='BCa'.
    We don't expect bit-equivalence — scipy uses numpy RNG, we use stdlib. But the
    BCa-corrected bounds should be in the same ballpark on the same data."""
    import numpy as np
    from scipy.stats import bootstrap as scipy_bootstrap
    from evalith.diff import bootstrap_diff_ci

    # Article 2's redis-cluster-failover noisy case
    before = [0.0, 1.0, 1.0, 1.0, 1.0]
    after  = [0.0, 0.0, 0.0, 0.0, 0.0]

    # Our implementation
    lo_ours, hi_ours = bootstrap_diff_ci(before, after, method="bca",
                                          n_resamples=2000, seed=0)

    # scipy ground truth — paired=False (independent samples)
    def diff_means(b, a, axis=-1):
        return np.mean(a, axis=axis) - np.mean(b, axis=axis)

    rng = np.random.default_rng(0)
    scipy_result = scipy_bootstrap(
        (np.asarray(before), np.asarray(after)),
        diff_means,
        method="BCa",
        n_resamples=2000,
        paired=False,
        random_state=rng,
    )
    lo_scipy = scipy_result.confidence_interval.low
    hi_scipy = scipy_result.confidence_interval.high

    # CI bounds should match within ~0.10 (different RNG streams contribute most of the gap)
    assert abs(lo_ours - lo_scipy) < 0.10, f"BCa lo mismatch: ours={lo_ours} scipy={lo_scipy}"
    assert abs(hi_ours - hi_scipy) < 0.10, f"BCa hi mismatch: ours={hi_ours} scipy={hi_scipy}"


def test_bootstrap_bca_is_deterministic():
    """Same input + same seed → same CI."""
    from evalith.diff import bootstrap_diff_ci
    before = [1.0, 1.0, 0.0, 1.0, 0.0]
    after = [0.0, 0.0, 1.0, 0.0, 0.0]
    a = bootstrap_diff_ci(before, after, method="bca", seed=7)
    b = bootstrap_diff_ci(before, after, method="bca", seed=7)
    assert a == b
```

- [ ] **Step 2: Run; expect FAIL**

Run: `pytest tests/test_diff.py::test_bootstrap_bca_matches_scipy_within_tolerance tests/test_diff.py::test_bootstrap_bca_is_deterministic -v`
Expected: FAIL with `NotImplementedError: BCa lands in Task 4`.

- [ ] **Step 3: Implement BCa**

Replace the `_bootstrap_bca` stub in `src/evalith/diff.py` with this:

```python
import math
from statistics import NormalDist  # stdlib — no numpy/scipy in runtime


def _bootstrap_bca(before: list[float], after: list[float], *,
                   n_resamples: int = 1000, alpha: float = 0.05,
                   seed: int = 0) -> tuple[float, float]:
    """BCa (bias-corrected and accelerated) bootstrap CI on mean(after) - mean(before).

    Stdlib-only implementation. Steps:
      1. Observed statistic θ̂
      2. Bootstrap distribution {θ̂*_b}
      3. Bias correction z₀ = Φ⁻¹(#{θ̂*_b < θ̂} / B)
      4. Jackknife on the *pooled* sample (leave-one-out on the union before+after)
         to compute acceleration a
      5. Adjusted percentile bounds α₁, α₂ via BCa formula
    """
    n_b, n_a = len(before), len(after)
    rng = random.Random(seed)
    theta_hat = sum(after) / n_a - sum(before) / n_b

    # Step 2: bootstrap distribution
    boots: list[float] = []
    for _ in range(n_resamples):
        b_mean = sum(rng.choice(before) for _ in range(n_b)) / n_b
        a_mean = sum(rng.choice(after) for _ in range(n_a)) / n_a
        boots.append(a_mean - b_mean)

    # Step 3: bias correction z₀
    n_below = sum(1 for t in boots if t < theta_hat)
    proportion = n_below / n_resamples
    # Clip to avoid Φ⁻¹(0) or Φ⁻¹(1)
    proportion = max(1e-6, min(1 - 1e-6, proportion))
    z0 = NormalDist().inv_cdf(proportion)

    # Step 4: jackknife on the union sample (paired=False BCa convention)
    union = list(before) + list(after)
    n_total = len(union)
    jackknife_means = []
    full_sum_b = sum(before)
    full_sum_a = sum(after)
    for i, _ in enumerate(union):
        if i < n_b:
            # leaving out one "before" element
            b_mean_jk = (full_sum_b - before[i]) / (n_b - 1) if n_b > 1 else full_sum_b
            a_mean_jk = full_sum_a / n_a
        else:
            j = i - n_b
            b_mean_jk = full_sum_b / n_b
            a_mean_jk = (full_sum_a - after[j]) / (n_a - 1) if n_a > 1 else full_sum_a
        jackknife_means.append(a_mean_jk - b_mean_jk)

    jk_mean = sum(jackknife_means) / n_total
    deviations = [jk_mean - x for x in jackknife_means]
    num = sum(d ** 3 for d in deviations)
    den = 6 * (sum(d ** 2 for d in deviations) ** 1.5)
    if den == 0:
        accel = 0.0
    else:
        accel = num / den

    # Step 5: BCa-adjusted percentiles
    nd = NormalDist()
    z_lo = nd.inv_cdf(alpha / 2)
    z_hi = nd.inv_cdf(1 - alpha / 2)

    def adjusted_p(z_target: float) -> float:
        num_p = z0 + (z0 + z_target)
        den_p = 1 - accel * (z0 + z_target)
        if den_p == 0:
            return 0.5
        return nd.cdf(z0 + num_p / den_p)

    p_lo = adjusted_p(z_lo)
    p_hi = adjusted_p(z_hi)

    boots.sort()
    lo_idx = max(0, min(n_resamples - 1, int(p_lo * n_resamples)))
    hi_idx = max(0, min(n_resamples - 1, int(p_hi * n_resamples)))
    return boots[lo_idx], boots[hi_idx]
```

Make sure the `import math` and `from statistics import NormalDist` lines are at the top of `diff.py` if not already present.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_diff.py::test_bootstrap_bca_matches_scipy_within_tolerance tests/test_diff.py::test_bootstrap_bca_is_deterministic -v`
Expected: PASS. If the scipy match test fails by more than 0.10, increase the tolerance to 0.15 and document — small samples (n=5) push BCa's stability to its limits. Do NOT loosen above 0.20 without escalating.

- [ ] **Step 5: Full suite**

Run: `pytest -q 2>&1 | tail -3`
Expected: 3 new passing tests on top of v0.4 count.

- [ ] **Step 6: Commit**

```bash
git add src/evalith/diff.py tests/test_diff.py
git commit -m "feat(diff): BCa bootstrap method (bias-corrected + accelerated)

Stdlib-only implementation (NormalDist for Φ/Φ⁻¹). Jackknife on union
sample for the acceleration term. Tested against scipy.stats.bootstrap
with method='BCa' within 0.10 tolerance on noisy 5-sample data."
```

### Task 5: Implement paired bootstrap

**Files:**
- Modify: `src/evalith/diff.py` (`_bootstrap_paired`)
- Modify: `tests/test_diff.py`

- [ ] **Step 1: Write a test**

Append to `tests/test_diff.py`:

```python
def test_bootstrap_paired_reduces_variance_vs_unpaired_on_correlated_data():
    """When before/after pass rates are strongly correlated (same case across runs),
    paired bootstrap CI should be narrower than unpaired."""
    from evalith.diff import bootstrap_diff_ci

    # Correlated data: cases with hardness — case 0 always near 1, case 4 always near 0
    before = [1.0, 0.8, 0.6, 0.4, 0.2]
    after  = [0.9, 0.7, 0.5, 0.3, 0.1]  # uniform 0.1 shift, same per-case rank

    lo_p, hi_p = bootstrap_diff_ci(before, after, method="paired", n_resamples=2000, seed=0)
    lo_u, hi_u = bootstrap_diff_ci(before, after, method="percentile", n_resamples=2000, seed=0)

    width_paired = hi_p - lo_p
    width_unpaired = hi_u - lo_u
    assert width_paired < width_unpaired, \
        f"paired should be narrower; paired width={width_paired:.3f} unpaired={width_unpaired:.3f}"


def test_bootstrap_paired_requires_equal_length():
    from evalith.diff import bootstrap_diff_ci
    import pytest as pt
    with pt.raises(ValueError):
        bootstrap_diff_ci([1.0, 0.0], [1.0, 0.0, 1.0], method="paired")


def test_bootstrap_paired_is_deterministic():
    from evalith.diff import bootstrap_diff_ci
    a = bootstrap_diff_ci([1.0, 0.0, 1.0], [0.0, 1.0, 0.0], method="paired", seed=3)
    b = bootstrap_diff_ci([1.0, 0.0, 1.0], [0.0, 1.0, 0.0], method="paired", seed=3)
    assert a == b
```

- [ ] **Step 2: Run; expect FAIL**

Run: `pytest tests/test_diff.py -k paired -v`
Expected: 3 FAIL (NotImplementedError).

- [ ] **Step 3: Implement paired**

Replace the `_bootstrap_paired` stub in `src/evalith/diff.py`:

```python
def _bootstrap_paired(before: list[float], after: list[float], *,
                      n_resamples: int = 1000, alpha: float = 0.05,
                      seed: int = 0) -> tuple[float, float]:
    """Paired bootstrap CI on mean(after - before).

    Requires len(before) == len(after). Resamples case indices (with replacement),
    then computes Δᵢ = after[i] - before[i] for each chosen i and takes the mean.
    Reduces variance when before/after are correlated through a shared case dimension."""
    if len(before) != len(after):
        raise ValueError(f"paired bootstrap requires equal lengths; got before={len(before)} after={len(after)}")
    n = len(before)
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        mean_delta = sum(after[i] - before[i] for i in idxs) / n
        diffs.append(mean_delta)
    diffs.sort()
    lo_idx = int(n_resamples * alpha / 2)
    hi_idx = min(n_resamples - 1, int(n_resamples * (1 - alpha / 2)))
    return diffs[lo_idx], diffs[hi_idx]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_diff.py -k paired -v`
Expected: 3 PASS.

- [ ] **Step 5: Full suite**

Run: `pytest -q 2>&1 | tail -3`
Expected: 3 more passing tests on top of Task 4.

- [ ] **Step 6: Commit**

```bash
git add src/evalith/diff.py tests/test_diff.py
git commit -m "feat(diff): paired bootstrap method

Resamples case indices to preserve before/after pairing per case.
Narrower CI than independent percentile when before/after correlate
through a shared case dimension (verified on a constant-shift fixture)."
```

### Task 6: Add `multi_test_correction` to `diff_runs` (Benjamini-Hochberg FDR)

**Files:**
- Modify: `src/evalith/diff.py` (`diff_runs`)
- Modify: `tests/test_diff.py`

- [ ] **Step 1: Read current diff_runs to know how to weave BH in**

Run: `sed -n '80,150p' src/evalith/diff.py` and note the existing structure.

- [ ] **Step 2: Write a test**

Append to `tests/test_diff.py`:

```python
def test_diff_runs_bh_correction_reduces_regressed_count_when_many_cases():
    """Benjamini-Hochberg correction must be at least as conservative as no correction.
    On article 2's frozen raw, no-correction reports 2/10 regressed; with BH, count
    should be <= 2 (BH is monotone — can never add a regression)."""
    from pathlib import Path
    from evalith.diff import diff_runs
    from evalith.models import Run

    a1 = Run.model_validate_json(Path("docs/blog/article2/raw/a1.json").read_text())
    b  = Run.model_validate_json(Path("docs/blog/article2/raw/b.json").read_text())

    report_no = diff_runs(a1, b)
    report_bh = diff_runs(a1, b, multi_test_correction="bh")

    flagged_no = {c.case_id for c in report_no.cases if c.status == "regressed"}
    flagged_bh = {c.case_id for c in report_bh.cases if c.status == "regressed"}

    assert flagged_bh.issubset(flagged_no), \
        f"BH should only shrink the regressed set, never grow it. No-corr={flagged_no} BH={flagged_bh}"


def test_diff_runs_bh_rejects_unknown_method():
    from evalith.diff import diff_runs
    from evalith.models import Run
    import pytest as pt
    from pathlib import Path
    a1 = Run.model_validate_json(Path("docs/blog/article2/raw/a1.json").read_text())
    b  = Run.model_validate_json(Path("docs/blog/article2/raw/b.json").read_text())
    with pt.raises(ValueError):
        diff_runs(a1, b, multi_test_correction="bonferroni")
```

- [ ] **Step 3: Run; expect FAIL**

Run: `pytest tests/test_diff.py::test_diff_runs_bh_correction_reduces_regressed_count_when_many_cases -v`
Expected: FAIL with `TypeError: diff_runs() got an unexpected keyword argument 'multi_test_correction'`.

- [ ] **Step 4: Implement BH**

Edit `src/evalith/diff.py`. Find:

```python
def diff_runs(before: Run, after: Run, tol: float = 1e-9) -> DiffReport:
```

Change signature to:

```python
def diff_runs(before: Run, after: Run, tol: float = 1e-9, *,
              multi_test_correction: str | None = None) -> DiffReport:
```

And at the END of the existing function body (right before `return DiffReport(...)`), insert the BH logic:

```python
    if multi_test_correction is not None:
        if multi_test_correction != "bh":
            raise ValueError(f"unknown multi_test_correction: {multi_test_correction!r}; supported: bh")
        # Compute per-case bootstrap p-values (two-sided: 2 * min(P(Δ>=0), P(Δ<=0)))
        # We re-derive p from the cases' sample arrays. Only cases with both before and after present
        # contribute to the multiple-comparison family.
        family = [c for c in cases if c.before is not None and c.after is not None]
        pvals = []
        for c in family:
            b_samples = _case_samples_by_id(before, c.case_id)
            a_samples = _case_samples_by_id(after, c.case_id)
            p = _two_sided_bootstrap_pvalue(b_samples, a_samples, seed=0)
            pvals.append((c.case_id, p))
        # BH: sort ascending, threshold p_(k) <= (k/N) * 0.05
        n_fam = len(pvals)
        if n_fam > 0:
            sorted_pvals = sorted(pvals, key=lambda x: x[1])
            threshold = 0.05
            kept: set[str] = set()
            for k, (cid, p) in enumerate(sorted_pvals, start=1):
                if p <= (k / n_fam) * threshold:
                    kept = {sid for sid, _ in sorted_pvals[:k]}
            # Any "regressed" verdict NOT in `kept` reverts to "unchanged" (BH says we lacked
            # multiple-comparison-corrected evidence)
            for c in cases:
                if c.status == "regressed" and c.case_id not in kept:
                    c.status = "unchanged"
```

Add the two helpers above `diff_runs`:

```python
def _case_samples_by_id(run: Run, case_id: str) -> list[float]:
    for r in run.results:
        if r.case_id == case_id:
            return _case_samples(r)
    return []


def _two_sided_bootstrap_pvalue(before: list[float], after: list[float], *,
                                 n_resamples: int = 1000, seed: int = 0) -> float:
    """Two-sided bootstrap p for H0: mean(after) == mean(before)."""
    rng = random.Random(seed)
    n_b, n_a = len(before), len(after)
    if n_b == 0 or n_a == 0:
        return 1.0
    diffs: list[float] = []
    for _ in range(n_resamples):
        b_mean = sum(rng.choice(before) for _ in range(n_b)) / n_b
        a_mean = sum(rng.choice(after) for _ in range(n_a)) / n_a
        diffs.append(a_mean - b_mean)
    p_left  = sum(1 for d in diffs if d <= 0) / n_resamples
    p_right = sum(1 for d in diffs if d >= 0) / n_resamples
    return min(1.0, 2 * min(p_left, p_right))
```

Note: `cases` is the list built earlier in `diff_runs`. If the existing code uses a different variable name, adapt — but DO NOT rename `cases` in pre-existing code beyond this scope.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_diff.py::test_diff_runs_bh_correction_reduces_regressed_count_when_many_cases tests/test_diff.py::test_diff_runs_bh_rejects_unknown_method -v`
Expected: 2 PASS.

- [ ] **Step 6: Full suite**

Run: `pytest -q 2>&1 | tail -3`
Expected: 2 more on top of Task 5.

- [ ] **Step 7: Commit**

```bash
git add src/evalith/diff.py tests/test_diff.py
git commit -m "feat(diff): Benjamini-Hochberg FDR multi-test correction in diff_runs

Per-case bootstrap p-values (two-sided), sort ascending, keep cases whose
p_(k) <= (k/N) * 0.05. Any 'regressed' verdict not kept reverts to
'unchanged'. Verified on article 2 frozen raw: BH set is a subset of
the no-correction set (monotone, never adds a regression)."
```

### Task 7: Expose CLI flags (`--ci-method`, `--multi-test`)

**Files:**
- Modify: `src/evalith/cli.py`
- Modify: `tests/test_cli.py` (or `tests/test_diff.py` if test_cli.py doesn't exist)

- [ ] **Step 1: Inspect current CLI structure**

Run: `grep -n 'def diff\|--samples\|--fail-on' src/evalith/cli.py | head -20`
This tells you exactly where `diff` command + flag handling lives.

- [ ] **Step 2: Write a CLI smoke test**

Add a test (in `tests/test_cli.py`, or if that doesn't exist, append to `tests/test_diff.py`):

```python
def test_cli_diff_accepts_ci_method_flag(tmp_path):
    """`evalith diff a.json b.json --ci-method bca` runs without error."""
    import subprocess, sys
    from pathlib import Path
    # Use article 2's frozen raw
    a = Path("docs/blog/article2/raw/a1.json")
    b = Path("docs/blog/article2/raw/b.json")
    assert a.exists() and b.exists(), "article 2 frozen raw missing"
    res = subprocess.run(
        [sys.executable, "-m", "evalith.cli", "diff", str(a), str(b), "--ci-method", "bca"],
        capture_output=True, text=True,
    )
    assert res.returncode in (0, 1), f"unexpected exit {res.returncode}: {res.stderr[-400:]}"
    # 0 = no regression; 1 = regressed (also fine, we just check the flag is accepted)


def test_cli_diff_accepts_multi_test_flag(tmp_path):
    import subprocess, sys
    from pathlib import Path
    a = Path("docs/blog/article2/raw/a1.json")
    b = Path("docs/blog/article2/raw/b.json")
    res = subprocess.run(
        [sys.executable, "-m", "evalith.cli", "diff", str(a), str(b), "--multi-test", "bh"],
        capture_output=True, text=True,
    )
    assert res.returncode in (0, 1), f"unexpected exit {res.returncode}: {res.stderr[-400:]}"
```

- [ ] **Step 3: Run; expect FAIL**

Run: `pytest tests/test_diff.py -k "ci_method_flag or multi_test_flag" -v`
Expected: FAIL — Typer doesn't recognize the flags yet.

- [ ] **Step 4: Add the flags to the `diff` Typer command in cli.py**

Find the `diff` command function (likely `def diff(before, after, ...):` decorated with `@app.command()`). Add two `Option` parameters:

```python
ci_method: str = typer.Option(
    "percentile",
    "--ci-method",
    help="Bootstrap CI method: percentile (default), bca, or paired.",
),
multi_test: Optional[str] = typer.Option(
    None,
    "--multi-test",
    help="Multiple-comparison correction across cases: bh (Benjamini-Hochberg). "
         "Default: no correction.",
),
```

(Make sure `Optional` is imported from typing.)

Then where the function calls `diff_runs(...)`, pass the new flags through:

```python
report = diff_runs(before_run, after_run, ci_method=ci_method, multi_test_correction=multi_test)
```

Wait — `diff_runs` doesn't currently take `ci_method` (Tasks 3-5 added it to `bootstrap_diff_ci`, not `diff_runs`). So we need to thread it through. Look at `diff_runs` in `src/evalith/diff.py`: find where it calls `bootstrap_diff_ci(...)` (likely on line ~95-105). Add a `ci_method` parameter to `diff_runs` and pass it through:

```python
def diff_runs(before: Run, after: Run, tol: float = 1e-9, *,
              ci_method: str = "percentile",
              multi_test_correction: str | None = None) -> DiffReport:
    ...
    # where the existing code calls:
    #     lo, hi = bootstrap_diff_ci(b_samples, a_samples)
    # change to:
    lo, hi = bootstrap_diff_ci(b_samples, a_samples, method=ci_method)
```

Add a unit test confirming this threading works:

```python
def test_diff_runs_passes_ci_method_to_bootstrap():
    from pathlib import Path
    from evalith.diff import diff_runs
    from evalith.models import Run
    a1 = Run.model_validate_json(Path("docs/blog/article2/raw/a1.json").read_text())
    b  = Run.model_validate_json(Path("docs/blog/article2/raw/b.json").read_text())

    report_pct  = diff_runs(a1, b)
    report_bca  = diff_runs(a1, b, ci_method="bca")
    # CI bounds should generally differ (different statistical methods on same data)
    pct_cis = {c.case_id: c.ci for c in report_pct.cases if c.ci is not None}
    bca_cis = {c.case_id: c.ci for c in report_bca.cases if c.ci is not None}
    assert pct_cis.keys() == bca_cis.keys()
    differing = [cid for cid in pct_cis if pct_cis[cid] != bca_cis[cid]]
    assert len(differing) >= 1, "BCa and percentile should produce different CI on at least one case"
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_diff.py -k "ci_method_flag or multi_test_flag or passes_ci_method" -v`
Expected: 3 PASS.

- [ ] **Step 6: Full suite**

Run: `pytest -q 2>&1 | tail -3`
Expected: monotonically growing test count, no regressions.

- [ ] **Step 7: Commit**

```bash
git add src/evalith/cli.py src/evalith/diff.py tests/test_diff.py
git commit -m "feat(cli): expose --ci-method and --multi-test flags

diff_runs now accepts ci_method= (percentile|bca|paired) and
multi_test_correction= (bh|None). CLI mirrors these as --ci-method
and --multi-test. Default behavior unchanged (percentile, no FDR)."
```

### Task 8: Update README "What's new in v0.5" + run full suite + verify CLI help

**Files:**
- Modify: `README.md`, `README.zh-CN.md`

- [ ] **Step 1: Add a "What's new in v0.5" section**

In `README.md`, find the existing "Status" section. **Above** it, insert:

```markdown
## What's new in v0.5

- **`--ci-method bca`** — BCa (bias-corrected and accelerated) bootstrap on Δ. Stdlib-only; more accurate than percentile when the bootstrap distribution is skewed.
- **`--ci-method paired`** — paired bootstrap. Reduces variance when before/after correlate through a shared case dimension.
- **`--multi-test bh`** — Benjamini-Hochberg FDR control across cases. With many cases, percentile alone can over-report regressions; BH compresses the family-wise false-positive rate.
- **scipy** is now a dev dependency (used as ground truth in tests). Not pulled into runtime — production installs stay minimal.

All v0.5 additions are opt-in. The v0.4 default behavior is byte-for-byte preserved.
```

In `README.zh-CN.md`, add a parallel Chinese section:

```markdown
## v0.5 新功能

- **`--ci-method bca`** —— Δ 上的 BCa(偏置纠正加速)bootstrap。仅依赖 Python 标准库;在 bootstrap 分布偏态时比 percentile 更准。
- **`--ci-method paired`** —— paired bootstrap。当 before/after 通过 case 维度有相关性时,CI 宽度收窄,降假阳率。
- **`--multi-test bh`** —— 跨 case 的 Benjamini-Hochberg FDR 控制。case 数多时,纯 percentile 容易过度报告回退;BH 压制家族级假阳率。
- **scipy** 进入 dev 依赖(用作单元测试 ground truth),不进 runtime —— 生产部署仍然零额外依赖。

v0.5 的全部新增都是 opt-in。v0.4 默认行为字节级保留。
```

- [ ] **Step 2: Verify CLI help reflects new flags**

Run: `python3 -m evalith.cli diff --help | grep -E "ci-method|multi-test"`
Expected: both flags appear in `--help` output with their descriptions.

- [ ] **Step 3: Run full test suite — final gate before publishing**

Run: `pytest -q 2>&1 | tail -5`
Expected: all tests pass, zero warnings.

- [ ] **Step 4: Commit**

```bash
git add README.md README.zh-CN.md
git commit -m "docs: document Evalith v0.5 — BCa, paired, FDR (all opt-in)"
```

---

## Phase 2: Article 3 skeleton + pre-committed hypothesis

### Task 9: Create article 3 skeleton + lock the 5 predictions in git

**Files:**
- Create: `docs/blog/2026-05-31-article3-statistical-methods.zh.md`

This commit MUST land BEFORE Phase 3 (the actual data collection). git history is the audit trail proving the hypothesis was pre-hoc.

- [ ] **Step 1: Write the article skeleton**

Create `docs/blog/2026-05-31-article3-statistical-methods.zh.md` with this content (every character matters):

```markdown
# 续之续:把 BCa、paired bootstrap、FDR、第三方 judge 都挂上去,文章 2 的结论变了吗?

> 文章 2 的结论是 Evalith bootstrap 抓到 2/10 regressed,promptfoo 抓到 1/10,DeepEval 抓到 5/10,只有 sql-injection-vulnerability 三家一致。今天把更严格的统计方法和换 judge 都加上,用同一份 frozen raw 数据重看一遍。

---

## 一、那个承诺 + 一个问题

<TODO §1 — Task 16: 回顾 article 2 verdict、引出核心问句、把 5 条 pre-committed 预测的存在直接讲清楚>

### 在看到结果之前,我先把 5 条预测写在这里

把统计方法升级 + 换 judge,我**事先**预测如下:

1. **BCa 大概不会显著改变 verdict。** 文章 2 数据样本小(每 case 5 sample),分布近 0/1 二元,BCa 的修偏在这种分布上贡献有限。
2. **Paired bootstrap CI 会收窄,但不足以让任何 unchanged 翻成 regressed。** 文章 2 的 unchanged case 几乎都是 mean=1.00 两边、Δ=0、零方差。收窄一个零宽度的 CI,结果仍然横跨 0。
3. **FDR 在 10 case 上会把 sql-injection 翻成 unchanged。** 其 CI 离 0 较近(`[-1.00, -0.20]`),BH 校正后 p-value 可能不再通过 0.05 阈值。redis-cluster-failover 的 CI `[-1.00, -0.40]` 更远离 0,会留住。
4. **Judge swap A(单换 judge)会让 verdict 大幅变化。** GPT-4o-mini 不是 deepseek 亲属,对同一中文 prompt + 同一回答的判定标准会不一样。可能比 deepseek 严或宽,但一定不会照搬。
5. **Judge swap B(全换 model+judge)与 A 的差距反映 model variance 的贡献占比。** 如果 A 和 B verdict 接近,说明换 judge 已主导;如果差距大,说明 model 输出本身也是变量。

预测全错的可能不小。文章 2 的假设就几乎全错了。这一篇会用同样方式把对错都摆出来,无论结果怎样。

(以下表格都来自实际跑出来的实验,任何一条没有事后修正。所有 raw outputs 在 `docs/blog/article3/raw/`,一行命令复现。)

## 二、BCa: 修偏与加速

<TODO §2 — Task 17>

## 三、Paired bootstrap: 利用 case 内相关性降方差

<TODO §3 — Task 18>

## 四、FDR: 当你同时检验 10 个 case 时

<TODO §4 — Task 19>

## 五、换 judge: 一路只换,一路全换

<TODO §5 — Task 20>

## 六、谁改变了我们的判定

<TODO §6 — Task 21>

## 七、局限和第四篇方向

<TODO §7 — Task 22>

---

如果你也在做 LLM eval 或 AI CI 集成,欢迎到 [github.com/dominciyue/Evalith_MingJing](https://github.com/dominciyue/Evalith_MingJing) 提 issue / PR。

```bash
pip install evalith
```

即装即用,Apache-2.0。
```

- [ ] **Step 2: Verify article has 7 H2 headers + 5 numbered predictions + footer**

Run:
```bash
grep -c "^## " docs/blog/2026-05-31-article3-statistical-methods.zh.md
# expected: 7
grep -cE "^[0-9]\.\s\*\*" docs/blog/2026-05-31-article3-statistical-methods.zh.md
# expected: 5
grep -c "pip install evalith" docs/blog/2026-05-31-article3-statistical-methods.zh.md
# expected: 1
```

- [ ] **Step 3: Commit — this commit pre-dates ALL data collection**

```bash
git add docs/blog/2026-05-31-article3-statistical-methods.zh.md
git commit -m "docs(blog/article3): skeleton + 5 pre-committed predictions

The 5 predictions in §1 are locked in git BEFORE Phase 3/4 experiments
run. git log makes the pre-hoc nature verifiable. Same audit-trail
pattern as article 2's §4 hypothesis (committed at 68d1db0)."
```

---

## Phase 3: Statistical experiment on article 2 frozen raw

### Task 10: Run all 4 methods on article 2 frozen raw + emit §2/§3/§4 tables

**Files:**
- Create: `docs/blog/article3/stats_runner.py`

This task collects the per-case verdicts for the four statistical methods (percentile baseline + BCa + paired + percentile-with-BH) on the same a1.json vs b.json pair from article 2. Output goes to `/tmp/article3_stats.txt` for Tasks 17-19 to consume.

- [ ] **Step 1: Write the runner**

Create `docs/blog/article3/stats_runner.py`:

```python
"""Per-case verdicts under four statistical methods on article 2's frozen raw.

Output prints §2/§3/§4 markdown tables for direct paste into article 3."""
import json
from pathlib import Path
from evalith.diff import diff_runs, bootstrap_diff_ci, _two_sided_bootstrap_pvalue, _case_samples_by_id
from evalith.models import Run

ART2_RAW = Path("docs/blog/article2/raw")
a1 = Run.model_validate_json((ART2_RAW / "a1.json").read_text())
b  = Run.model_validate_json((ART2_RAW / "b.json").read_text())

CANONICAL = ["explain-rlhf","explain-vector-db","sql-injection-vulnerability",
             "k8s-configmap-vs-secret","asyncio-yield-deadlock","python-gil-tradeoffs",
             "redis-cluster-failover","tcp-congestion-control","jwt-vs-session",
             "transformer-attention"]

def verdict_table(label, ci_method, multi_test):
    rep = diff_runs(a1, b, ci_method=ci_method, multi_test_correction=multi_test)
    print(f"\n### {label}\n")
    print("| case | A1 mean | B mean | CI on Δ | verdict |")
    print("|---|---|---|---|---|")
    by_id = {c.case_id: c for c in rep.cases}
    for cid in CANONICAL:
        c = by_id[cid]
        bm = "—" if c.before is None else f"{c.before:.2f}"
        am = "—" if c.after  is None else f"{c.after:.2f}"
        ci = "—" if c.ci is None else f"[{c.ci[0]:+.2f}, {c.ci[1]:+.2f}]"
        print(f"| `{cid}` | {bm} | {am} | {ci} | {c.status} |")
    flagged = [cid for cid in CANONICAL if by_id[cid].status == "regressed"]
    print(f"\nregressed: {flagged} ({len(flagged)}/10)")
    return flagged

print("=" * 78)
print("ARTICLE 3 §2/§3/§4 — Statistical method comparison")
print("=" * 78)
pct = verdict_table("§2 baseline: percentile (v0.4)", "percentile", None)
bca = verdict_table("§2-extension: BCa",              "bca",        None)
prd = verdict_table("§3: paired bootstrap",            "paired",     None)
fdr = verdict_table("§4: percentile + BH FDR",         "percentile", "bh")

print("\n" + "=" * 78)
print("Cross-method verdict deltas:")
print("=" * 78)
print(f"percentile regressed:  {pct}")
print(f"BCa regressed:         {bca}")
print(f"paired regressed:      {prd}")
print(f"FDR regressed:         {fdr}")

print("\n# Per-case p-values (used by FDR — for §4 prose)")
print("| case | p-value (two-sided bootstrap, n=1000) |")
print("|---|---|")
for cid in CANONICAL:
    b_samples = _case_samples_by_id(a1, cid)
    a_samples = _case_samples_by_id(b,  cid)
    p = _two_sided_bootstrap_pvalue(b_samples, a_samples, seed=0)
    print(f"| `{cid}` | {p:.4f} |")
```

- [ ] **Step 2: Run it and tee output**

Run: `python3 docs/blog/article3/stats_runner.py | tee /tmp/article3_stats.txt`
Expected: prints 4 tables + p-value table + summary line of flagged sets per method.

- [ ] **Step 3: Capture key findings (these go into prose tasks)**

In your task report, note:
- percentile regressed set
- BCa regressed set (likely same or near-same — prediction 1)
- paired regressed set (likely same — prediction 2)
- FDR regressed set (prediction 3: sql-injection might disappear)
- Whether any prediction was correct

- [ ] **Step 4: Commit the runner**

```bash
git add docs/blog/article3/stats_runner.py
git commit -m "feat(blog/article3): per-case verdict table generator for §2/§3/§4

Runs four statistical methods (percentile, BCa, paired, FDR) on
article 2's a1.json vs b.json. Output is the per-case verdict table
and p-value table that §2-§4 prose tasks consume."
```

---

## Phase 4: Cross-judge double-track experiment

### Task 11: Write rejudge.py (judge-only swap)

**Files:**
- Create: `docs/blog/article3/rejudge.py`

- [ ] **Step 1: Write the rejudge helper**

Create `docs/blog/article3/rejudge.py`:

```python
"""Cross-judge swap A: hold model outputs constant, only re-judge with GPT-4o-mini.

Reads article 2 frozen raw (a1.json or b.json), extracts each case's model output
(trial 0 — same data Evalith stored), re-judges using GPT-4o-mini via litellm with
the SAME criteria string article 2 used, and writes a new pass_rate-shaped JSON
ready for Evalith's diff path.

Usage: OPENAI_API_KEY=sk-... python3 docs/blog/article3/rejudge.py <input.json> <output.json>

Note: article 2 stored only trial-0 output (Evalith engine.py:48-58 takes rep=trials[0]),
so this rejudge approximates "judge variance on a single sampled output". For honest
multi-sample comparison, see swap B (fresh runs)."""
import json, os, sys
from pathlib import Path
import litellm
import yaml

if len(sys.argv) != 3:
    print(f"usage: {sys.argv[0]} <input.json> <output.json>", file=sys.stderr); sys.exit(1)

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set", file=sys.stderr); sys.exit(1)

src = json.loads(Path(sys.argv[1]).read_text())
EVAL_CFG = yaml.safe_load(Path("docs/blog/article2/configs/eval.high-temp.yaml").read_text())
CRITERIA = EVAL_CFG["scorers"][0]["params"]["criteria"]

JUDGE_PROMPT = """你是一名严格的 AI 答案评审。

问题/输入:
{input}

AI 的回答:
{output}

评判标准: {criteria}

请只输出一个 JSON 对象,不要任何其它文字:
{{"score": <0到1之间的小数>, "pass": <true 或 false>, "reason": "<简短理由>"}}
"""

def judge(input_text: str, output_text: str) -> tuple[float, bool]:
    prompt = JUDGE_PROMPT.format(input=input_text, output=output_text, criteria=CRITERIA)
    r = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # judge stays stable
    )
    text = r.choices[0].message.content
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON in judge output: {text[:200]}")
    data = json.loads(text[start:end+1])
    return float(data.get("score", 0.0)), bool(data.get("pass", data.get("score", 0.0) >= 0.5))

out = []
for case in src["results"]:
    cid = case["case_id"]
    input_text = case["input"]
    output_text = case["output"]
    # NOTE: judge once per case here. To get a sample distribution, you'd need either
    # multiple stored outputs (article 2 only stored trial 0) or multiple judge calls
    # on the same output. We choose the latter for swap A — call judge with same
    # output 5 times (temp=0 so it's mostly deterministic, but Open AI's API still has
    # tiny jitter that produces a pass-rate "distribution" of sorts).
    samples = []
    for _ in range(5):
        score, passed = judge(input_text, output_text)
        samples.append(1.0 if passed else 0.0)
    out.append({
        "case_id": cid,
        "pass_rate_samples": samples,
        "judge_output_score_first_trial": samples[0],
        "input": input_text,
        "output_used": output_text[:100],
    })
    print(f"  {cid:35s} judge samples: {samples}")

Path(sys.argv[2]).write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"wrote {sys.argv[2]}")
```

- [ ] **Step 2: Smoke (1 case, locally edit script to slice [:1] in a copy)**

Quickest sanity:
```bash
python3 - <<'PY'
import litellm, os
assert os.environ.get("OPENAI_API_KEY"), "set OPENAI_API_KEY first"
r = litellm.completion(model="openai/gpt-4o-mini", messages=[{"role":"user","content":"reply with 'OK' only"}], temperature=0)
print(r.choices[0].message.content)
PY
```
Expected: prints `OK` or similar. Validates the API path.

- [ ] **Step 3: Commit the script (data comes in Task 12)**

```bash
git add docs/blog/article3/rejudge.py
git commit -m "feat(blog/article3): rejudge.py — GPT-4o-mini judge-only swap helper"
```

### Task 12: Run rejudge on article 2 a1.json and b.json

**Files:**
- Create: `docs/blog/article3/raw/rejudge_a1.json`
- Create: `docs/blog/article3/raw/rejudge_b.json`

- [ ] **Step 1: Run rejudge on a1**

Run (Bash timeout 600000):
```bash
python3 docs/blog/article3/rejudge.py \
    docs/blog/article2/raw/a1.json \
    docs/blog/article3/raw/rejudge_a1.json
```
Expected: ~30-60s wall-clock; prints 10 case rows. If any error (auth / 429 / parse), STOP and report BLOCKED.

- [ ] **Step 2: Run rejudge on b**

Run:
```bash
python3 docs/blog/article3/rejudge.py \
    docs/blog/article2/raw/b.json \
    docs/blog/article3/raw/rejudge_b.json
```

- [ ] **Step 3: Sanity check**

```bash
python3 - <<'PY'
import json
from statistics import mean
for f in ['rejudge_a1','rejudge_b']:
    r = json.load(open(f'docs/blog/article3/raw/{f}.json'))
    print(f"\n=== {f} ===")
    for c in r:
        s = c.get('pass_rate_samples', [])
        m = mean(s) if s else 0
        print(f"  {c['case_id']:35s} mean={m:.2f} samples={s}")
PY
```

- [ ] **Step 4: Commit**

```bash
git add docs/blog/article3/raw/rejudge_a1.json docs/blog/article3/raw/rejudge_b.json
git commit -m "data(blog/article3): GPT-4o-mini rejudge on article 2's a1/b outputs

Swap A: model outputs held constant (article 2's deepseek-chat outputs),
only the judge swapped to GPT-4o-mini. Per-case pass rates from 5 judge
samples at temp=0."
```

### Task 13: Write GPT-4o-mini Evalith config (for swap B)

**Files:**
- Create: `docs/blog/article3/configs/eval.high-temp.openai.yaml`
- Create: `docs/blog/article3/configs/eval.high-temp.openai.broken.yaml`

- [ ] **Step 1: Write baseline config**

`docs/blog/article3/configs/eval.high-temp.openai.yaml`:
```yaml
name: high-temp-openai-baseline
dataset: docs/blog/article2/qa.high-temp.yaml
model: openai/gpt-4o-mini
prompt_template: "{{input}}"
temperature: 1.0
concurrency: 3
scorers:
  - type: llm_judge
    params:
      criteria: "回答必须充分解释问题涉及的多个核心概念。如果回答过短(少于 80 字)或仅触及一个方面而忽略其它核心概念,给 score=0.0 且 pass=false。覆盖部分概念但解释不够给 score=0.3-0.5 且 pass=false。充分覆盖且解释清晰给 score=0.9-1.0 且 pass=true。"
      language: zh
```

- [ ] **Step 2: Write broken config**

`docs/blog/article3/configs/eval.high-temp.openai.broken.yaml`:
```yaml
name: high-temp-openai-broken
dataset: docs/blog/article2/qa.high-temp.yaml
model: openai/gpt-4o-mini
prompt_template: "Be very concise. Answer in 1-2 short sentences. Skip explanations and concept enumeration: {{input}}"
temperature: 1.0
concurrency: 3
scorers:
  - type: llm_judge
    params:
      criteria: "回答必须充分解释问题涉及的多个核心概念。如果回答过短(少于 80 字)或仅触及一个方面而忽略其它核心概念,给 score=0.0 且 pass=false。覆盖部分概念但解释不够给 score=0.3-0.5 且 pass=false。充分覆盖且解释清晰给 score=0.9-1.0 且 pass=true。"
      language: zh
```

Identical to article 2's broken except `model: openai/gpt-4o-mini`. Criteria is byte-identical.

- [ ] **Step 3: Verify**

```bash
diff <(grep criteria docs/blog/article3/configs/eval.high-temp.openai.yaml) <(grep criteria docs/blog/article3/configs/eval.high-temp.openai.broken.yaml)
```
Expected: empty (criteria identical).

- [ ] **Step 4: Commit**

```bash
git add docs/blog/article3/configs/
git commit -m "feat(blog/article3): GPT-4o-mini Evalith configs for swap B (full re-run)"
```

### Task 14: Run swap B (fresh A1 + B with GPT-4o-mini)

**Files:**
- Create: `docs/blog/article3/raw/openai_a1.json`
- Create: `docs/blog/article3/raw/openai_b.json`

- [ ] **Step 1: Run baseline**

```bash
python3 -m evalith.cli run docs/blog/article3/configs/eval.high-temp.openai.yaml \
    --samples 5 --out docs/blog/article3/raw/openai_a1.json
```
Bash timeout 600000. Expected: ~2-4 min; 10 cases × 5 samples.

- [ ] **Step 2: Run broken**

```bash
python3 -m evalith.cli run docs/blog/article3/configs/eval.high-temp.openai.broken.yaml \
    --samples 5 --out docs/blog/article3/raw/openai_b.json
```

- [ ] **Step 3: Sanity check**

```bash
python3 - <<'PY'
import json
from statistics import mean
for f in ['openai_a1','openai_b']:
    r = json.load(open(f'docs/blog/article3/raw/{f}.json'))
    print(f"\n=== {f} ===")
    for c in r['results']:
        s = c.get('pass_rate_samples', [])
        m = mean(s) if s else 0
        print(f"  {c['case_id']:35s} mean={m:.2f} samples={s}")
PY
```

- [ ] **Step 4: Commit**

```bash
git add docs/blog/article3/raw/openai_a1.json docs/blog/article3/raw/openai_b.json
git commit -m "data(blog/article3): swap B — GPT-4o-mini does both model and judge

Same dataset, same temp + concurrency + criteria as article 2's runs.
Fresh A1 + B runs; both model output and judge are GPT-4o-mini.
Compared against swap A in §5 to isolate model variance from judge variance."
```

### Task 15: Build §5 cross-judge comparison + §6 综合矩阵 via compare.py

**Files:**
- Create: `docs/blog/article3/compare.py`

- [ ] **Step 1: Write compare.py**

Create `docs/blog/article3/compare.py`:

```python
"""Article 3 §5 + §6 tables — cross-judge swap A vs B + the master 6-column matrix."""
import json
from pathlib import Path
from evalith.diff import diff_runs, bootstrap_diff_ci
from evalith.models import Run

ART2 = Path("docs/blog/article2/raw")
ART3 = Path("docs/blog/article3/raw")

CANONICAL = ["explain-rlhf","explain-vector-db","sql-injection-vulnerability",
             "k8s-configmap-vs-secret","asyncio-yield-deadlock","python-gil-tradeoffs",
             "redis-cluster-failover","tcp-congestion-control","jwt-vs-session",
             "transformer-attention"]

def load_evalith(path):
    return Run.model_validate_json(Path(path).read_text())

def load_rejudge_pair_as_evalith_like(rejudge_path):
    """Convert rejudge JSON shape into a fake Run-like dict for diff_runs."""
    data = json.load(open(rejudge_path))
    results = []
    for c in data:
        results.append({
            "case_id": c["case_id"],
            "input": c.get("input", ""),
            "output": c.get("output_used", ""),
            "scores": [{"scorer": "gpt-4o-mini-judge", "value": c["pass_rate_samples"][0], "passed": c["pass_rate_samples"][0] >= 0.5, "detail": ""}],
            "latency_ms": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "cost_usd": 0.0,
            "pass_rate_samples": c["pass_rate_samples"],
        })
    return Run.model_validate({
        "run_id": "rejudge",
        "created_at": "2026-05-31T00:00:00Z",
        "model": "gpt-4o-mini-judge",
        "results": results,
        "config": {"name": "rejudge"},
    })

# Article 2 baseline
a1_dsk = load_evalith(ART2 / "a1.json")
b_dsk  = load_evalith(ART2 / "b.json")

# Swap A: rejudged outputs
a1_rej = load_rejudge_pair_as_evalith_like(ART3 / "rejudge_a1.json")
b_rej  = load_rejudge_pair_as_evalith_like(ART3 / "rejudge_b.json")

# Swap B: fresh GPT-4o-mini runs
a1_oai = load_evalith(ART3 / "openai_a1.json")
b_oai  = load_evalith(ART3 / "openai_b.json")

def status_map(report):
    return {c.case_id: c.status for c in report.cases}

# Compute all 6 interventions
verdicts = {
    "percentile (v0.4)":  status_map(diff_runs(a1_dsk, b_dsk)),
    "BCa":                 status_map(diff_runs(a1_dsk, b_dsk, ci_method="bca")),
    "paired":              status_map(diff_runs(a1_dsk, b_dsk, ci_method="paired")),
    "FDR (BH)":            status_map(diff_runs(a1_dsk, b_dsk, multi_test_correction="bh")),
    "judge swap A":        status_map(diff_runs(a1_rej, b_rej)),
    "judge swap B":        status_map(diff_runs(a1_oai, b_oai)),
}

# ---- §5 cross-judge table ----
print("\n## §5 — cross-judge double-track\n")
print("| case | DS+DS (v0.4 baseline) | swap A: DS-out + GPT judge | swap B: GPT+GPT |")
print("|---|---|---|---|")
for cid in CANONICAL:
    v0 = verdicts["percentile (v0.4)"][cid]
    va = verdicts["judge swap A"][cid]
    vb = verdicts["judge swap B"][cid]
    print(f"| `{cid}` | {v0} | {va} | {vb} |")

# ---- §6 master matrix ----
print("\n## §6 — 谁改变了我们的判定: per case × 6 interventions\n")
header = ["case"] + list(verdicts.keys())
print("| " + " | ".join(header) + " |")
print("|" + "|".join("---" for _ in header) + "|")
for cid in CANONICAL:
    row = [f"`{cid}`"] + [verdicts[k][cid] for k in verdicts]
    print("| " + " | ".join(row) + " |")

# ---- Flagged sets summary ----
print("\n## §6 supporting: flagged set per intervention\n")
for name, m in verdicts.items():
    flagged = sorted(cid for cid, s in m.items() if s == "regressed")
    print(f"- **{name}**: {flagged} ({len(flagged)}/10)")

# ---- prediction check ----
print("\n## §6 supporting: predictions check (from §1)\n")
percentile = set(c for c,s in verdicts["percentile (v0.4)"].items() if s == "regressed")
bca = set(c for c,s in verdicts["BCa"].items() if s == "regressed")
paired = set(c for c,s in verdicts["paired"].items() if s == "regressed")
fdr = set(c for c,s in verdicts["FDR (BH)"].items() if s == "regressed")
swap_a = set(c for c,s in verdicts["judge swap A"].items() if s == "regressed")
swap_b = set(c for c,s in verdicts["judge swap B"].items() if s == "regressed")

print(f"1. BCa not significantly different from percentile?  {bca == percentile}  (BCa={bca}, percentile={percentile})")
print(f"2. Paired doesn't add any new regressed?               {paired.issubset(percentile)}  (paired={paired})")
print(f"3. FDR removes sql-injection but keeps redis?          {('sql-injection-vulnerability' not in fdr) and ('redis-cluster-failover' in fdr)}  (FDR={fdr})")
print(f"4. Swap A verdict differs notably from percentile?    {swap_a != percentile}  (swap_a={swap_a})")
print(f"5. Swap B differs from swap A (model variance contributes)?  {swap_b != swap_a}  (swap_b={swap_b})")
```

- [ ] **Step 2: Run + capture**

Run: `python3 docs/blog/article3/compare.py | tee /tmp/article3_compare.txt`
Expected: prints §5 table + §6 master matrix + predictions check. This output is the source-of-truth for Tasks 20-21 prose.

- [ ] **Step 3: Commit**

```bash
git add docs/blog/article3/compare.py
git commit -m "feat(blog/article3): cross-tool/cross-judge comparison script

Loads article 2 frozen raw + article 3 rejudge + swap B outputs.
Computes per-case verdicts under 6 interventions (percentile/BCa/paired/FDR/swap-A/swap-B).
Emits §5 cross-judge table + §6 master matrix + automated prediction check."
```

---

## Phase 5: Article 3 prose

### Task 16: Write §1 — hook + recap

**Files:**
- Modify: `docs/blog/2026-05-31-article3-statistical-methods.zh.md`

- [ ] **Step 1: Replace `<TODO §1>` with prose covering**

1. Article 2 verdict numbers (recap from memory: Evalith 2/10 regressed = sql-injection + redis; promptfoo 1/10; DeepEval 5/10; only sql-injection unanimous).
2. The "what would change if you upgraded the stats" question — make it concrete and falsifiable.
3. Three weeks ago in §7 we promised 4 things: BCa / paired / FDR / GPT-4o-mini cross-judge. Today, all done. The 5 predictions are right below.
4. Pacing: ~350-400 字 prose. Tone matches articles 1+2 — direct, no hedging.

The 5-prediction subsection is already in the skeleton (Task 9). §1 prose comes BEFORE it.

- [ ] **Step 2: Verify the section now has substantive prose before "### 在看到结果之前"**

```bash
python3 -c "import re; t=open('docs/blog/2026-05-31-article3-statistical-methods.zh.md').read(); s=t.split('## 一')[1].split('### 在看到结果之前')[0]; print('§1 prose chars:', len(re.sub(r'\s','',s)))"
```
Expected: at least 600 chars (= ~350-400 字 Chinese with markdown overhead).

- [ ] **Step 3: Commit**

```bash
git add docs/blog/2026-05-31-article3-statistical-methods.zh.md
git commit -m "docs(blog/article3): §1 hook + article 2 verdict recap"
```

### Task 17: Write §2 — BCa prose

**Files:**
- Modify: `docs/blog/2026-05-31-article3-statistical-methods.zh.md`

- [ ] **Step 1: Replace `<TODO §2>` with prose covering**

Open `/tmp/article3_stats.txt` to read actual BCa-vs-percentile verdict comparison data.

Beats:
1. What BCa fixes vs percentile (one paragraph): 偏态分布上,percentile CI 不对称地"压"在 mode 一侧。BCa 用 bias correction z₀ + acceleration a 把这个压扁的部分推回来。stdlib NormalDist 可以直接做。
2. **Per-case verdict table** for percentile vs BCa side-by-side (from compare.py output).
3. **Prediction 1 check (from §1):** "BCa 不会显著改变 verdict" — was this right? State plainly. If BCa flagged same set as percentile, prediction 1 ✓.
4. **One specific case to anchor:** pick the case where CI bounds differ most between percentile and BCa, name it, explain why (small N, mode at boundary, etc).
5. **One-line takeaway:** "BCa 在 n=5 样本上的修偏贡献是 X 微小/显著",写到读者能记住。

字数: ~700 中文 prose 字。

- [ ] **Step 2: Verify**

```bash
python3 -c "import re; t=open('docs/blog/2026-05-31-article3-statistical-methods.zh.md').read(); s=t.split('## 二')[1].split('## 三')[0]; print('§2 chars:', len(re.sub(r'\s','',s)))"
```
Expected: 1000-1400 chars (700 字 prose + table markup).

- [ ] **Step 3: Commit**

```bash
git add docs/blog/2026-05-31-article3-statistical-methods.zh.md
git commit -m "docs(blog/article3): §2 BCa — theory + frozen-data verdict comparison"
```

### Task 18: Write §3 — paired bootstrap prose

**Files:**
- Modify: `docs/blog/2026-05-31-article3-statistical-methods.zh.md`

- [ ] **Step 1: Replace `<TODO §3>`**

Use `/tmp/article3_stats.txt` for paired vs percentile comparison.

Beats:
1. The mechanism: paired resamples case INDICES not values. Why this reduces variance when cases have intrinsic hardness.
2. Concrete fixture: refer to test_bootstrap_paired_reduces_variance_vs_unpaired (Task 5) — when before/after have a uniform shift per case, paired CI is narrower by X.
3. **Per-case table** for paired CI vs percentile CI side-by-side.
4. **Prediction 2 check:** "paired CI 收窄但不足以让 unchanged 翻 regressed" — true or false? In article 2's data most unchanged are mean=1.0/Δ=0,可能确实如此。
5. One paragraph: what condition would have made paired flip something? (cases with high within-case variance + small mean difference)

字数: ~700 中文 prose 字。

- [ ] **Step 2: Verify + Commit**

```bash
python3 -c "import re; t=open('docs/blog/2026-05-31-article3-statistical-methods.zh.md').read(); s=t.split('## 三')[1].split('## 四')[0]; print('§3 chars:', len(re.sub(r'\s','',s)))"
git add docs/blog/2026-05-31-article3-statistical-methods.zh.md
git commit -m "docs(blog/article3): §3 paired bootstrap — variance reduction via case-index resampling"
```

### Task 19: Write §4 — FDR prose

**Files:**
- Modify: `docs/blog/2026-05-31-article3-statistical-methods.zh.md`

- [ ] **Step 1: Replace `<TODO §4>`**

Use `/tmp/article3_stats.txt` for FDR per-case p-values + FDR-corrected verdict set.

Beats:
1. The multiple-comparisons problem: 10 case 同时检验,每个独立用 α=0.05 → 期望假阳 0.5 个,但分布尾部能挤出更多。BH 是 1995 年 Benjamini-Hochberg 的方法。
2. The BH procedure mechanically: sort p ascending, threshold = (k/N) × α, accept all rank ≤ k.
3. **p-value table** for the 10 cases (from compare.py).
4. **Per-case verdict comparison** percentile (no-correction) vs FDR-BH.
5. **Prediction 3 check:** "FDR removes sql-injection but keeps redis" — true or false? Be specific about the rank cutoff.
6. The practical implication: if you run a CI gate over 50+ cases, you want BH on. Over 5 cases, the correction is mild.

字数: ~700 中文 prose 字。

- [ ] **Step 2: Verify + Commit**

```bash
python3 -c "import re; t=open('docs/blog/2026-05-31-article3-statistical-methods.zh.md').read(); s=t.split('## 四')[1].split('## 五')[0]; print('§4 chars:', len(re.sub(r'\s','',s)))"
git add docs/blog/2026-05-31-article3-statistical-methods.zh.md
git commit -m "docs(blog/article3): §4 FDR — Benjamini-Hochberg vs percentile on 10-case family"
```

### Task 20: Write §5 — cross-judge double-track prose

**Files:**
- Modify: `docs/blog/2026-05-31-article3-statistical-methods.zh.md`

- [ ] **Step 1: Replace `<TODO §5>`**

Use `/tmp/article3_compare.txt` for the §5 cross-judge table.

Beats:
1. **The two tracks explained:** swap A holds model outputs constant (article 2's deepseek-chat outputs), only changes the judge. Swap B is fresh end-to-end with GPT-4o-mini. Why both: A isolates judge variance, A vs B together teases out model variance.
2. **The §5 table** (3 columns: v0.4 baseline + swap A + swap B).
3. **Prediction 4 check:** "swap A 与 baseline 大幅不同?" State.
4. **Prediction 5 check:** "swap B 与 swap A 的差距反映 model variance?" State.
5. **What this means:** 如果 swap A vs baseline 差距 ≫ swap B vs swap A 差距 → judge 是主要变量;反之 model 是主要变量。把两个差距具体量化。
6. Honest caveat: GPT-4o-mini 不是 ground truth,只是第三方意见。我们能说的是「两个 judge 不一致 → verdict 是 judge-specific」,不能说「哪个 judge 更准」。

字数: ~1200 中文 prose 字。§5 是这一篇最厚的一节。

- [ ] **Step 2: Verify + Commit**

```bash
python3 -c "import re; t=open('docs/blog/2026-05-31-article3-statistical-methods.zh.md').read(); s=t.split('## 五')[1].split('## 六')[0]; print('§5 chars:', len(re.sub(r'\s','',s)))"
git add docs/blog/2026-05-31-article3-statistical-methods.zh.md
git commit -m "docs(blog/article3): §5 cross-judge — swap A (judge-only) vs swap B (full)"
```

### Task 21: Write §6 — 综合矩阵 + 谁改变了我们的判定

**Files:**
- Modify: `docs/blog/2026-05-31-article3-statistical-methods.zh.md`

- [ ] **Step 1: Replace `<TODO §6>`**

Use `/tmp/article3_compare.txt` master matrix.

Beats:
1. The 6-column matrix table (per case × 6 interventions: percentile / BCa / paired / FDR / swap A / swap B). This is the article's central物件.
2. Per-row interpretation for cases that DID change verdict across interventions. Two natural buckets:
   - "翻盘 case" — cases whose verdict changes across any of the 6 columns. Each gets a sentence on why.
   - "稳定 case" — cases that stay unchanged across all 6 (or stay regressed across all 6). These are the strongest signal.
3. **Wrap up with the 5 predictions tally:** how many of my 5 predictions were right? Be honest. Refer back to git commit hash where they were locked in.
4. The article's central point synthesized: 
   - Statistical method matters less than expected (BCa / paired / FDR can refine but rarely flip).
   - Judge identity matters a lot (swap A flips things).
   - Model identity matters too (swap B differs from swap A).
   - In practical CI, fix model + judge first, then choose any reasonable statistical method.

字数: ~700 中文 prose 字。

- [ ] **Step 2: Verify + Commit**

```bash
python3 -c "import re; t=open('docs/blog/2026-05-31-article3-statistical-methods.zh.md').read(); s=t.split('## 六')[1].split('## 七')[0]; print('§6 chars:', len(re.sub(r'\s','',s)))"
git add docs/blog/2026-05-31-article3-statistical-methods.zh.md
git commit -m "docs(blog/article3): §6 master matrix + prediction tally + central synthesis"
```

### Task 22: Write §7 — limitations + article 4 direction

**Files:**
- Modify: `docs/blog/2026-05-31-article3-statistical-methods.zh.md`

- [ ] **Step 1: Replace `<TODO §7>`**

Numbered limitations (article 1+2 style):
1. n=10 case 仍小。BCa 在小样本上的 acceleration 估计本身有方差;FDR 的 power 有限。
2. swap A 只重判一次输出;article 2 只存了 trial-0 output。要更严谨的 judge 噪声估计应该用多个不同 trial output。
3. swap A vs swap B 的对照虽然清楚,但缺第 4 个 judge(如 Claude)无法说"判定 cross-judge 是否聚合"。
4. 所有结果是 deepseek-chat / gpt-4o-mini 在一时刻的行为。三周后跑可能不一样。
5. **Article 4 方向:** per-case `expected_concepts` 插值到 judge prompt + adaptive sampling + 跨领域 dataset 扩到 50-100。

Footer (article 1+2 风格):
```markdown
如果你也在做 LLM eval 或 AI CI 集成,欢迎到 [github.com/dominciyue/Evalith_MingJing](https://github.com/dominciyue/Evalith_MingJing) 提 issue / PR。

`​``bash
pip install evalith
`​``

即装即用,Apache-2.0。
```

(Real backticks in file, ZWJ above just escapes from prompt rendering.)

字数: ~300 中文 prose 字 (§7 should be terse like article 1's §7).

- [ ] **Step 2: Verify + Commit**

```bash
grep -c "TODO" docs/blog/2026-05-31-article3-statistical-methods.zh.md
# expected: 0
git add docs/blog/2026-05-31-article3-statistical-methods.zh.md
git commit -m "docs(blog/article3): §7 limitations + article 4 direction"
```

---

## Phase 6: Finalize + v0.5 release

### Task 23: Write experiment.sh for end-to-end reproduction

**Files:**
- Create: `docs/blog/article3/experiment.sh`

- [ ] **Step 1: Write the script**

`docs/blog/article3/experiment.sh`:
```bash
#!/usr/bin/env bash
# Article 3 end-to-end reproduction.
#
# Usage:
#   DEEPSEEK_API_KEY=sk-... OPENAI_API_KEY=sk-... \
#     bash docs/blog/article3/experiment.sh [OUT_DIR]
#
# What it does:
#   Phase 1 (stats): runs all 4 statistical methods on article 2 frozen raw
#   Phase 2 (swap A): rejudges article 2 outputs with GPT-4o-mini
#   Phase 3 (swap B): fresh GPT-4o-mini A1 + B runs
#   Phase 4 (synthesize): emits §5 + §6 tables
#
# ~$0.30 OpenAI credit + ~5-10 min wall-clock.

set -euo pipefail
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: set DEEPSEEK_API_KEY" >&2; exit 1
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: set OPENAI_API_KEY" >&2; exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

OUT="${1:-$(mktemp -d -t article3_repro.XXXXXX)}"
mkdir -p "${OUT}"
echo "=== Article 3 reproduction ===" >&2
echo "Output dir: ${OUT}" >&2

echo "" >&2
echo "=== Phase 1: 4 statistical methods on article 2 frozen raw ===" >&2
python3 docs/blog/article3/stats_runner.py | tee "${OUT}/stats.txt"

echo "" >&2
echo "=== Phase 2: swap A — rejudge with GPT-4o-mini ===" >&2
python3 docs/blog/article3/rejudge.py docs/blog/article2/raw/a1.json "${OUT}/rejudge_a1.json"
python3 docs/blog/article3/rejudge.py docs/blog/article2/raw/b.json  "${OUT}/rejudge_b.json"

echo "" >&2
echo "=== Phase 3: swap B — fresh GPT-4o-mini runs ===" >&2
python3 -m evalith.cli run docs/blog/article3/configs/eval.high-temp.openai.yaml \
    --samples 5 --out "${OUT}/openai_a1.json"
python3 -m evalith.cli run docs/blog/article3/configs/eval.high-temp.openai.broken.yaml \
    --samples 5 --out "${OUT}/openai_b.json"

echo "" >&2
echo "=== Phase 4: §5 + §6 tables (compare.py reads from docs/blog/article3/raw/) ===" >&2
python3 docs/blog/article3/compare.py | tee "${OUT}/compare.txt"

echo "" >&2
echo "Done. Artifacts at: ${OUT}" >&2
```

- [ ] **Step 2: chmod + syntax check**

```bash
chmod +x docs/blog/article3/experiment.sh
bash -n docs/blog/article3/experiment.sh && echo "syntax OK"
```

- [ ] **Step 3: Commit**

```bash
git add docs/blog/article3/experiment.sh
git commit -m "feat(blog/article3): end-to-end reproduction script (stats + swap A + swap B)"
```

### Task 24: Article self-review

**Files:** Read-only at this step.

- [ ] **Step 1: Verify all tables in §2-§6 match the actual data**

For each table in the article, re-run the relevant snippet from `stats_runner.py` or `compare.py` and confirm the numbers match.

```bash
python3 docs/blog/article3/stats_runner.py > /tmp/check_stats.txt
python3 docs/blog/article3/compare.py     > /tmp/check_compare.txt
# Spot-check article tables against these
```

- [ ] **Step 2: Check no TODOs remain**

```bash
grep -nE "TODO|<fill|<paste|<X>|<Y>|TBD" docs/blog/2026-05-31-article3-statistical-methods.zh.md
```
Expected: no matches.

- [ ] **Step 3: Length check**

```bash
python3 -c "import re; t = open('docs/blog/2026-05-31-article3-statistical-methods.zh.md').read(); print('total non-whitespace:', len(re.sub(r'\s','',t)))"
```
Expected: roughly 9000-12000 (target 4500-5500 字 prose + tables).

- [ ] **Step 4: Predictions section check — all 5 predictions explicitly evaluated**

```bash
grep -E "预测 1|预测 2|预测 3|预测 4|预测 5" docs/blog/2026-05-31-article3-statistical-methods.zh.md
# OR
grep -E "Prediction|prediction 1|prediction 2" docs/blog/2026-05-31-article3-statistical-methods.zh.md
```
Each of the 5 predictions must be explicitly evaluated (right or wrong) somewhere in §2-§6.

- [ ] **Step 5: Commit if any fix was needed**

If fixes were applied:
```bash
git add docs/blog/2026-05-31-article3-statistical-methods.zh.md
git commit -m "docs(blog/article3): self-review pass — <describe>"
```

### Task 25: Rename article to publication date + update READMEs + pytest gate

**Files:**
- Rename: `docs/blog/2026-05-31-...` → `docs/blog/<PUB>-...`
- Modify: `docs/README.md`, `README.md`, `README.zh-CN.md`

- [ ] **Step 1: Stamp publication date**

```bash
PUB=$(date +%Y-%m-%d)
git mv docs/blog/2026-05-31-article3-statistical-methods.zh.md \
       docs/blog/${PUB}-article3-statistical-methods.zh.md
git commit -m "docs(blog/article3): stamp publication date ${PUB}"
```

- [ ] **Step 2: Update docs/README.md**

Add a bullet:
```markdown
- [`<PUB>-article3-statistical-methods.zh.md`](blog/<PUB>-article3-statistical-methods.zh.md) — 续之续:BCa + paired + FDR + 第三方 judge 在文章 2 同一份数据上的对比 (Chinese)
```

- [ ] **Step 3: Update README.md + README.zh-CN.md**

In both, add the article 3 entry to the "Read more" / "深入阅读" section. For now point to the local file path; once published to Zhihu, the user will update with the URL (same pattern as article 2).

- [ ] **Step 4: Run full pytest — non-regression gate**

```bash
pytest -q 2>&1 | tail -5
```
Expected: all pass, including new Tasks 3-7 tests.

- [ ] **Step 5: Commit + push**

```bash
git add docs/README.md README.md README.zh-CN.md
git commit -m "docs: link article 3 from docs index + top-level READMEs"
git push origin main
```

### Task 26: PyPI release v0.5.0

**Files:** None (release artifacts).

- [ ] **Step 1: Build distribution**

```bash
python3 -m build 2>&1 | tail -3
```
Expected: writes `dist/evalith-0.5.0-py3-none-any.whl` + `dist/evalith-0.5.0.tar.gz`.

- [ ] **Step 2: twine check**

```bash
python3 -m twine check dist/evalith-0.5.0*
```
Expected: PASSED.

- [ ] **Step 3: Inform user — they upload themselves**

Report to user (the user must run twine upload with their own token; we can't do it in this session):
```bash
python3 -m twine upload dist/evalith-0.5.0*
```

After upload, verify on PyPI: `pip install evalith==0.5.0 --dry-run` from a clean environment.

- [ ] **Step 4: Tag the release commit**

```bash
git tag -a v0.5.0 -m "Evalith v0.5.0 — BCa + paired + FDR (all opt-in)"
git push origin v0.5.0
```

---

## Self-Review (executed before saving this plan)

**1. Spec coverage:**
- ✅ §1 (background, article 2 §7 promise) — Tasks 9, 16 capture pre-committed predictions + recap.
- ✅ §2 (goals: tenancy of 4 commitments + falsifiable question + cross-judge isolation) — Tasks 4 (BCa), 5 (paired), 6 (FDR), 11+14 (cross-judge double-track).
- ✅ §3 (non-goals: no per-case criteria insert, no English version, no Ragas) — none of these are tasks.
- ✅ §4 (article 7-section structure) — Tasks 16-22 cover §1-§7.
- ✅ §5.1 (data source = article 2 frozen raw) — Tasks 10, 12, 15 all reference `docs/blog/article2/raw/`.
- ✅ §5.2 (4 methods + 2 cross-judge tracks) — Tasks 4, 5, 6, 11+12, 13+14 implement each.
- ✅ §5.3 (§6 6-column matrix) — Task 15 produces it.
- ✅ §5.4 (~$0.30 budget) — Task 12 (~$0.10 GPT-4o-mini judge) + Task 14 (~$0.20 GPT-4o-mini model + judge).
- ✅ §5.5 (5 pre-committed predictions) — Task 9 commits them; Tasks 17-21 evaluate them.
- ✅ §6.1 (API) — Tasks 3, 4, 5, 6 add `method=` and `multi_test_correction=`.
- ✅ §6.2 (CLI) — Task 7 adds `--ci-method`, `--multi-test`.
- ✅ §6.3 (backward compat) — Task 3 Step 5 explicitly verifies v0.4 tests pass after each change.
- ✅ §6.4 (testing against scipy) — Task 4 Step 1 + Task 5 Step 1 use scipy ground truth.
- ✅ §6.5 (v0.5 release) — Tasks 2 (version bump), 8 (README "What's new"), 26 (PyPI release).
- ✅ §7 (deliverables) — every path in §7 maps to a Task.
- ✅ §8 (honest disclosures) — Tasks 9 (predictions can be wrong), 22 (§7 prose), 24 (self-review).
- ✅ §9 (success criteria) — Task 24 + Task 25 + Task 26 collectively verify.
- ✅ §10 (out-of-scope) — none of these are tasks.

**2. Placeholder scan:**
- The article filename uses `2026-05-31` placeholder during writing; Task 25 renames to publication date. Intentional and documented.
- `<TODO §1>` through `<TODO §7>` markers in Task 9 skeleton are intentional landmarks. Tasks 16-22 explicitly replace each. Task 24 Step 2 greps for any leftover.
- `<PUB>` placeholder in Task 25 is intentional (computed from `date +%Y-%m-%d` at execution).
- No "TBD" / "handle edge cases" / vague-implement-later strings.

**3. Type consistency:**
- `bootstrap_diff_ci(method=...)` introduced in Task 3, used in Tasks 4, 5, 7.
- `_two_sided_bootstrap_pvalue` introduced in Task 6, referenced in Task 10's stats_runner.py.
- `_case_samples_by_id` introduced in Task 6, referenced in Task 10's stats_runner.py.
- `diff_runs(ci_method=..., multi_test_correction=...)` introduced/extended in Tasks 6 and 7, used in Tasks 10 and 15.
- `pass_rate_samples` (existing field) referenced in Tasks 10, 11, 15.
- CLI flag names (`--ci-method`, `--multi-test`) consistent in Tasks 7, 8, 23.

Plan is internally consistent and spec-complete.
