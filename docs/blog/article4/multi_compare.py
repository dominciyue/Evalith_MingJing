"""Cross-judge (§3) and cross-model (§4) analysis for article 4.

§3 Cross-judge: hold model = deepseek, vary judge in {ds, qw, glm}. How far does
the verdict move when ONLY the judge changes? Pairwise Cohen's kappa on per-case
×trial pass/fail labels quantifies inter-judge agreement.

§4 Cross-model: hold judge = deepseek, vary model in {ds, qw, glm}. Does the
baseline→broken regression get flagged consistently across model families?

Reads the 10 judged JSONs in raw/ (written by runners/multi_judge.py).
"""
import json
import statistics
from itertools import combinations
from pathlib import Path

RAW = Path("docs/blog/article4/raw")
JUDGES = ["ds", "qw", "glm"]
MODELS = ["ds", "qw", "glm"]


def load(name):
    return json.loads((RAW / name).read_text())


def case_mean(case):
    s = case.get("pass_rate_samples") or []
    return statistics.mean(s) if s else 0.0


def overall(run):
    return statistics.mean(case_mean(c) for c in run["results"])


def trial_labels(run):
    """Flat per-(case,trial) binary labels, ordered case-then-trial."""
    out = []
    for c in run["results"]:
        for v in (c.get("pass_rate_samples") or []):
            out.append(1 if v >= 0.5 else 0)
    return out


def cohen_kappa(a, b):
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def hr(title):
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------- §3
hr("§3 CROSS-JUDGE  (model = deepseek held constant, judge varies)")
for variant, tag in [("a1", "baseline"), ("b", "broken")]:
    runs = {j: load(f"j_ds_by_{j}_{variant}.json") for j in JUDGES}
    print(f"\n  ds-{tag}: overall pass-rate by judge")
    for j in JUDGES:
        print(f"    judge={j:4s}  pass={overall(runs[j]):.3f}")
    print("    pairwise Cohen's kappa (per-case×trial pass/fail):")
    for j1, j2 in combinations(JUDGES, 2):
        k = cohen_kappa(trial_labels(runs[j1]), trial_labels(runs[j2]))
        print(f"      {j1:4s} vs {j2:4s}: kappa={k:+.3f}")

print("\n  Verdict (Δ = broken − baseline pass-rate) per judge:")
for j in JUDGES:
    a = overall(load(f"j_ds_by_{j}_a1.json"))
    b = overall(load(f"j_ds_by_{j}_b.json"))
    print(f"    judge={j:4s}  baseline={a:.3f}  broken={b:.3f}  Δ={b - a:+.3f}")

# ---------------------------------------------------------------- §4
print()
hr("§4 CROSS-MODEL  (judge = deepseek held constant, model varies)")
print("\n  baseline→broken Δ per model family (judge=ds):")
for m in MODELS:
    a = overall(load(f"j_{m}_by_ds_a1.json"))
    b = overall(load(f"j_{m}_by_ds_b.json"))
    flagged = "REGRESSION" if (b - a) < -0.05 else "no-flag"
    print(f"    model={m:4s}  baseline={a:.3f}  broken={b:.3f}  Δ={b - a:+.3f}  [{flagged}]")

# ---------------------------------------------------------------- dataset
print()
hr("DATASET domain breakdown (ds baseline, judged by ds)")
ref = load("j_ds_by_ds_a1.json")
by_dom = {}
for c in ref["results"]:
    by_dom.setdefault(c.get("domain") or "?", []).append(case_mean(c))
for dom, vals in sorted(by_dom.items()):
    print(f"    {dom:12s}  n={len(vals):2d}  pass={statistics.mean(vals):.3f}")
