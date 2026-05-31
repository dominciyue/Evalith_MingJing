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
