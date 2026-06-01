"""Article 3 §5 + §6 tables — cross-judge swap A vs B + the master 6-column matrix.

Reads:
  - docs/blog/article2/raw/{a1,b}.json (Evalith Run-format, deepseek+deepseek)
  - docs/blog/article3/raw/rejudge_{a1,b}.json (custom list shape, swap A)
  - docs/blog/article3/raw/openai_{a1,b}.json (Run-format, swap B)

Emits to stdout:
  - §5 cross-judge table (3 columns: baseline / swap A / swap B verdicts)
  - §6 master matrix (per case × 6 interventions: percentile / BCa / paired / FDR / swap A / swap B)
  - Per-tool flagged sets + cross-method agreement summary
  - Automated check of the 5 predictions from article §1 (commit cc98b85)"""
import json
from pathlib import Path
from datetime import datetime, timezone

from evalith.diff import diff_runs
from evalith.models import Run, CaseResult, Score

ART2 = Path("docs/blog/article2/raw")
ART3 = Path("docs/blog/article3/raw")

CANONICAL = ["explain-rlhf","explain-vector-db","sql-injection-vulnerability",
             "k8s-configmap-vs-secret","asyncio-yield-deadlock","python-gil-tradeoffs",
             "redis-cluster-failover","tcp-congestion-control","jwt-vs-session",
             "transformer-attention"]

def load_run(path: Path) -> Run:
    return Run.model_validate_json(path.read_text())

def rejudge_as_run(path: Path, name: str) -> Run:
    """Adapt the rejudge list shape to a Run object so diff_runs can consume it."""
    raw = json.loads(path.read_text())
    results = []
    for c in raw:
        first_rate = c["pass_rate_samples"][0] if c["pass_rate_samples"] else 0.0
        results.append(CaseResult(
            case_id=c["case_id"],
            input=c.get("input", ""),
            output=c.get("output_used", ""),
            scores=[Score(scorer="gpt-5-mini-judge", value=first_rate, passed=first_rate >= 0.5, detail="")],
            latency_ms=0.0,
            prompt_tokens=0, completion_tokens=0, total_tokens=0, cost_usd=0.0,
            pass_rate_samples=c["pass_rate_samples"],
        ))
    return Run(id=f"rejudge-{name}", name=f"rejudge-{name}",
               created_at=datetime.now(timezone.utc),
               model="gpt-5-mini-rejudge", results=results, config={})

# Load all sources
a1_dsk = load_run(ART2 / "a1.json")
b_dsk  = load_run(ART2 / "b.json")

a1_rej = rejudge_as_run(ART3 / "rejudge_a1.json", "a1")
b_rej  = rejudge_as_run(ART3 / "rejudge_b.json",  "b")

a1_oai = load_run(ART3 / "openai_a1.json")
b_oai  = load_run(ART3 / "openai_b.json")

def status_map(report) -> dict[str, str]:
    return {c.case_id: c.status for c in report.cases}

verdicts = {
    "percentile (v0.4)":   status_map(diff_runs(a1_dsk, b_dsk)),
    "BCa":                 status_map(diff_runs(a1_dsk, b_dsk, ci_method="bca")),
    "paired":              status_map(diff_runs(a1_dsk, b_dsk, ci_method="paired")),
    "FDR (BH)":            status_map(diff_runs(a1_dsk, b_dsk, multi_test_correction="bh")),
    "swap A (DS-out+GPT-judge)": status_map(diff_runs(a1_rej, b_rej)),
    "swap B (GPT+GPT)":          status_map(diff_runs(a1_oai, b_oai)),
}

# ===== §5 cross-judge table =====
print("\n## §5 — cross-judge double-track\n")
print("| case | DS+DS (v0.4 baseline) | swap A: DS-out + GPT judge | swap B: GPT+GPT |")
print("|---|---|---|---|")
for cid in CANONICAL:
    v0 = verdicts["percentile (v0.4)"][cid]
    va = verdicts["swap A (DS-out+GPT-judge)"][cid]
    vb = verdicts["swap B (GPT+GPT)"][cid]
    print(f"| `{cid}` | {v0} | {va} | {vb} |")

# ===== §6 master matrix =====
print("\n## §6 — 谁改变了我们的判定: per case × 6 interventions\n")
header = ["case"] + list(verdicts.keys())
print("| " + " | ".join(header) + " |")
print("|" + "|".join("---" for _ in header) + "|")
for cid in CANONICAL:
    row = [f"`{cid}`"] + [verdicts[k][cid] for k in verdicts]
    print("| " + " | ".join(row) + " |")

# ===== flagged sets summary =====
print("\n## flagged sets per intervention\n")
for name, m in verdicts.items():
    flagged = sorted(cid for cid, s in m.items() if s == "regressed")
    print(f"- **{name}**: {flagged} ({len(flagged)}/10)")

# ===== predictions check =====
print("\n## §1 5 predictions vs reality\n")

percentile = {c for c,s in verdicts["percentile (v0.4)"].items() if s == "regressed"}
bca        = {c for c,s in verdicts["BCa"].items() if s == "regressed"}
paired     = {c for c,s in verdicts["paired"].items() if s == "regressed"}
fdr        = {c for c,s in verdicts["FDR (BH)"].items() if s == "regressed"}
swap_a     = {c for c,s in verdicts["swap A (DS-out+GPT-judge)"].items() if s == "regressed"}
swap_b     = {c for c,s in verdicts["swap B (GPT+GPT)"].items() if s == "regressed"}

print(f"1. BCa not significantly different from percentile?  {bca == percentile}  (BCa={sorted(bca)}, percentile={sorted(percentile)})")
print(f"2. Paired doesn't add any new regressed (subset of percentile)? {paired.issubset(percentile)}  (paired={sorted(paired)})")
print(f"3. FDR removes sql-injection but keeps redis?  {('sql-injection-vulnerability' not in fdr) and ('redis-cluster-failover' in fdr)}  (FDR={sorted(fdr)})")
print(f"4. Swap A verdict differs notably from percentile?  {swap_a != percentile}  (swap_a={sorted(swap_a)})")
print(f"5. Swap B differs from swap A (model variance contributes)?  {swap_b != swap_a}  (swap_b={sorted(swap_b)})")

# ===== meta: gpt-5-mini baseline divergence vs deepseek =====
print("\n## meta: judge mean disagreement on a1 baseline\n")
print("(gpt-5-mini judging deepseek's a1 outputs vs deepseek-chat's own a1 judgments)\n")
from statistics import mean
a1_dsk_means = {c.case_id: (mean(c.pass_rate_samples) if c.pass_rate_samples else 0.0) for c in a1_dsk.results}
a1_rej_means = {c.case_id: (mean(c.pass_rate_samples) if c.pass_rate_samples else 0.0) for c in a1_rej.results}
print("| case | deepseek-chat judge | gpt-5-mini judge | gap |")
print("|---|---|---|---|")
for cid in CANONICAL:
    dsk = a1_dsk_means.get(cid, 0.0)
    rej = a1_rej_means.get(cid, 0.0)
    gap = rej - dsk
    print(f"| `{cid}` | {dsk:.2f} | {rej:.2f} | {gap:+.2f} |")
