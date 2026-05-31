"""Cross-tool comparison metrics for article 2 §5.

Reads the raw outputs from all three tools, normalizes per-case pass rates,
and emits the verdict table + aggregate metrics that the article §5 prose
references. Designed to be re-runnable: re-running this script after the
raw outputs land must produce identical output (assuming raw outputs are
frozen as they should be)."""
import json
from pathlib import Path
from evalith.diff import diff_runs
from evalith.models import Run

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # repo root
RAW  = Path(__file__).resolve().parent / "raw"

# ---- Pre-committed hypothesis (from article §4) ----
STRONG = {"explain-rlhf","explain-vector-db","python-gil-tradeoffs","tcp-congestion-control","transformer-attention"}
BORDER = {"k8s-configmap-vs-secret","jwt-vs-session"}
SAFE   = {"sql-injection-vulnerability","asyncio-yield-deadlock","redis-cluster-failover"}
PREDICTED_AFFECTED = STRONG | BORDER
ALL_CASES = STRONG | BORDER | SAFE
CANONICAL_ORDER = ["explain-rlhf","explain-vector-db","sql-injection-vulnerability",
                   "k8s-configmap-vs-secret","asyncio-yield-deadlock","python-gil-tradeoffs",
                   "redis-cluster-failover","tcp-congestion-control","jwt-vs-session",
                   "transformer-attention"]

# ---- Load and normalize ----
def evalith_pass_rates(path):
    r = Run.model_validate_json(Path(path).read_text())
    return {c.case_id: sum(c.pass_rate_samples)/len(c.pass_rate_samples) if c.pass_rate_samples else None
            for c in r.results}

def evalith_verdicts(before_path, after_path):
    b = Run.model_validate_json(Path(before_path).read_text())
    a = Run.model_validate_json(Path(after_path).read_text())
    return {c.case_id: c.status for c in diff_runs(b, a).cases}

def promptfoo_pass_rates(path):
    raw = json.load(open(path))
    results = raw['results']['results']
    by_case = {}
    for t in results:
        cid = t['vars']['_case_id']
        by_case.setdefault(cid, []).append(1 if t.get('success') else 0)
    return {cid: sum(v)/len(v) for cid, v in by_case.items()}

def deepeval_pass_rates(path):
    return {c['case_id']: c['pass_rate'] for c in json.load(open(path))}

# ---- Pull data ----
ev_a1  = evalith_pass_rates(RAW / 'a1.json')
ev_a2  = evalith_pass_rates(RAW / 'a2.json')
ev_b   = evalith_pass_rates(RAW / 'b.json')
ev_a1a2_verdict = evalith_verdicts(RAW / 'a1.json', RAW / 'a2.json')
ev_a1b_verdict  = evalith_verdicts(RAW / 'a1.json', RAW / 'b.json')

pf_a1 = promptfoo_pass_rates(RAW / 'promptfoo_a1.json')
pf_b  = promptfoo_pass_rates(RAW / 'promptfoo_b.json')

de_a1 = deepeval_pass_rates(RAW / 'deepeval_a1.json')
de_b  = deepeval_pass_rates(RAW / 'deepeval_b.json')

# ---- Section 1: per-case verdict table ----
print("## §5 表 1:三工具对 A1-vs-B 的 per-case 判定\n")
print("| case | A1 mean(Evalith / promptfoo / DeepEval) | B mean(Evalith / promptfoo / DeepEval) | Evalith bootstrap |")
print("|---|---|---|---|")
for cid in CANONICAL_ORDER:
    a1s = f"{ev_a1.get(cid,float('nan')):.2f} / {pf_a1.get(cid,float('nan')):.2f} / {de_a1.get(cid,float('nan')):.2f}"
    bs  = f"{ev_b.get(cid,float('nan')):.2f} / {pf_b.get(cid,float('nan')):.2f} / {de_b.get(cid,float('nan')):.2f}"
    v   = ev_a1b_verdict.get(cid, '?')
    # Bold if any tool's B < 0.8
    flag = "**" if any(x < 0.8 for x in [ev_b.get(cid,1), pf_b.get(cid,1), de_b.get(cid,1)]) else ""
    print(f"| {flag}`{cid}`{flag} | {a1s} | {bs} | {v} |")

# ---- Section 2: per-tool flagged cases ----
TH = 0.8  # below this counts as "this tool said something's off"
def flagged_below(rates, th=TH):
    return sorted([cid for cid,r in rates.items() if r < th])

ev_flagged_strict = sorted([cid for cid,v in ev_a1b_verdict.items() if v == 'regressed'])
ev_flagged_loose  = flagged_below(ev_b)
pf_flagged = flagged_below(pf_b)
de_flagged = flagged_below(de_b)

print("\n## §5 表 2:每工具在 A1-vs-B 上的标记结果\n")
print(f"- **Evalith bootstrap 严格判 regressed:** {ev_flagged_strict}  ({len(ev_flagged_strict)}/10)")
print(f"- **Evalith B mean < 0.8 (放宽口径):** {ev_flagged_loose}  ({len(ev_flagged_loose)}/10)")
print(f"- **promptfoo B pass_rate < 0.8:** {pf_flagged}  ({len(pf_flagged)}/10)")
print(f"- **DeepEval B pass_rate < 0.8:** {de_flagged}  ({len(de_flagged)}/10)")

# ---- Section 3: cross-tool agreement (loose < 0.8) ----
e = set(ev_flagged_loose); p = set(pf_flagged); d = set(de_flagged)
print("\n## §5 表 3:两两/三方一致情况(B mean < 0.8 口径)\n")
print(f"- Evalith ∩ promptfoo: {sorted(e & p)}")
print(f"- Evalith ∩ DeepEval:  {sorted(e & d)}")
print(f"- promptfoo ∩ DeepEval: {sorted(p & d)}")
print(f"- 三家全部命中:       {sorted(e & p & d)}")
print(f"- 只有 Evalith 命中:   {sorted(e - p - d)}")
print(f"- 只有 promptfoo 命中: {sorted(p - e - d)}")
print(f"- 只有 DeepEval 命中:  {sorted(d - e - p)}")

# ---- Section 4: vs hypothesis ----
print("\n## §5 表 4:每工具 flagged 集合 vs 预先承诺的假设 (commit 68d1db0)\n")
for name, flagged in [("Evalith bootstrap (strict)", set(ev_flagged_strict)),
                       ("Evalith (B<0.8 loose)", e),
                       ("promptfoo (B<0.8)", p),
                       ("DeepEval (B<0.8)", d)]:
    print(f"\n**{name}** ({len(flagged)} flagged):")
    print(f"  - 与 STRONG 预测重叠: {sorted(flagged & STRONG)}")
    print(f"  - 与 BORDER 预测重叠: {sorted(flagged & BORDER)}")
    print(f"  - 落在 SAFE 预测里的 (即被预测安全却挂了): {sorted(flagged & SAFE)}")

# ---- Section 5: A1-vs-A2 noise floor (Evalith only) ----
ev_a1a2_flagged = sorted([cid for cid,v in ev_a1a2_verdict.items() if v == 'regressed'])
print("\n## §5 表 5:A1-vs-A2 噪声基线(Evalith only — promptfoo/DeepEval 未跑 A2)\n")
print(f"- Evalith bootstrap 假阳率: {len(ev_a1a2_flagged)}/10  flagged={ev_a1a2_flagged}")

# ---- Footer for the article writer ----
print("\n---\n(End of compare.py output. Paste relevant tables into article §5.)")
