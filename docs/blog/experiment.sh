#!/usr/bin/env bash
# Run the two blog-post experiments against real DeepSeek. Outputs two markdown
# tables ready to paste into docs/blog/2026-05-28-bootstrap-ci-for-ai-eval.zh.md.
#
# Usage:  DEEPSEEK_API_KEY=sk-... bash docs/blog/experiment.sh
#
# What it does:
#   A) Two identical runs of examples/eval.deepseek.yaml with --samples 5;
#      diff them. Expectation: all `unchanged`, CIs span 0 -> no false alarms.
#   B) Baseline vs a deliberately broken prompt (forces English answer so the
#      Chinese expected-substring fails). Expectation: regressions captured by
#      bootstrap with CIs fully below zero.

set -euo pipefail

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is not set. Run as:"
    echo "    DEEPSEEK_API_KEY=sk-... bash docs/blog/experiment.sh"
    exit 1
fi

cd "$(dirname "$0")/../.."
EXP=$(mktemp -d)

# Make a "broken" config: same dataset, but ask the model to answer in English
# only -> output won't contain the Chinese `expected` strings.
cat > "$EXP/eval.broken.yaml" <<'YAML'
name: deepseek-broken
dataset: examples/qa.yaml
model: deepseek-chat
prompt_template: "Answer only in English, even if the question is Chinese: {{input}}"
temperature: 0.0
concurrency: 3
scorers:
  - type: contains
  - type: llm_judge
    params:
      criteria: "回答是否准确、直接地回应了问题"
      language: zh
YAML

echo "=== A) Noise-floor: identical config, two runs ==="
python3 -m evalith.cli run examples/eval.deepseek.yaml --samples 5 \
    --out "$EXP/a1.json" --store "$EXP/.store"
python3 -m evalith.cli run examples/eval.deepseek.yaml --samples 5 \
    --out "$EXP/a2.json" --store "$EXP/.store"

echo "=== B) Real regression: baseline vs broken prompt ==="
python3 -m evalith.cli run "$EXP/eval.broken.yaml" --samples 5 \
    --out "$EXP/broken.json" --store "$EXP/.store"

python3 - "$EXP/a1.json" "$EXP/a2.json" "$EXP/broken.json" <<'PY'
import sys
from pathlib import Path
from evalith.diff import diff_runs
from evalith.models import Run

def load(p): return Run.model_validate_json(Path(p).read_text(encoding="utf-8"))

def table(report, title):
    print(f"\n### {title}\n")
    print("| case | mean before | mean after | bootstrap CI on Δ | status |")
    print("|---|---|---|---|---|")
    for c in report.cases:
        b = "—" if c.before is None else f"{c.before:.2f}"
        a = "—" if c.after is None else f"{c.after:.2f}"
        ci = "—" if c.ci is None else f"[{c.ci[0]:+.2f}, {c.ci[1]:+.2f}]"
        print(f"| `{c.case_id}` | {b} | {a} | {ci} | {c.status} |")

a1, a2, broken = map(load, sys.argv[1:4])
print("\n" + "=" * 72)
print("PASTE THE TWO TABLES BELOW INTO THE BLOG POST'S SECTION 五")
print("=" * 72)
table(diff_runs(a1, a2),     "实验 A — 同一 prompt 跑两次 (噪声基线)")
table(diff_runs(a1, broken), "实验 B — 基线 vs 故意破坏的 prompt (真实回退)")
PY

echo ""
echo "Done. Artifacts at: $EXP"
