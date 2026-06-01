#!/usr/bin/env bash
# Article 3 end-to-end reproduction.
#
# Usage:
#   OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.sublyx.org/v1 \
#     bash docs/blog/article3/experiment.sh [OUT_DIR]
#
# What it does:
#   Phase 1 (stats): runs 4 statistical methods on article 2 frozen raw
#                    → §2/§3/§4 tables (percentile / BCa / paired / FDR)
#   Phase 2 (swap A): rejudges article 2 outputs with gpt-5-mini
#                    → rejudge_a1.json, rejudge_b.json
#   Phase 3 (swap B): fresh gpt-5-mini A1 + B runs
#                    → openai_a1.json, openai_b.json
#   Phase 4 (synthesis): runs compare.py for §5/§6 tables + predictions check
#
# Cost: ~$0.40 OpenAI (sublyx proxy); ~10-15 min wall-clock.
# Article 2's DeepSeek runs are not re-executed — we use the frozen raw.

set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: set OPENAI_API_KEY" >&2; exit 1
fi
: "${OPENAI_BASE_URL:=https://api.sublyx.org/v1}"
export OPENAI_BASE_URL

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

OUT="${1:-$(mktemp -d -t article3_repro.XXXXXX)}"
mkdir -p "${OUT}"
echo "=== Article 3 reproduction ===" >&2
echo "Repo root: ${REPO_ROOT}" >&2
echo "Output dir: ${OUT}" >&2
echo "OpenAI base URL: ${OPENAI_BASE_URL}" >&2

echo "" >&2
echo "=== Phase 1: 4 statistical methods on article 2 frozen raw ===" >&2
python3 docs/blog/article3/stats_runner.py | tee "${OUT}/stats.txt"

echo "" >&2
echo "=== Phase 2: swap A — rejudge with gpt-5-mini ===" >&2
python3 docs/blog/article3/rejudge.py \
    docs/blog/article2/raw/a1.json "${OUT}/rejudge_a1.json"
python3 docs/blog/article3/rejudge.py \
    docs/blog/article2/raw/b.json  "${OUT}/rejudge_b.json"

echo "" >&2
echo "=== Phase 3: swap B — fresh gpt-5-mini runs ===" >&2
python3 docs/blog/article3/openai_runner.py baseline "${OUT}/openai_a1.json"
python3 docs/blog/article3/openai_runner.py broken   "${OUT}/openai_b.json"

echo "" >&2
echo "=== Phase 4: §5/§6 tables + predictions check ===" >&2
python3 docs/blog/article3/compare.py | tee "${OUT}/compare.txt"

echo "" >&2
echo "Done. Artifacts: ${OUT}" >&2
echo "" >&2
echo "Note: compare.py reads from docs/blog/article3/raw/ (the article's" >&2
echo "frozen verdicts). If you want the §5/§6 numbers computed against" >&2
echo "YOUR fresh runs in ${OUT}, replace docs/blog/article3/raw/ contents" >&2
echo "with those files first, or modify compare.py's RAW_PATHS constant." >&2
