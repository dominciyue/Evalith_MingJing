"""Run the same 10-case dataset through DeepEval's GEval scorer, 5 samples per
case, both baseline prompt and regression-injected prompt. Output a per-case
pass rate JSON in the same shape Evalith uses, so the comparison script
downstream can be tool-agnostic.

Judge model: deepseek-chat via DeepEval's native DeepSeekModel class
  (deepeval.models.llms.deepseek_model.DeepSeekModel). DeepEval 4.0.5 ships
  first-class DeepSeek support — no custom wrapper needed. GEval automatically
  falls back from log-prob weighted scoring to schema-based scoring because
  deepseek-chat does not expose log probs (supports_log_probs() → False).

Test-output model: deepseek/deepseek-chat at temp=1.0 via litellm — identical
  to the Evalith and promptfoo runs, so results are directly comparable.

API key: read from DEEPSEEK_API_KEY env var (required).
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import litellm
import yaml

from deepeval.metrics import GEval
from deepeval.models.llms.deepseek_model import DeepSeekModel
from deepeval.test_case import LLMTestCase
from deepeval.test_case.llm_test_case import SingleTurnParams

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # /home/ygwang/newproject
ARTICLE_DIR = REPO_ROOT / "docs" / "blog" / "article2"

DATASET = yaml.safe_load((ARTICLE_DIR / "qa.high-temp.yaml").read_text())
EVAL_CFG = yaml.safe_load(
    (ARTICLE_DIR / "configs" / "eval.high-temp.yaml").read_text()
)

SAMPLES = 5
TEMP = 1.0
LITELLM_MODEL = "deepseek/deepseek-chat"  # for generating test outputs

BASELINE_PREFIX = ""
BROKEN_PREFIX = (
    "Be very concise. Answer in 1-2 short sentences. "
    "Skip explanations and concept enumeration: "
)

# Same judge criteria string used by Evalith and promptfoo runs
JUDGE_CRITERIA: str = EVAL_CFG["scorers"][0]["params"]["criteria"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        sys.exit("ERROR: DEEPSEEK_API_KEY env var is not set.")
    return key


def model_call(prompt: str, api_key: str) -> str:
    """Generate a model response via litellm (deepseek-chat, temp=1.0)."""
    r = litellm.completion(
        model=LITELLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMP,
        api_key=api_key,
    )
    return r.choices[0].message.content


def make_judge(api_key: str) -> GEval:
    """Build a new GEval instance backed by deepseek-chat as judge."""
    judge_model = DeepSeekModel(
        model="deepseek-chat",
        api_key=api_key,
        # Judge calls are deterministic — temperature=0 for reproducibility
        temperature=0.0,
    )
    return GEval(
        name="Concept coverage",
        criteria=JUDGE_CRITERIA,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge_model,
        threshold=0.5,
        # async_mode=False avoids nested-event-loop issues in some envs
        async_mode=False,
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run(prefix: str, tag: str, api_key: str) -> None:
    cases = DATASET["cases"]
    out = []

    for case in cases:
        passes = 0
        outputs = []

        for i in range(SAMPLES):
            prompt = prefix + case["input"]
            answer = model_call(prompt, api_key)
            outputs.append(answer)

            # GEval judges this single (input, answer) pair
            metric = make_judge(api_key)
            tc = LLMTestCase(input=case["input"], actual_output=answer)
            metric.measure(tc)

            if metric.score >= 0.5:
                passes += 1

        record = {
            "case_id": case["id"],
            "pass_rate": passes / SAMPLES,
            "samples": SAMPLES,
            "first_output_preview": outputs[0][:200],
        }
        out.append(record)
        print(f"  {case['id']:40s} {passes}/{SAMPLES} pass  (score last sample: {metric.score:.3f})")

    out_path = ARTICLE_DIR / "raw" / f"deepeval_{tag}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    api_key = get_api_key()

    print("=== DeepEval baseline (a1) ===")
    run(BASELINE_PREFIX, "a1", api_key)

    print("\n=== DeepEval regression-injected (b) ===")
    run(BROKEN_PREFIX, "b", api_key)
