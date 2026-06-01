"""Cross-judge swap B: gpt-5-mini does BOTH model output and judge scoring.

Loads docs/blog/article2/qa.high-temp.yaml as dataset, runs each case 5 times
through gpt-5-mini at temp=1 (forced by gpt-5 family), then has gpt-5-mini judge
each output against article 2's same criteria string at temp=1 (also forced).

Writes a Run-shaped JSON consumable by `Run.model_validate_json(...)` so the
downstream compare.py + diff_runs can read it like an Evalith run.

Deviation from plan documented in article §5/§7: judge temp=1 instead of temp=0;
also a different model family (gpt-5-mini) than article 2's gpt-4o-mini target.

Usage:
  OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.sublyx.org/v1 \\
    python3 docs/blog/article3/openai_runner.py <variant> <out.json>

  variant = "baseline" (passthrough prompt) | "broken" (prompt with bias injection)
"""
import json, os, sys, uuid
from datetime import datetime, timezone
from pathlib import Path
import litellm
import yaml

if len(sys.argv) != 3:
    print(f"usage: {sys.argv[0]} <baseline|broken> <out.json>", file=sys.stderr); sys.exit(1)

VARIANT = sys.argv[1]
OUT_PATH = sys.argv[2]
if VARIANT not in {"baseline", "broken"}:
    print(f"variant must be 'baseline' or 'broken', got {VARIANT!r}", file=sys.stderr); sys.exit(1)

API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.sublyx.org/v1")
if not API_KEY:
    print("ERROR: OPENAI_API_KEY not set", file=sys.stderr); sys.exit(1)

DATASET = yaml.safe_load(Path("docs/blog/article2/qa.high-temp.yaml").read_text())
EVAL_CFG = yaml.safe_load(Path("docs/blog/article2/configs/eval.high-temp.yaml").read_text())
CRITERIA = EVAL_CFG["scorers"][0]["params"]["criteria"]

BASELINE_PREFIX = ""
BROKEN_PREFIX = "Be very concise. Answer in 1-2 short sentences. Skip explanations and concept enumeration: "
PROMPT_PREFIX = BASELINE_PREFIX if VARIANT == "baseline" else BROKEN_PREFIX

MODEL = "openai/gpt-5-mini"
TEMP = 1.0

JUDGE_PROMPT = """你是一名严格的 AI 答案评审。

问题/输入:
{input}

AI 的回答:
{output}

评判标准: {criteria}

请只输出一个 JSON 对象,不要任何其它文字:
{{"score": <0到1之间的小数>, "pass": <true 或 false>, "reason": "<简短理由>"}}
"""

def model_call(prompt: str) -> str:
    r = litellm.completion(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=TEMP, api_base=BASE_URL,
    )
    return r.choices[0].message.content

def judge(input_text: str, output_text: str) -> tuple[float, bool, str]:
    prompt = JUDGE_PROMPT.format(input=input_text, output=output_text, criteria=CRITERIA)
    r = litellm.completion(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=TEMP, api_base=BASE_URL,
    )
    text = r.choices[0].message.content
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return 0.0, False, f"no JSON in judge: {text[:120]}"
    data = json.loads(text[start:end+1])
    score = float(data.get("score", 0.0))
    passed = bool(data.get("pass", score >= 0.5))
    reason = str(data.get("reason", ""))
    return score, passed, reason

SAMPLES = 5
results = []
for case in DATASET["cases"]:
    cid = case["id"]
    input_text = case["input"]
    trial_outputs = []
    pass_samples = []
    first_score = 0.0
    first_passed = False
    first_reason = ""
    for trial in range(SAMPLES):
        try:
            model_out = model_call(PROMPT_PREFIX + input_text)
            score, passed, reason = judge(input_text, model_out)
        except Exception as e:
            print(f"  WARN {cid} trial {trial}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            model_out, score, passed, reason = "", 0.0, False, f"error: {type(e).__name__}"
        trial_outputs.append(model_out)
        pass_samples.append(1.0 if passed else 0.0)
        if trial == 0:
            first_score, first_passed, first_reason = score, passed, reason
    # CaseResult shape — Evalith engine takes trial[0] as representative
    results.append({
        "case_id": cid,
        "input": input_text,
        "output": trial_outputs[0],
        "scores": [
            {"scorer": "llm_judge", "value": first_score, "passed": first_passed, "detail": first_reason}
        ],
        "latency_ms": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "pass_rate_samples": pass_samples,
    })
    print(f"  {cid:35s} pass_rate_samples={pass_samples}")

run = {
    "id": uuid.uuid4().hex[:12],
    "name": f"openai-swap-b-{VARIANT}",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "model": MODEL,
    "results": results,
    "config": {},
}

Path(OUT_PATH).write_text(json.dumps(run, indent=2, ensure_ascii=False))
print(f"wrote {OUT_PATH}")
