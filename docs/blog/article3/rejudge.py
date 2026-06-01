"""Cross-judge swap A: hold model outputs constant, only re-judge with gpt-5-mini.

Reads article 2 frozen raw (a1.json or b.json), extracts each case's model output
(trial 0 — same data Evalith stored), re-judges using gpt-5-mini via the sublyx
proxy with the SAME criteria string article 2 used, and writes a new pass_rate
shaped JSON ready for downstream comparison.

Deviation from original plan (documented in article §5/§7):
- Model: gpt-5-mini (originally gpt-4o-mini; sublyx proxy via ChatGPT-Codex
  account doesn't expose gpt-4o-mini)
- Judge temperature: 1.0 (forced; gpt-5 family rejects temp=0)
  This adds judge noise that wasn't in article 2's baseline (deepseek-chat
  judge at temp=0); §5 prose accounts for this conflation.

Usage:
  OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.sublyx.org/v1 \\
    python3 docs/blog/article3/rejudge.py <input.json> <output.json>

Note: article 2 stored only trial-0 output (engine.py:48-58 takes rep=trials[0]).
So this rejudge gets 5 judge samples on the SAME single output — capturing judge
noise (the LLM judge's own jitter) but not model output noise. For full multi-trial
judging we'd need stored multi-trial outputs, which is article 4 territory."""
import json, os, sys
from pathlib import Path
import litellm
import yaml

if len(sys.argv) != 3:
    print(f"usage: {sys.argv[0]} <input.json> <output.json>", file=sys.stderr); sys.exit(1)

API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.sublyx.org/v1")
if not API_KEY:
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
        model="openai/gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,  # forced by gpt-5 family
        api_base=BASE_URL,
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
    samples = []
    for trial in range(5):
        try:
            score, passed = judge(input_text, output_text)
            samples.append(1.0 if passed else 0.0)
        except Exception as e:
            print(f"  WARN {cid} trial {trial}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            samples.append(0.0)  # conservative: failed judge call → fail
    out.append({
        "case_id": cid,
        "pass_rate_samples": samples,
        "input": input_text,
        "output_used": output_text[:120],
    })
    print(f"  {cid:35s} judge samples: {samples}")

Path(sys.argv[2]).write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"wrote {sys.argv[2]}")
