"""Cross-judge swap A: hold model outputs constant, only re-judge with qwen-plus.

Reads article 2 frozen raw (a1.json or b.json), extracts each case's model output
(trial 0 — same data Evalith stored), re-judges using qwen-plus via DashScope's
OpenAI-compatible endpoint with the SAME criteria string article 2 used, and
writes a new pass_rate shaped JSON ready for downstream comparison.

Cross-judge design:
- Judge family swap (deepseek-chat → qwen-plus); a different LLM vendor
- Judge temperature: 0.0 (matches article 2's evalith llm_judge default exactly)
- Only one variable changes (judge family), so verdict differences vs article 2
  are attributable to judge identity alone — clean cross-judge isolation.

Usage:
  DASHSCOPE_API_KEY=sk-... python3 docs/blog/article3/rejudge.py <input.json> <output.json>
"""
import json, os, sys
from pathlib import Path
import litellm
import yaml

if len(sys.argv) != 3:
    print(f"usage: {sys.argv[0]} <input.json> <output.json>", file=sys.stderr); sys.exit(1)

API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
if not API_KEY:
    print("ERROR: set DASHSCOPE_API_KEY (or OPENAI_API_KEY fallback)", file=sys.stderr); sys.exit(1)

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
        model="openai/qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        api_key=API_KEY,
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
            samples.append(0.0)
    out.append({
        "case_id": cid,
        "pass_rate_samples": samples,
        "input": input_text,
        "output_used": output_text[:120],
    })
    print(f"  {cid:35s} judge samples: {samples}")

Path(sys.argv[2]).write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"wrote {sys.argv[2]}")
