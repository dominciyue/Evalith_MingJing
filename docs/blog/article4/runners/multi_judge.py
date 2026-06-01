"""Judge an existing model-output JSON with a chosen judge family.

Reads a JSON written by multi_model.py (which has per-case `outputs: list[str]`)
and produces a JSON adding `pass_rate_samples: list[float]` per case based on
how many of the trial outputs the judge passes.

The criteria string is fixed (matches article 2/3) and per-case expected_concepts
are appended as a checklist in the judge prompt."""
import os, sys, json, argparse
from pathlib import Path
import litellm
import yaml

FAMILIES = {
    "ds":  {"model": "deepseek/deepseek-chat",       "api_key_env": "DEEPSEEK_API_KEY",    "api_base": None},
    "qw":  {"model": "openai/qwen-plus",              "api_key_env": "DASHSCOPE_API_KEY",   "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
    "glm": {"model": "openai/zai-org/GLM-4.5-Air",    "api_key_env": "SILICONFLOW_API_KEY", "api_base": "https://api.siliconflow.cn/v1"},
}

# Match article 2/3 wording for direct comparability
EVAL_CFG = yaml.safe_load(open("docs/blog/article2/configs/eval.high-temp.yaml"))
BASE_CRITERIA = EVAL_CFG["scorers"][0]["params"]["criteria"]

JUDGE_PROMPT = """你是一名严格的 AI 答案评审。

问题/输入:
{input}

AI 的回答:
{output}

评判标准: {criteria}

请只输出一个 JSON 对象,不要任何其它文字:
{{"score": <0到1之间的小数>, "pass": <true 或 false>, "reason": "<简短理由>"}}
"""

def build_criteria(base, expected_concepts):
    if not expected_concepts:
        return base
    block = "\n".join(f"- {c}" for c in expected_concepts)
    return f"{base}\n\n核心概念清单(回答须覆盖):\n{block}"

def judge_call(cfg, prompt):
    kwargs = {"model": cfg["model"],
              "messages": [{"role":"user","content":prompt}],
              "temperature": 0.0,
              "api_key": os.environ.get(cfg["api_key_env"])}
    if cfg["api_base"]:
        kwargs["api_base"] = cfg["api_base"]
    r = litellm.completion(**kwargs)
    text = r.choices[0].message.content
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        return 0.0, False
    try:
        data = json.loads(text[s:e+1])
    except json.JSONDecodeError:
        return 0.0, False
    score = float(data.get("score", 0.0))
    return score, bool(data.get("pass", score >= 0.5))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge", choices=list(FAMILIES.keys()), required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = FAMILIES[args.judge]
    if not os.environ.get(cfg["api_key_env"]):
        print(f"ERROR: {cfg['api_key_env']} not set", file=sys.stderr); sys.exit(1)

    src = json.loads(Path(args.input).read_text())
    for ci, case in enumerate(src["results"]):
        criteria = build_criteria(BASE_CRITERIA, case.get("expected_concepts") or [])
        samples = []
        for trial_i, out in enumerate(case.get("outputs", [])):
            if not out:
                samples.append(0.0); continue
            prompt = JUDGE_PROMPT.format(input=case["input"], output=out, criteria=criteria)
            try:
                score, passed = judge_call(cfg, prompt)
            except Exception as e:
                print(f"  WARN {case['case_id']} trial {trial_i}: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
                passed = False
            samples.append(1.0 if passed else 0.0)
        case["pass_rate_samples"] = samples
        print(f"  {ci+1:2d}/{len(src['results'])} {case['case_id']:35s} judge={args.judge} samples={samples}")

    src["judge_family"] = args.judge
    src["judge_id"] = cfg["model"]
    Path(args.out).write_text(json.dumps(src, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()

