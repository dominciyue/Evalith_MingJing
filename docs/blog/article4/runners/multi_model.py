"""Generate model outputs for one (family, variant) combo.

Output schema mirrors Evalith's Run, with `pass_rate_samples` left empty (judge
fills it later via multi_judge.py)."""
import os, sys, json, uuid, argparse
from datetime import datetime, timezone
from pathlib import Path
import litellm
import yaml

FAMILIES = {
    "ds":  {"model": "deepseek/deepseek-chat",          "api_key_env": "DEEPSEEK_API_KEY",   "api_base": None},
    "qw":  {"model": "openai/qwen-plus",                 "api_key_env": "DASHSCOPE_API_KEY",  "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
    "glm": {"model": "openai/zai-org/GLM-4.5-Air",       "api_key_env": "SILICONFLOW_API_KEY","api_base": "https://api.siliconflow.cn/v1"},
}
BASELINE_PREFIX = ""
BROKEN_PREFIX = "Be very concise. Answer in 1-2 short sentences. Skip explanations and concept enumeration: "

def model_call(family_cfg, prompt):
    kwargs = {"model": family_cfg["model"],
              "messages": [{"role":"user","content":prompt}],
              "temperature": 1.0,
              "api_key": os.environ.get(family_cfg["api_key_env"])}
    if family_cfg["api_base"]:
        kwargs["api_base"] = family_cfg["api_base"]
    r = litellm.completion(**kwargs)
    return r.choices[0].message.content

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=list(FAMILIES.keys()), required=True)
    p.add_argument("--variant", choices=["baseline","broken"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--dataset", default="docs/blog/article4/qa.small.yaml")
    args = p.parse_args()

    cfg = FAMILIES[args.family]
    if not os.environ.get(cfg["api_key_env"]):
        print(f"ERROR: {cfg['api_key_env']} not set", file=sys.stderr); sys.exit(1)

    ds = yaml.safe_load(open(args.dataset))
    prefix = BASELINE_PREFIX if args.variant == "baseline" else BROKEN_PREFIX

    results = []
    for ci, case in enumerate(ds["cases"]):
        cid = case["id"]
        input_text = case["input"]
        ec = case.get("expected_concepts") or []
        outputs = []
        for trial in range(args.samples):
            try:
                out = model_call(cfg, prefix + input_text)
            except Exception as e:
                print(f"  WARN {cid} trial {trial}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
                out = ""
            outputs.append(out)
        results.append({
            "case_id": cid,
            "input": input_text,
            "domain": case.get("domain"),
            "expected_concepts": ec,
            "outputs": outputs,
            # downstream judging fills pass_rate_samples
        })
        print(f"  {ci+1:2d}/{len(ds['cases'])} {cid:35s} {[o[:30] for o in outputs[:1]]}")

    run = {
        "id": uuid.uuid4().hex[:12],
        "name": f"{args.family}-{args.variant}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_family": args.family,
        "model_id": cfg["model"],
        "variant": args.variant,
        "samples_per_case": args.samples,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(run, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
