# 明镜 / Evalith

**English** | [中文](README.zh-CN.md)

> Catch AI regressions before your users do.

A neutral, local-first **AI regression-testing** tool. Define a test set, run it
against any model (DeepSeek / Qwen / OpenAI / Claude / …), score every case, and
**diff two runs to see exactly what got better or worse** — instead of shipping on vibes.

## Why Evalith

- **Neutral & open.** Your evaluation harness decides which model wins, so it
  shouldn't be owned by a model vendor. Evalith is vendor-independent and open
  source (Apache-2.0).
- **Local-first.** The core workflow runs entirely on your machine — no account,
  no upload, no network. Your prompts and test data stay with you.
- **China models first-class.** DeepSeek, Qwen, 豆包, GLM and global models are
  all just a `model:` string away (via [LiteLLM](https://docs.litellm.ai/docs/providers)).
- **Regressions, not vibes.** `diff` tells you which cases improved, regressed,
  or broke when you change a prompt, a model, or a version.

## Install

Requires Python ≥ 3.10.

```bash
pip install -e .            # core: pydantic, pyyaml, typer (no network needed)
pip install -e ".[litellm]" # optional: real models (DeepSeek/Qwen/OpenAI/Claude/...)
```

## Quickstart (offline, no API key)

```bash
# 1. Run the example eval — uses the offline `echo` model, passes 2/2
mingjing run examples/eval.yaml

# 2. Tweak your prompt/model in examples/eval.yaml, then run again
mingjing run examples/eval.yaml

# 3. List runs, then diff the two newest to spot regressions
mingjing list
mingjing diff <OLDER_RUN_ID> <NEWER_RUN_ID>
```

## Using a real model (DeepSeek example)

`examples/eval.deepseek.yaml` evaluates `deepseek/deepseek-chat` on a few factual
questions, scoring each with both a `contains` check and an `llm_judge` pass:

```bash
pip install -e ".[litellm]"
export DEEPSEEK_API_KEY=sk-...
mingjing run examples/eval.deepseek.yaml
# -> Run <id> saved to .mingjing/runs/<id>.json — 6/6 checks passed
```

Swap `model:` for any LiteLLM-supported model — `qwen/qwen-max`, `gpt-4o-mini`,
`claude-3-5-sonnet`, etc. — and set that provider's API key.

## Scorers

| type | passes when |
|------|-------------|
| `exact_match` | output equals the case's `expected` |
| `contains`    | output contains `params.text` (or the case's `expected`) |
| `regex`       | output matches `params.pattern` |
| `llm_judge`   | an LLM grades the output against `params.criteria` |

## How it works

`run` evaluates a config against a model and saves a **Run** — a JSON snapshot of
every case's output and scores — to `.mingjing/runs/`. `diff` compares two saved
runs case-by-case and labels each **improved / regressed / unchanged / new /
removed**, so a prompt or model change can't silently break things.

## Status

v0.1 — single-turn prompt evaluation, file-based run store, run-to-run diff, and a
provider layer for Chinese + global models. Team/cloud features are on the roadmap.
Issues and PRs welcome.

## License

[Apache-2.0](LICENSE). Copyright © 2026 Evalith (明镜) Authors.
