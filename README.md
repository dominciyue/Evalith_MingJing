# 明镜 / Evalith

A neutral, local-first **AI regression-testing** tool. Define a test set, run it
against any model (DeepSeek / Qwen / OpenAI / Claude / …), score each case, and
**diff two runs to see exactly what got better or worse** — instead of shipping on vibes.

## Install

Requires Python >=3.10. From source (v0.1):

```bash
pip install -e .            # core: pydantic, pyyaml, typer
pip install -e ".[litellm]" # optional: real models (DeepSeek/Qwen/OpenAI/Claude/...)
```

## Quickstart

```bash
# 1. Run the example eval (offline `echo` model, no API key needed)
mingjing run examples/eval.yaml

# 2. Change your prompt/model in examples/eval.yaml, run again
mingjing run examples/eval.yaml

# 3. List runs, then diff the two newest to spot regressions
mingjing list
mingjing diff <OLDER_RUN_ID> <NEWER_RUN_ID>
```

## Using a real model

Set the provider API key (see LiteLLM docs) and set `model:` in `examples/eval.yaml`,
e.g. `deepseek/deepseek-chat`, `qwen/qwen-max`, `gpt-4o-mini`, or `claude-3-5-sonnet`.

```bash
export DEEPSEEK_API_KEY=sk-...
# set  model: deepseek/deepseek-chat  in eval.yaml
mingjing run examples/eval.yaml
```

## Scorers

- `exact_match` — output equals the case's `expected`
- `contains` — output contains `params.text` (or the case's `expected`)
- `regex` — output matches `params.pattern`
- `llm_judge` — an LLM grades the output against `params.criteria`
