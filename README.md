# 明镜 / Evalith

**English** | [中文](README.zh-CN.md)

> Catch AI regressions before your users do.

A neutral, local-first **AI regression-testing** tool. Define a test set, run it
against any model (DeepSeek / Qwen / OpenAI / Claude / …), score every case, and
**diff two runs to see exactly what got better or worse** — then **gate CI** so a
prompt or model change can't silently break your product.

## Why Evalith

- **Neutral & open.** Your evaluation harness decides which model wins, so it
  shouldn't be owned by a model vendor. Evalith is vendor-independent and open
  source (Apache-2.0).
- **Local-first.** The core workflow runs entirely on your machine — no account,
  no upload, no network. Your prompts and test data stay with you.
- **China models first-class.** DeepSeek, Qwen and global models are first-class
  aliases (`evalith models`); a Chinese `llm_judge` ships in the box.
- **Regressions, not vibes.** `diff` and `--fail-on-regression` tell you which
  cases improved, regressed, or broke when you change a prompt, model, or version.

## Install

Requires Python ≥ 3.10.

```bash
pip install evalith              # core: pydantic, pyyaml, typer
pip install "evalith[litellm]"   # optional: real models (DeepSeek/Qwen/OpenAI/Claude/...)
```

Or from source: `git clone https://github.com/dominciyue/Evalith_MingJing` then `pip install -e ".[litellm]"`.

## Quickstart (offline, no API key)

```bash
# 1. Run the example eval — uses the offline `echo` model, passes 2/2
evalith run examples/eval.yaml

# 2. Tweak your prompt/model in examples/eval.yaml, then run again
evalith run examples/eval.yaml

# 3. List runs, then diff the two newest to spot regressions
evalith list
evalith diff <OLDER_RUN_ID> <NEWER_RUN_ID>
```

## Gate CI on regressions

Fail a build when quality drops — two ways:

```bash
# Absolute gate: fail if fewer than 90% of checks pass (no baseline needed)
evalith run examples/eval.yaml --fail-under 0.9

# Relative gate: fail if any case regressed vs a baseline run
evalith diff <BASELINE_RUN_ID> <NEW_RUN_ID> --fail-on-regression
```

Both exit non-zero on failure, so CI stops the PR. This repo ships a **composite
GitHub Action** — drop this into `.github/workflows/eval.yml`:

```yaml
name: AI eval gate
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dominciyue/Evalith_MingJing@main
        with:
          config: examples/eval.yaml
          fail-under: "0.9"
```

(See `.github/workflows/eval-example.yml` for a working copy using the offline demo.)

**Catch regressions vs a baseline.** `diff` accepts run IDs *or* `.json` file paths, so CI needs no shared state — bless a baseline once, commit it, then compare each PR's fresh run against it:

```bash
evalith run examples/eval.yaml --out baseline.json   # bless once, commit baseline.json
# then, in CI on a PR:
evalith run examples/eval.yaml --out current.json
evalith diff baseline.json current.json --fail-on-regression
```

## Tame LLM noise with sampling

LLM outputs can drift between calls (even at temperature 0 with some providers). To stop random noise from looking like a regression, run each case multiple times and let Evalith bootstrap a 95% confidence interval on Δ:

```bash
evalith run examples/eval.yaml --samples 5 --out current.json
evalith diff baseline.json current.json --fail-on-regression
# -> a case is only flagged "regressed" when the 95% CI on (after − before) is fully below zero
```

Single-shot runs (`--samples 1`, the default) behave exactly as before. The diff report adds a `Δ 95% CI` column when sampling was used.

## Shareable reports

Turn a run or a diff into Markdown (for PR comments) or a self-contained HTML page:

```bash
evalith report <RUN_ID> --format md                       # Markdown to stdout
evalith report <RUN_ID> --format html --output report.html # standalone HTML file
evalith diff <A> <B> --format md --output diff.md          # diff as Markdown
```

Reports include the pass rate, mean score, and — for real models — **cost, token
count, and latency**.

## Using real models (国产 first-class)

```bash
evalith models          # list first-class aliases + the env var each needs
export DEEPSEEK_API_KEY=sk-...
evalith run examples/eval.deepseek.yaml --concurrency 3
# -> Run <id> saved to .evalith/runs/<id>.json — 6/6 checks passed
```

Set `model:` to an alias (`deepseek-chat`, `deepseek-reasoner`, `qwen-max`,
`qwen-plus`) or any LiteLLM id directly (`gpt-4o-mini`, `claude-3-5-sonnet`, …),
then set that provider's API key. The `llm_judge` scorer can grade in Chinese with
`params: {language: zh}`.

## Scale

- `--concurrency N` runs cases in parallel (provider calls are I/O-bound), or set
  `concurrency:` in the config. Order of results is always preserved.
- Datasets load from **YAML, JSON, CSV, or JSONL** (`examples/qa.jsonl`).

## Scorers

| type | passes when |
|------|-------------|
| `exact_match` | output equals the case's `expected` |
| `contains`    | output contains `params.text` (or the case's `expected`) |
| `regex`       | output matches `params.pattern` |
| `llm_judge`   | an LLM grades the output against `params.criteria` (`params.language: en\|zh`) |

## How it works

`run` evaluates a config against a model and saves a **Run** — a JSON snapshot of
every case's output, scores, tokens, cost, and latency — to `.evalith/runs/`.
`diff` compares two saved runs case-by-case and labels each **improved / regressed
/ unchanged / new / removed**.

## Status

v0.4 — single-turn prompt evaluation, file-based run store, run-to-run diff with
per-case output comparison **and bootstrap 95% CI on Δ (`--samples N`) so LLM
noise can't masquerade as a regression**, CI gating (`--fail-under`,
`--fail-on-regression`, file-based baselines, GitHub Action), Markdown/HTML
reports, concurrency with per-case error isolation, cost/token/latency tracking,
and 国产 model aliases with a Chinese judge. Team/cloud features are on the
roadmap. Issues and PRs welcome.

## License

[Apache-2.0](LICENSE). Copyright © 2026 Evalith (明镜) Authors.
