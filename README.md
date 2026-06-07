# 明镜 / Evalith

**English** | [中文](README.zh-CN.md)

[![PyPI](https://img.shields.io/pypi/v/evalith.svg)](https://pypi.org/project/evalith/)
[![Python](https://img.shields.io/pypi/pyversions/evalith.svg)](https://pypi.org/project/evalith/)
[![CI](https://github.com/dominciyue/Evalith_MingJing/actions/workflows/ci.yml/badge.svg)](https://github.com/dominciyue/Evalith_MingJing/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/evalith.svg)](https://github.com/dominciyue/Evalith_MingJing/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/evalith)](https://pepy.tech/project/evalith)

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

## What's new in v0.7

- **Judge consensus panel** — attach extra judges to one eval; get per-case disagreement, pairwise Cohen's κ, per-domain agreement and ⚠ low-consensus flags. Primary judge still gates; the panel never blocks CI.

## What's new in v0.6

- **Per-case `expected_concepts` in `llm_judge`.** Each dataset case can now declare `expected_concepts: [...]` and the judge prompt automatically appends them as a coverage checklist. Closes the limitation noted in articles 2 and 3 (judge had no per-case checklist). Fully backward compatible: cases without `expected_concepts` behave identically to v0.5.
- **Adaptive sampling.** `evalith run --adaptive --min-samples 2 --max-samples 10 --ci-tolerance 0.2` runs each case until the bootstrap CI on its pass-rate samples is narrower than the tolerance (or `max_samples` reached). Stable cases stop early — saves API cost without losing statistical signal on noisy ones.

## What's new in v0.5

- **`--ci-method bca`** — BCa (bias-corrected and accelerated) bootstrap on Δ. Stdlib-only; more accurate than percentile when the bootstrap distribution is skewed.
- **`--ci-method paired`** — paired bootstrap. Reduces variance when before/after correlate through a shared case dimension.
- **`--multi-test bh`** — Benjamini-Hochberg FDR control across cases. With many cases, percentile alone can over-report regressions; BH compresses the family-wise false-positive rate.
- **scipy** is now a dev dependency (used as ground truth in tests). Not pulled into runtime — production installs stay minimal.

All v0.5 additions are opt-in. The v0.4 default behavior is byte-for-byte preserved.

## Status

v0.4 — single-turn prompt evaluation, file-based run store, run-to-run diff with
per-case output comparison **and bootstrap 95% CI on Δ (`--samples N`) so LLM
noise can't masquerade as a regression**, CI gating (`--fail-under`,
`--fail-on-regression`, file-based baselines, GitHub Action), Markdown/HTML
reports, concurrency with per-case error isolation, cost/token/latency tracking,
and 国产 model aliases with a Chinese judge. Team/cloud features are on the
roadmap. Issues and PRs welcome.

## Read more

- **Deep dive (Chinese):** [AI 回归测试需要统计显著性: 用 bootstrap CI 抗 LLM 噪声](https://zhuanlan.zhihu.com/p/2043351926964848178) — why point-to-point eval diffs are statistically wrong, the math behind Evalith's `--samples N`, and a reproducible DeepSeek A/B experiment.
- **Follow-up (Chinese):** [续:LLM 当 judge 自己也在抖](https://zhuanlan.zhihu.com/p/2044542154400322098) — noise-immunity validation under high temperature + llm_judge, plus a 3-tool comparison against promptfoo and DeepEval. Source + raw data in [`docs/blog/article2/`](docs/blog/article2/).
- **Second follow-up (Chinese):** [续之续:更严的统计方法 + 第三方 judge,article 2 的结论变了吗?](https://zhuanlan.zhihu.com/p/2044820946721231928) — Adds BCa, paired bootstrap, BH FDR to Evalith and runs a double-track Qwen-plus cross-judge experiment on article 2's frozen raw data. v0.5 release. Source + raw data in [`docs/blog/article3/`](docs/blog/article3/).
- **Third follow-up (Chinese):** [续之续之续:换三个 judge、三个模型、五个领域,judge 的分歧到底藏在哪?](https://zhuanlan.zhihu.com/p/2046436508459070253) — 30 case × 5 domains × 3 judges × 3 models maps the *domain structure* of judge disagreement (code = max disagreement, safety/concept = consensus). Productized in v0.7's judge consensus panel. Source + raw data in [`docs/blog/article4/`](docs/blog/article4/).
- **Design docs & TDD plans:** [`docs/`](docs/) — v0.1 spec, v0.1/v0.2 task-by-task plans, and the blog source.

## License

[Apache-2.0](LICENSE). Copyright © 2026 Evalith (明镜) Authors.
