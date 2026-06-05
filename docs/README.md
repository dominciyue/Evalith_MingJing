# Evalith Docs

This folder contains two kinds of documentation:

## `blog/` — public technical writeups

Long-form articles + reproducible experiments aimed at external readers.
Each post is self-contained and links back to the package on PyPI / repo on GitHub.

- [`2026-05-28-bootstrap-ci-for-ai-eval.zh.md`](blog/2026-05-28-bootstrap-ci-for-ai-eval.zh.md) — Why AI regression tests need statistical significance; bootstrap CI implementation in Evalith, with a real DeepSeek A/B experiment. (Chinese)
- [`2026-05-31-llm-judge-noise-bootstrap.zh.md`](blog/2026-05-31-llm-judge-noise-bootstrap.zh.md) — 续:LLM 当 judge 自己也在抖 —— bootstrap CI 抗噪与三个 OSS eval 工具的同台对照 (Chinese, [Zhihu](https://zhuanlan.zhihu.com/p/2044542154400322098))
- [`2026-06-01-article3-statistical-methods.zh.md`](blog/2026-06-01-article3-statistical-methods.zh.md) — 续之续:把 BCa、paired bootstrap、FDR、第三方 judge 都挂上去,文章 2 的结论变了吗?(Chinese, [Zhihu](https://zhuanlan.zhihu.com/p/2044820946721231928))
- [`2026-06-05-article4-cross-judge-cross-model.zh.md`](blog/2026-06-05-article4-cross-judge-cross-model.zh.md) — 续之续之续:换三个 judge、三个模型、五个领域,judge 的分歧到底藏在哪?30 case × 5 领域 × 3 judge × 3 model,judge 分歧的领域结构。(Chinese, 未发布)
- [`experiment.sh`](blog/experiment.sh) — one-liner reproduction of the article's DeepSeek experiment (`DEEPSEEK_API_KEY=sk-... bash docs/blog/experiment.sh`).

## `design/` — design specs and implementation plans

Historical design artifacts that drove the build. Useful if you want to
understand *why* Evalith looks the way it does, or contribute a feature in
the same TDD style.

- [`specs/2026-05-25-ai-eval-regression-tool-design.md`](design/specs/2026-05-25-ai-eval-regression-tool-design.md) — v0.1 product/architecture spec.
- [`plans/2026-05-25-ai-eval-tool-v0.1.md`](design/plans/2026-05-25-ai-eval-tool-v0.1.md) — v0.1 task-by-task TDD plan (MVP: CLI + scorers + diff).
- [`plans/2026-05-26-ai-eval-tool-v0.2.md`](design/plans/2026-05-26-ai-eval-tool-v0.2.md) — v0.2 task-by-task TDD plan (CI gating, reports, concurrency, cost tracking).

> Newer hardening (v0.3 per-case isolation / output diff / `--out` baseline, v0.4 bootstrap CI) is documented in git history + the blog post; no separate plan was written.
