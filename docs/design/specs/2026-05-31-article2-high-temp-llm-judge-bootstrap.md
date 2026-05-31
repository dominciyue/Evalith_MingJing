# Spec: 第二篇技术文章 — 高 temperature + llm_judge 下的 bootstrap CI 实验 + OSS eval 工具横评

**Date:** 2026-05-31
**Author:** PattyWoods (with Claude collaboration)
**Status:** Approved — moving to writing-plans
**Targets:** Zhihu 知乎 (primary). English version is a follow-up, out of scope here.

---

## 1. Background and motivation

The first article (`docs/blog/2026-05-28-bootstrap-ci-for-ai-eval.zh.md`, published on
Zhihu at <https://zhuanlan.zhihu.com/p/2043351926964848178>) argued that
point-to-point eval diffing is statistically wrong and proposed Evalith's
percentile bootstrap CI as the fix. It included a real DeepSeek experiment at
`temperature=0` on three factual short-answer questions.

That experiment honestly closed with a limitation:

> temperature 0 + 简单事实题让 DeepSeek 这次表现极其确定,5 次采样里几乎没观测到 LLM 抖动(所以 CI 都很窄)。bootstrap 的"噪声免疫"优势在 `llm_judge` 评分器、高 temperature、或更接近边界的开放式问答里才会真正显著,下一篇文章会专门做这类实验。

This article delivers on that promise, and stacks an additional layer on top:
**a side-by-side comparison of Evalith vs `promptfoo` vs `DeepEval` on the same
high-noise dataset and the same injected regression** — so the "bootstrap CI
matters" claim isn't just self-validating, it's verified against the two most
widely used OSS LLM eval tools today.

## 2. Goals

1. **Close the limitation from article 1** — produce experimental data where LLM
   noise is unmistakable, and show bootstrap CI both refusing to false-alarm on
   noise (experiment A) and accurately capturing a real regression (experiment B).
2. **Run the same experiment data through promptfoo and DeepEval**, and quantify
   how each tool handles the same noise floor and the same injected regression.
3. **Stay credible.** Three integrity rails:
   - All raw model outputs and tool configs land in the repo.
   - Each competitor tool is configured per its own official-recommended usage.
   - Article explicitly names dimensions where promptfoo / DeepEval beat Evalith.

## 3. Non-goals (explicitly excluded to prevent scope creep)

- No new statistical machinery in Evalith (BCa, paired bootstrap, FDR control).
  Those are reserved for a future article #3.
- No RAG-specific experiment.
- No creative-generation tasks (taglines, openers).
- No simultaneous English version. English release tracked as separate follow-up.
- No comparison against OpenAI Evals or Ragas (out-of-scope: OpenAI Evals is
  narrow; Ragas is RAG-specific).

## 4. Article structure

Total target length: **3,500-4,500 字** (中文).

**Working title:** 《续:LLM 当 judge 自己也在抖 —— bootstrap CI 抗噪与三个 OSS eval 工具的同台对照》

**Hook:** "上篇没说清的那一半。" Continuation framing — re-uses the existing
Zhihu readership; opens by quoting article 1's own honest limitation paragraph.

| § | Heading | Approx. 字数 | Role |
|---|---|---|---|
| 1 | 开篇:上篇留下的那一半 | ~300 | Re-quote article 1's limitation; promise of resolution |
| 2 | 重新设计实验:让 LLM 真的开始抖 | ~500 | Dataset choice (10 open-ended technical Q&A), temp=1.0, llm_judge rationale |
| 3 | 实验 A:噪声基线 — 同 config 重跑 | ~600 | Per-case single-sample fluctuation chart + bootstrap CIs all spanning 0 |
| 4 | 实验 B:注入隐式 prompt 偏差 | ~600 | Before/after table + which cases CI captured + honest about which it didn't |
| 5 | 同台对照:promptfoo / DeepEval 跑同一套 | ~800 | Three-tool side-by-side judgment table + false-alarm / capture-rate comparison |
| 6 | 这告诉我们什么 | ~400 | Two synthesized points: statistical significance is non-optional; LLM-as-judge is not a neutral observer |
| 7 | 局限 + 下一步 | ~300 | Same style as article 1; honest about paired/BCa/FDR not done |

## 5. Experiment design

### 5.1 Dataset

New file: `docs/blog/article2/qa.high-temp.yaml`

10 medium-depth technical open-ended questions, topic-mixed:

| # | case_id | Category |
|---|---|---|
| 1 | explain-rlhf | 概念解释 |
| 2 | explain-vector-db | 概念解释 |
| 3 | sql-injection-vulnerability | 安全识别 |
| 4 | k8s-configmap-vs-secret | 系统设计辨析 |
| 5 | asyncio-yield-deadlock | 代码问题诊断 |
| 6 | python-gil-tradeoffs | 概念权衡 |
| 7 | redis-cluster-failover | 系统行为预测 |
| 8 | tcp-congestion-control | 协议机制 |
| 9 | jwt-vs-session | 系统设计辨析 |
| 10 | transformer-attention | 概念解释 |

Each case has an `expected_concepts` list (2–4 core concepts) that the judge
prompt references when scoring.

### 5.2 Run configuration

| Parameter | Value | Rationale |
|---|---|---|
| `model` | `deepseek-chat` | Same provider as article 1; cheap; consistent baseline |
| `temperature` | `1.0` | Sweet spot — visible noise without hallucination collapse |
| `samples` | `5` per case | Matches article 1; statistically meaningful at this article's scope |
| `scorer` | `llm_judge` only (no `contains`) | Pure judge-driven scoring — exposes both layers of noise (model + judge) |
| `judge model` | `deepseek-chat`, `temperature=0` | Stabilizes judge so observed judge noise comes from boundary-case interpretation, not random sampling |
| Runs | 3: A1 (baseline), A2 (identical re-run = noise-floor control), B (regression-injected) | Same shape as article 1 |
| Total model calls | ~300 (10 × 5 × 3 × 2 [model + judge]) | ~$0.20 on DeepSeek pricing |

### 5.3 Regression injection (experiment B)

Same dataset as A. Only `prompt_template` changes.

- **Baseline (A1, A2):**
  ```
  {{input}}
  ```
- **Regression (B):**
  ```
  Assume the user is a senior engineer who knows the basics. Skip foundational
  explanations and focus on the non-obvious parts. Be concise: {{input}}
  ```

This is a "product-manager-tweaked-one-line" change — subtle but real. The
judge criterion "覆盖关键概念 + 解释清楚" will systematically penalize the
terse outputs, but not on every sample, creating exactly the noise-vs-signal
boundary that bootstrap CI is designed for.

### 5.4 Ground truth

Before running the experiment, write down (in the article) a hypothesis on
which of the 10 cases the regression *should* affect (likely 4–6 of the more
concept-heavy ones; arithmetic-style/short-answer ones may shrug it off).
Then judge each tool against this prediction.

This is documented up-front so the article cannot post-hoc rationalize.

## 6. Comparison framework (the §5 horizontal comparison)

### 6.1 Fairness invariants

The same `outputs.json` (raw model + judge outputs from the Evalith run) is
fed to all three tools' diff/comparison logic. **The only thing that varies is
each tool's statistical inference / report**.

- **Evalith:** `evalith diff a1.json b.json` (bootstrap CI path, default).
- **promptfoo:** `promptfoo eval` with its standard scoring/compare flow on
  the same outputs. Use its official-recommended config.
- **DeepEval:** `assert_test` + `compare_test_results` (or equivalent latest
  API). Use its official-recommended config.

### 6.2 Shared metrics

For each tool, on each comparison (A1 vs A2, A1 vs B):

| Metric | Definition |
|---|---|
| Per-case verdict | regressed / unchanged / improved (or each tool's nearest equivalent) |
| False-alarm rate | On A1 vs A2 (noise floor): cases flagged as regressed ÷ 10 |
| Capture rate | On A1 vs B: cases flagged as regressed ∩ ground-truth-affected ÷ \|ground-truth-affected\| |
| Capture precision | On A1 vs B: cases flagged as regressed ∩ ground-truth-affected ÷ cases flagged as regressed |

### 6.3 Three credibility rails

1. **Raw outputs committed.** All three runs' JSON outputs land in
   `docs/blog/article2/raw/` (~50KB; reviewable).
2. **Tool configs committed side-by-side.** `docs/blog/article2/configs/`
   contains the Evalith YAML, the promptfoo YAML, and the DeepEval Python
   harness. Anyone can audit whether they're configured fairly.
3. **Named-and-credited section.** The article includes an explicit paragraph
   acknowledging where promptfoo and DeepEval each outperform Evalith
   (distribution, UI, scorer breadth, ecosystem, etc.). No hidden agenda.

## 7. Repo deliverables (commits, not just article)

New files (single PR / commit batch):

```
docs/blog/2026-XX-XX-llm-judge-noise-bootstrap.zh.md     # article body
docs/blog/article2/
  qa.high-temp.yaml                                       # 10-case dataset
  eval.high-temp.yaml                                     # Evalith baseline config
  eval.high-temp.broken.yaml                              # Evalith regression-injected config
  configs/
    promptfoo.yaml                                        # promptfoo equivalent
    deepeval_compare.py                                   # DeepEval harness
  experiment.sh                                           # one-liner reproduction
  raw/
    a1.json                                               # frozen baseline run #1
    a2.json                                               # frozen baseline run #2
    b.json                                                # frozen regression run
```

`experiment.sh` runs the full pipeline end-to-end given `DEEPSEEK_API_KEY`,
emits the three markdown tables that go in §3, §4, §5 of the article.

**Evalith core gets no new features in this PR.** The article is delivered
through existing v0.4 capabilities. This is an honesty constraint — no
"feature sneaked in via blog post."

## 8. Risks and honest disclosures (planned to land in the article body)

1. **n=10 cases is a small sample.** Horizontal-comparison conclusions should
   not be over-generalized. State this in §5 prose.
2. **promptfoo / DeepEval may perform differently under configurations we did
   not try.** Cite their docs; invite issue/PR if a better config exists.
3. **Same judge model judging same model's outputs (DeepSeek judging DeepSeek)
   has affinity-bias risk.** Discuss in §6. Sanity-check spot: run one case's
   judge with GPT-4o-mini and note whether the verdict flips.
4. **Prompt-bias injection is artificial.** Real-world regressions can be more
   subtle. Frame this as a *lower bound* demonstration, not a worst case.

## 9. Success criteria

The article is "done" when:

1. All 7 sections are written, 中文 prose quality matches article 1.
2. All three tables (§3 noise floor, §4 regression, §5 cross-tool) are
   populated from **real experiment runs**, not synthetic.
3. `experiment.sh` runs clean end-to-end on a fresh checkout with
   `DEEPSEEK_API_KEY` set.
4. Raw outputs and all three tool configs are committed.
5. The "where promptfoo / DeepEval are better than Evalith" paragraph exists
   and is written in good faith.
6. Self-review passes (no placeholders, no contradictions, no scope creep).

## 10. Out-of-scope follow-ups (tracked for later)

- **Article 3:** BCa bootstrap, paired bootstrap, FDR / multiple-comparison
  correction — promised at end of this article.
- **English version** for dev.to / Show HN / Reddit.
- **GPT-4o-mini cross-judge sanity check** as a fuller experiment if §8.3
  finding is interesting enough to warrant a §5.5 sub-section.

---

## Appendix A: Open questions to resolve during implementation

These are intentionally left open here — they'll be decided in the
writing-plans phase or during execution, not at spec time:

- Exact wording of `expected_concepts` per case (will be drafted in
  implementation, reviewed before running experiment).
- Exact wording of the `llm_judge` criteria string (Chinese; will mirror
  article 1's pattern of "回答是否准确直接").
- Whether to render the per-case noise-floor fluctuation as a markdown table
  or an ASCII chart (decided in writing phase).
- Final article title (working title in §4 is a candidate, not final).
