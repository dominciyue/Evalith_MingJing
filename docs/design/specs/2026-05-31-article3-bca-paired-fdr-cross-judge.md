# Spec: 第三篇文章 + Evalith v0.5 — BCa / paired bootstrap / FDR + GPT-4o-mini cross-judge

**Date:** 2026-05-31
**Author:** PattyWoods (with Claude collaboration)
**Status:** Approved — moving to writing-plans
**Targets:** Zhihu 知乎 + 掘金(中文)。同时为 arXiv preprint 提供方法论核心内容。

---

## 1. Background and motivation

文章 2(`docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md`,发布于
<https://zhuanlan.zhihu.com/p/2044542154400322098>)在 §7 公开承诺了四件事:

1. BCa(bias-corrected and accelerated)bootstrap 加进 Evalith;
2. Paired bootstrap 加进 Evalith;
3. FDR(false discovery rate)多重比较校正加进 Evalith;
4. 用 GPT-4o-mini 当第三方 judge 做跨 judge sanity check。

第三篇文章兑现这个承诺,以 article 2 的 frozen raw 数据(`docs/blog/article2/raw/`)
为基线,对比四种统计方法在同一份数据上的 verdict 漂移,并用双路 cross-judge 实验
拆开「judge 噪声」和「model 噪声」对结论的相对贡献。

文章 3 同时是 Evalith v0.5 的 release 文档(单 PR / 单 release,backward-compatible),
也是 arXiv preprint 第 3-5 节(方法 + 实验)的核心材料。

## 2. Goals

1. **兑现 §7 公开承诺的全部 4 项工作。** 文章 1 / 2 建立了「公开承诺 → 兑现」
   的诚信链条,文章 3 必须继续这条链条。
2. **回答一个可证伪的问题:** 文章 2 报的 verdict(Evalith 2/10, promptfoo 1/10,
   DeepEval 5/10)在更严格的统计方法下变了吗?在第三方 judge 下变了吗?
3. **完整、可复现的实验链。** 所有新代码 + 新 raw outputs + 复现脚本都在
   `docs/blog/article3/` 目录里,与 article 2 同等水平的透明度。
4. **Evalith v0.5 release** — 三个新统计方法 + cross-judge helper 一次性 ship,
   percentile 仍为默认,新方法全部 opt-in。
5. **三件诚信保险:**
   - 所有 v0.5 新方法的 unit test 以 `scipy.stats` 为 ground truth(不自己 reinvent);
   - 文章 3 与 article 2 共用同一份 frozen raw outputs,任何 verdict 漂移可归因于方法变化;
   - §6 综合矩阵展示 per case × 6 种干预的完整结果,**不挑数据、不藏失败**。

## 3. Non-goals(明确剔除,避免范围蠕变)

- 不再做 promptfoo / DeepEval 横评(article 2 已做,重复无意义)。
- 不扩 dataset 到 20+ case(在 article 2 frozen 数据上做方法对比才纯净;dataset 扩展留给 arXiv paper)。
- 不实现 per-case `expected_concepts` 插值到 judge prompt(article 2 §3 发现这个 limitation,
  但与方法对照混在一起会污染实验;留给 v0.6 / 文章 4)。
- 不做英文版(继续中文优先;英文版独立任务)。
- 不接入 Ragas / OpenAI Evals(同 article 2,scope 不可比)。
- 不引入第四种 LLM judge(Claude 等)。GPT-4o-mini 是足够的 sanity check。

## 4. Article structure

**Working title:** 《续之续:把 BCa、paired bootstrap、FDR、第三方 judge 都挂上去,文章 2 的结论变了吗?》

**Hook:** "变了吗?变了什么?为什么?" —— 上篇 verdict 摆在那里,这篇用四种干预重新检验之。

**Length target:** 4500-5500 字中文 prose(不含表格);总非空白字符 ~10000-11000(参考 article 2 去 AI 化后的 12444)。

| § | 标题 | 字数 | Role |
|---|---|---|---|
| 1 | 那个承诺 + 一个问题 | ~400 | 回顾 article 2 的 2/1/5 verdict,引出"四件事都做了,结论变了吗"的核心问句 |
| 2 | BCa: 修偏与加速 | ~700 | 理论简介、jackknife 实现陷阱、在 frozen 数据上的 verdict diff |
| 3 | Paired bootstrap: 利用 case 内相关性降方差 | ~700 | 同上结构 |
| 4 | FDR: 同时检验 10 个 case 时 | ~700 | 同上结构;包含 p-value 生成的实现细节 |
| 5 | 换 judge: 一路只换,一路全换 | ~1200 | 方案 A(judge-only swap)+ 方案 B(全部重跑)双路对照,拆"judge variance vs model variance" |
| 6 | 谁改变了我们的判定 | ~700 | per case × 6 种干预的横向矩阵 + 综合解读 |
| 7 | 局限 + 第四篇方向 | ~300 | 诚实承认 |

## 5. Experiment design

### 5.1 Data source

**严格使用 article 2 的 frozen raw outputs:**
- `docs/blog/article2/raw/a1.json`、`a2.json`、`b.json` — Evalith runs(deepseek-chat 输出 + deepseek-chat judge)
- `docs/blog/article2/raw/promptfoo_a1.json`、`promptfoo_b.json` — promptfoo runs
- `docs/blog/article2/raw/deepeval_a1.json`、`deepeval_b.json` — DeepEval runs

§1 会显式声明:"下面所有 verdict 漂移都跑在同一份数据上,任何漂移可归因于方法或 judge 变化"。

### 5.2 Four statistical methods + two cross-judge tracks

| Intervention | 怎么算 | 在 Evalith 中的 surface |
|---|---|---|
| Percentile bootstrap(baseline,v0.4 已有) | 1000 resamples, 95% percentile CI on Δ | 默认行为,无需 flag |
| BCa bootstrap | Percentile + bias correction z₀(observed Δ 在 bootstrap 分布的分位数)+ acceleration a(jackknife 派生)| `--ci-method bca` |
| Paired bootstrap | Resample case 索引;对每个采样索引取 (before[i], after[i]) 配对;算 Δ[i],取 mean。Reduces variance when before/after 在同 case 上相关 | `--ci-method paired` |
| FDR (Benjamini-Hochberg) | Per case 算 bootstrap p-value(Pr(Δ ≥ 0)),N case 同时做检验时套 BH 调整 | `--multi-test bh` |
| Cross-judge A: judge-only swap | 从 article 2 a1.json/b.json 提 model outputs,inline 调 GPT-4o-mini judge(同一 criteria string)→ 新 pass_rate_samples → 重算 verdict | `docs/blog/article3/rejudge.py` (新一次性脚本) |
| Cross-judge B: model + judge full swap | 用 GPT-4o-mini 既出 model output(temperature=1.0,匹配 article 2)又评分(temperature=0,匹配 evalith llm_judge 默认),fresh A1/B 跑一遍 → 与路 A 对照拆 model vs judge 各自贡献 | `evalith run` with `model: gpt-4o-mini` + `judge: gpt-4o-mini` |

### 5.3 Output: §6 综合矩阵

中心物件(填空后是 §6 的核心表):

| case | percentile(v0.4) | + BCa | + paired | + FDR | + judge swap A | + judge swap B |
|---|---|---|---|---|---|---|
| explain-rlhf | unchanged | ? | ? | ? | ? | ? |
| explain-vector-db | unchanged | ? | ? | ? | ? | ? |
| sql-injection-vulnerability | regressed | ? | ? | ? | ? | ? |
| k8s-configmap-vs-secret | unchanged | ? | ? | ? | ? | ? |
| asyncio-yield-deadlock | unchanged | ? | ? | ? | ? | ? |
| python-gil-tradeoffs | unchanged | ? | ? | ? | ? | ? |
| redis-cluster-failover | regressed | ? | ? | ? | ? | ? |
| tcp-congestion-control | unchanged | ? | ? | ? | ? | ? |
| jwt-vs-session | unchanged | ? | ? | ? | ? | ? |
| transformer-attention | unchanged | ? | ? | ? | ? | ? |

§6 prose 是这张表的解读 —— 哪些 case 在哪些干预下翻盘、为什么。

### 5.4 Budget

- ~120 个 GPT-4o-mini 调用(judge-only + full swap 双路合计),~$0.30 OpenAI 额度
- Statistical methods 本身离线计算,零 API 成本
- 所有重跑都使用 frozen raw outputs 作 input,deepseek-chat 不会被再次调用(节省成本 + 保持可比性)

### 5.5 Pre-committed hypothesis

效法 article 2 §4,文章 §1 末尾会**事先**承诺以下 5 条预测(在数据出来之前 commit 到 git,与统计方法 / cross-judge 实验跑出来之前形成可审计的 pre-hoc 锚点):

1. **BCa 大概不会显著改变 verdict**(article 2 数据样本小、分布近 0/1 二元,BCa 的修偏在这个分布上贡献有限)。
2. **Paired bootstrap CI 会收窄**,但不足以让任何 unchanged 翻成 regressed —— 因为 article 2 的 unchanged case 几乎都是 mean=1.00 两边,Δ=0 没有方差可降。
3. **FDR 在 10 case 上会让 sql-injection / redis 这两个原本 regressed 的判定**之一翻成 unchanged(可能 sql-injection,因为其 CI 离 0 较近)。
4. **judge swap A**(单换 judge)会让 verdict 大幅变化,因为 GPT-4o-mini 不是 deepseek 亲属,对中文 prompt 的判定标准不同。
5. **judge swap B**(全换)与 A 的差异反映 model variance 的贡献占比。

预测可能全错,这正是 article 2 的传统:把假设 lock 在 git commit 里,事后用数据检验。

## 6. Evalith v0.5 engineering

### 6.1 API design

```python
# Existing (v0.4):
bootstrap_diff_ci(before, after, n_resamples=1000, alpha=0.05, seed=0) -> tuple[float, float]

# New (v0.5):
bootstrap_diff_ci(
    before, after, *,
    method: Literal["percentile", "bca", "paired"] = "percentile",
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]
```

对于 paired,要求 `len(before) == len(after)` 且每个 index 对应同一 case 的同一 sample。
对于 BCa,内部跑 jackknife(N replicates,each leaving one sample out)算 acceleration。

```python
# Existing (v0.4):
diff_runs(before: Run, after: Run, tol: float = 1e-9) -> DiffReport

# New (v0.5):
diff_runs(
    before, after,
    tol: float = 1e-9,
    *,
    ci_method: Literal["percentile", "bca", "paired"] = "percentile",
    multi_test_correction: Literal[None, "bh"] = None,
) -> DiffReport
```

### 6.2 CLI

```bash
# Default: percentile (backward compatible)
evalith diff a.json b.json

# Opt-in new methods:
evalith diff a.json b.json --ci-method bca
evalith diff a.json b.json --ci-method paired
evalith diff a.json b.json --multi-test bh
evalith diff a.json b.json --ci-method paired --multi-test bh
```

### 6.3 Backward compatibility

- `--samples N`(包括默认 `--samples 1`)的全部 v0.4 行为完全不动:default `--ci-method=percentile` 时 CLI 输出与 v0.4 byte-for-byte 一致(同一 seed)。
- `bootstrap_diff_ci(before, after)` 不带 method 参数等价于 `method="percentile"`。
- `diff_runs(before, after)` 不带新参数等价于 v0.4 行为(无 multi-test correction、无 CI method override)。
- v0.4 的所有 CLI flag(`--samples`、`--fail-under`、`--fail-on-regression`、`--format`、`--out`、`--store` 等)仍然有效。
- 所有 v0.4 的 frozen run JSONs(包括 article 2 的)能被 v0.5 直接读取,且默认参数下产出的 verdict 与 article 2 文章中的表对得上(零漂移)。
- v0.5 的所有 v0.4 测试必须通过,不修不破。

### 6.4 Testing

每个新方法的 unit test 以 **scipy.stats** 作 ground truth:

- BCa: 与 `scipy.stats.bootstrap(method="BCa")` 在相同 seed 下结果应在小误差(~0.05)内
- Paired: 与 `scipy.stats.bootstrap(paired=True)` 比对
- FDR: 与 `scipy.stats.false_discovery_control(method="bh")` 比对 p-value 排序结果

`scipy` 加入 `dev` extras(不进 core dependency,避免老用户被迫装)。

### 6.5 Release

v0.4 → **v0.5**(单 release,所有方法 + helper 一次进)。
PyPI 同步发布,README 加 "What's new in v0.5" 小节。
GitHub Action `eval-example.yml` 不动(继续 default percentile)。

## 7. Repo deliverables

新增 / 修改:

```
src/evalith/diff.py                # 加 BCa + paired branch + multi_test_correction
src/evalith/cli.py                 # 加 --ci-method / --multi-test
tests/test_diff.py                 # 加新方法的 scipy-ground-truth 测试
pyproject.toml                     # version → 0.5.0; dev extras 加 scipy
README.md / README.zh-CN.md        # What's new in v0.5
docs/blog/2026-XX-XX-statistical-deep-dive.zh.md   # 文章正文
docs/blog/article3/
  rejudge.py                       # judge-only swap helper
  configs/
    eval.high-temp.openai.yaml     # 给 swap B 用,GPT-4o-mini 做 model + judge
  raw/
    rejudge_a1.json                # judge-only swap on a1 outputs
    rejudge_b.json                 # judge-only swap on b outputs
    openai_a1.json                 # swap B: GPT-4o-mini full re-run baseline
    openai_b.json                  # swap B: regression
  compare.py                       # 计算 §6 综合矩阵
  experiment.sh                    # 一行复现
```

## 8. Risks and honest disclosures(写进文章 §7)

1. **n=10 cases 是小样本** —— BCa 在小样本下的 acceleration 估计本身有方差;FDR 在 10 个 hypothesis 上的统计 power 有限。
2. **预测可能全错** —— §5.5 列出的 5 条预测如果全部对了,文章会说"运气好";如果全错,文章会说"我对统计方法的直觉也不可靠,这是 article 2 教训的延伸"。
3. **GPT-4o-mini 仍是 LLM,不是 ground truth judge** —— cross-judge 实验只能说 "judge 变了 verdict 变了",不能说 "GPT-4o-mini 更对"。这是 sanity check,不是裁判。
4. **frozen raw 数据有特定时刻的 deepseek 行为印记** —— 三周后跑同 prompt 可能不一样,所以 article 3 的所有 verdict 是 "deepseek 在那一刻 + 那个判分体系" 下的判定。
5. **scipy 加进 dev extras** —— 一些非常严格的 minimal-dep 部署方案可能不喜欢,但只是 dev,不影响 runtime。

## 9. Success criteria

文章 + 代码 + release 一起算"done"当且仅当:

1. 文章 3 所有 7 个 section 都从真实实验数据写出,不含 placeholder。
2. §6 综合矩阵 6 列每一格都有真实 verdict(不能空、不能编)。
3. `evalith` v0.5.0 在 PyPI 可装,`pip install evalith==0.5.0` 后 `evalith diff a.json b.json --ci-method bca` 能跑。
4. `docs/blog/article3/experiment.sh` 在干净环境上一行复现端到端。
5. 所有 v0.4 的 frozen runs 能被 v0.5 加载(`evalith diff` 在 article 2 的 a1.json + b.json 上仍然能输出 percentile CI,数字与 article 2 的表对得上)。
6. v0.4 的 63 个测试 + 新增 v0.5 测试全部通过(零 warning)。
7. §5.5 的 5 条预测的对照(对 / 错)在文章里**直接陈述**,不藏。

## 10. Out-of-scope follow-ups(显式记下,留给后续)

- **arXiv preprint:** 文章 3 + article 2 + article 1 合并 → ~10 页 paper,加 related work + abstract + LaTeX。1-2 周。投 NeurIPS 2026 LLM Evals Workshop 或 ICLR 2027 Workshop。
- **文章 4:** Per-case criteria 插值到 llm_judge prompt + adaptive sampling + larger dataset(50-100 case 跨领域)。约 1 个月。
- **英文版:** Article 1 + 2 + 3 的英文合集 → dev.to / Hacker News Show HN。
- **Adaptive sampling:** 动态决定每 case 跑多少次,直到 CI 收敛或超时。
- **Per-case criteria 插值:** llm_judge 把 `expected_concepts` 注入 judge prompt(article 2 §3 发现的 limitation)。
- **更广的 cross-judge:** Claude / Qwen / Mistral 当 judge,Ragas-style judge 集成。

---

## Appendix A: Open questions to resolve during implementation

- BCa 的 jackknife 在 paired 数据上要怎么算(每次 leave-one-pair-out vs leave-one-sample-out)?implementation 阶段定。
- FDR 的 p-value 生成具体公式:`min(p_left, p_right) * 2` 还是 `min(p_left, p_right)`?二者在双边检验下有细微差,implementation 阶段对照 scipy。
- Cross-judge swap A 的 `rejudge.py` 是 inline script 还是要变成 `evalith rejudge` CLI 命令?偏向 inline script(不让 article-specific 逻辑污染 core CLI),implementation 阶段确认。
- §5.5 的预测段在哪里 commit:文章 §1 末尾还是单独 commit?为追求 audit trail,**单独的 pre-experiment commit** 是更干净的(同 article 2 §4 的做法)。
