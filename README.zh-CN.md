# 明镜 / Evalith

[English](README.md) | **中文**

[![PyPI](https://img.shields.io/pypi/v/evalith.svg)](https://pypi.org/project/evalith/)
[![Python](https://img.shields.io/pypi/pyversions/evalith.svg)](https://pypi.org/project/evalith/)
[![CI](https://github.com/dominciyue/Evalith_MingJing/actions/workflows/ci.yml/badge.svg)](https://github.com/dominciyue/Evalith_MingJing/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/evalith.svg)](https://github.com/dominciyue/Evalith_MingJing/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/evalith)](https://pepy.tech/project/evalith)

> 在用户发现之前,先抓住 AI 的质量回退(regression)。

一个**中立、本地优先**的 **AI 回归测试**工具。定义一个测试集,针对任意模型
(DeepSeek / Qwen / OpenAI / Claude / …)跑一遍,为每条用例打分,
**对比两次运行,精确看出哪里变好、哪里变差**,再**在 CI 里设卡**——让 prompt
或模型的改动无法悄悄搞坏你的产品。

## 为什么是明镜

- **中立 & 开源。** 评测工具决定了"哪个模型更好",所以它不该被某个模型厂商
  拥有。明镜不绑定任何厂商,且完全开源(Apache-2.0)。
- **本地优先。** 核心流程完全在本机运行——无需账号、无需上传、无需联网,
  你的 prompt 和测试数据始终在自己手里。
- **国产模型一等公民。** DeepSeek、Qwen 与海外模型都是一等别名(`evalith models`);
  内置中文 `llm_judge`。
- **看回归,不看感觉。** `diff` 与 `--fail-on-regression` 会告诉你:改了 prompt、
  换了模型或升了版本后,哪些用例变好、变差或直接坏了。

## 安装

需要 Python ≥ 3.10。

```bash
pip install evalith              # 核心依赖:pydantic、pyyaml、typer
pip install "evalith[litellm]"   # 可选:接入真实模型(DeepSeek/Qwen/OpenAI/Claude/...)
```

或从源码安装:`git clone https://github.com/dominciyue/Evalith_MingJing` 后 `pip install -e ".[litellm]"`。

## 快速开始(离线,无需 API key)

```bash
# 1. 跑示例评测 —— 使用离线 `echo` 模型,2/2 通过
evalith run examples/eval.yaml

# 2. 在 examples/eval.yaml 里改 prompt/模型,再跑一次
evalith run examples/eval.yaml

# 3. 列出运行记录,对比最新两次以发现回归
evalith list
evalith diff <较旧的_RUN_ID> <较新的_RUN_ID>
```

## 在 CI 里拦住回归

质量下降时让构建失败——两种方式:

```bash
# 绝对门槛:通过率低于 90% 就失败(无需基线)
evalith run examples/eval.yaml --fail-under 0.9

# 相对门槛:相比基线运行,只要有用例回退就失败
evalith diff <基线_RUN_ID> <新的_RUN_ID> --fail-on-regression
```

两者失败时都以非零码退出,CI 会拦下该 PR。本仓库自带一个**复合 GitHub Action**——
把下面这段放进 `.github/workflows/eval.yml`:

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

(可用的离线示例见 `.github/workflows/eval-example.yml`。)

**相对基线拦回归。** `diff` 既接受 run ID,也接受 `.json` 文件路径,所以 CI 无需共享状态——先生成一份基线并提交,之后每个 PR 用新跑的结果跟它比:

```bash
evalith run examples/eval.yaml --out baseline.json   # 生成一次基线,提交 baseline.json
# 然后在 CI 的 PR 上:
evalith run examples/eval.yaml --out current.json
evalith diff baseline.json current.json --fail-on-regression
```

## 抗 LLM 噪声: 多次采样

LLM 同一 prompt 跑两次结果可能不同(即便 temperature=0 有些 provider 也会抖)。
为避免把随机噪声判成回退,可以让每条用例跑多次,Evalith 会对 Δ 做 bootstrap
95% 置信区间:

```bash
evalith run examples/eval.yaml --samples 5 --out current.json
evalith diff baseline.json current.json --fail-on-regression
# -> 只有当 (after - before) 的 95% 置信区间整体 < 0,才判为 regressed
```

单样本运行(`--samples 1`,默认)行为与之前完全一致。一旦启用采样,diff 报告会
多出一列 `Δ 95% CI`。

## 可分享的报告

把一次 run 或一次 diff 导出为 Markdown(贴到 PR)或自包含的 HTML 页面:

```bash
evalith report <RUN_ID> --format md                        # Markdown 输出到 stdout
evalith report <RUN_ID> --format html --output report.html # 独立 HTML 文件
evalith diff <A> <B> --format md --output diff.md           # diff 导出为 Markdown
```

报告包含通过率、平均分,以及——使用真实模型时——**成本、token 数、延迟**。

## 接入真实模型(国产一等公民)

```bash
evalith models          # 列出一等别名 + 各自需要的环境变量
export DEEPSEEK_API_KEY=sk-...
evalith run examples/eval.deepseek.yaml --concurrency 3
# -> Run <id> saved to .evalith/runs/<id>.json — 6/6 checks passed
```

`model:` 可填别名(`deepseek-chat`、`deepseek-reasoner`、`qwen-max`、`qwen-plus`)
或任意 LiteLLM id(`gpt-4o-mini`、`claude-3-5-sonnet` …),再设置对应厂商的
API key。`llm_judge` 评分器可用 `params: {language: zh}` 进行中文评审。

## 规模化

- `--concurrency N` 并行执行各用例(provider 调用是 I/O 密集型),也可在配置里写
  `concurrency:`。结果顺序始终保持不变。
- 数据集支持 **YAML、JSON、CSV、JSONL**(见 `examples/qa.jsonl`)。

## 评分器(Scorers)

| 类型 | 通过条件 |
|------|----------|
| `exact_match` | 输出等于用例的 `expected` |
| `contains`    | 输出包含 `params.text`(或用例的 `expected`) |
| `regex`       | 输出匹配 `params.pattern` |
| `llm_judge`   | 由 LLM 依据 `params.criteria` 打分(`params.language: en\|zh`) |

## 工作原理

`run` 针对某个模型执行配置,并把一次 **Run**(每条用例的输出、分数、token、成本、
延迟的 JSON 快照)保存到 `.evalith/runs/`。`diff` 逐条比较两次保存的 Run,把每条
用例标记为**改进 / 回退 / 不变 / 新增 / 移除**。

## v0.7 新功能

- **多 judge 共识面板** —— 一次 eval 挂多个 judge:per-case 分歧、pairwise Cohen's κ、分领域一致性、⚠ 低共识标记。主 judge 照常 gate,panel 只诊断不拦截 CI。

## v0.6 新功能

- **`llm_judge` 支持 per-case `expected_concepts`。** 每个 dataset case 可声明 `expected_concepts: [...]`,judge prompt 自动把它当作"必须覆盖的概念清单"插入。补上 article 2/3 提到的 judge 没有 per-case checklist 的局限。完全向后兼容:不写这个字段的 case 与 v0.5 行为一致。
- **Adaptive sampling(动态采样)。** `evalith run --adaptive --min-samples 2 --max-samples 10 --ci-tolerance 0.2` 每个 case 跑到 bootstrap CI 宽度小于阈值即停(或到 max_samples)。稳定 case 早停,省 API 钱;噪声 case 仍跑满。

## v0.5 新功能

- **`--ci-method bca`** —— Δ 上的 BCa(偏置纠正加速)bootstrap。仅依赖 Python 标准库;在 bootstrap 分布偏态时比 percentile 更准。
- **`--ci-method paired`** —— paired bootstrap。当 before/after 通过 case 维度有相关性时,CI 宽度收窄,降假阳率。
- **`--multi-test bh`** —— 跨 case 的 Benjamini-Hochberg FDR 控制。case 数多时,纯 percentile 容易过度报告回退;BH 压制家族级假阳率。
- **scipy** 进入 dev 依赖(用作单元测试 ground truth),不进 runtime —— 生产部署仍然零额外依赖。

v0.5 的全部新增都是 opt-in。v0.4 默认行为字节级保留。

## 状态

v0.4 —— 单轮 prompt 评测、基于文件的运行存储、带逐用例输出对比**和 Δ 上的 bootstrap
95% 置信区间(`--samples N`,让 LLM 噪声不再被误判为回退)**的运行间 diff、CI 设卡
(`--fail-under`、`--fail-on-regression`、文件基线、GitHub Action)、Markdown/HTML 报告、
带逐用例容错的并发、成本/token/延迟统计,以及带中文评审的国产模型别名。团队/云端能力在路线图上。
欢迎提 Issue 和 PR。

## 深入阅读

- **技术深度文(中文):** [AI 回归测试需要统计显著性: 用 bootstrap CI 抗 LLM 噪声](https://zhuanlan.zhihu.com/p/2043351926964848178) —— 为什么点对点比较的 eval CI gate 在统计上就是错的、Evalith `--samples N` 背后的数学,以及一组可复现的 DeepSeek A/B 实验。
- **续篇(中文):** [续:LLM 当 judge 自己也在抖](https://zhuanlan.zhihu.com/p/2044542154400322098) —— bootstrap CI 抗噪与三个 OSS eval 工具的同台对照。源文件 + 原始数据见 [`docs/blog/article2/`](docs/blog/article2/)。
- **第三篇(中文):** [续之续:更严的统计方法 + 第三方 judge,article 2 的结论变了吗?](https://zhuanlan.zhihu.com/p/2044820946721231928) —— 把 BCa、paired bootstrap、BH FDR 加进 Evalith,在 article 2 同一份 frozen raw 数据上跑双路 qwen-plus cross-judge 实验。v0.5 release。源文件 + 原始数据见 [`docs/blog/article3/`](docs/blog/article3/)。
- **设计文档与 TDD 计划:** [`docs/`](docs/) —— v0.1 spec、v0.1/v0.2 任务级 plan,以及博客原文。

## 许可证

[Apache-2.0](LICENSE)。Copyright © 2026 Evalith(明镜) Authors。
