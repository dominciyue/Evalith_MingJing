# 明镜 / Evalith

[English](README.md) | **中文**

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

## 状态

v0.2 —— 单轮 prompt 评测、基于文件的运行存储、运行间 diff、CI 设卡
(`--fail-under`、`--fail-on-regression`、GitHub Action)、Markdown/HTML 报告、
并发、成本/token/延迟统计,以及带中文评审的国产模型别名。团队/云端能力在路线图上。
欢迎提 Issue 和 PR。

## 许可证

[Apache-2.0](LICENSE)。Copyright © 2026 Evalith(明镜) Authors。
