# 明镜 / Evalith

[English](README.md) | **中文**

> 在用户发现之前,先抓住 AI 的质量回退(regression)。

一个**中立、本地优先**的 **AI 回归测试**工具。定义一个测试集,针对任意模型
(DeepSeek / Qwen / OpenAI / Claude / …)跑一遍,为每条用例打分,并
**对比两次运行,精确看出哪里变好、哪里变差**——而不是凭感觉发版。

## 为什么是明镜

- **中立 & 开源。** 评测工具决定了"哪个模型更好",所以它不该被某个模型厂商
  拥有。明镜不绑定任何厂商,且完全开源(Apache-2.0)。
- **本地优先。** 核心流程完全在你本机运行——无需账号、无需上传、无需联网,
  你的 prompt 和测试数据始终留在自己手里。
- **国产模型一等公民。** DeepSeek、Qwen、豆包、GLM 以及海外模型,都只是一个
  `model:` 字符串的距离(通过 [LiteLLM](https://docs.litellm.ai/docs/providers))。
- **看回归,不看感觉。** 当你改了 prompt、换了模型或升了版本,`diff` 会告诉你
  哪些用例变好了、变差了、或直接坏了。

## 安装

需要 Python ≥ 3.10。

```bash
pip install -e .            # 核心依赖:pydantic、pyyaml、typer(无需联网)
pip install -e ".[litellm]" # 可选:接入真实模型(DeepSeek/Qwen/OpenAI/Claude/...)
```

## 快速开始(离线,无需 API key)

```bash
# 1. 跑示例评测 —— 使用离线 `echo` 模型,2/2 通过
mingjing run examples/eval.yaml

# 2. 在 examples/eval.yaml 里改 prompt/模型,再跑一次
mingjing run examples/eval.yaml

# 3. 列出运行记录,对比最新两次以发现回归
mingjing list
mingjing diff <较旧的_RUN_ID> <较新的_RUN_ID>
```

## 接入真实模型(DeepSeek 示例)

`examples/eval.deepseek.yaml` 用 `deepseek/deepseek-chat` 评测几道事实题,每条
用例同时用 `contains` 校验和一次 `llm_judge` 打分:

```bash
pip install -e ".[litellm]"
export DEEPSEEK_API_KEY=sk-...
mingjing run examples/eval.deepseek.yaml
# -> Run <id> saved to .mingjing/runs/<id>.json — 6/6 checks passed
```

把 `model:` 换成任意 LiteLLM 支持的模型——`qwen/qwen-max`、`gpt-4o-mini`、
`claude-3-5-sonnet` 等——并设置对应厂商的 API key 即可。

## 评分器(Scorers)

| 类型 | 通过条件 |
|------|----------|
| `exact_match` | 输出等于用例的 `expected` |
| `contains`    | 输出包含 `params.text`(或用例的 `expected`) |
| `regex`       | 输出匹配 `params.pattern` |
| `llm_judge`   | 由一个 LLM 依据 `params.criteria` 给输出打分 |

## 工作原理

`run` 针对某个模型执行配置,并把一次 **Run**(每条用例的输出与分数的 JSON 快照)
保存到 `.mingjing/runs/`。`diff` 逐条比较两次保存的 Run,把每条用例标记为
**改进 / 回退 / 不变 / 新增 / 移除**,让 prompt 或模型的改动无法悄悄搞坏东西。

## 状态

v0.1 —— 单轮 prompt 评测、基于文件的运行存储、运行间 diff,以及面向国产 + 海外
模型的 provider 层。团队/云端能力在路线图上。欢迎提 Issue 和 PR。

## 许可证

[Apache-2.0](LICENSE)。Copyright © 2026 Evalith(明镜) Authors。
