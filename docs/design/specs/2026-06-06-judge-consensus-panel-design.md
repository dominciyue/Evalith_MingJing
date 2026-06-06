# Judge Consensus Panel (v0.7) — Design Spec

Date: 2026-06-06
Status: approved-pending-review

## 背景与动机

文章 4 的实验结论:judge 之间的分歧有领域结构(code 领域 qwen 把 deepseek 输出的
0.80 判成 0.00;safety/概念解释是共识区),且 judge 换人改变分数 level 但基本不改
回归 direction。这个发现目前只活在博客里,没有反哺到工具。

本特性把它产品化:一次 eval 可以挂 N 个 judge,主 judge 负责评分与 gate(语义不
变),其余 judge 纯诊断——报告 per-case 分歧、pairwise Cohen's kappa、领域分组
一致性,低共识 case 自动标 ⚠"判分不可靠"。这是同类 OSS 工具(promptfoo /
deepeval / openai-evals)都没有的差异化能力。

顺带修一个现状 bug:CLI 跑 eval 时 `judge_provider` 从未传入,llm_judge 实际用
被测模型自判自(`cli.py` → `run_eval(cfg, get_provider(cfg.model))`)。
(实现注:为保持向后兼容,缺省行为不变——`judge_model` 未配置时仍回退到被测模型;
该问题的修复方式是"可经 `judge_model` 配置显式指定",而非改变缺省。)

## 已确认的设计决策(与用户对齐)

1. **聚合语义**:主 judge 评分 + 其余诊断。主 judge 分数进 `scores` /
   `pass_rate_samples`,gate 行为与现在完全一致;panel judge 不参与 pass/fail。
2. **CI 行为**:低共识仅报告标记,不影响 exit code。后续版本再考虑
   `--consensus-fail` 之类的 strict 模式。
3. **实现路线**:engine 内置 panel(每个 trial 的输出当场被所有 judge 各判一次),
   而非事后重判——因为 `CaseResult` 多 sample 时只存代表性 output,事后重判会丢
   (N-1)/N 的数据。

## 配置

`llm_judge` scorer 的 `params` 新增两个字段:

```yaml
scorers:
  - type: llm_judge
    params:
      criteria: 回答应当准确、覆盖核心概念
      language: zh
      judge_model: deepseek-chat          # 主 judge(新增;缺省=被测模型,向后兼容)
      panel: [qwen-plus, glm-4.5-air]     # 诊断 judge 列表(新增;缺省空=行为不变)
```

- `judge_model` / `panel` 元素均走现有 `get_provider()` 解析(支持国产 alias)。
- `panel` 中如果含 `judge_model` 本身,自动去重,不重复调用计费。
- 不加顶层 `judges:` 配置段,避免两套入口。

## 数据模型(models.py)

`CaseResult` 新增三个字段,均有缺省值,旧 run JSON 可正常加载:

```python
panel_samples: dict[str, list[float]] = Field(default_factory=dict)
# {judge名: [每trial通过率]};主 judge 不在其中(它在 pass_rate_samples)
# panel judge 某 trial 调用失败 → 该位置记 -1.0(哨兵),统计时跳过并在报告标注

panel_details: dict[str, str] = Field(default_factory=dict)
# {judge名: 代表性 reason}——取该 judge 与主 judge 分歧最大的那个 trial 的 reason;
# 无分歧时取第一个 trial。让 ⚠ 标记可解释、可行动

judge_cost_usd: float = 0.0
judge_tokens: int = 0
# 主 judge + panel 全部判分调用的成本/token 合计(现状是 0,judge 成本被丢弃)
```

`Run` 新增属性 `total_judge_cost_usd`(judge 成本合计);既有
`total_cost_usd` 含义不变(仅生成成本),避免改变旧字段语义。报告中
"生成成本"与"评分成本"分列,合计行 = 两者之和。

## Engine(engine.py)

- `_eval_once` 内,模型输出产生后:主 judge 照常打分进 `scores`;每个 panel judge
  对同一输出各打一次分。
- **相同输出缓存**:同一 case 内,`(judge名, output文本)` 为 key 的 dict 缓存,
  trial 输出完全相同的只判一次复用(低温模型常见,纯省钱)。
- **韧性**:单个 panel judge 调用异常只在该 judge 该 trial 记缺失(-1.0),不影响
  主 judge、其他 panel、其他 case(延续 v0.3 per-case 韧性原则)。
- 多 sample / adaptive 聚合时,把各 trial 的 panel 分数按 judge 收拢成
  `panel_samples`;judge 成本累加进 `judge_cost_usd`。
- 不做 panel 内嵌套并发:case 级 ThreadPoolExecutor 已提供吞吐。

## 共识统计(新模块 consensus.py)

把 `docs/blog/article4/multi_compare.py` 中已验证的统计逻辑搬进包内:

```python
def case_spread(case: CaseResult) -> float
    # 主 judge + 各 panel judge 的 per-case 均值的 max - min;缺失 judge 跳过

def cohen_kappa(a: list[int], b: list[int]) -> float
    # per-(case,trial) 二值标签的 Cohen's kappa
    # pe = pa1*pb1 + (1-pa1)*(1-pb1);pe >= 1.0 时返回 1.0(退化保护)

def pairwise_kappa(run: Run) -> dict[tuple[str, str], float]
    # 所有 judge 两两(含主 judge),标签对齐到 (case, trial) 粒度;
    # 任一方缺失的 trial 成对剔除

def domain_agreement(run: Run) -> dict[str, dict]
    # case.metadata["domain"] 存在时按领域分组,给出每领域的均值 spread 与 kappa;
    # 无 domain 标签时整体作一组
```

低共识判定:`case_spread >= threshold`,threshold 缺省 0.5,可经
`llm_judge.params.consensus_threshold` 覆盖。

kappa 退化(某 judge 近常量标签时 κ 塌向 0)在报告中附一句固定说明,
和文章 4 的处理一致。

## 报告与 CLI 输出

- `run` 命令结束摘要追加一行(仅当 panel 非空):
  `panel: 3 judges, 4/30 low-consensus cases, min pairwise κ=0.31`
- md/html 报告(report.py)新增 "Judge Consensus" 节,仅当 panel 非空时渲染:
  - 总表:每个 judge 的 overall 均值
  - pairwise kappa 矩阵(附退化说明)
  - 领域分组表(有 domain 标签时)
  - 低共识 case 列表:case_id、各 judge 分数、各 judge 的 reason
    (`panel_details`)、⚠ 标记
  - 评分成本小计(与生成成本分列)
- panel judge 有调用失败时,表格相应单元格记 `n/a` 并加脚注。

## 测试策略(TDD)

- **FakeProvider 注入分歧**:测试用 fake judge providers 返回预设分数,断言
  `panel_samples` / spread / ⚠ 标记 / 主 judge gate 不受影响。
- **kappa 回归测试**:用文章 4 raw 数据中已人工核验过的数字
  (如 ds×qw、ds×glm 的 κ)做 golden 测试,防统计逻辑搬运走样。
- **韧性测试**:panel judge 抛异常 → 该 judge 记缺失,主 judge 分数完整,
  exit code 不变。
- **兼容测试**:无 `panel` 配置时,run JSON 与 v0.6 字节级语义一致
  (新字段为缺省值);v0.6 的旧 run JSON 能被 `diff` / `report` 正常加载。
- **缓存测试**:同 case 两个 trial 输出相同 → 每个 judge 只被调用一次。
- **去重测试**:panel 含 judge_model → 实际只调用一次。

## 范围外(明确不做)

- panel 分数参与 gate(mean/vote 集成)——留待用户反馈
- `--consensus-fail` strict 模式
- trial 全量输出落盘
- panel 内嵌套并发
- 事后 `evalith panel` 重判命令

## 验收标准

1. 旧配置(无 panel)行为与 v0.6 完全一致,全部既有测试通过。
2. 配 `panel: [qwen-plus, glm-4.5-air]` 跑文章 4 的 30-case 数据集,报告能复现
   "code 领域低共识、safety/概念解释高共识"的领域结构。
3. judge 成本在报告中可见,与生成成本分列。
4. 低共识 case 在 md/html 报告中带各 judge 分数与 reason。
