# 续:LLM 当 judge 自己也在抖 —— bootstrap CI 抗噪与三个 OSS eval 工具的同台对照

> 上一篇 [《AI 回归测试需要统计显著性》](https://zhuanlan.zhihu.com/p/2043351926964848178)在结尾承认了一件事:实验用的 temperature=0 + 简单事实题让 DeepSeek 太确定,bootstrap 的"噪声免疫"优势没有真正显现。这一篇,我们把那张牌翻开。

---

## 一、上篇没说清的那一半

<TODO §1 prose — Task 13>

## 二、重新设计实验:让 LLM 真的开始抖

<TODO §2 prose — Task 13>

## 三、实验 A:噪声基线 —— 同 config 重跑

用第二节的 dataset + temperature=1.0 + 每 case 5 次采样,跑两遍**完全相同**的 config,得到 A1 和 A2。没有改任何 prompt,没有换模型,没有动 judge 标准。预期:两次跑出来的 mean 应该高度接近,bootstrap CI 应该全部跨过 0,假阳率为 0。

| case | A1 mean | A2 mean | A1 单样本极差 | bootstrap CI on Δ | status |
|---|---|---|---|---|---|
| `asyncio-yield-deadlock` | 1.00 | 1.00 | [1.0, 1.0] | [+0.00, +0.00] | unchanged |
| `explain-rlhf` | 1.00 | 1.00 | [1.0, 1.0] | [+0.00, +0.00] | unchanged |
| `explain-vector-db` | 1.00 | 1.00 | [1.0, 1.0] | [+0.00, +0.00] | unchanged |
| `jwt-vs-session` | 1.00 | 1.00 | [1.0, 1.0] | [+0.00, +0.00] | unchanged |
| `k8s-configmap-vs-secret` | 1.00 | 1.00 | [1.0, 1.0] | [+0.00, +0.00] | unchanged |
| `python-gil-tradeoffs` | 1.00 | 1.00 | [1.0, 1.0] | [+0.00, +0.00] | unchanged |
| `redis-cluster-failover` | 0.80 | 0.60 | [0.0, 1.0] | [-0.80, +0.40] | unchanged |
| `sql-injection-vulnerability` | 1.00 | 1.00 | [1.0, 1.0] | [+0.00, +0.00] | unchanged |
| `tcp-congestion-control` | 1.00 | 1.00 | [1.0, 1.0] | [+0.00, +0.00] | unchanged |
| `transformer-attention` | 1.00 | 1.00 | [1.0, 1.0] | [+0.00, +0.00] | unchanged |

假阳率:0/10。

9 个 case 的 bootstrap CI 全部收缩到 [+0.00, +0.00],直接判 unchanged。这不是奇迹,是模型特性:对这类有明确知识边界的技术问答,DeepSeek 在 temperature=1.0 下依然高度稳定——5 次采样打出 5 次 1.0,采样方差本身就是 0,bootstrap 再怎么重采也出不了区间宽度。

真正有意思的只有最后一行:`redis-cluster-failover`。

在 A1 里,5 个 sample 里有 1 个拿到 0(其余 4 个均为 1.0),mean=0.80,单样本极差是 [0.0, 1.0]——同一套问题同一个模型,一次打满分,一次完全挂掉。在 A2 里,2 个 sample 挂掉、3 个通过,mean 降到 0.60。

如果你在 CI 里用单样本点对点比较,这条记录会被判成 **"0.80 → 0.60,回退 25%,阻断 PR"**。但那个 0.80 本身就不稳定——下次重跑它可能变 1.0,也可能变 0.6。判断依据是一个掷硬币的结果,而不是模型能力的变化。

bootstrap CI 给出 [-0.80, +0.40],跨过 0,判 unchanged。这才是正确的裁决:这两次跑的差异完全处于采样噪声范围内,没有任何证据表明 config 本身发生了变化。

为什么偏偏是 `redis-cluster-failover` 不稳定?Redis Cluster 的 failover 流程涉及多个合理的描述路径——节点探活的时序、epoch 更新、slot 迁移确认——不同采样路径给出的回答侧重不同,LLM judge 在"概念覆盖完整性"上的判定本身也有边界模糊区,5 次里出现 1-2 次"刚好漏掉 judge 看重的那个步骤"并不罕见。这不是模型变差,这是 judge 评分的天然噪声。

这一节验证的事情很简单:上篇没展示出来的 LLM 抖动,在 temperature=1.0 + 开放性技术题的条件下真实存在,而且一旦存在,单样本比较就会产生假阳。bootstrap CI 在这里做的事,就是把"采样引起的均值漂移"和"config 引起的真实回退"区分开来。

## 四、实验 B:注入隐式 prompt 偏差

### 在看到数据之前,我先把假设写在这里

把 baseline prompt 加一句 `Assume the user is a senior engineer who knows the basics. Skip foundational explanations and focus on the non-obvious parts. Be concise:`,我**事先**预测哪些 case 会被这条隐式偏差打中:

**应该被命中(judge 因"概念缺失"扣分):**
- `explain-rlhf` — "SFT → reward model → PPO" 这条主线本身就属于"基础",会被跳过
- `explain-vector-db` — ANN / embedding / 索引结构是基础概念
- `python-gil-tradeoffs` — 解释 GIL 本身是基础,会被简化掉
- `tcp-congestion-control` — 慢启动 vs 拥塞避免的对比需要解释,容易被压缩
- `transformer-attention` — Q/K/V 的定义就是"基础",最容易丢

**边界(可能掉可能不掉):**
- `k8s-configmap-vs-secret` — 区别本身不算基础,但完整说清要点篇幅会缩
- `jwt-vs-session` — 同上

**应该不被命中(因为"答案"就在非基础部分):**
- `sql-injection-vulnerability` — "改成参数化查询" 这个修法不是基础
- `asyncio-yield-deadlock` — 诊断本身就是答案
- `redis-cluster-failover` — failover 流程不是"基础概念"

**预测的命中数量:5(强预期) + 2(边界)= 5-7 / 10。**

(下面的数据表来自实际跑出来的实验,任何一条都没有事后修正。所有 raw outputs 在 `docs/blog/article2/raw/`,一行命令可复现。)

<TODO §4 table + post-data prose — Task 8>

## 五、同台对照:promptfoo / DeepEval 跑同一套

<TODO §5 prose + cross-tool table — Task 11/12>

## 六、这告诉我们什么

<TODO §6 synthesis — Task 14>

## 七、局限和下一步

<TODO §7 limitations + article 3 promise — Task 14>

---

如果你也在做 LLM eval 或 AI CI 集成,欢迎到 **[github.com/dominciyue/Evalith_MingJing](https://github.com/dominciyue/Evalith_MingJing)** 提 issue / PR。

```bash
pip install evalith
```

即装即用,Apache-2.0。
