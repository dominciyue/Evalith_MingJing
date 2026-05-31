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

### 真实数据

| case | A1 mean | B mean | bootstrap CI on Δ | status |
|---|---|---|---|---|
| `asyncio-yield-deadlock` | 1.00 | 0.60 | [-0.80, +0.00] | unchanged |
| `explain-rlhf` | 1.00 | 1.00 | [+0.00, +0.00] | unchanged |
| `explain-vector-db` | 1.00 | 0.60 | [-0.80, +0.00] | unchanged |
| `jwt-vs-session` | 1.00 | 1.00 | [+0.00, +0.00] | unchanged |
| `k8s-configmap-vs-secret` | 1.00 | 1.00 | [+0.00, +0.00] | unchanged |
| `python-gil-tradeoffs` | 1.00 | 1.00 | [+0.00, +0.00] | unchanged |
| `redis-cluster-failover` | 0.80 | 0.00 | [-1.00, -0.40] | regressed |
| `sql-injection-vulnerability` | 1.00 | 0.40 | [-1.00, -0.20] | regressed |
| `tcp-congestion-control` | 1.00 | 1.00 | [+0.00, +0.00] | unchanged |
| `transformer-attention` | 1.00 | 1.00 | [+0.00, +0.00] | unchanged |

Evalith bootstrap 确认回退(regressed):**`redis-cluster-failover`、`sql-injection-vulnerability`**,共 2/10。

---

### 假设 vs 现实:我几乎全猜错了

我事先写下的假设,公开锁在 git commit `68d1db0`,几乎全错。

**强预期(5 个)的命中情况:**

- `explain-vector-db` — mean 从 1.00 跌至 0.60 ✓ 确实下滑,但 bootstrap CI 为 [-0.80, +0.00],刚好触碰 0,判 unchanged。统计保守一步,没报警。
- `explain-rlhf` — mean 纹丝不动,1.00 → 1.00。❌ 完全没命中。
- `python-gil-tradeoffs` — 同上,1.00 → 1.00。❌
- `tcp-congestion-control` — 1.00 → 1.00。❌
- `transformer-attention` — 1.00 → 1.00。❌

强预期 5 个,bootstrap 确认回退的:0 个。就算放宽到"均值有下滑"也只有 1 个(`explain-vector-db`)。

**边界 case(2 个):**

`k8s-configmap-vs-secret` 和 `jwt-vs-session` 均纹丝不动,1.00 → 1.00。我预测"可能掉可能不掉",它们选择了不掉。

**应该不被命中的(3 个)——反而 3/3 全被打中:**

- `sql-injection-vulnerability` — mean 从 1.00 跌至 0.40,bootstrap 确认回退。❌ **假安全。**
- `asyncio-yield-deadlock` — mean 从 1.00 跌至 0.60,CI 触碰 0,判 unchanged——但下滑是真实的。❌ **假安全。**
- `redis-cluster-failover` — mean 从 0.80 跌至 0.00,5/5 全挂,bootstrap 确认回退,CI 完全在 0 以下。❌ **灾难性失手。**

---

### 为什么我的直觉是错的

事后可以做合理化解释——但这正是要警惕的陷阱。解释放在这里,是为了说明"直觉错在哪",而不是为自己辩护。

**`explain-rlhf` 为什么幸存:**"SFT → 奖励模型 → PPO"这三个词本身就是高频训练语料里的核心 token。即便 prompt 要求"跳过基础",模型在极度压缩输出时也会本能地保留这条主干,因为它们就是答案的骨架,不是铺垫。judge 看到 SFT/reward model/PPO 三个节点都在,照样给过。我的预测错在把"知道这个概念的读者觉得是基础"等同于"模型在压缩时会省掉它"——两者不是一回事。

**`redis-cluster-failover` 为什么灾难性崩溃:**Failover 流程的每一个步骤都需要具体陈述——epoch 更新机制、gossip 协议广播时序、slot 重定向的 MOVED vs ASK 区别。这些步骤之间没有一个可以被"略去基础"这条指令优雅地省掉:每一步都是"非基础的",但省掉任何一步都会让回答不完整。实际 B 跑的输出只有 101 个字,把所有步骤压进三句话,judge 在"流程完整性"上给出全 0。这不是因为模型对 Redis 理解不够,而是"长度约束 + 多步骤流程"的组合必然导致截断。

**`sql-injection-vulnerability` 和 `asyncio-yield-deadlock` 为什么被打中:**这两题都是"诊断 + 修复"双轨结构。`sql-injection` 的正确回答需要说清楚两件事:漏洞为何危险(字符串拼接让 SQL 元字符逃逸)和修复手段(参数化查询)。1-2 句话只够说"改成参数化查询",why-not 那一半被 prompt 的"concise"指令砍掉了——judge 在"解释漏洞成因"这个维度上扣分。`asyncio-yield-deadlock` 同理:正确答案不只是"`asyncio.sleep` 替代 `time.sleep`",还要说清楚 `time.sleep` 是同步阻塞不会让出事件循环控制权。B 跑的输出 146 字,这层 why 被压缩成一半,judge 扣了分。

---

### 这正是最重要的一点

我事先写下的假设,公开锁在 git commit `68d1db0`,几乎全错。强预期命中的只有 0 个(放宽到"均值有下滑"也只有 1 个),应该不被命中的反而 3/3 全部出现下滑。**人对 LLM 行为的直觉本身就不可靠。**

这不是谦虚,这是实测结论。我在这个领域浸泡了足够长时间,我对 10 个 case 的预测还是在关键方向上全错了。任何没有数据的"我感觉这个改动不影响质量"都应该被当成零信息处理。

但 bootstrap CI 不需要我的预测才能工作,它只读数据。两个被确认回退的 case,bootstrap CI 全部在 0 以下,mean 分别降至 0.00 和 0.40,判定坚实。均值没动的 case(包括那些我强预期会命中的),bootstrap 照样给出 [+0.00, +0.00],一条都没有误报。这才是正确的评测行为:不问"我以为哪里会出问题",只问"数据说哪里出了问题"。

## 五、同台对照:promptfoo / DeepEval 跑同一套

### 横评公平性框架

这一节用同一套数据,同时跑了三个工具:Evalith、promptfoo、DeepEval。公平性边界严格对齐:同一份 10 条 dataset,同一个模型(DeepSeek)加同一个 temperature=1.0,同一份 judge criteria 字符串,同一份隐式偏差 prompt 注入(B 组)。三个工具之间唯一的变量,是**如何调用 judge、如何对分数打分、如何把多次采样聚合成最终结论**。所有配置文件公开在 `docs/blog/article2/configs/`,raw outputs 在 `docs/blog/article2/raw/`,一行命令可复现。

### 每 case 三方判定总表

| case | A1 mean (E / pf / De) | B mean (E / pf / De) | Evalith bootstrap |
|---|---|---|---|
| `explain-rlhf` | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | unchanged |
| **`explain-vector-db`** | 1.00 / 1.00 / 1.00 | 0.60 / 1.00 / 0.20 | unchanged |
| **`sql-injection-vulnerability`** | 1.00 / 1.00 / 1.00 | 0.40 / 0.60 / 0.60 | regressed |
| `k8s-configmap-vs-secret` | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | unchanged |
| **`asyncio-yield-deadlock`** | 1.00 / 1.00 / 1.00 | 0.60 / 1.00 / 1.00 | unchanged |
| `python-gil-tradeoffs` | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | unchanged |
| **`redis-cluster-failover`** | 0.80 / 1.00 / 1.00 | 0.00 / 1.00 / 0.60 | regressed |
| **`tcp-congestion-control`** | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 0.60 | unchanged |
| `jwt-vs-session` | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | unchanged |
| **`transformer-attention`** | 1.00 / 1.00 / 1.00 | 1.00 / 0.80 / 0.60 | unchanged |

列标注:E = Evalith,pf = promptfoo,De = DeepEval。B mean < 0.8 的 case 粗体标注。

### 三个工具,同一份数据,三种故事

把三家的"B mean < 0.8"集合各自列出来,就能看清楚分歧有多大:

- **Evalith bootstrap(严格,CI 完全在 0 以下):** `sql-injection-vulnerability`、`redis-cluster-failover` — **2/10**
- **Evalith(放宽,B mean < 0.8):** `sql-injection-vulnerability`、`redis-cluster-failover`、`asyncio-yield-deadlock`、`explain-vector-db` — **4/10**
- **promptfoo(B mean < 0.8):** `sql-injection-vulnerability` — **1/10**
- **DeepEval(B mean < 0.8):** `sql-injection-vulnerability`、`redis-cluster-failover`、`explain-vector-db`、`tcp-congestion-control`、`transformer-attention` — **5/10**

归类之后,有四种模式:

**三家全部命中(可信程度最高):** 只有 `sql-injection-vulnerability` 一个。所有工具在 B 组都看到了这条 case 下滑,不管评分逻辑多么不同,结论一致。如果只能相信一个回退结论,就是这个。

**两方一致命中(较可信):** Evalith 和 DeepEval 都抓到了 `redis-cluster-failover`(Evalith: 0.80 → 0.00,DeepEval: 1.00 → 0.60)和 `explain-vector-db`(Evalith: 1.00 → 0.60,DeepEval: 1.00 → 0.20)。promptfoo 在这两条上都给出 1.00,没有任何下滑信号。两家同意、一家不同意。

**单方独立标记:** DeepEval 额外抓到了 `tcp-congestion-control` 和 `transformer-attention`——Evalith 和 promptfoo 在这两条上都判 unchanged,B mean 均为 1.00。只有 DeepEval 在这两条上给出了低于 0.8 的分数。反过来,Evalith(放宽标准)抓到了 `asyncio-yield-deadlock`(0.60),DeepEval 和 promptfoo 都说 unchanged。

**promptfoo 是三家里最保守的:** 10 个 case 里只标记了 1 个。这不一定是坏事——少报警也意味着少打扰开发者——但它放过了所有其他工具都发现了问题的那些 case。

### 为什么同一份 judge 会给出三种答案

三个工具共用的是同一个 judge 模型和同一份 criteria 字符串,但**调用 judge 的方式本质上不同**。

promptfoo 的 `llm-rubric` 在内部有自己的 prompt 模板,默认倾向于二元判断("通过 / 不通过"),对"看起来合理但不完整"的短回答更容易给过;它在 `redis-cluster-failover` 上给出了 1.00,而实际上 B 组的输出只有 101 字且流程严重截断——这说明 promptfoo 默认模板在"完整性"这个维度上的敏感度较低。

DeepEval 的 `GEval` 是带 chain-of-thought 的 weighted scoring,judge 先逐步推理再打分,对"短回答 / 信息缺失"的惩罚更重——这解释了为什么它额外标记了 `transformer-attention` 和 `tcp-congestion-control`:这两条的 B 组输出字数有限,GEval 的推理链更容易发现"某个关键点没有被展开"。

Evalith 的做法是直接让 judge 给出 score + pass,然后对多次采样的 pass_rate 做 bootstrap 置信区间,在均值差值上判断统计显著性。它的判定逻辑是"均值差值的 CI 是否完全低于 0",而不是"pass rate 是否超过某个阈值"。

同一个评分目标,落实到三套不同的 prompt engineering 和聚合机制,verdict 就会分叉。这是结构性的,不是 bug,也不能靠调参消除——除非三家对齐到完全相同的 judge prompt,但那就不再是"三个工具的横评"了。

### 诚实地承认:这场对照里 Evalith 不"赢"

没有 ground truth,无法说哪个工具最准。

三家一致判定的 `sql-injection-vulnerability` 大概率是真回退。但其余意见分歧的 case——比如 `tcp-congestion-control` 和 `transformer-attention` 是否真的下滑了——谁对谁错全凭信念,不凭数据。DeepEval 说 yes,Evalith 和 promptfoo 说 no,我们没有独立的真相可以裁决。

如果说这场对照有什么发现,那是:**同一个评测问题,用不同工具会得出不同结论**。这是 LLM eval 这个领域最难看清楚的一点——不是"哪个工具更好",而是"你以为在测同一件事,其实不是"。

Evalith 有一个其他两家没有汇报的数据点:A1 vs A2 假阳率为 0/10。这是因为 Evalith 是唯一跑了两份 baseline 的工具,promptfoo 和 DeepEval 只跑了一份 baseline,所以它们的"假阳率"这个指标根本就是缺失的,而不是零。bootstrap CI 让"无信号"和"有信号"有了清晰的统计边界,这是 Evalith 的具体贡献。

但 DeepEval 在"轻微下滑"上更敏感,如果你更在乎不放过任何下滑信号,它的 5/10 覆盖率是一种选择。promptfoo 在"避免噪声报警"上最保守,如果你的团队更怕误报,它的 1/10 是另一种选择。各有价值,没有绝对正确。

### promptfoo 和 DeepEval 在哪些方面强于 Evalith

这里需要直说:在分发量、社区生态、scorer 种类、UI 体验这几个维度上,promptfoo 和 DeepEval 目前仍然遥遥领先于 Evalith。

promptfoo 有成熟的 web UI、数十种内置 scorer(包括语义相似度、毒性检测、代码执行正确性等)、完善的 CI/CD 集成文档和活跃的社区。DeepEval 有 Confident AI 云平台、丰富的评测维度(faithfulness、contextual precision、hallucination 等)以及对 RAG 评测的原生支持。这两个工具在 LLM eval 生态里处于主流位置是有道理的。

Evalith 不是要取代它们。Evalith 目前补的那一块是:**基于 bootstrap 置信区间的统计显著性 CI gate**——当你需要的不是"打分",而是"这两份 config 之间的差值是否在统计上显著",那是 Evalith 专门设计的场景。

一个可能合理的组合姿势:用 promptfoo 起步做日常 eval,熟悉各种 scorer 和 UI;在 CI 阶段需要阻断 PR 时,用 Evalith 的 bootstrap CI 做最终裁决,避免把采样噪声引起的均值漂移误判为质量回退。两者互补,不互斥。

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
