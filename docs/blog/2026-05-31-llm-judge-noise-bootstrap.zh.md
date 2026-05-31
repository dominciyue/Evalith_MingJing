# 续:LLM 当 judge 自己也在抖 —— bootstrap CI 抗噪与三个 OSS eval 工具的同台对照

> 上一篇 [《AI 回归测试需要统计显著性》](https://zhuanlan.zhihu.com/p/2043351926964848178)在结尾承认了一件事:实验用的 temperature=0 + 简单事实题让 DeepSeek 太确定,bootstrap 的"噪声免疫"优势没有真正显现。这一篇,我们把那张牌翻开。

---

## 一、上篇没说清的那一半

上篇最后有一段"需要诚实承认的局限":temperature=0 + 简单事实题让 DeepSeek 这次表现极其确定,5 次采样里几乎没观测到 LLM 抖动,所以 bootstrap CI 都很窄。那篇文章建立了方法论框架,却用了一组"几乎不会抖"的数据来演示"抗抖动能力"——这个矛盾我当时在文末点了出来,承诺"下一篇文章会专门做这类实验"。

这篇就是那个实验。

bootstrap 的核心论点是:它能把"采样噪声引起的均值漂移"和"真实回退"区分开。这个论点在上篇里靠推理支撑,靠数学支撑,但没有靠实证支撑——因为噪声太小,没有什么东西需要被区分。读者有权问:如果 LLM 真的开始抖,bootstrap 还能好好工作吗?judge 本身也是 LLM、它也在抖,这时候又会怎样?

这一篇做三件具体的事来回答这些问题。

第一,重新设计实验:换成中等技术深度的开放式问答,temperature 调到 1.0,评分器改成纯 llm\_judge。这让 LLM 真的开始产生方差,judge 也真的面对语义边界模糊的回答。

第二,跑两组对照实验:A 组用完全相同的 config 跑两遍(验证 bootstrap 不误报),B 组注入一条隐式 prompt 偏差(验证 bootstrap 能识别真实回退)。在看到数据之前,我把对 B 组的预测逐 case 写在了文档里,公开提交进了 git。

第三,把同一份数据交给 Evalith、promptfoo、DeepEval 三个工具分别跑,看三家是否给出相同的结论。

结果是:我事先写下的假设几乎全错了,而三个工具的回答各不相同。这才是更真实的 AI eval 实况。

## 二、重新设计实验:让 LLM 真的开始抖

### 换题:从事实问答到开放式技术问答

上篇的测试集是三道事实题:首都、算术、日出方向。这类题的答案空间极窄——"北京"就是对,"Paris"就是错,contains 一查就出结论。DeepSeek 在这种题上的输出几乎是确定性的,LLM 的随机性来不及表达就已经被收敛掉了。

换成开放式技术问答之后情况完全不同。这次的 dataset 是 10 道中等技术深度的问题:RLHF 的流程、向量数据库的原理、asyncio 死锁的诊断、Redis Cluster failover 的步骤、JWT 与 session 的对比……每个问题都有多条合理的回答路径,答案的字面表达可以差别很大,没有任何一种表述是唯一正确的。完整 10 道题的原文在 `docs/blog/article2/qa.high-temp.yaml`,看一眼选题列表就能感受到题型逻辑:都是"懂的人知道对不对,但说法千变万化"的那类问题。

这个换型有两个作用。一是让 LLM 的输出真的产生方差:temperature=1.0 下,同一道关于 Redis Cluster 的问题,五次采样可能分别侧重节点探活时序、epoch 更新、slot 迁移确认的不同角度。二是让 judge 真的面临判断难题:回答的差异不是"对/错",而是"这个角度的覆盖足不足够",这正是 llm\_judge 类评分器噪声的真正来源。

### 评分器:砍掉 contains,只保留 llm\_judge

上篇用的是 `contains + llm_judge` 双评分器,contains 作为兜底。这次把 contains 完全去掉,只留 llm\_judge 一个。

原因很简单:开放式问答根本没法用 contains 兜底。"充分解释了 RLHF 的三个阶段"这样的评分标准,没有任何固定字串可以匹配。

但这个改动还有一个更重要的含义,是上篇没说清楚的核心机制:**用 LLM 评 LLM 时,噪声是两层叠加的**。模型输出本身在抖(temperature=1.0),judge 的评分本身也在抖(对边界 case 的语义判定不一致)。上篇的实验里这两层噪声都被事实题的确定性压平了,这次我们把两层都暴露出来。

这不是为了制造混乱,而是因为这才是生产环境里 llm\_judge eval 实际工作的状态。

### Temperature=1.0:不是 0,也不是 1.5

temperature 定在 1.0 有具体理由。

temperature=0 在上篇已经证明没有意义——模型太确定,看不到噪声。temperature=1.5 走向另一个极端:模型偶尔会产生 hallucination 性质的输出,judge 一律给 0 分,"抖动"的信号被"整体变烂"的信号盖过,bootstrap 区分噪声和真实回退的价值就消失了。1.0 是"输出明显有方差、但质量仍可读可判"的甜区——模型在 1.0 下会选择不同的表达路径和侧重点,但不会瞎说。

采样数保持每 case 5 次,与上篇一致,便于读者直接类比上篇的 CI 计算逻辑。

### Judge 配置:保持 temperature=0

judge 用同一个 deepseek-chat,但 temperature=0——这是 Evalith llm\_judge 的默认行为(llm\_judge.py 第 57 行写死)。

这是有意为之:judge 自己保持稳定,这样在 §3 里观察到的任何 judge 噪声完全来自"边界 case 的语义判定不一致",而不是 judge 自己的随机抖动。两层噪声里,这次我们只让模型输出那层抖,judge 层固定住,这样更容易定位噪声来源。

### 横评公平性:byte-for-byte 一致的 criteria

三个工具共用的 judge criteria 字符串完全相同——不是"表意一样",是字节级相同:

> 回答必须充分解释问题涉及的多个核心概念,包括关键机制、适用场景或对比维度。仅给出结论或只列举步骤而不解释原理的回答不应得满分。

这是横评公平性的核心保障。三个工具的 judge 调用方式、prompt 模板、聚合逻辑各不相同——但这是变量,不是噪声。我们控制的是"评分标准",让它不成为三家分歧的来源。所有配置文件公开在 `docs/blog/article2/configs/`。

### 成本与复现

整套实验涉及约 300 个 deepseek-chat 调用(A1、A2、B 三组 × 3 个工具 × 10 case × 5 次采样 + judge 调用),总花费不到 $0.20。任何人设好 `DEEPSEEK_API_KEY` 之后一行命令即可复现:

```bash
bash docs/blog/article2/experiment.sh
```

接下来 §3 是噪声基线(A 组)、§4 是回退注入(B 组)与假设对账、§5 是三工具同台对照。每张表里的每个数都来自实际跑出来的实验,任何一条都没有事后修正——commit history 可以证明假设在数据之前写下。

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

### 统计显著性不是可选项

§3 的 `redis-cluster-failover` 用数字把这件事说清楚了:在 llm_judge 评分体系下,同一道题、同一个模型、同一份 config,单样本 pass rate 可以在 0.0 和 1.0 之间跳动。A1 跑出来是 0.80(5 次里 1 次挂),A2 跑出来是 0.60(5 次里 2 次挂)——单样本极差 [0.0, 1.0],意味着任何一次单跑都可能落在这个区间的任何一点上。

如果你在 CI 里用单样本点对点比较,这条记录会被判成"A1 → A2,回退 25%,阻断 PR"。但那个 0.80 本身就不稳定——下次重跑可能变 1.0,也可能变 0.4。你的阻断依据是一枚偏心硬币的正反面,而不是模型能力的变化。bootstrap CI 给出 [-0.80, +0.40],跨过 0,判 unchanged——这才是正确的裁决。

这不再是上篇那种"理论上 bootstrap 抗噪"的论述。数字摆在这里:A1 单样本极差 [0.0, 1.0],A2 跑出 5/5 里 2 个挂掉。单样本比较会把这两次的差异判成"回退 25%",bootstrap CI 跨过 0 判 unchanged。两者的判定结果完全相反,但只有一个符合事实——config 根本没动。

这里的核心机制不是 bootstrap 本身有多复杂,而是:**从 CI gate 的设计开始,就必须把噪声当一等公民对待。** 点对点比较是今天大多数 eval 工具的默认行为——这个默认值本身是个 bug,不是因为工具不够好,而是因为单样本不能描述总体。任何不引入采样 + CI 的 gate,在 llm_judge 评分场景下只能是"过严则误报频繁、过松则真回退漏过"——两种调法都让工程师不信任 gate 本身。

### LLM-as-judge 是第二个噪声源,这件事改变了 eval 工具的设计

上篇说过"llm_judge 本身也在抖",但没有把这个结论推到工具设计层面。这一篇的横评数据给了更具体的证据。

当 judge 也是 LLM 时,同一个答案被多次评分得到的结果会不一样——尤其是边界 case,比如"回答正确但不完整"、"覆盖了主干但漏了一个关键步骤"这类情况。这意味着 eval 的不确定性来自两层叠加:模型输出抖动 × judge 评分抖动。两层都在,两层都不可忽视。

§5 的对照数据直接证明了这一点的工具间影响。三家工具用了同一份 criteria 字符串、同一个 deepseek-chat judge、同一份 dataset,verdict 仍然分叉成 2/1/5 三种结果——Evalith 严格 2 个,promptfoo 保守 1 个,DeepEval 敏感 5 个。分歧完全来自各家把 criteria 落实到 LLM 提示工程和聚合阈值时的工程选择不同:promptfoo 的默认模板对"完整性"维度敏感度低,DeepEval 的 chain-of-thought 对短回答惩罚更重,Evalith 的 bootstrap CI 在均值差值上做统计显著性检验。

这说明:**"用 LLM 评 LLM"不是一个中立、可比较的测量行为**,它本身是工具实现细节强相关的。同一个评测问题,落实到不同工具,就是不同的测量。这不是 bug,也不能靠调参消除——除非三家对齐到完全相同的 judge prompt 和聚合逻辑,但那就不再是独立工具的横评了。

推论直接影响选工具的决策逻辑:如果你团队在做 LLM eval CI,选工具不是选"哪家最准",而是选"我团队对回退的定义和这家工具的判定语义最匹配"。DeepEval 的 5/10 覆盖率意味着它对轻微下滑更敏感——如果你更在乎不放过任何信号,这是一种选择。promptfoo 的 1/10 意味着它更保守——如果你的团队更怕误报打扰开发节奏,这是另一种选择。Evalith 的 bootstrap CI 提供的是统计边界,让"无信号"和"有信号"有明确的置信度区分。没有绝对正确,只有哪个语义和你的需求更匹配。

## 七、局限和下一步

几个需要诚实说清楚的边界:

1. **n=10 cases 是小样本。** 这整篇是 lower bound 演示,不是 LLM eval 工具的全面性能基准。10 道题只能说明"在这类开放式技术问答上,三家工具在噪声处理和判定语义上存在分歧"——不能泛化成"哪家工具更好"的结论。横评结论请别过度解读。

2. **prompt 偏差注入是人为的。** `Be very concise. 1-2 short sentences.` 是一条教学场景的合成回退,真实业务里的质量下滑可能隐蔽得多——模型版本静默升级、上下文长度悄悄裁剪、retrieval 召回率下降。即便如此,"教学难度"的注入让三个工具给出了三种不同的 verdict,说明信号可识别性本身不是瓶颈,工具语义才是。如果换成更隐蔽的回退,分歧只会更大,不会更小。

3. **judge 与被评模型同源,存在亲缘偏差。** 同一个 deepseek-chat 既出答案又评分,是 affinity-bias 的标准场景——模型倾向于给自己风格接近的答案打高分。第三篇会用 GPT-4o-mini 当第三方 judge 在同一份 dataset 上跑一遍,看 verdict 是否大幅迁移。如果迁移幅度大,那"哪家工具在这次实验里报了几个回退"的讨论本身就要重做——因为 judge 换了,测量对象就变了。

4. **统计工具栈仍然是最朴素的 percentile bootstrap。** BCa(偏置纠正加速)在样本分布偏态时比 percentile 更准;paired bootstrap 利用同 case 配对可以降低方差;FDR(假发现率控制)在 10 个 case 同时检验时能压制多重比较带来的假阳率膨胀——这些都还没进 Evalith。**第三篇会把这些一一加进 Evalith 并对照本篇的同一份 raw 数据重跑,看哪些指标实际改变,哪些只是工具箱里"听起来更高级"的东西。** 你可以把本篇的结果当基线,第三篇的结果当对照。

5. **横评只跑了三家。** Ragas 专门针对 RAG 评测、OpenAI Evals 深度绑定 OpenAI 接口——两家的 scope 和这次实验不可比,所以没接入,§5 开头已说明理由。dev.to / Show HN 的英文版会单独做,工具范围和实验设计会相应调整。

---

如果你也在做 LLM eval 或 AI CI 集成,欢迎到 [github.com/dominciyue/Evalith_MingJing](https://github.com/dominciyue/Evalith_MingJing) 提 issue / PR。

```bash
pip install evalith
```

即装即用,Apache-2.0。
