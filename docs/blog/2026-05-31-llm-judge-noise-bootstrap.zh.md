# 续:LLM 当 judge 自己也在抖 —— bootstrap CI 抗噪与三个 OSS eval 工具的同台对照

> 上一篇 [《AI 回归测试需要统计显著性》](https://zhuanlan.zhihu.com/p/2043351926964848178)在结尾承认了一件事:实验用的 temperature=0 + 简单事实题让 DeepSeek 太确定,bootstrap 的"噪声免疫"优势没有真正显现。这一篇,我们把那张牌翻开。

---

## 一、上篇没说清的那一半

<TODO §1 prose — Task 13>

## 二、重新设计实验:让 LLM 真的开始抖

<TODO §2 prose — Task 13>

## 三、实验 A:噪声基线 —— 同 config 重跑

<TODO §3 prose + table from Evalith diff A1-vs-A2 — Task 7>

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
