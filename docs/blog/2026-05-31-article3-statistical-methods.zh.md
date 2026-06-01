# 续之续:把 BCa、paired bootstrap、FDR、第三方 judge 都挂上去,文章 2 的结论变了吗?

> 文章 2 的结论是 Evalith bootstrap 抓到 2/10 regressed,promptfoo 抓到 1/10,DeepEval 抓到 5/10,只有 sql-injection-vulnerability 三家一致。今天把更严格的统计方法和换 judge 都加上,用同一份 frozen raw 数据重看一遍。

---

## 一、那个承诺 + 一个问题

上篇数字:Evalith bootstrap 抓到 **2/10** regressed,promptfoo 1/10,DeepEval 5/10,三家唯一一致命中的只有 sql-injection-vulnerability。其余 case 三家各说各的。这就产生一个必须回答的问题:Evalith 漏掉的,到底是别家误报,还是真的有问题而 Evalith 没抓到?

上篇文末承诺了 4 件事——BCa 修偏、paired bootstrap 利用 case 内相关性收窄区间、FDR 控制 10 个 case 同时检验时的假阳率膨胀、第三方 judge 验证跨家族判定是否一致。今天把这四件事挂上去,在同一份 frozen raw 数据上重看一遍。

实验环境有一个偏离需要事先说清楚:可用的 OpenAI 代理只放行 gpt-5-mini,且强制 temperature=1。上篇 deepseek judge 跑在 temp=0;这次 judge 本身带了采样噪声。对结论的影响,§5 展开,§7 再提。

5 条预测就摆在下面。文章 2 的假设几乎全错,这次数据说了算。

### 在看到结果之前,我先把 5 条预测写在这里

把统计方法升级 + 换 judge,我**事先**预测如下:

1. **BCa 大概不会显著改变 verdict。** 文章 2 数据样本小(每 case 5 sample),分布近 0/1 二元,BCa 的修偏在这种分布上贡献有限。
2. **Paired bootstrap CI 会收窄,但不足以让任何 unchanged 翻成 regressed。** 文章 2 的 unchanged case 几乎都是 mean=1.00 两边、Δ=0、零方差。收窄一个零宽度的 CI,结果仍然横跨 0。
3. **FDR 在 10 case 上会把 sql-injection 翻成 unchanged。** 其 CI 离 0 较近(`[-1.00, -0.20]`),BH 校正后 p-value 可能不再通过 0.05 阈值。redis-cluster-failover 的 CI `[-1.00, -0.40]` 更远离 0,会留住。
4. **Judge swap A(单换 judge)会让 verdict 大幅变化。** GPT-4o-mini 不是 deepseek 亲属,对同一中文 prompt + 同一回答的判定标准会不一样。可能比 deepseek 严或宽,但一定不会照搬。
5. **Judge swap B(全换 model+judge)与 A 的差距反映 model variance 的贡献占比。** 如果 A 和 B verdict 接近,说明换 judge 已主导;如果差距大,说明 model 输出本身也是变量。

预测全错的可能不小。文章 2 的假设就几乎全错了。这一篇会用同样方式把对错都摆出来,无论结果怎样。

(以下表格都来自实际跑出来的实验,任何一条没有事后修正。所有 raw outputs 在 `docs/blog/article3/raw/`,一行命令复现。)

## 二、BCa: 修偏与加速

Percentile bootstrap 直接切 bootstrap 分布的第 α/2 和第 1-α/2 分位数,在分布对称时没问题。当分布偏态,分位数切出来的区间会往偏的那侧倾斜,不补偿这个偏置。BCa 加了两个修正:`z₀` 衡量观察统计量在 bootstrap 分布里的相对位置并把边界往中心扳,`a` 用 jackknife 估计分布的二阶弯曲度。两项合在一起,让 CI 在小样本、偏态场景下更接近名义覆盖率。

Evalith v0.5 的 BCa 是纯标准库实现:`statistics.NormalDist` 提供 Φ/Φ⁻¹,jackknife 在 A1∪B union 样本上逐次删一跑。dev extras 里挂 `scipy.stats.bootstrap(method='BCa')` 作 ground truth;在 redis-cluster-failover 这类噪声 fixture 上,两者 CI 边界差 < 0.10。

**文章 2 frozen raw 的 BCa 与 percentile 对照:**

| case | percentile CI | BCa CI |
|---|---|---|
| `explain-rlhf` | [+0.00, +0.00] | [+0.00, +0.00] |
| `explain-vector-db` | [-0.80, +0.00] | [-0.80, +0.00] |
| `sql-injection-vulnerability` | [-1.00, -0.20] | [-1.00, -0.20] |
| `k8s-configmap-vs-secret` | [+0.00, +0.00] | [+0.00, +0.00] |
| `asyncio-yield-deadlock` | [-0.80, +0.00] | [-0.80, +0.00] |
| `python-gil-tradeoffs` | [+0.00, +0.00] | [+0.00, +0.00] |
| `redis-cluster-failover` | [-1.00, -0.40] | [-1.00, -0.20] |
| `tcp-congestion-control` | [+0.00, +0.00] | [+0.00, +0.00] |
| `jwt-vs-session` | [+0.00, +0.00] | [+0.00, +0.00] |
| `transformer-attention` | [+0.00, +0.00] | [+0.00, +0.00] |

9 行 CI 完全一样。这批 case 的 bootstrap 分布要么是单点(Δ=0,零方差),要么对称;z₀ 和 a 都接近 0,修正量可以忽略。

唯一有漂移的是 `redis-cluster-failover`。Percentile 上界 `-0.40`,**BCa 上界移到 `-0.20`**。A1 mean 0.80、B mean 0.00,bootstrap 差值大量落在 -1.00 附近、上端稀疏,是单边偏态;z₀ 把边界往 0 拉了一段。CI 仍然完全在 0 以下,verdict 不变,还是 **regressed**。

**预测 1** 说"BCa 大概不会显著改变 verdict"。结果吻合:回归集合与 percentile 完全相同,{sql-injection-vulnerability, redis-cluster-failover},2/10。教材说"小样本时 BCa 比 percentile 准"指的是覆盖率准确性,预测 1 说的是 verdict 层。CI 边界移了,判定没变,两件事都成立。

## 三、Paired bootstrap: 利用 case 内相关性降方差

<TODO §3 — Task 18>

## 四、FDR: 当你同时检验 10 个 case 时

<TODO §4 — Task 19>

## 五、换 judge: 一路只换,一路全换

<TODO §5 — Task 20>

## 六、谁改变了我们的判定

<TODO §6 — Task 21>

## 七、局限和第四篇方向

<TODO §7 — Task 22>

---

如果你也在做 LLM eval 或 AI CI 集成,欢迎到 [github.com/dominciyue/Evalith_MingJing](https://github.com/dominciyue/Evalith_MingJing) 提 issue / PR。

```bash
pip install evalith
```

即装即用,Apache-2.0。
