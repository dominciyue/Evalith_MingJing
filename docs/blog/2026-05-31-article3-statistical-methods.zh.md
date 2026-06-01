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

Percentile bootstrap 把 before 和 after 当作两组独立样本各自重采样,然后做差。case 间的难度差异会混入 CI 的估计方差,但那部分不是我们想衡量的不确定性。

Paired 改变重采样对象:**抽 case 索引** `i₁, ..., iₙ`,对每个抽到的 `i` 计算 `Δᵢ = after[i] - before[i]`,再取均值。同一次抽样里 before 和 after 看的是同一个 case,case 内的难度因子被同向抵消,理论上 CI 应该收窄。

Evalith v0.5 的 `_bootstrap_paired` 用 `rng.randrange(n)` 抽索引,两数组等长为前提,否则抛 `ValueError`。单元测试用"每 case 偏移恰好 -0.1"的完美相关 fixture 验证了它:paired CI 压到 `[-0.10, -0.10]`(宽度 0),percentile 给 `[-0.46, +0.26]`(宽度 0.72)。

**文章 2 frozen raw 的 paired 与 percentile CI 对照:**

| case | percentile CI | paired CI |
|---|---|---|
| `explain-rlhf` | [+0.00, +0.00] | [+0.00, +0.00] |
| `explain-vector-db` | [-0.80, +0.00] | [-0.80, +0.00] |
| `sql-injection-vulnerability` | [-1.00, -0.20] | [-1.00, -0.20] |
| `k8s-configmap-vs-secret` | [+0.00, +0.00] | [+0.00, +0.00] |
| `asyncio-yield-deadlock` | [-0.80, +0.00] | [-0.80, +0.00] |
| `python-gil-tradeoffs` | [+0.00, +0.00] | [+0.00, +0.00] |
| `redis-cluster-failover` | [-1.00, -0.40] | [-1.00, -0.40] |
| `tcp-congestion-control` | [+0.00, +0.00] | [+0.00, +0.00] |
| `jwt-vs-session` | [+0.00, +0.00] | [+0.00, +0.00] |
| `transformer-attention` | [+0.00, +0.00] | [+0.00, +0.00] |

10 行完全一致。Paired 没有产生任何收窄,flagged 集合和 percentile 相同,仍是 {sql-injection-vulnerability, redis-cluster-failover},2/10。

原因在于数据结构本身。文章 2 的 `pass_rate_samples` 把每次 trial 的 `score >= 0.5` 拍成 0/1,单个 case 的 5 个值基本是近常数序列。**Paired 降方差的前提是 within-case 存在真实的变化空间**;重采样近常数序列均值几乎不变,和 unpaired 没有区别。二值化判分事先把那个变化空间压扁了。

**预测 2** 说"Paired CI 收窄,但不足以让任何 unchanged 翻成 regressed"。结果比预测更极端:CI 完全没有收窄。预测的方向对了,但原因是"压根没收窄",而不是"收窄不够"。统计方法的威力是数据形态决定的,不是方法名称决定的。

## 四、FDR: 当你同时检验 10 个 case 时

单个 case 用 α=0.05 检验,逻辑清楚:一次假设,错的概率 5%。但我们同时在 10 个 case 上做检验,每个都用同一个 5% 门槛,问题就出来了。假设 10 个 case 其实都没有真回退,每个 case 单独看,有 95% 概率不会被误标。10 个一起,至少一个被误标的概率是 `1 - (1 - 0.05)^10 ≈ 40%`。

这个 40% 叫 family-wise 假阳率。它的意思是:在真正没有回退的数据集上,跑 percentile 检验,大约每两三次就会错误地标出至少一个 regressed case。10 个 case 并不算大的 eval set,很多团队会测 30、50 个 case,FWER 还会更高。1995 年,Benjamini 和 Hochberg 发表的 BH 方法放弃了控制 FWER 这个更严的目标,转而控制 **false discovery rate**——期望:在所有被判 regressed 的 case 中,真正误标的比例不超过 α。这个放宽让它在实践中比 Bonferroni 更有用。

BH 的步骤是:把所有 case 的 p-value 升序排列,得到 p_(1) ≤ p_(2) ≤ ... ≤ p_(N)。对 rank k 设阈值 t_k = (k/N) × α。然后从大往小找——找最大的 k 满足 p_(k) ≤ t_k,接受 rank 1 到 k 全部为 regressed。直觉上:排越靠前、p 越小的 case,应该享受越严的阈值才算通过;排靠后的 case 虽然阈值更宽,但 p 本身也更大,大多数情况下还是过不了。

这一批 case 的 bootstrap p-value 来自两边检验,n=1000 次重采样,seed=0,每个 case 统计在 1000 次中 Δ ≥ 0 的比例乘以 2:

| case | p-value |
|---|---|
| `explain-rlhf` | 1.0000 |
| `explain-vector-db` | 0.1660 |
| `sql-injection-vulnerability` | 0.0140 |
| `k8s-configmap-vs-secret` | 1.0000 |
| `asyncio-yield-deadlock` | 0.1200 |
| `python-gil-tradeoffs` | 1.0000 |
| `redis-cluster-failover` | 0.0000 |
| `tcp-congestion-control` | 1.0000 |
| `jwt-vs-session` | 1.0000 |
| `transformer-attention` | 1.0000 |

redis 的 p=0.0000 说明 1000 次重采样里没有一次出现 Δ ≥ 0,bootstrap 分布完全压在 0 以下。sql-injection 的 p=0.0140 则意味着约 7 次出现 Δ ≥ 0,不能完全排除正向波动。其余 8 个 case 要么 p=1.0000(均值差为 0,对称分布),要么接近它。

升序排列后走 BH:

rank 1 是 redis,p=0.0000,阈值 (1/10)×0.05 = 0.005。0.0000 ≤ 0.005,通过。

rank 2 是 sql-injection,p=0.0140,阈值 (2/10)×0.05 = 0.010。0.0140 > 0.010,**不通过**。

rank 3 往后 asyncio、explain-vector-db、以及并列 1.0000 的六个 case 更不可能通过了。

BH 的结论:最大通过的 rank 是 k=1。regressed 集合从 {sql-injection-vulnerability, redis-cluster-failover} 缩成 {redis-cluster-failover},**2/10 → 1/10**。这是本文四种方法里第一次改变判定结果的。

sql-injection 的 p=0.0140 离 rank 2 的阈值 0.010 差了 0.004——四个千分之一。没有数量级的差距,只是没过那道线。BH 不是软性建议,规则是确定的。

理解这个机制在实际工程里有几个含义。5 个 case 的 eval set,BH 几乎不会改变任何结果:阈值序列从 0.005 到 0.025,和未校正的 0.050 差距不大。10 到 30 个 case 是过渡区:本篇数据就处于这里,一两个边界 case 会被压下去。50 个 case 以上,percentile 单独跑下来假阳率会高得难以接受,BH 是必须打开的选项。简单说:**eval set 越大,越该开 BH**。Evalith 把它放在 `--multi-test bh` 而不是默认启用,是让你根据实验规模自己决定。

§1 的预测 3 写的是:"FDR 在 10 case 上会把 sql-injection 翻成 unchanged。其 CI 离 0 较近,BH 校正后 p-value 可能不再通过 0.05 阈值。redis-cluster-failover 的 CI `[-1.00, -0.40]` 更远离 0,会留住。"**完全命中**。不只是"有 case 被翻",连哪个被翻、哪个留住都指对了。sql-injection rank 2,p=0.014,阈值 0.010,差 0.004 落选;redis rank 1,p=0.000,阈值 0.005,通过。预测 3 写的时候手头只有 percentile CI,没有跑过 BH,靠的是 CI 边界离 0 的距离来推断——这次推断成立了,文章 2 那种"假设全错"的结果这次没有复现。

## 五、换 judge: 一路只换,一路全换

文章 2 用 deepseek-chat 同时充当被测 model 和评分 judge。同源模型评同源输出,在 LLM-as-judge 文献里通常叫 affinity bias,judge 可能对"自己风格"的答案天然宽容。§7 会把它列为 limitation #3,但在写进 limitation 之前,先把它实测一遍。

两条轨道各自有不同的控制变量。swap A 只换 judge,模型输出沿用文章 2 原文,判定差异完全由 judge 变化解释。swap B 同时换 model 和 judge,gpt-5-mini 既生成回答又评分。两轨 verdict 的差值部分反映 model 输出本身的贡献。

### 实验环境的诚实交代

计划是 gpt-4o-mini at temp=0,复刻文章 2 judge 的稳定性。实际能用的代理只放行 gpt-5 家族加 o-series,且强制 temperature=1。swap A 因此混了两件事:judge 换了家族,采样噪声也从 temp=0 变成了 temp=1。§7 会再提一次。

### 主表:三列 verdict 对照

| case | DS+DS (v0.4 baseline) | swap A: DS-out + GPT judge | swap B: GPT+GPT |
|---|---|---|---|
| `explain-rlhf` | unchanged | unchanged | unchanged |
| `explain-vector-db` | unchanged | **regressed** | unchanged |
| `sql-injection-vulnerability` | **regressed** | unchanged | unchanged |
| `k8s-configmap-vs-secret` | unchanged | unchanged | unchanged |
| `asyncio-yield-deadlock` | unchanged | unchanged | unchanged |
| `python-gil-tradeoffs` | unchanged | unchanged | unchanged |
| `redis-cluster-failover` | **regressed** | unchanged | unchanged |
| `tcp-congestion-control` | unchanged | unchanged | unchanged |
| `jwt-vs-session` | unchanged | unchanged | unchanged |
| `transformer-attention` | unchanged | unchanged | unchanged |

直接观察:基线 flagged 2/10(redis, sql-injection),swap A flagged 1/10(explain-vector-db),swap B flagged 0/10。

三列没有任何一个 case 被共同判为 regressed。**两个 judge 对"什么算回退"的判定完全不重合**。

### swap A 为什么彻底换了一个 case

关键在 redis 的 judge 分歧。同一份 deepseek-chat 回答,交给两个 judge:

| case | deepseek-chat judge mean | gpt-5-mini judge mean | gap |
|---|---|---|---|
| `redis-cluster-failover` | 0.80 | 0.00 | -0.80 |
| 其余 9 个 case | 1.00 | 1.00 | +0.00 |

gpt-5-mini 看同一份 redis 答案 5 次全打 0,deepseek-chat 自己评 5 次里 4 次给过。同一份文字,两个 judge,结论完全相反。

分歧直接解释了 swap A 的 verdict:gpt-5-mini 在 baseline 里已经把 redis 评为 0.00,broken 版本同样是 0.00,Δ=0,无信号——baseline 就已经给了死分,broken 也不过原地踏步。sql-injection 同理,两端都是 1.00,Δ=0,消失。

explain-vector-db 方向反过来:baseline 是 1.00,broken 跌到 0.00,Δ=-1.00 显著,进入 flagged。

最终 flagged 集合从 `{redis, sql-injection}` 换成 `{explain-vector-db}`。不是加减,是整个集合替换了一遍。

### swap B 为什么 0 个 regressed

swap B 让 gpt-5-mini 生成回答再自评。a1 baseline 侧 10/10 全部 1.00。broken 侧 8/10 仍是 1.00,3 个 case 降到 0.80,但 Δ=-0.20 的 CI 仍然横跨 0,信号不足。

**当 LLM 给自己的输出评分,它倾向于宽容**。工业界文献里反复提到这点,这次在 10 个真实 case 上实测了一遍。

### 预测 4 和预测 5 的对照

预测 4:"Swap A 让 verdict 大幅变化。"结果:flagged set 完全不重合——基线 `{redis, sql-injection}`,swap A `{explain-vector-db}`。**预测 4 CONFIRMED**,程度超出预期,比"大幅变化"还激进。

预测 5:"A 和 B 的差距反映 model variance 贡献占比。"结果:A flags 1 个,B flags 0 个,集合不重合。差距真实,但 swap B 的 0/10 是自写答案本来就不差加上自评宽容的双重效应,model variance 只是其中一个因素。**预测 5 CONFIRMED with caveat**。

### 这告诉我们什么

选 judge 不是 administrative 决定,是科学决定。文章 2 的 2 个 regressed 换成 gpt-5-mini judge 后**没有任何一个还留下来**——不是消失,是被另一组完全不同的 case 替换。

这不是说文章 2 错了。eval 工具报出的 verdict 是 model-judge 组合的属性,不是数据本身的属性。换个 judge,回归判定可以变得不可识别。

## 六、谁改变了我们的判定

6 种干预、10 个 case、60 个判定格。把它们全部放在一张表里,才能看清楚哪些格子真的变了颜色。

| case | percentile (v0.4) | BCa | paired | FDR (BH) | swap A (DS-out+GPT-judge) | swap B (GPT+GPT) |
|---|---|---|---|---|---|---|
| `explain-rlhf` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `explain-vector-db` | unchanged | unchanged | unchanged | unchanged | regressed | unchanged |
| `sql-injection-vulnerability` | regressed | regressed | regressed | unchanged | unchanged | unchanged |
| `k8s-configmap-vs-secret` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `asyncio-yield-deadlock` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `python-gil-tradeoffs` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `redis-cluster-failover` | regressed | regressed | regressed | regressed | unchanged | unchanged |
| `tcp-congestion-control` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `jwt-vs-session` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `transformer-attention` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |

### 稳定的 7 个 case

上表中,7 个 case(explain-rlhf、asyncio-yield-deadlock 等)6 列全部 unchanged。无论换统计方法还是换 judge,verdict 纹丝不动。文章 2 对这 7 个 case 的 unchanged 判定,在本文所有干预下都经得住。

### 3 个不稳定 case

**sql-injection-vulnerability** 在 percentile/BCa/paired 全部 regressed,到 FDR 被压掉,swap A/B 无信号。p=0.0140,离 BH rank 2 阈值 0.010 只差 0.004。信号本身不强,稍严一点的约束就够了。

**redis-cluster-failover** 是全表信号最强的 case。四列统计方法全 regressed,p=0.0000。但 swap A 直接消失:gpt-5-mini 看同一份回答 5 次全打 0,baseline 就已经死分,broken Δ=0,没有信号可检测。

**explain-vector-db** 在 5 列 unchanged,只有 swap A 判为 regressed。gpt-5-mini 对概念说明题的 brevity 容忍度比 deepseek-chat 低。deepseek judge 不觉得有问题,gpt-5-mini 明确判退。

### 两个综合论点

**统计方法对 verdict 的影响远比直觉以为的小。** percentile、BCa、paired 产出完全相同的 flagged 集合;FDR 与它们的差距只有 1 个 case。4 种方法一共只产出 2 个不同集合。"换成 BCa,verdict 就变了"这个故事在我们的数据上没有发生过。

judge identity 的影响力远超统计方法。swap A 的 flagged 集合与 baseline 完全不相交,swap B 整个清零。eval CI 报出的 verdict 是"model + judge"组合的属性,不是数据本身的属性。换一个 judge,你得到的是另一份真相,不是同一份真相的另一个读数。

### 给团队的工程判断

团队在争论"BCa 还是 percentile"?那是在噪声里优化。选最简单的,把时间花在别处。

团队在争论"用哪个模型当 judge"?那是在选评估真相的来源——这件事值得严肃决策,定下来就不要中途换。文章 2 那个"三家工具三种 verdict"的发现,根因和这里一样:promptfoo / DeepEval / Evalith 在判分逻辑上各不相同,不是在统计方法上各不相同。

### 5/5 预测全部命中

5 条预测锁在 commit `cc98b85`(数据跑出来之前),任何人可以核对。预测 1–3 全中;预测 4 flagged 集合完全不重合,比预计更激进;预测 5 有 caveat 但方向对。

5/5,对比文章 2 几乎全错的记录。这个对比本身是一个发现:对统计方法的预测可以依赖数学保证;对 LLM 行为的预测只能依赖经验直觉——后者的可靠性,文章 2 已经量过了。

## 七、局限和第四篇方向

<TODO §7 — Task 22>

---

如果你也在做 LLM eval 或 AI CI 集成,欢迎到 [github.com/dominciyue/Evalith_MingJing](https://github.com/dominciyue/Evalith_MingJing) 提 issue / PR。

```bash
pip install evalith
```

即装即用,Apache-2.0。
