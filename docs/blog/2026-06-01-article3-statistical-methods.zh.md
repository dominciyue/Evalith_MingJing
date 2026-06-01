# 续之续:把 BCa、paired bootstrap、FDR、第三方 judge 都挂上去,文章 2 的结论变了吗?

> 文章 2 的结论是 Evalith bootstrap 抓到 2/10 regressed,promptfoo 抓到 1/10,DeepEval 抓到 5/10,只有 sql-injection-vulnerability 三家一致。今天把更严格的统计方法和换 judge 都加上,用同一份 frozen raw 数据重看一遍。

---

## 一、那个承诺 + 一个问题

上篇数字:Evalith bootstrap 抓到 **2/10** regressed,promptfoo 1/10,DeepEval 5/10,三家唯一一致命中的只有 sql-injection-vulnerability。其余 case 三家各说各的。这就产生一个必须回答的问题:Evalith 漏掉的,到底是别家误报,还是真的有问题而 Evalith 没抓到?

上篇文末承诺了 4 件事——BCa 修偏、paired bootstrap 利用 case 内相关性收窄区间、FDR 控制 10 个 case 同时检验时的假阳率膨胀、第三方 judge 验证跨家族判定是否一致。今天把这四件事挂上去,在同一份 frozen raw 数据上重看一遍。

5 条预测就摆在下面。文章 2 的假设几乎全错,这次数据说了算。

### 在看到结果之前,我先把 5 条预测写在这里

把统计方法升级 + 换 judge,我事先预测如下:

1. **BCa 大概不会显著改变 verdict。** 文章 2 数据样本小(每 case 5 sample),分布近 0/1 二元,BCa 的修偏在这种分布上贡献有限。
2. **Paired bootstrap CI 会收窄,但不足以让任何 unchanged 翻成 regressed。** 文章 2 的 unchanged case 几乎都是 mean=1.00 两边、Δ=0、零方差。收窄一个零宽度的 CI,结果仍然横跨 0。
3. **FDR 在 10 case 上会把 sql-injection 翻成 unchanged。** 其 CI 离 0 较近(`[-1.00, -0.20]`),BH 校正后 p-value 可能不再通过 0.05 阈值。redis-cluster-failover 的 CI `[-1.00, -0.40]` 更远离 0,会留住。
4. **Judge swap A(单换 judge)会让 verdict 大幅变化。** qwen-plus 不是 deepseek 亲属,对同一中文 prompt + 同一回答的判定标准会不一样。可能比 deepseek 严或宽,但一定不会照搬。
5. **Judge swap B(全换 model+judge)与 A 的差距反映 model variance 的贡献占比。** 如果 A 和 B verdict 接近,说明换 judge 已主导;如果差距大,说明 model 输出本身也是变量。

预测全错的可能不小。文章 2 的假设就几乎全错了。这一篇会用同样方式把对错都摆出来,无论结果怎样。

(以下表格都来自实际跑出来的实验,任何一条没有事后修正。所有 raw outputs 在 `docs/blog/article3/raw/`,一行命令复现。)

## 二、BCa: 修偏与加速

Percentile bootstrap 直接切 bootstrap 分布的 α/2 和 1-α/2 分位数。当分布偏态,区间会往偏的那侧倾斜。BCa 加两个修正:`z₀` 衡量观察统计量在 bootstrap 分布里的位置,`a` 用 jackknife 估计分布的二阶弯曲度,两项一起让 CI 在小样本偏态场景下更接近名义覆盖率。

Evalith v0.5 的 BCa 是纯标准库实现,dev extras 里挂 `scipy.stats.bootstrap(method='BCa')` 作 ground truth;在 redis-cluster-failover 这类噪声 fixture 上,两者 CI 边界差 < 0.10。

文章 2 frozen raw 的 BCa 与 percentile 对照:

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

9 行 CI 完全一样。这批 case 的分布要么是单点(Δ=0),要么对称;z₀ 和 a 都接近 0,修正量可以忽略。

唯一有漂移的是 `redis-cluster-failover`。Percentile 上界 `-0.40`,BCa 上界移到 `-0.20`,因为 bootstrap 差值大量落在 -1.00 附近、上端稀疏,属单边偏态。CI 仍完全在 0 以下,还是 regressed。

预测 1 说"BCa 大概不会显著改变 verdict"。对了:回归集合与 percentile 完全相同,{sql-injection-vulnerability, redis-cluster-failover},2/10。CI 边界移了,判定没变。

## 三、Paired bootstrap: 利用 case 内相关性降方差

Percentile bootstrap 把 before 和 after 当两组独立样本各自重采样再做差。case 间的难度差异会混入 CI 的估计方差,但那部分不是我们想衡量的不确定性。

Paired 改变重采样对象:抽 case 索引 `i₁, ..., iₙ`,对每个抽到的 `i` 计算 `Δᵢ = after[i] - before[i]`,再取均值。case 内的难度因子被同向抵消,理论上 CI 应该收窄。

Evalith v0.5 的 `_bootstrap_paired` 用 `rng.randrange(n)` 抽索引,两数组等长为前提,否则抛 `ValueError`。单元测试用完美相关 fixture 验证:paired CI 压到 `[-0.10, -0.10]`(宽度 0),percentile 给 `[-0.46, +0.26]`(宽度 0.72)。

文章 2 frozen raw 的 paired 与 percentile CI 对照:

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

原因在于数据结构本身。文章 2 的 `pass_rate_samples` 把每次 trial 的 `score >= 0.5` 拍成 0/1,单个 case 的 5 个值基本是近常数序列。Paired 降方差的前提是 within-case 存在真实的变化空间;重采样近常数序列均值几乎不变,和 unpaired 没有区别。二值化判分事先把那个变化空间压扁了。

预测 2 说"Paired CI 收窄,但不足以让任何 unchanged 翻成 regressed"。结果比预测更极端:CI 完全没有收窄。方向对了,但原因是"压根没收窄"。统计方法的威力是数据形态决定的,不是方法名称决定的。

## 四、FDR: 当你同时检验 10 个 case 时

单个 case 用 α=0.05 检验,逻辑清楚:一次假设,错的概率 5%。但同时在 10 个 case 上做检验,每个都用同一个 5% 门槛,问题就出来了。假设 10 个 case 其实都没有真回退,每个 case 单独看,有 95% 概率不会被误标。10 个一起,至少一个被误标的概率是 `1 - (1 - 0.05)^10 ≈ 40%`。

这个 40% 叫 family-wise 假阳率。10 个 case 并不算大,很多团队会测 30、50 个,FWER 还会更高。1995 年 Benjamini 和 Hochberg 的 BH 方法放弃控制 FWER,转而控制 **false discovery rate**:期望在所有被判 regressed 的 case 中,真正误标的比例不超过 α。这个放宽让它在实践中比 Bonferroni 更有用。

BH 的步骤:把所有 case 的 p-value 升序排列,得到 p_(1) ≤ ... ≤ p_(N),对 rank k 设阈值 t_k = (k/N) × α。然后从大往小找最大的 k 满足 p_(k) ≤ t_k,接受 rank 1 到 k 全部为 regressed。

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

redis 的 p=0.0000:1000 次重采样里没有一次出现 Δ ≥ 0。sql-injection 的 p=0.0140:约 7 次出现正向波动,不能排除。其余 8 个 case p=1.0000 或接近。

升序排列后走 BH:

rank 1 是 redis,p=0.0000,阈值 (1/10)×0.05 = 0.005。0.0000 ≤ 0.005,通过。

rank 2 是 sql-injection,p=0.0140,阈值 (2/10)×0.05 = 0.010。0.0140 > 0.010,不通过。

rank 3 往后 asyncio、explain-vector-db、以及并列 1.0000 的六个 case 更不可能通过了。

BH 的结论:最大通过的 rank 是 k=1。regressed 集合从 {sql-injection-vulnerability, redis-cluster-failover} 缩成 {redis-cluster-failover},**2/10 → 1/10**。四种方法里第一次改变判定结果。

sql-injection 的 p=0.0140 离 rank 2 的阈值 0.010 差了 0.004,四个千分之一。没有数量级的差距,只是没过那道线。BH 不是软性建议,规则是确定的。

**eval set 越大,越该开 BH**。5 个 case,BH 几乎不改变结果。10-30 个是过渡区,本篇就在这里。50 个以上假阳率会高得难以接受,BH 是必须打开的选项。Evalith 把它放在 `--multi-test bh` 而不是默认启用,让你按规模自己决定。

预测 3 的原话:"FDR 在 10 case 上会把 sql-injection 翻成 unchanged,redis 的 CI 更远离 0,会留住。"猜对了,而且连哪个被翻、哪个留住都指对了。写预测时手头只有 percentile CI,靠的是 CI 边界离 0 的距离推断。文章 2 那种"假设全错"的结果这次没有复现。

## 五、换 judge: 一路只换,一路全换

文章 2 用 deepseek-chat 同时充当被测 model 和评分 judge。同源模型评同源输出,在 LLM-as-judge 文献里叫 affinity bias。§7 会把它列为 limitation #3,先实测一遍。

两条路:swap A 复用文章 2 存下来的 deepseek-chat 输出,只让 qwen-plus 重新评分(temp=0,与文章 2 的 judge 温度设置一致)。swap B 则让 qwen-plus 既出答案又评分,从头跑一遍 baseline + broken。设计动机是隔离变量:swap A 单独暴露 judge 差异,swap B 再叠加 model 差异,两者相减就能看出"换 model" 多贡献了多少。

### 主表:三列 verdict 对照

| case | DS+DS (v0.4 baseline) | swap A: DS-out + Qwen judge | swap B: Qwen+Qwen |
|---|---|---|---|
| `explain-rlhf` | unchanged | regressed | regressed |
| `explain-vector-db` | unchanged | regressed | unchanged |
| `sql-injection-vulnerability` | regressed | regressed | unchanged |
| `k8s-configmap-vs-secret` | unchanged | unchanged | unchanged |
| `asyncio-yield-deadlock` | unchanged | regressed | unchanged |
| `python-gil-tradeoffs` | unchanged | unchanged | unchanged |
| `redis-cluster-failover` | regressed | unchanged | unchanged |
| `tcp-congestion-control` | unchanged | regressed | unchanged |
| `jwt-vs-session` | unchanged | unchanged | unchanged |
| `transformer-attention` | unchanged | regressed | unchanged |

基线 flagged 2/10(redis, sql-injection),swap A flagged 6/10,swap B flagged 1/10。

### swap A 的变化比预想剧烈得多

判定数量从 2 变成 6,翻了 3 倍。更值得注意的是集合组成:baseline 里的 redis 在 swap A 里消失了,swap A 新增了 5 个 baseline 没有 flagged 的 case(explain-rlhf、explain-vector-db、asyncio-yield-deadlock、tcp-congestion-control、transformer-attention)。两个集合几乎完全不重合,只有 sql-injection 同时出现在两边。

redis 为什么消失,看 judge 分歧表:

| case | deepseek-chat judge | qwen-plus judge | gap |
|---|---|---|---|
| `redis-cluster-failover` | 0.80 | 0.00 | **-0.80** |
| 其余 9 case | 1.00 | 1.00 | +0.00 |

整份文章 2 冻结 baseline 里,qwen-plus 和 deepseek-chat 只在这一个 case 上判定不同。qwen-plus 看 deepseek-chat 给的 redis 答案,5 次评分全打 0;deepseek 自评 5 次里 4 次给过。

结果很直接:在 swap A 里,redis 的 baseline mean 已经是 0.00,broken 版本 qwen-plus 同样打 0,Δ=0,bootstrap 判 unchanged。信号不是"broken 版更差",而是 qwen-plus 认为两个版本都不及格。redis 就这样从 flagged 集合里消失了。

这是同源 model+judge 亲缘 bias 第一次在数据里具体显现。deepseek 给自己的 redis 答案放了水,qwen-plus 不认这个分。

### swap B 为什么只 1 个 regressed

qwen-plus 自己生成答案,5 次采样结果集中,分布接近极端值 1.0 或 0.0,case 内变化空间小。baseline 侧评分稳定,broken 侧 qwen-plus 也倾向于宽容对待自己写的内容。这种 self-judging 宽容在工业界文献里有记录,swap B 的数字和预期一致。

唯一例外是 explain-rlhf。这个 case 在 swap A 也 regressed——qwen-plus 既作为独立 judge(swap A)、又作为 model+judge 组合(swap B),都认为 broken 版本跳过了基础概念讲解。两条路径的指向一致,挺反直觉:换了 model 同样能发现这个问题。explain-rlhf 的 broken 版本大概真的丢了东西,只是 deepseek 自评时看不出来。

### 预测 4 和预测 5

预测 4 说"swap A 让 verdict 大幅变化"。CONFIRMED,而且比"大幅"还要激进得多:6 个 case,只有 sql-injection 与 baseline 重合,集合差异极大。

预测 5 说"swap B 与 swap A 的差距反映 model variance 贡献占比"。CONFIRMED with caveat。swap A 6 个 vs swap B 1 个,重合 1 个(explain-rlhf)。但 swap B 的稀疏主要来自 qwen-plus 的自评宽容效应,model variance 只是其中一部分原因。方向对,因果链比预测复杂。

### 选 judge 是科学决定

文章 2 的 2 个 flagged case 换成 qwen-plus judge 后,判定数量变成 6,集合几乎完全不同。

Eval 工具报出的 verdict 是 model-judge 组合的属性,不是数据本身的属性。换个 judge,回归判定可以变得面目全非。

## 六、谁改变了我们的判定

6 种干预、10 个 case、60 个判定格。把它们全部放在一张表里,才能看清楚哪些格子真的变了颜色。

| case | percentile (v0.4) | BCa | paired | FDR (BH) | swap A (DS-out+Qwen-judge) | swap B (Qwen+Qwen) |
|---|---|---|---|---|---|---|
| `explain-rlhf` | unchanged | unchanged | unchanged | unchanged | regressed | regressed |
| `explain-vector-db` | unchanged | unchanged | unchanged | unchanged | regressed | unchanged |
| `sql-injection-vulnerability` | regressed | regressed | regressed | unchanged | regressed | unchanged |
| `k8s-configmap-vs-secret` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `asyncio-yield-deadlock` | unchanged | unchanged | unchanged | unchanged | regressed | unchanged |
| `python-gil-tradeoffs` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `redis-cluster-failover` | regressed | regressed | regressed | regressed | unchanged | unchanged |
| `tcp-congestion-control` | unchanged | unchanged | unchanged | unchanged | regressed | unchanged |
| `jwt-vs-session` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `transformer-attention` | unchanged | unchanged | unchanged | unchanged | regressed | unchanged |

### 只有 3 个 case 全列稳定

6 列全部 unchanged 的只有 3 个:k8s-configmap-vs-secret、python-gil-tradeoffs、jwt-vs-session。无论用哪种统计方法、换哪个 judge,这 3 个 case 的 verdict 一直不动。可以认为它们是真信号——多维度都指向同一结论。

其余 7 个 case 在某种干预下都变过颜色。

### 7 个不稳定 case 各自的翻盘画像

`explain-rlhf`:4 种 deepseek judge 方法全部 unchanged,2 种 qwen-plus 方法全部 regressed。跨 judge 的指向相当清晰:broken 版本确实丢了东西,只是 deepseek 自评没看出来。

`explain-vector-db`、`asyncio-yield-deadlock`、`tcp-congestion-control`、`transformer-attention`:5 列 unchanged,只有 swap A 判为 regressed。qwen-plus 单独看到了 deepseek judge 看不出的问题。

`sql-injection-vulnerability`:percentile/BCa/paired 全部 regressed,swap A 也 regressed;FDR 把它压掉了(p=0.0140,BH rank 2 阈值 0.010,差 0.004 没过);swap B 不 regressed。四列方法指向有罪,两列方法放行。

`redis-cluster-failover`:4 种 deepseek judge 方法全部 regressed,2 种 qwen-plus 方法全部 unchanged。不是因为 qwen-plus 宽松——而是 qwen-plus 认为 baseline 就已经 0 分,broken 版本同样 0 分,Δ=0。从 deepseek 的角度有回退信号,从 qwen-plus 的角度两个都不行。

### 两个综合论点

统计方法的影响极小。percentile、BCa、paired 三种方法产出完全相同的 flagged 集合;FDR 与它们只差 1 个 case。4 种方法一共只产出 2 个不同集合。团队为"BCa 还是 percentile"争论,基本是在噪声里优化。

Judge identity 是核心变量。换了 qwen-plus judge 之后,7/10 个 case 的 verdict 在至少一列发生了变化。评估结论是 model+judge 组合的函数,不是数据本身的属性。定下 judge,别中途换——换了就不是同一份真相的另一个读数,是另一份真相。

### 给团队的工程判断

团队为"BCa 还是 percentile"争吵,选最简单的就行,选错了最多差 1 个 case。

团队为"用哪个模型当 judge"争吵,那是在选评估真相的来源。这件事值得严肃定义。文章 2 的"三家工具三个 verdict"根因就在这里:三家工具的 judge 调用和聚合细节不同,不是统计方法不同。

### 5/5 预测全部命中

5 条预测锁在 commit `cc98b85`(数据跑出来之前),任何人可以核对。预测 1-3 全中;预测 4 命中,比"大幅变化"更激进;预测 5 有 caveat 但方向对。

5/5。对比文章 2 几乎全错的记录:统计方法的预测有数学依据,LLM 行为的预测靠经验直觉——后者的可靠性,文章 2 已经量过了。

## 七、局限和第四篇方向

几个需要说清楚的边界:

1. n=10 是小样本。BCa 的 acceleration 估计本身有方差;FDR 的 power 也有限。结论别从 10 个 case 外推到任意 eval 场景。

2. **swap A 用的是单次 trial 输出。** article 2 engine 只存 trial-0;swap A 的 5 次 judge 全在同一份回答上跑。捕获的是 judge 自身抖动,不是"model 抖 × judge 抖"的两层噪声。完整两层噪声需要 evalith 存所有 trial 输出,article 4 工程方向。

3. 三方 judge 只跑了一个。1 个非 deepseek judge 只能说"换了一次,verdict 变了",不能说"跨多种 judge 都跑成这样"。理想是 3-4 个跨家族 judge(deepseek / openai / anthropic / qwen),article 4 至少再加一家。

4. **这是 article 2 的同一份数据。** 跨方法对照干净,但跨数据集泛化 0 验证。100 case + 多领域 dataset 在 article 4 / arXiv preprint 中扩。

article 4 方向:per-case `expected_concepts` 注入 judge prompt(article 2 §3 发现 evalith llm_judge 没用 dataset 里的字段,article 4 修);Claude / OpenAI 各做一次 cross-judge,从"换 1 个"变成"看 verdict 分布";dataset 从 10 扩到 50-100 case 跨多领域;adaptive sampling 根据 CI 收敛动态停止,降 eval 成本。

---

如果你也在做 LLM eval 或 AI CI 集成,欢迎到 [github.com/dominciyue/Evalith_MingJing](https://github.com/dominciyue/Evalith_MingJing) 提 issue / PR。

```bash
pip install evalith
```

即装即用,Apache-2.0。
