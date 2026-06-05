# 续之续之续:换三个 judge、三个模型、五个领域,judge 的分歧到底藏在哪?

> 文章 3 的结论:换一个 judge,回归判定可以面目全非。但那是 n=10、单数据集、只换了一个第三方 judge。这一篇把数据集扩到 30 case × 5 领域,judge 从 1 个第三方加到 3 个跨家族(deepseek / qwen / glm),被测 model 也换 3 家。问题很简单:judge 之间的分歧,在更大、更杂的数据上还在不在?如果在,它到底藏在哪里?

---

## 一、上一篇留下的问题

文章 3 跑完,有一个结论硬得没法绕开:eval 工具报出的 verdict,是 model+judge 组合的属性,不是数据本身的属性。换一个 judge,10 个 case 里 7 个的判定在某一列翻过盘。

但那份实验有三个一眼可见的窟窿,文章 3 自己在 §7 列了出来:

- n=10,小样本,而且全是后端工程类的概念题,领域单一。
- 只换了一个第三方 judge(qwen-plus)。换一次能说"换了就变",换不出"跨多种 judge 都怎么变"。
- 单数据集,跨数据集泛化零验证。

这一篇就来堵这三个窟窿。数据集从 10 扩到 30,横跨 5 个领域;judge 从 deepseek 自评 + 一个 qwen,扩到 deepseek / qwen / glm 三个跨家族 judge;被测 model 也是这三家,各跑 baseline 和 broken 两版。

我事先有几个预期,下面照例写在前面。但得先说清楚一件事:这次和文章 3 不一样,数据是先跑完、我再回头复盘的,不是把预测锁进 commit 再开跑。文章 3 那套"5 条预测锁死、事后逐条对账"的玩法,前提是预测写下来的时候我还不知道答案。这次做不到了:多模型多 judge 的实验跑一轮要几个小时,我是看着数字一点点出来的。所以下面这些与其叫"预测",不如叫"我原本以为会怎样,实际怎样",诚实归类成 post-hoc 复盘。能对账的地方我标对,猜错的地方我标错,但不假装它是盲测。

我原本以为:

1. **三个 judge 的总体严格程度会有差异,但不会差太多。** 文章 3 里 qwen 比 deepseek 严,我以为加进 glm 之后三家会落在一个不太宽的区间里。
2. **回归(baseline→broken)的方向三家 judge 都会抓到。** broken 版本是被显式削弱过的,再宽容的 judge 也该看出退化。
3. **换 model 比换 judge 影响小。** 文章 3 的 swap A/B 已经透出这个味道:judge 换了 verdict 大变,model 换了变化没那么剧烈。
4. **judge 分歧大概率是均匀的噪声。** 这条我后来发现错得最离谱(见 §3)。

(以下所有表格都来自实跑,raw outputs 在 `docs/blog/article4/raw/`,`multi_compare.py` 一条命令复现。)

## 二、数据集、被测设定,以及这期间 Evalith 的两处改动

### 数据集:30 case,5 领域,benchmark 子集 + 自造

文章 2/3 的 10 个 case 全是后端概念题。这次按领域分层,30 个 case 均匀落在 5 个领域,每域 6 个:

| 领域 | 来源 | 例子 |
|---|---|---|
| `code` | HumanEval 子集 | 补全 `double_the_difference`、`concatenate` 等函数 |
| `math` | GSM8K 子集 | 小学应用题,需要分步算 |
| `knowledge` | MMLU 子集 | machine_learning / computer_security 多选与问答 |
| `safety` | TruthfulQA 子集 | 容易诱发常见误解的事实题 |
| `concept-explanation` | 自造 | quantization、LoRA、bloom filter、JWT 吊销等概念解释 |

前四类是公开 benchmark 抽样(seed=42 固定),最后一类仿照前面的风格自造,补齐"需要主观判断质量"的开放式解释题。完整构造脚本在 `build_dataset.py`(65 case 全集)和 `build_subset.py`(分层抽 30 case 子集)。

每个 case 都带 `expected_concepts` 字段,声明这道题的回答必须覆盖哪些核心概念点。code 题是"函数正确实现 / 返回类型正确 / 覆盖 docstring 所有情形",concept 题是各自的概念清单。这个字段不是摆设,它直接喂进了 judge,下面讲。

### baseline vs broken 怎么造

每个 model 跑两版。baseline 是正常 prompt;broken 在 prompt 前面加一句削弱前缀:

> Be very concise. Answer in 1-2 short sentences. Skip explanations and concept enumeration:

逼模型把答案压成一两句、跳过概念讲解。这是人为制造的"质量回退",目的是给 eval 工具一个确定有退化的信号,看它抓不抓得到。每 case 采样 5 次,temperature=1.0,让 LLM 的自然抖动也进来。

### 这期间 Evalith 做了两处优化(v0.5 → v0.6)

文章 3 发布时 Evalith 是 v0.5。写这一篇之前,工具加了两个东西,正好都是为这次实验铺路的。

**第一,`llm_judge` 支持 per-case `expected_concepts`。** 文章 2 §3 当时就发现一个问题:Evalith 的 llm_judge 只用一句全局 criteria 评所有 case,完全没用上 dataset 里每个 case 自带的核心概念字段。judge 评一道 code 题和评一道 safety 题,看到的标准是同一句话,它根本不知道"这道题具体该覆盖什么"。v0.6 把这个补上了:case 若声明了 `expected_concepts`,judge prompt 会自动把它拼成一段"核心概念清单(回答须覆盖)"插进评分标准里。

```
评判标准: {全局 criteria}

核心概念清单(回答须覆盖):
- 函数 double_the_difference 的正确实现
- 返回值类型正确
- 处理 docstring 中描述的所有情形
```

这正是这一篇能做"分领域评分"的前提:每个 case 的 judge 有了 per-case 的锚点,而不是用一把尺子量所有题。完全向后兼容:不写这个字段的 case,行为和 v0.5 字节一致。

**第二,adaptive sampling(动态停止采样)。** 固定 `--samples 5` 对所有 case 一刀切:三家 judge 都打 1.00 的稳定 case,跑 5 次纯属烧 API 钱;judge 之间撕裂的噪声 case,5 次可能还不够。v0.6 加了 `--adaptive`,每个 case 跑到它 pass-rate 的 bootstrap CI 宽度低于阈值就停(或到 `--max-samples` 上限):

```bash
evalith run config.yaml --adaptive --min-samples 2 --max-samples 10 --ci-tolerance 0.2
```

稳定 case 早停省钱,噪声 case 自动跑满保信号。放到这次的数据里看特别贴切:safety 域三家 judge 全 1.00,正是该早停的;code 域 judge 互相撕,正是该跑满的。这是 opt-in 选项,默认行为不变。

(为了让这一篇和文章 2/3 跨篇可比,正式实验里我还是用固定 `--samples 5`,没开 adaptive,变量只留一个。adaptive 是工具能力的展示,不混进对照实验。)

## 三、Cross-judge:同一份输出,三个 judge 各打各的

第一个问题,也是文章 3 的核心问题在更大数据上的重演:把被测 model 固定成 deepseek,只换 judge,verdict 会差多少?

### 总体:qwen 是个严苛的异类

deepseek 的 baseline 输出,分别让三个 judge 评:

| judge | 总体 pass-rate |
|---|---|
| deepseek(自评) | 0.840 |
| glm | 0.827 |
| qwen | 0.460 |

deepseek 自评和 glm 几乎一样(0.84 / 0.83),qwen 直接腰斩到 0.46。预期 1 说"三家差异不会太大",错了,qwen 和另外两家差了将近一倍。

但这个总体数字会骗人。真正有意思的不是"qwen 严",而是 qwen 严在哪里。

### 把分歧拆到领域:它根本不是均匀的

这是这一篇相对文章 3 最大的增量。把上面那个总体 pass-rate 按领域拆开:

| 领域 | deepseek judge | qwen judge | glm judge |
|---|---|---|---|
| `code` | 0.80 | 0.00 | 0.93 |
| `knowledge` | 0.83 | 0.30 | 0.83 |
| `math` | 1.00 | 0.53 | 0.80 |
| `concept-explanation` | 0.57 | 0.57 | 0.57 |
| `safety` | 1.00 | 0.90 | 1.00 |

qwen 的"严苛"几乎全部砸在 `code` 上:deepseek 写的代码,deepseek 自评 0.80、glm 给 0.93,qwen 直接打 0.00,一道都不过。往下 knowledge(0.83→0.30)、math(1.00→0.53)也明显被压,但没归零。

而到了 `concept-explanation`,三家完全一致,都是 0.57。`safety` 三家几乎一致(0.90~1.00)。

预期 4 我原以为"judge 分歧是均匀的噪声",这是错得最彻底的一条。分歧不是噪声,它有领域结构。客观可判的题(safety 是事实对错、concept-explanation 这批题的概念点相对明确)三家 judge 高度一致。需要主观判断"这段代码/这个回答够不够好"的题,judge 之间撕裂得最厉害,code 是重灾区。

为什么是 code?翻 qwen 在 code 域的逐条评分理由,它对代码的"可运行性 / 边界处理 / 风格"卡得极死,deepseek 给的代码答案在它眼里普遍"没覆盖全 docstring 描述的情形"。同一段代码,glm 觉得能跑就行,qwen 觉得不够严谨。这不是谁对谁错,是两个 judge 对"code 题的及格线"理解不同。而 safety 题的及格线("有没有说出那个正确事实")客观得多,judge 之间没有发挥空间,自然就一致了。

### 逐 case 看:13 个大分歧,15 个共识

把 30 个 case 按"三家 judge 之间的最大分差(spread)"排序,极端的几个:

| case | 领域 | ds | qw | glm | spread |
|---|---|---|---|---|---|
| `mmlu-machine_learning-93` | knowledge | 1.00 | 0.00 | 1.00 | 1.00 |
| `he-humaneval-70` | code | 1.00 | 0.00 | 1.00 | 1.00 |
| `he-humaneval-28` | code | 0.40 | 0.00 | 1.00 | 1.00 |
| `gsm-1232` | math | 1.00 | 0.00 | 0.40 | 1.00 |
| `he-humaneval-62` | code | 0.80 | 0.00 | 1.00 | 1.00 |
| … | | | | | |
| `tq-159` (safety) | safety | 1.00 | 1.00 | 1.00 | 0.00 |
| `gsm-13` (math) | math | 1.00 | 1.00 | 1.00 | 0.00 |
| `self-bloom-filter` | concept | 1.00 | 1.00 | 1.00 | 0.00 |

30 个 case 里,13 个 spread ≥ 0.6(三家严重分歧),15 个 spread ≤ 0.2(三家基本共识)。分歧和共识几乎对半,而且哪些落在哪边,跟领域强相关:大分歧集中在 code 和部分 knowledge/math,共识集中在 safety 和已经定型的 concept 题。

### Cohen's kappa:数字也证实 qwen 是孤立的

把每个 (case, trial) 的 pass/fail 当二元标签,算三家 judge 两两之间的 Cohen's kappa(衡量超出随机一致的吻合度,1 是完全一致,0 是只有随机水平):

| | baseline | broken |
|---|---|---|
| ds vs glm | 0.568 | 0.252 |
| ds vs qw | 0.254 | 0.027 |
| qw vs glm | 0.252 | 0.019 |

baseline 上,deepseek 和 glm 是中等一致(κ=0.57),qwen 对这两家都只有 0.25 上下,在三角里是孤立的那个顶点。

broken 上 kappa 集体崩塌(ds-qw 掉到 0.027)。但这里要小心,低 kappa 不等于"判得更乱"。broken 版本里 qwen 几乎把所有 case 都判 fail(它本来就严,面对被削弱的答案更是一个不放过),标签退化成近乎全 0 的常数序列。kappa 的分母里有个"期望一致率",当一方接近常数,期望一致率本身就拉满,kappa 公式会塌向 0,哪怕两个 judge 其实都判了 fail、表面一致率很高。所以 broken 的 kappa 低,反映的是 qwen 判定退化成常数、kappa 这个指标在这种分布下本身失灵,不是 judge 突然乱判。这是用 kappa 评 judge 一致性时一个真实的坑,值得单独记一笔。

## 四、Cross-model:换被测模型,verdict 方向稳不稳?

第二个问题反过来:judge 固定成 deepseek,换被测 model,baseline→broken 的回归还抓得到吗?

### 三家 model 全部被抓到回归

| model | baseline | broken | Δ | 判定 |
|---|---|---|---|---|
| deepseek | 0.840 | 0.247 | −0.593 | REGRESSION |
| qwen | 0.927 | 0.407 | −0.520 | REGRESSION |
| glm | 0.747 | 0.087 | −0.660 | REGRESSION |

三家全部、毫无悬念地被判回归,Δ 在 −0.52 到 −0.66 之间。预期 2 说"回归方向三家都会抓到",对了。

### 拆到领域:回归在每个领域都是负的

再把 cross-model 的 Δ 按领域摊开(judge 都是 deepseek):

| 领域 | deepseek Δ | qwen Δ | glm Δ |
|---|---|---|---|
| `code` | −0.60 | −0.63 | −0.40 |
| `math` | −0.80 | −0.33 | −1.00 |
| `knowledge` | −0.47 | −0.63 | −0.70 |
| `concept-explanation` | −0.17 | −0.13 | −0.27 |
| `safety` | −0.93 | −0.87 | −0.93 |

15 个格子(3 model × 5 领域)全是负的,没有一个例外。回归是真实存在、跨模型跨领域稳健的。退化最浅的是 concept-explanation(Δ −0.13 ~ −0.27):这批开放式解释题 baseline 本来就只有 0.5 上下,削弱前缀让它"少说几句",但概念点还在,judge 没把它判到地板。safety 和 math 这种"要么对要么错"的题,削弱前缀一逼简短就直接崩,Δ 接近 −1。

## 五、谁是变量:judge 改 level,model 不改方向

把 §3 和 §4 并排,结论很对称:

- §4 换 model:三家 model、五个领域,15/15 全部判回归,方向完全一致。换被测模型,不改变"有没有回归"这个判定。
- §3 换 judge:同一份 deepseek 输出,总体 pass-rate 从 0.84(ds)到 0.46(qw)差了一倍,code 域从 0.80 到 0.00 直接归零。换 judge,把同一份答案的"分数水平"整个搬家。

文章 3 的结论是"换 judge,verdict 面目全非"。这一篇在更大数据上给它加了两个限定词,让它更准确:

1. **换 judge 主要改的是 level(分数高低),不是 direction(有没有退化)。** 三家 judge 总体严格程度差一倍,但 baseline→broken 的 Δ 三家都是大负数,方向一致(judge=ds −0.59、qw −0.45、glm −0.54)。也就是说:你想知道"这版比上版差了没",换哪个 judge 答案都是"差了";你想知道"这版到底有多好,能不能上线",换个 judge 数字能差一倍,这才是真正危险的地方。

2. **judge 分歧是 domain-structured 的,不是均匀噪声。** 这是文章 3 没看到、这一篇靠分领域数据才挖出来的。code、需要主观质量判断的题,是 judge 撕裂的重灾区;safety、事实对错明确的题,是 judge 共识区。换句话说:你的 eval set 越偏向"开放式、要判质量"的题,换 judge 带来的数字漂移就越大;越偏向"客观对错"的题,judge 选谁都差不多。

### 给团队的工程判断

文章 3 说"定下 judge,别中途换"。这一篇能说得更细:

- 如果你的 eval 主要是回归门禁("这个 PR 有没有让质量掉下去"),judge 选谁没那么要命,三家都判得出方向。`--fail-on-regression` 这种相对门槛对 judge 选择不敏感。
- 如果你的 eval 要给绝对分数("我们模型在这个 benchmark 上有 0.8 还是 0.46"),judge identity 是头等大事,尤其当你的题偏 code / 开放式。换个 judge,同一份答案能从"优秀"判成"勉强及格"。`--fail-under 0.9` 这种绝对门槛对 judge 选择极度敏感,定 judge 要像定 benchmark 一样慎重。

一句话:相对门禁看方向,judge 可换;绝对评分看水平,judge 必须钉死,而且要钉一个你信得过它评那个领域的 judge。

## 六、原本以为 vs 实际:post-hoc 对账

照例把开头几条复盘一遍,但这次诚实标注:这是 post-hoc,不是盲测。

1. "三家 judge 总体差异不会太大":错。qwen 0.46 对另两家 0.84/0.83,差近一倍。我低估了 judge 之间的离散度。
2. "回归方向三家都抓得到":对。§4 里 15/15 全负,毫无例外。
3. "换 model 比换 judge 影响小":对,而且 §3/§4 的对称性把它讲清楚了,model 不改方向,judge 改 level。
4. "judge 分歧是均匀噪声":错得最离谱,而这条错出了全篇最有价值的发现。分歧有领域结构,code 撕裂、safety 共识。如果这条当初猜对了,我反而挖不到 domain-structured 这层。

4 条里 2 对 2 错。和文章 3 那种"5/5 全中"比,这一篇的对账成色差一些。但一部分原因正是它不是盲测,我没法假装第 4 条是事前写下的。把对错如实摆出来,比凑一个漂亮的命中率重要。

## 七、局限和下一步

几个必须说清的边界:

1. **三家 judge 全是中文国产模型(deepseek / qwen / glm)。** "跨模型家族"这个说法成立,它们是三个独立训练的模型,不是同一家的不同尺寸。但"跨地域 / 跨文化 judge"这次没验证:三家都主要在中文语料上训练,面对中文 prompt + 中文评分标准,它们共享的偏置可能比想象的多。一个英文为主的 judge(GPT / Claude 一类)会不会在 safety、concept 这些"共识区"也分裂?这次答不了。坦白讲,原本是想加一个英文家族的 judge 的,因为支付渠道的现实问题没跑成,留作下一步。

2. **broken 是人造前缀,不是自然回归。** 真实的质量回退是改 prompt、换模型版本带来的细微漂移,不是一句"答简短点"的暴力削弱。本篇的 Δ 普遍很大(−0.5 以上),是因为退化信号被人为放大了。自然回归会微妙得多,那种场景下 §3 的 level 漂移会不会把一个真回归淹掉,值得专门做一次。

3. **n=30 仍是小样本。** 每域只有 6 个 case,domain-structured 这个发现的强度受限于此。code 撕裂、safety 共识的对比很清楚,但"6 个 code 题里 qwen 全打 0"外推到"所有 code 题"还需要更大的样本。65 case 全集(`qa.large.yaml`)已经在仓库里,下一步在它上面重跑,把每域样本翻倍。

4. **concept-explanation 的 baseline 偏低(0.57),三家一致。** 这批自造题可能本身就偏难,或者 `expected_concepts` 清单订得偏苛。三家 judge 一致地给 0.57,既可能是"题确实难",也可能是"清单写法让三家都卡同一个点"。自造数据的标注质量需要再校。

下一步方向:把实验搬到 65 case 全集;想办法加一个非中文家族的 judge,把"跨地域"这一格补上;adaptive sampling 在真实多领域数据上的省钱效果做一次量化(稳定的 safety 域早停能省多少 call);以及把文章 2/3/4 三篇的方法和数据整理成一份英文 preprint。

---

如果你也在做 LLM eval 或 AI CI 集成,欢迎到 [github.com/dominciyue/Evalith_MingJing](https://github.com/dominciyue/Evalith_MingJing) 提 issue / PR。

```bash
pip install evalith
```

即装即用,Apache-2.0。raw 数据和 `multi_compare.py` 都在 `docs/blog/article4/`,一条命令复现全部表格。
