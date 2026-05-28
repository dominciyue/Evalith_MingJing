# AI 回归测试需要统计显著性: 用 bootstrap CI 抗 LLM 噪声

> 你说"这次改 prompt 把 case X 搞挂了",你怎么知道不是 LLM 自己抖的? 这是一份 Python 实现 + 数学推导,以及为什么主流 AI eval 工具今天都在裸奔。

---

## 一、问题: 你看到的"回退",一半可能是噪声

设想你给 LLM 应用写了一个简单的 CI gate: 每次 PR 触发都跑一遍 eval 测试集,通过率比 baseline 下降就 fail PR。今天 baseline 是 0.85,新 PR 一跑只剩 0.80。回退了 5 个点,合理。

但**没改任何代码**重新触发一次,通过率变成 0.83。再跑一次 0.78。

你的 eval 通过率在 ±5 个点的区间里随机游走。原因不止一个:

- temperature 即便设为 0,**部分 provider 也不保证完全确定** —— KV cache 命中差异、batch 大小不同、硬件层 fp 舍入,都会让同一 prompt 产生不同 output。
- `llm_judge` 类 scorer 本身就是 LLM,自带噪声。
- 边界 case 在 `contains` / `regex` scorer 上也可能因为 output 措辞细微变化而 flip。

**这意味着:** 一个真实的 prompt 改动把通过率从 0.85 拉到 0.80,和一次纯噪声把通过率从 0.85 抖到 0.80,**单次数字上不可分**。

而今天主流 AI eval 工具 —— Promptfoo、DeepEval、Ragas、OpenAI Evals —— 在这个问题上几乎都是裸奔: 直接点对点比较两个数字,不区分信号和噪声。

CI gate 因此处在一个尴尬位置:

- 调严了 → 每隔几个 PR 就因为噪声 fail,工程师开始 ignore 它。
- 调松了 → 真回退漏过去,gate 变成摆设。

唯一靠谱的出路是引入**统计显著性**: 给"回退"这个判定挂上一个置信区间,只有当 CI 整体在 0 以下,才算真回退。

## 二、统计直觉: 把每个 case 看成伯努利试验

每条用例的评分本质上是一个 0/1 二元变量(`passed=True/False`)。对同一 case 重复跑 N 次,得到的就是伯努利序列:

```
case_id="customer-support-1"  ->  [1, 1, 0, 1, 1, 0, 1, 1]  (pass rate ≈ 0.75)
```

我们真正想知道的不是"这个 case 真实的 pass rate 是 0.75 还是 0.85",而是:

> **after 的 pass rate,是否显著低于 before?**

记 `p_b`、`p_a` 为 before / after 两次跑的(未知的)真实 pass rate,我们要给

`Δ = p_a − p_b`

估一个 95% 置信区间。**CI 整体 < 0 才判 regressed**;CI 跨过 0 就保守地说"噪声范围内,放行"。

## 三、为什么是 bootstrap

直觉上你会想用 z 检验、t 检验或 Wilson 区间,但有几个工程问题:

1. **每个 case 的 N 通常很小**(3-10 次,因为每次调用要钱),正态近似不准。
2. **pass rate 是 0/1 binary**,严格意义上不连续。
3. 我们要的是**对 Δ 的 CI**,不是单边均值 CI。
4. **不想做强配对**: 同一 case 在 before / after 的 N 次试验之间没有天然 pairing(每次 sample 独立)。

**百分位 bootstrap (percentile bootstrap)** 几乎是为此而生:

- 不假设任何分布
- 对小样本鲁棒(N=3 也能跑)
- 可以对任何统计量(差值、ratio、median)直接算 CI
- **实现 < 20 行 Python,不依赖 scipy**

算法本身是教科书级的简单:

```
对 b = 1, 2, ..., B (典型 B = 1000):
    1. 从 before 样本里有放回采样 |before| 次 -> before*
    2. 从 after  样本里有放回采样 |after|  次 -> after*
    3. Δ_b* = mean(after*) - mean(before*)
排序 {Δ_1*, ..., Δ_B*}
95% CI = [Δ_{α/2}*, Δ_{1-α/2}*]
```

直觉是: 用观测样本本身近似真实分布,反复"自助"重采样,看看 Δ 在哪个区间内浮动。

## 四、Evalith 里的实现

把上面的算法实现成生产可用代码:

```python
import random

def bootstrap_diff_ci(before: list[float], after: list[float], *,
                      n_resamples: int = 1000, alpha: float = 0.05,
                      seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI on (mean(after) - mean(before))."""
    rng = random.Random(seed)
    n_b, n_a = len(before), len(after)
    diffs = []
    for _ in range(n_resamples):
        b_mean = sum(rng.choice(before) for _ in range(n_b)) / n_b
        a_mean = sum(rng.choice(after) for _ in range(n_a)) / n_a
        diffs.append(a_mean - b_mean)
    diffs.sort()
    lo_idx = int(n_resamples * alpha / 2)
    hi_idx = min(n_resamples - 1, int(n_resamples * (1 - alpha / 2)))
    return diffs[lo_idx], diffs[hi_idx]
```

几个工程决定值得说明:

- **seeded RNG**: bootstrap 本身是随机算法,但**同一份 input 必须给同一份 CI**。否则每次 CI 检查 CI 在抖,工程师没法 reproduce、没法信任。`seed=0` 保证确定性。
- **不依赖 numpy / scipy**: evalith 核心 dep 只有 `pydantic + pyyaml + typer`。这段统计只要 stdlib `random` 就够了,把依赖图保持小,便于 CI 容器里部署。
- **单样本快路径**: 当 `samples=1` 时根本不走 bootstrap,直接点对点比较。N ≥ 2 才启用 CI。**老用户零额外开销**。

在 diff 引擎里集成:

```python
b_samples = br.pass_rate_samples or [case_score(br)]
a_samples = ar.pass_rate_samples or [case_score(ar)]
if max(len(b_samples), len(a_samples)) >= 2:
    lo, hi = bootstrap_diff_ci(b_samples, a_samples)
    if hi < -tol:
        status = "regressed"
    elif lo > tol:
        status = "improved"
    else:
        status = "unchanged"   # CI 跨过 0 -> 没信心说回退,放行
```

用户层 API 只多一个 flag:

```bash
evalith run examples/eval.yaml --samples 5 --out current.json
evalith diff baseline.json current.json --fail-on-regression
```

差异化非常具体: **只有当 CI 整体低于 0,CI gate 才 fail PR**。噪声彻底过滤。

## 五、实验

### 例子 1 — 真噪声不应被判 regressed

```
before pass rates: [1, 0, 1, 0, 1]   (mean 0.6, 高方差)
after  pass rates: [0, 1, 0, 1, 0]   (mean 0.4, 高方差)
```

- 朴素点对点 diff: `0.6 → 0.4`,判 regressed,fail PR。**误报**。
- Evalith bootstrap: 95% CI ≈ `[−0.6, +0.4]`,**跨过 0**,status = `unchanged`,放行。**正确**。

### 例子 2 — 真回退仍被精确捕获

```
before: [1, 1, 1, 1, 1, 1, 1, 1]   (mean 1.0)
after:  [0, 0, 0, 0, 0, 0, 0, 0]   (mean 0.0)
```

Bootstrap CI: `[−1.00, −1.00]`,**完全在 0 以下**,status = `regressed`,fail PR。**正确**。

两个极端都得到了符合直觉的判定。这正是 CI gate 想要的行为: **见噪声而不动,见信号即抓**。

这两个极端都得到了符合直觉的判定: **见噪声不动,见信号即抓**。

### 真实 DeepSeek 上的对照实验

把 `evalith` 接到 **DeepSeek-V3** (`deepseek-chat`, temperature 0.0) 上,在一个 3 case 的中文事实问答集(中国首都 / 一道算术 / 太阳升起方向)上跑两组对照,每个 case `--samples 5`:

**A 组(噪声基线)** —— 同一个 config,**不改一个字**,跑两次,diff。预期: 没有任何 case 应判回退。

| case | mean before | mean after | bootstrap CI on Δ | status |
|---|---|---|---|---|
| `arithmetic` | 1.00 | 1.00 | [+0.00, +0.00] | unchanged |
| `capital-cn` | 1.00 | 1.00 | [+0.00, +0.00] | unchanged |
| `sunrise`    | 1.00 | 1.00 | [+0.00, +0.00] | unchanged |

**B 组(注入真实回退)** —— baseline vs 故意把 `prompt_template` 改成 `"Answer only in English, even if the question is Chinese: {{input}}"`,强制模型用英文回答 —— 中文 `expected`(如 "北京")会被 `contains` scorer 漏掉。

| case | mean before | mean after | bootstrap CI on Δ | status |
|---|---|---|---|---|
| `arithmetic` | 1.00 | 1.00 | [+0.00, +0.00] | unchanged |
| `capital-cn` | 1.00 | 0.50 | [−0.50, −0.50] | **regressed** |
| `sunrise`    | 1.00 | 1.00 | [+0.00, +0.00] | unchanged |

三点观察:

1. **A 组零假阳率** —— 同配置重跑没有任何 case 被判回退。这是工程师最关心的 "gate 不误伤" 性质。
2. **B 组只命中真正坏掉的 case** —— `capital-cn`: 输出从 "北京" 变成 "Beijing",`contains` 失分而 `llm_judge` 还给分,pass rate 从 1.0 跌到 0.5,CI `[−0.50, −0.50]` 完全在零下,**精确判为 regressed**;`arithmetic` 没受影响("42" 跨语言相同);`sunrise` 也没受影响 —— DeepSeek 这次没听 "只用英文" 这条指令,仍然输出了 "东",所以 contains 仍通过。这反过来暴露了我们 prompt 干预的不完美,但 **bootstrap 没把这种"实际没变化"误报为变化**,正是想要的。
3. **需要诚实承认的局限** —— temperature 0 + 简单事实题让 DeepSeek 这次表现极其确定,5 次采样里几乎没观测到 LLM 抖动(所以 CI 都很窄)。**bootstrap 的"噪声免疫"优势在 `llm_judge` 评分器、高 temperature、或更接近边界的开放式问答里才会真正显著**,下一篇文章会专门做这类实验。

> 实验完全可复现 —— 设 `DEEPSEEK_API_KEY` 后一行命令:
> ```bash
> DEEPSEEK_API_KEY=sk-... bash docs/blog/experiment.sh
> ```
> 脚本会输出上面两张 markdown 表。

## 六、局限和下一步

bootstrap CI 不是银弹,几个已知方向:

1. **BCa (bias-corrected and accelerated) bootstrap**: 样本分布偏态时,百分位 CI 略有偏差,BCa 修正之。evalith 当前用更简单的 percentile,后续会加。
2. **配对 vs 独立**: 同一组 case 重复跑时,理论上可以做 paired bootstrap (resample case index, not result),方差更小。当前是 unpaired,简化但保守。
3. **Sequential testing**: 现在是固定 N 次采样。理想是动态判断 —— 不确定就再多跑几次,确信就早停。Active CI 的研究方向之一。
4. **Multiple-comparison correction**: 50 个 case 同时检验,family-wise 假阳率会膨胀。要么 Bonferroni 校正 alpha,要么 BH 控制 FDR。下一步要加。
5. **ROPE (Region of Practical Equivalence)**: 某些场景下 0.001 的回退不值得 gate。让 `tol` 用户可配,evalith 已支持。

## 七、结语

LLM 应用 CI 化的趋势是确定的,但**点对点比较的 CI gate 在统计上就是错的** —— 要么过敏,要么过钝,两种结局工程师都没法信任。引入哪怕最朴素的 bootstrap CI,假阳率立刻下降,gate 才能真正成为研发流程的一环。

工业界的 LLM 测试栈急需更深的统计基础。这只是开始。

---

如果你也在做 LLM eval 或 AI CI 集成,欢迎到 **[github.com/dominciyue/Evalith_MingJing](https://github.com/dominciyue/Evalith_MingJing)** 提 issue / PR。

```bash
pip install evalith
```

即装即用,Apache-2.0。
