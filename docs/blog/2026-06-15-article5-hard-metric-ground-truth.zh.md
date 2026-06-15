# 当 judge 们吵起来时，别再投票了：用执行结果给 code eval 一个 ground truth

> 文章 4 最刺眼的数字不是某个模型掉了多少分，而是 code 域里三个 judge 几乎各说各话。v0.7 的共识面板能告诉我们“这里有分歧”，但它还不能回答“谁对”。这篇补的是后半句：能跑单测的题，不该继续让 LLM 投票。

---

## 一、文章 4 留下的那个洞

上一篇把数据集扩到 30 case、5 个领域、3 个 judge、3 个被测模型。结果比我预想得更有结构：judge 分歧不是均匀噪声，它集中在某些领域。

最典型的是 code。固定被测模型为 deepseek，只换 judge，看同一批 HumanEval 输出，三个 judge 给出的 code 域均分是：

| judge | code 域 pass-rate |
|---|---:|
| deepseek | 0.80 |
| qwen | 0.00 |
| glm | 0.93 |

逐题看更扎眼：

| case | deepseek judge | qwen judge | glm judge | spread |
|---|---:|---:|---:|---:|
| `he-humaneval-151` | 1.00 | 0.00 | 1.00 | 1.00 |
| `he-humaneval-28` | 0.40 | 0.00 | 1.00 | 1.00 |
| `he-humaneval-163` | 0.80 | 0.00 | 0.80 | 0.80 |
| `he-humaneval-108` | 0.80 | 0.00 | 0.80 | 0.80 |
| `he-humaneval-62` | 0.80 | 0.00 | 1.00 | 1.00 |
| `he-humaneval-70` | 1.00 | 0.00 | 1.00 | 1.00 |

这是很难靠“多挂几个 judge”解决的分歧。

如果 deepseek 说 0.80，qwen 说 0.00，glm 说 0.93，我们当然可以做 majority vote。问题是，vote 出来的不是事实，只是另一个聚合规则。三个人站在一段代码旁边争“能不能过”，最直接的方法不是开会，是运行。

这就是 v0.8 的动机。

文章 4 的 panel 解决了前半个问题：发现 judge 之间在吵。v0.8 要解决后半个问题：在能客观判定的领域，别让 judge 继续吵。

## 二、为什么 code 域不该继续用 LLM 裁判

LLM judge 评代码时经常混在一起看三件事：

1. 代码是不是语法正确。
2. 代码是不是覆盖了题意。
3. 代码风格、解释、边界情况看起来是不是像一个“好答案”。

第三件事有时有用，但 HumanEval 这种题真正关心的是前两件事。函数能不能处理 docstring 里的输入，返回值对不对，边界条件过不过。它不是作文题。

上一篇里 qwen judge 把 deepseek 的 code 输出全部打成 0。也许 qwen 对代码质量更严，也许它过度惩罚了风格和边界描述，也许 deepseek 和 glm 太宽。单看 judge 理由，没有办法裁决。

但 HumanEval 自己带测试：

```python
def check(candidate):
    assert candidate([5, 4]) == 25
    assert candidate([0.1, 0.2, 0.3]) == 0
    assert candidate([-10, -20, -30]) == 0
```

这类题有一个朴素到不性感的判据：把模型输出当成函数实现，拼上测试，跑一下。全过就是过，assert 挂了就是挂。

这不是说单测等于全部真理。HumanEval 的测试也可能不完备，隐藏边界也可能漏。但它至少比“另一个 LLM 觉得像不像对”多了一层可复现的物理约束。代码真的执行了，异常真的抛了，返回值真的错了。这个事实不依赖 judge 的口味。

## 三、v0.8 做了什么

v0.8 加了两个 hard-metric scorer：

| scorer | 解决的问题 |
|---|---|
| `code_exec` | 把模型输出的 Python 代码放进子进程，跑 HumanEval 式单测 |
| `numeric_match` | 从输出中抽最后一个数字，按 `rel_tol` / `abs_tol` 和 `expected` 比对 |

`numeric_match` 是 math 域的补丁。以前用 `exact_match` 时，“3.14”、“3.1400”、“答案约为 3.14”会被字符串格式牵着走。数值题应该比数值，不该比标点。

`code_exec` 是这篇的主角。它的配置长这样：

```yaml
scorers:
  - type: code_exec
    params:
      timeout: 5
      memory_mb: 256
```

用例不需要改 `TestCase` 模型，只把 HumanEval 的 `entry_point` 和 `test` 放到 metadata：

```yaml
- id: he-humaneval-151
  domain: code
  input: "补全下面的 Python 函数..."
  metadata:
    entry_point: double_the_difference
    test: |
      def check(candidate):
          assert candidate([]) == 0
          assert candidate([5, 4]) == 25
```

scorer 做四步：

1. 从模型输出里抽第一段 fenced code；没有围栏就取整段输出。
2. 拼成 `模型代码 + test + check(entry_point)`。
3. 放进受限子进程执行。
4. exit code 为 0 就 pass，否则 fail，并把 timeout / AssertionError / stderr 尾部写进 detail。

这条路径没有改 engine、report、diff。它仍然只是一个普通 scorer，产出普通 `Score`。这点很重要，因为 Evalith 的核心抽象不应该因为一个新评分器变复杂。

## 四、为什么要有 `EVALITH_ALLOW_CODE_EXEC=1`

执行模型代码不是普通 scorer。`contains` 最坏只是误判，`code_exec` 是真的在本机跑不可信代码。

所以 v0.8 没有让它静默启用。配置里写了 `code_exec`，但没显式设置环境变量时，`build_scorer` 会直接报错：

```bash
EVALITH_ALLOW_CODE_EXEC=1 evalith run examples/eval.code-exec.yaml
```

这个开关有点啰嗦，但我宁愿它啰嗦。一个 eval 工具如果在用户没意识到的情况下执行模型生成的代码，那就是设计错误。

sandbox 这层做了几件防护：

- 每次执行都起独立 Python 子进程。
- 用 `-I` 隔离 Python 环境，减少用户 site/env 的影响。
- 在子进程内注入 resource limit：CPU、地址空间、文件大小。
- 在前导代码里禁掉一批危险调用，比如 `os.system`、`os.remove`、`os.kill`、`subprocess.run`、`shutil.rmtree`。
- 设定 wall-clock timeout，死循环会被主进程杀掉。

有一个实现细节值得单独说：没有用 `preexec_fn`。

Evalith 的 engine 会在线程池里并发跑 case。Python 文档明确警告，多线程程序里用 `preexec_fn` 有死锁风险。传统写法是在 `subprocess.Popen(..., preexec_fn=set_limits)` 里给子进程设 RLIMIT，但这条路在这里不合适。

v0.8 的做法是把 `resource.setrlimit` 注入到子进程代码最前面。它仍然发生在用户代码之前，但不经过 `preexec_fn`。这不是为了炫技，是为了避开一个真实的多线程坑。

当然，这不是 Docker，不是强安全沙箱。它是一个面向本地 eval 的最小隔离层。边界要说清楚：别拿它跑恶意对抗样本，别把它当云端代码执行服务。它解决的是“模型偶尔写死循环、乱分配内存、误调危险函数时不要拖死整轮 eval”，不是解决所有安全问题。

## 五、验收数据集：复用文章 4 的同一批 HumanEval 题

为了让 v0.8 和文章 4 接上，我没有另选题，而是把文章 4 的 6 个 code case 原样映射回 HumanEval：

| article 4 case | entry point |
|---|---|
| `he-humaneval-151` | `double_the_difference` |
| `he-humaneval-28` | `concatenate` |
| `he-humaneval-163` | `generate_integers` |
| `he-humaneval-108` | `count_nums` |
| `he-humaneval-62` | `derivative` |
| `he-humaneval-70` | `strange_sort_list` |

新的文件是：

```text
examples/code.humaneval.yaml
docs/blog/article4/build_code_exec_dataset.py
docs/blog/article4/configs/eval.code-exec-accept.yaml
```

这组数据的意义不是“又多了一个 demo”。它让文章 4 的 judge 分歧有了一个后续裁判面板：

| 层次 | 问的问题 |
|---|---|
| v0.7 judge consensus panel | judge 之间是否分歧，分歧落在哪些领域 |
| v0.8 code_exec | 对 code 题，模型输出到底能不能通过单测 |

之前只能写：

> qwen 认为这 6 题全不行，glm 认为几乎都行，deepseek 介于中间。

现在可以继续写：

> 把同一批题的模型输出拿去跑 HumanEval 单测，再看三个 judge 各自偏离 execution ground truth 多远。

这一步我还没有把 article 4 的历史 raw outputs 全部重跑成执行表，所以这篇不伪装成“最终实验结论”。目前仓库里已经有的是数据集、scorer、验收配置和全量测试。真正的 judge-vs-exec 对照表，应该作为下一轮实验跑出来，而不是在文章里脑补。

这点要老实。没有跑过的表，不写。

## 六、TDD 结果：这次更像补地基，不像加功能

v0.8 的实现拆成 8 个小任务：

1. `extract_code` 剥围栏纯函数。
2. `sandbox.run_program` 隔离子进程。
3. `CodeExec` scorer。
4. `NumericMatch` scorer。
5. `build_scorer` 接线和环境变量闸门。
6. engine 端到端测试，证明生产引擎零改动。
7. 用真实 HumanEval 重建 article 4 同题号验收集。
8. 版本、示例和 README。

我更喜欢这次的地方在于，engine 没动。报告也没动。diff 也没动。

这说明 scorer 抽象承住了新能力。`code_exec` 看上去是一个很不一样的东西，实际上对外仍然是：

```python
score(case, output) -> Score
```

一个 eval 工具如果每加一种评分方式都要改 engine，后面很快会变成一锅汤。v0.8 没有走到那一步。

测试覆盖也基本沿着风险来：

- fenced code、裸代码、多代码块、空输出。
- assert 失败、死循环 timeout、内存炸弹、危险调用。
- 缺 metadata 时优雅失败。
- `EVALITH_ALLOW_CODE_EXEC` 未开启时拒绝构建 scorer。
- engine 通过 echo provider 做一次完整 end-to-end。
- `numeric_match` 覆盖 exact、容差内、容差外、无数字、expected 非数字。

服务器上那次全量回归跑到 134 passed，只剩一个旧版本号 smoke test。这个测试后来从 `0.1.0` 同步到 `0.8.0`。发布前还应该再用服务器环境跑一遍全量，这是最后的门闩。

## 七、这篇真正想说的不是“我们支持代码执行了”

如果只把 v0.8 理解成“Evalith 加了 code_exec scorer”，那它有点小。

它真正补上的，是这条 eval 工作流：

```text
先用 LLM judge / panel 找到可疑分歧
再在有客观判据的领域切换到 hard metric
最后只把没有客观判据的部分留给 judge
```

这比“多找几个 judge 投票”稳得多。

对 code，跑测试。  
对 math，比数字。  
对事实题，尽量用可检索答案或结构化 gold。  
只有开放式解释、审美、语气、安全边界这类确实没有单一答案的问题，再交给 LLM judge，而且最好挂 panel 看分歧。

LLM judge 不是不能用。文章 2、3、4 其实都在用它。但它应该放在合适的位置：处理那些 hard metric 覆盖不到的语义判断。能不用它的地方，就别用它。

这是我从前四篇里越来越确定的一点。

## 八、给团队的工程判断

如果你的 eval set 里有 code 题，不要只用 LLM judge。至少为关键 case 加一层 `code_exec`。

如果你的 eval set 里有数值题，不要只用字符串匹配。至少用 `numeric_match` 抹平格式差异。

如果你的 eval set 里混着 code、math、knowledge、safety、open-ended explanation，不要幻想一个 judge criteria 能公平量所有题。先按领域拆，再决定每个领域该用什么 scorer。

一个比较实用的组合是：

| 领域 | 首选 scorer | 辅助信号 |
|---|---|---|
| code | `code_exec` | judge panel 看可读性/解释，但不做主裁判 |
| math | `numeric_match` | llm_judge 只看推理过程质量 |
| knowledge | `contains` / `regex` / gold answer | llm_judge 看解释完整性 |
| safety | llm_judge panel | 人审抽样 |
| concept explanation | llm_judge panel + expected_concepts | bootstrap / adaptive sampling |

这样做会让 eval 配置更复杂一点，但复杂度是诚实的。问题本来就不是同一种题，硬压成一个 scorer 只是把复杂度藏进误判里。

## 九、局限和下一步

这次 v0.8 还有几个边界：

1. `code_exec` 只支持 Python/HumanEval 式函数补全，不支持 stdin/stdout 竞赛题。
2. sandbox 是本地最小隔离，不是强安全容器。
3. HumanEval 单测不是形式化证明，测试覆盖不到的 bug 仍然可能漏。
4. 文章 4 的历史模型输出还没有全部通过 `code_exec` 生成 judge-vs-ground-truth 对照表。
5. Windows 上没有 `resource` 模块，当前实现的 RLIMIT 路径主要面向 Linux 发布/CI 环境。

下一步最值得做的不是继续加 scorer，而是把 article 4 的 code raw outputs 全部接到 `code_exec` 上，生成一张表：

| case | deepseek judge | qwen judge | glm judge | code_exec |
|---|---:|---:|---:|---:|
| `he-humaneval-151` | 1.00 | 0.00 | 1.00 | ? |
| `he-humaneval-28` | 0.40 | 0.00 | 1.00 | ? |

这张表才会真正回答上一篇最想回答的问题：

> code 域里，到底是谁更接近 ground truth？

如果 qwen 全打 0 但 `code_exec` 大多通过，说明 qwen 在这批 code 题上过严。  
如果 qwen 全打 0 且 `code_exec` 也大多失败，说明 deepseek/glm 在自评或宽松 judge 上放水。  
如果三者都和 execution 有系统偏差，那就更有意思，说明 LLM judge 对代码正确性的口味和真实执行之间存在结构性错位。

这会是下一篇实验文最有价值的表。

## 十、结论

前四篇一路走下来，Evalith 做了几层防护：

- 文章 1：不要点对点看回归，要用 bootstrap CI 处理 LLM 抖动。
- 文章 2：LLM judge 自己也在抖，工具之间的判定语义会分叉。
- 文章 3：统计方法影响有限，judge identity 才是核心变量。
- 文章 4：judge 分歧有领域结构，code 分歧最大，safety 更容易共识。

文章 5 的结论更朴素：

> 当题目有客观判据时，不要让 judge 投票决定事实。

能执行就执行，能算数就算数，能查 gold 就查 gold。LLM judge 留给那些真的需要语义判断的地方。

v0.7 的 panel 像烟雾报警器。它告诉你哪里烧起来了。  
v0.8 的 `code_exec` 是灭火器的一种。它不负责所有火情，但在 code 这个房间里，比继续开会有效。

```bash
pip install -U evalith
EVALITH_ALLOW_CODE_EXEC=1 evalith run examples/eval.code-exec.yaml
```

代码、验收数据集和复现实验都在：

<https://github.com/dominciyue/Evalith_MingJing>
