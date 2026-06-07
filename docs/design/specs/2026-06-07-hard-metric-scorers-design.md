# Hard-Metric Scorers (v0.8) — Design Spec

Date: 2026-06-07
Status: approved-pending-review

## 背景与动机

文章 4 的核心结论是 judge 分歧有领域结构,且 **code 领域分歧最大**——qwen 把
deepseek 输出的代码从 0.80 判成 0.00,judge 之间几乎不一致(κ 最低)。v0.7 的共识
面板把这个分歧*检测*出来了,但没有给出*ground truth*:面板只能告诉你"judge 们吵
起来了",不能告诉你"谁对"。

代码题恰恰有客观判据——**跑一下单测就知道**。本特性给 Evalith 加确定性、非 LLM 的
硬指标 scorer,把"judge 投票"换成"执行 / 数值比对"的铁证。这直接回应文章 4 留下
的开放问题,也是第五篇文章"检测分歧(面板)+ 解决分歧(执行)"的后半段素材。

新增两个 scorer:
- **`code_exec`** —— HumanEval 式:用例自带单测,模型输出当作函数实现,执行后单测
  全过即 pass。旗舰功能。
- **`numeric_match`** —— 从输出抽数值,带容差比对。服务 math 领域(`exact_match`
  在 "3.14"/"3.140"/"约 3.14" 这类格式差异上会误杀)。

## 已确认的设计决策(与用户对齐)

1. **执行契约**:HumanEval 式单测(用例带 `test` 代码 + `entry_point`),不做
   stdin/stdout 竞赛判题模式。
2. **隔离级别**:子进程 + 超时 + 资源限制(RLIMIT),非 Docker、非同进程 exec。
3. **数据存放**:`test` / `entry_point` 放进用例的 `metadata` 块(不加 TestCase
   类型字段),与"用例级任意数据走 metadata"的现有约定一致。
4. **执行器结构**:每次执行起一个子进程(非常驻 worker 池;后者属过早优化)。
5. **安全闸门**:环境变量 `EVALITH_ALLOW_CODE_EXEC=1` 显式开启;未设时
   `build_scorer` 直接报错中止整个 run(明确,不静默把所有用例判失败)。

## 配置

### code_exec

```yaml
scorers:
  - type: code_exec
    params:
      timeout: 5        # 单次执行墙钟超时(秒),缺省 5
      memory_mb: 256    # 子进程地址空间上限(RLIMIT_AS),缺省 256
```

用例侧(metadata):

```yaml
- id: he-humaneval-151
  domain: code
  input: "补全下面的 Python 函数 ..."
  metadata:
    entry_point: double_the_difference
    test: |
      def check(candidate):
          assert candidate([1, 3, 2, 0]) == 10
          assert candidate([-1, -2, 0]) == 0
          assert candidate([9, -2]) == 81
          assert candidate([0]) == 0
```

### numeric_match

```yaml
scorers:
  - type: numeric_match
    params:
      rel_tol: 1e-3     # 相对容差,缺省 1e-3
      abs_tol: 0.0      # 绝对容差,缺省 0.0
```

数值来源:与待比对值取自 `case.expected`(解析为浮点)。

## code_exec 判定流程

1. **抽代码**:模型输出常被 ```python ... ``` 围栏包住。抽取规则:存在围栏块则取
   第一个 ```` ```python ```` / ```` ``` ```` 块的内容;否则取整段 strip 后的输出。
2. **组装程序**:`extracted_code + "\n" + test + "\n" + f"check({entry_point})\n"`,
   头部再注入 reliability_guard 前导(见下)。
3. **执行**:子进程跑该程序。
4. **判定**:退出码 0 → `passed=True, value=1.0`;否则失败。
   `Score.detail` 给原因:`ok` / `AssertionError: ...`(取 stderr 末尾) /
   `timeout after 5s` / `no code block found` / `missing entry_point or test`。

缺 `entry_point` 或 `test` 的用例:返回失败 Score(detail 标注),不抛异常
(延续 per-case 韧性)。

## 沙箱(新模块 `src/evalith/scorers/sandbox.py`)

单一职责:把一段 Python 源在受限子进程里跑出"通过/失败/原因"。对外接口:

```python
def run_program(source: str, *, timeout: float, memory_mb: int) -> tuple[bool, str]:
    """Run source in an isolated subprocess. Returns (passed, detail)."""
```

实现要点:
- `subprocess.run([sys.executable, "-I", "-c", program], timeout=timeout,
  cwd=<临时目录>, capture_output=True, text=True, env=<精简 env>)`。
  `-I` 隔离模式(忽略环境/用户 site)。
- `preexec_fn` 设(Linux):`RLIMIT_CPU`(≈timeout+1)、`RLIMIT_AS`(memory_mb)、
  `RLIMIT_FSIZE`(限制写文件大小)。
- **reliability_guard 前导**:在 program 最前面注入一段,禁用最危险的调用——
  `os.system`、`os.remove`、`os.kill`、`subprocess.*`、`shutil.rmtree`、内建
  `exec`/`eval` 之外不动正常运行所需(参照 openai/human-eval 的 reliability_guard,
  按需裁剪)。这是纵深防御,不替代子进程/RLIMIT 隔离。
- 退出码 0 → `(True, "ok")`;`TimeoutExpired` → `(False, "timeout after Ns")`;
  非 0 → `(False, <stderr 末尾若干行>)`;spawn 失败等异常 → `(False, <异常信息>)`,
  绝不向上抛。

## scorer 实现(`src/evalith/scorers/`)

两个新类,放进 `rules.py`(或视体量拆 `code_exec.py`;实现时按文件职责单一原则决定):

```python
class CodeExec:
    name = "code_exec"
    def __init__(self, timeout: float = 5, memory_mb: int = 256): ...
    def score(self, case: TestCase, output: str) -> Score: ...

class NumericMatch:
    name = "numeric_match"
    def __init__(self, rel_tol: float = 1e-3, abs_tol: float = 0.0): ...
    def score(self, case: TestCase, output: str) -> Score: ...
```

代码抽取逻辑(剥围栏)做成可独立测试的纯函数 `extract_code(output) -> str | None`。

## build_scorer 接线(`rules.py`)

```python
if cfg.type == "code_exec":
    import os
    if os.environ.get("EVALITH_ALLOW_CODE_EXEC") != "1":
        raise ValueError(
            "code_exec runs untrusted model code; set EVALITH_ALLOW_CODE_EXEC=1 to enable")
    return CodeExec(timeout=cfg.params.get("timeout", 5),
                    memory_mb=cfg.params.get("memory_mb", 256))
if cfg.type == "numeric_match":
    return NumericMatch(rel_tol=cfg.params.get("rel_tol", 1e-3),
                        abs_tol=cfg.params.get("abs_tol", 0.0))
```

## 引擎集成

零改动。两者都满足 `Scorer` 协议(`.score(case, output) -> Score`),直接进现有
scorer 列表,沿用 case 级 ThreadPoolExecutor 并发。code_exec 较慢但不做嵌套并发。

## 报告与 CLI

零改动。两者产出普通 `Score`,自动进现有 pass-rate / md/html 报告 /
diff。本轮不加专门报告节(YAGNI)。

## 验收数据集

新建 `examples/code.humaneval.yaml`(或 `docs/blog/article4/` 下),复用**文章 4 同
一批 HumanEval 题号**(he-151 / 28 / 163 / 108 / 62 / 70),每个用例在 metadata 里带
真实的 `entry_point` 与 `test`(取自 HumanEval 原始数据集)。这样能在同一题上直接对照
"judge 分数 vs 执行 ground truth",成为第五篇文章的核心证据。

## 测试策略(TDD)

- **extract_code 纯函数**:```python``` 围栏 / 裸 ``` 围栏 / 无围栏裸代码 /
  多块取首块 / 空输出 → None。
- **sandbox.run_program**:通过的程序 → (True, "ok");assert 失败 → (False, 含
  AssertionError);死循环 → (False, timeout);内存炸弹 → (False);`os.system`
  等被 guard 拦截 → (False)。全部用极小内联程序,零网络。
- **CodeExec.score**:canonical solution → pass;错误实现 → fail;缺
  entry_point/test → 优雅失败;围栏内代码能被正确抽取执行。
- **NumericMatch.score**:精确相等 / 容差内 / 容差外 / 无数值 / expected 非数。
- **build_scorer 闸门**:未设 `EVALITH_ALLOW_CODE_EXEC` → 构建报错;设了 → 正常构建。
- **引擎端到端**:用 `echo:` provider 喂 canonical solution 的 code_exec 用例跑
  完整 run → passed,且与既有 scorer 共存不互相影响。
- **兼容**:不配新 scorer 时,既有全部测试通过,run JSON 语义不变。

## 范围外(明确不做)

- 非 Python 语言执行。
- Docker / 容器隔离。
- stdin/stdout 竞赛判题模式。
- 联网执行(子进程 env 精简,不主动放开网络)。
- 代码执行专门报告节(detail 已足够)。
- 常驻 worker 池。

## 验收标准

1. 未设 `EVALITH_ALLOW_CODE_EXEC=1` 时配 code_exec → run 明确报错中止,不静默失败。
2. 设了开关,跑 `examples/code.humaneval.yaml`:canonical solution 全 pass;故意改
   错一题 → 该题 fail,detail 给出 AssertionError。
3. 死循环 / 内存炸弹用例 → 被 timeout / RLIMIT 终止并判 fail,不拖垮整个 run。
4. `numeric_match`(expected=3.14159, rel_tol=1e-3):输出 "答案约 3.0" → 抽出 3.0,
   相对误差 ≈4.5% 超容差 → fail;输出 "结果是 3.1416" → 相对误差 ≈1e-5 在容差内 →
   pass。
5. 不配新 scorer 时,既有 112 测试全部通过。
