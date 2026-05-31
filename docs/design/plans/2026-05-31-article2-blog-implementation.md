# Article 2: High-temp + llm_judge Bootstrap CI Experiment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a publish-ready Chinese technical blog article (~4000 字) that delivers on article 1's "high-temp + llm_judge" limitation and includes a side-by-side comparison of Evalith vs promptfoo vs DeepEval on the same dataset + regression injection, with all raw data and reproduction scripts committed.

**Architecture:** Five sequential phases — (0) pre-flight, (1) dataset + Evalith configs, (2) Evalith data collection (the three runs A1/A2/B), (3) competitor tool runs, (4) article writing, (5) finalize. No new Evalith features. All experimental tables filled from real runs only.

**Tech Stack:** Evalith v0.4 (existing), DeepSeek API (via LiteLLM), promptfoo (npm), DeepEval (pip), bash + Python for orchestration.

**Spec reference:** `docs/design/specs/2026-05-31-article2-high-temp-llm-judge-bootstrap.md` (commit `9b7e2aa`).

**Article date stamp:** Use the date the experiment runs clean end-to-end (probably 2026-06-01 or later). Plan uses `<PUB-DATE>` placeholder for the filename until then.

---

## Phase 0: Pre-flight

### Task 1: Confirm pre-conditions and credentials

**Files:** None. Pure verification.

- [ ] **Step 1: Confirm DeepSeek API key is available**

Run: `echo "${DEEPSEEK_API_KEY:0:7}..."`
Expected: prints `sk-XXXX...`. If unset, stop and ask user — the user has a dedicated quota-limited key (per session memory).

- [ ] **Step 2: Confirm `llm_judge` judge-temperature is already 0**

Run: `grep -n "temperature=0.0" src/evalith/scorers/llm_judge.py`
Expected: matches line 57 (`resp = self.provider.complete(prompt, temperature=0.0)`). This resolves spec Appendix A — no Evalith code change needed.

- [ ] **Step 3: Confirm Evalith v0.4 CLI works**

Run: `python3 -m evalith.cli --help | head -20`
Expected: shows `run`, `diff`, `report`, `models`, `list` subcommands.

- [ ] **Step 4: Smoke-run an existing real-model eval to validate provider wiring**

Run: `python3 -m evalith.cli run examples/eval.deepseek.yaml --samples 2 2>&1 | tail -5`
Expected: completes successfully, prints a summary line like `3 cases, … passed`. If this fails the rest of the plan can't run.

- [ ] **Step 5: Create the article2 workspace directory**

Run:
```bash
mkdir -p docs/blog/article2/{configs,raw}
ls docs/blog/article2/
```
Expected: shows `configs/  raw/`.

- [ ] **Step 6: Commit the empty scaffolding to lock the path**

```bash
touch docs/blog/article2/.gitkeep docs/blog/article2/configs/.gitkeep docs/blog/article2/raw/.gitkeep
git add docs/blog/article2/
git commit -m "chore(blog): scaffold article2 workspace"
```

---

## Phase 1: Dataset and Evalith configs

### Task 2: Author the 10-case dataset

**Files:**
- Create: `docs/blog/article2/qa.high-temp.yaml`

- [ ] **Step 1: Write the dataset file with exactly this content**

```yaml
name: qa-high-temp
# 10 mid-depth open-ended technical questions. Each answer is judged purely by
# llm_judge against an expected-concepts checklist — no `contains` scorer, so
# scoring is fully judge-driven (the article's whole point is that both layers
# of noise — model output + judge verdict — are now in play).
cases:
  - id: explain-rlhf
    input: "用 4-6 句话解释 RLHF(从人类反馈中强化学习)是什么、为什么需要它,以及典型流程的主要环节。"
    expected_concepts: ["监督微调 SFT", "奖励模型 reward model", "PPO 或 DPO 优化", "对齐 alignment 目标"]
  - id: explain-vector-db
    input: "用 4-6 句话说明向量数据库与传统关系型数据库的核心区别,以及在 LLM 应用栈里的典型作用。"
    expected_concepts: ["embedding 向量表示", "近似最近邻 ANN", "相似度搜索", "索引结构 HNSW 或 IVF"]
  - id: sql-injection-vulnerability
    input: "下列伪代码 `db.query(\"SELECT * FROM users WHERE name = '\" + name + \"'\")` 在什么情况下不安全?该如何修?用 4-6 句话回答。"
    expected_concepts: ["未参数化的用户输入拼接", "参数化查询 prepared statement", "ORM 或绑定变量替代字符串拼接", "转义不可靠"]
  - id: k8s-configmap-vs-secret
    input: "Kubernetes 的 ConfigMap 和 Secret 区别是什么,选择哪一个的依据是什么?用 4-6 句话说明。"
    expected_concepts: ["ConfigMap 存非敏感配置", "Secret 存敏感数据(默认 base64)", "etcd 中 Secret 默认未真正加密", "RBAC 限制访问"]
  - id: asyncio-yield-deadlock
    input: "在 Python asyncio 协程中调用 `time.sleep(1)` 会发生什么?为什么不能这样写?用 4-6 句话说明。"
    expected_concepts: ["阻塞事件循环", "同步阻塞 vs 协程 await", "应改用 await asyncio.sleep", "并发被破坏"]
  - id: python-gil-tradeoffs
    input: "用 4-6 句话说明 Python 的 GIL 是什么、它对 CPU 密集型多线程任务的影响,以及常见绕开方式。"
    expected_concepts: ["全局解释器锁", "保证字节码操作原子性", "多线程 CPU 任务无法真正并行", "多进程或 C 扩展规避"]
  - id: redis-cluster-failover
    input: "Redis Cluster 中,一个主节点宕机后,从节点是如何被选举为新主、客户端流量是如何被重定向的?用 4-6 句话说明流程。"
    expected_concepts: ["gossip 协议感知故障", "从节点投票/选举", "epoch 递增", "客户端槽位重定向 MOVED 或 ASK"]
  - id: tcp-congestion-control
    input: "TCP 拥塞控制中,慢启动和拥塞避免阶段的核心区别是什么?用 4-6 句话说明。"
    expected_concepts: ["指数增长 cwnd", "线性增长 cwnd", "拥塞窗口 cwnd 概念", "阈值 ssthresh 切换"]
  - id: jwt-vs-session
    input: "JWT 与传统服务端 session 在认证上的核心权衡是什么?用 4-6 句话说明。"
    expected_concepts: ["无状态 vs 有状态", "服务端存储成本", "撤销与失效难度", "横向扩展性"]
  - id: transformer-attention
    input: "Transformer 中的 self-attention 是怎么工作的?Q/K/V 各自的作用是什么?用 4-6 句话说明。"
    expected_concepts: ["Query 向量", "Key 向量", "Value 向量", "softmax 归一化的注意力权重"]
```

- [ ] **Step 2: Sanity-check it parses**

Run: `python3 -c "import yaml; d = yaml.safe_load(open('docs/blog/article2/qa.high-temp.yaml')); print(len(d['cases']), 'cases')"`
Expected: prints `10 cases`.

- [ ] **Step 3: Commit**

```bash
git add docs/blog/article2/qa.high-temp.yaml
git commit -m "feat(blog/article2): 10-case high-temp dataset (open-ended technical Q&A)"
```

### Task 3: Author the two Evalith configs (baseline + regression-injected)

**Files:**
- Create: `docs/blog/article2/configs/eval.high-temp.yaml`
- Create: `docs/blog/article2/configs/eval.high-temp.broken.yaml`

- [ ] **Step 1: Write the baseline config**

`docs/blog/article2/configs/eval.high-temp.yaml`:
```yaml
name: high-temp-baseline
dataset: docs/blog/article2/qa.high-temp.yaml
model: deepseek-chat
prompt_template: "{{input}}"
temperature: 1.0
concurrency: 3
scorers:
  - type: llm_judge
    params:
      criteria: "回答是否充分覆盖了核心概念、解释清楚、且没有跑题或显著省略。完整覆盖给 1.0,部分覆盖给 0.5 左右,严重缺失或跑题给 0.0。"
      language: zh
```

- [ ] **Step 2: Write the regression-injected config**

`docs/blog/article2/configs/eval.high-temp.broken.yaml`:
```yaml
name: high-temp-broken
dataset: docs/blog/article2/qa.high-temp.yaml
model: deepseek-chat
prompt_template: "Assume the user is a senior engineer who knows the basics. Skip foundational explanations and focus on the non-obvious parts. Be concise: {{input}}"
temperature: 1.0
concurrency: 3
scorers:
  - type: llm_judge
    params:
      criteria: "回答是否充分覆盖了核心概念、解释清楚、且没有跑题或显著省略。完整覆盖给 1.0,部分覆盖给 0.5 左右,严重缺失或跑题给 0.0。"
      language: zh
```

(Identical to baseline except for `name` and `prompt_template`. **Crucially, the judge criteria is identical** — so a verdict difference comes purely from model output quality, not from scoring goalposts moving.)

- [ ] **Step 3: Confirm both parse**

Run:
```bash
for f in docs/blog/article2/configs/eval.high-temp*.yaml; do python3 -c "import yaml; yaml.safe_load(open('$f'))" && echo "OK: $f"; done
```
Expected: two `OK:` lines.

- [ ] **Step 4: Commit**

```bash
git add docs/blog/article2/configs/eval.high-temp.yaml docs/blog/article2/configs/eval.high-temp.broken.yaml
git commit -m "feat(blog/article2): Evalith baseline + regression-injected configs"
```

### Task 4: Mini smoke test (1 case, 2 samples) to catch wiring bugs cheap

**Files:** Temporary; do NOT commit smoke output.

- [ ] **Step 1: Make a single-case subset config to keep the smoke cheap**

```bash
python3 - <<'PY'
import yaml, copy
src = yaml.safe_load(open('docs/blog/article2/configs/eval.high-temp.yaml'))
ds = yaml.safe_load(open('docs/blog/article2/qa.high-temp.yaml'))
ds_smoke = {"name": "qa-smoke-single", "cases": ds['cases'][:1]}
yaml.safe_dump(ds_smoke, open('/tmp/qa.smoke.yaml','w'), allow_unicode=True, sort_keys=False)
cfg = copy.deepcopy(src)
cfg['dataset'] = '/tmp/qa.smoke.yaml'
cfg['name'] = 'smoke'
yaml.safe_dump(cfg, open('/tmp/eval.smoke.yaml','w'), allow_unicode=True, sort_keys=False)
print("smoke configs written")
PY
```
Expected: `smoke configs written`.

- [ ] **Step 2: Run it**

Run: `python3 -m evalith.cli run /tmp/eval.smoke.yaml --samples 2 --out /tmp/smoke.json`
Expected: completes (~10s); writes `/tmp/smoke.json`. Stdout shows `1 case`, with a score.

- [ ] **Step 3: Verify samples + judge score are present**

```bash
python3 - <<'PY'
import json
r = json.load(open('/tmp/smoke.json'))
case = r['results'][0]
print("samples recorded:", case.get('pass_rate_samples'))
print("scores:", [(s['scorer'], s['value']) for s in case['scores']])
PY
```
Expected: `samples recorded: [<f>, <f>]` (two floats), and judge score present. If samples list is empty or scorer is missing, stop and fix before any real run.

- [ ] **Step 4: Clean up smoke artifacts**

Run: `rm /tmp/qa.smoke.yaml /tmp/eval.smoke.yaml /tmp/smoke.json`

### Task 5: Write the article skeleton with §4 ground-truth hypothesis pre-committed

**Files:**
- Create: `docs/blog/<PUB-DATE>-llm-judge-noise-bootstrap.zh.md`

(For now use `2026-05-31` as the date placeholder; the final task will rename to publication date.)

- [ ] **Step 1: Create the article skeleton with hypothesis filled but data sections empty**

`docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md`:
```markdown
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
```

- [ ] **Step 2: Commit — this commit timestamps the hypothesis BEFORE the experiment runs**

```bash
git add docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md
git commit -m "docs(blog/article2): article skeleton + pre-committed hypothesis

Hypothesis (which cases the prompt-bias regression should hit) is committed
BEFORE the A1-vs-B experiment runs — git history makes the pre-hoc nature
verifiable. Per spec §5.4."
```

---

## Phase 2: Evalith data collection

### Task 6: Run the three Evalith experiments (A1, A2, B)

**Files:**
- Create: `docs/blog/article2/raw/a1.json`
- Create: `docs/blog/article2/raw/a2.json`
- Create: `docs/blog/article2/raw/b.json`

- [ ] **Step 1: Run experiment A1 (baseline)**

Run:
```bash
python3 -m evalith.cli run docs/blog/article2/configs/eval.high-temp.yaml \
    --samples 5 \
    --out docs/blog/article2/raw/a1.json
```
Expected: ~3-5 minutes; prints `10 cases, … passed`; writes `a1.json`.

- [ ] **Step 2: Run experiment A2 (identical config — noise-floor control)**

Run:
```bash
python3 -m evalith.cli run docs/blog/article2/configs/eval.high-temp.yaml \
    --samples 5 \
    --out docs/blog/article2/raw/a2.json
```
Expected: ~3-5 minutes; same shape as a1.json.

- [ ] **Step 3: Run experiment B (regression-injected)**

Run:
```bash
python3 -m evalith.cli run docs/blog/article2/configs/eval.high-temp.broken.yaml \
    --samples 5 \
    --out docs/blog/article2/raw/b.json
```
Expected: ~3-5 minutes; writes `b.json`.

- [ ] **Step 4: Sanity-check all three have 10 cases × 5 samples each**

```bash
python3 - <<'PY'
import json
for f in ['a1','a2','b']:
    r = json.load(open(f'docs/blog/article2/raw/{f}.json'))
    cases = r['results']
    n_samples = [len(c.get('pass_rate_samples', [])) for c in cases]
    print(f, 'cases:', len(cases), 'samples per case:', n_samples)
PY
```
Expected: all three lines show `cases: 10`, all samples-per-case lists are 5s. If anything else, re-run that experiment before continuing.

- [ ] **Step 5: Commit raw outputs (these are frozen — never re-run after commit)**

```bash
git add docs/blog/article2/raw/a1.json docs/blog/article2/raw/a2.json docs/blog/article2/raw/b.json
git commit -m "data(blog/article2): frozen raw outputs for A1, A2, B Evalith runs

10 cases × 5 samples × 3 runs; temp=1.0; judge=deepseek-chat at temp=0.
These outputs are the ground truth referenced by all three tools'
comparison logic. Never regenerated after this commit."
```

### Task 7: Build §3 noise-floor table (Evalith diff A1-vs-A2)

**Files:**
- Modify: `docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md` (replace `<TODO §3 …>`)

- [ ] **Step 1: Generate the noise-floor diff table**

```bash
python3 - <<'PY'
from pathlib import Path
from evalith.diff import diff_runs
from evalith.models import Run

a1 = Run.model_validate_json(Path('docs/blog/article2/raw/a1.json').read_text())
a2 = Run.model_validate_json(Path('docs/blog/article2/raw/a2.json').read_text())
report = diff_runs(a1, a2)

# Also compute: per-case single-sample range (max - min within a1) to
# show how wild point-compare on 1 sample would be.
import json
def sample_range(run, cid):
    for r in run.results:
        if r.case_id == cid:
            s = r.pass_rate_samples or []
            return (min(s), max(s)) if s else (None, None)
    return (None, None)

print("\n| case | A1 mean | A2 mean | A1 单样本极差 | bootstrap CI on Δ | status |")
print("|---|---|---|---|---|---|")
for c in report.cases:
    lo,hi = sample_range(a1, c.case_id)
    rng = f"[{lo:.1f}, {hi:.1f}]" if lo is not None else "—"
    b = "—" if c.before is None else f"{c.before:.2f}"
    a = "—" if c.after is None else f"{c.after:.2f}"
    ci = "—" if c.ci is None else f"[{c.ci[0]:+.2f}, {c.ci[1]:+.2f}]"
    print(f"| `{c.case_id}` | {b} | {a} | {rng} | {ci} | {c.status} |")

flagged = sum(1 for c in report.cases if c.status == 'regressed')
print(f"\n**A1-vs-A2 假阳率: {flagged}/10**")
PY
```
Expected: a 10-row markdown table prints, plus a "假阳率" line. **The expected behavior is 0 or near 0 regressed** — if it's >2, something is off and stop to investigate.

- [ ] **Step 2: Paste the table into §3 of the article**

Replace `<TODO §3 prose + table from Evalith diff A1-vs-A2 — Task 7>` in `docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md` with:

```markdown
我们用上一节描述的 dataset、temp=1.0、5 次采样,跑两遍**完全相同**的 config,得到 A1 和 A2 两组 run。预期:bootstrap CI 应该几乎都跨过 0 → 0 假阳。

<PASTE THE TABLE FROM STEP 1>

观察:
- 「A1 单样本极差」一列展示了**如果只采样一次会有多么不稳定** —— 多个 case 的单样本 pass rate 在 0.0 和 1.0 之间反复横跳。如果一个 CI gate 只跑一次就比较,会被这样的噪声反复误报。
- bootstrap CI 全部(或绝大多数)横跨 0,正确判定为 `unchanged`。这是"gate 不误伤"的基础性质。
- 真正有意思的不是这张表本身,而是它告诉你**单样本 point-compare 在 llm_judge 评分下几乎不可能保持稳定**。这是上篇没演示出来的核心论据。
```

(If the data shows >0 false alarms, adapt the prose honestly — name which cases and theorize why; do NOT hide it.)

- [ ] **Step 3: Commit**

```bash
git add docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md
git commit -m "docs(blog/article2): §3 noise floor — Evalith A1-vs-A2 results"
```

### Task 8: Build §4 regression-injection table (Evalith diff A1-vs-B)

**Files:**
- Modify: `docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md` (replace `<TODO §4 …>`)

- [ ] **Step 1: Generate the regression diff table**

```bash
python3 - <<'PY'
from pathlib import Path
from evalith.diff import diff_runs
from evalith.models import Run

a1 = Run.model_validate_json(Path('docs/blog/article2/raw/a1.json').read_text())
b  = Run.model_validate_json(Path('docs/blog/article2/raw/b.json').read_text())
report = diff_runs(a1, b)

print("\n| case | mean before | mean after | bootstrap CI on Δ | status |")
print("|---|---|---|---|---|")
for c in report.cases:
    bm = "—" if c.before is None else f"{c.before:.2f}"
    am = "—" if c.after  is None else f"{c.after:.2f}"
    ci = "—" if c.ci is None else f"[{c.ci[0]:+.2f}, {c.ci[1]:+.2f}]"
    print(f"| `{c.case_id}` | {bm} | {am} | {ci} | {c.status} |")

flagged = [c.case_id for c in report.cases if c.status == 'regressed']
print(f"\n**Evalith 命中的 regressed case:** {flagged}")
PY
```
Expected: 10-row table; bottom line names which cases were flagged.

- [ ] **Step 2: Compare flagged cases to the pre-committed hypothesis**

Hypothesis (from §4 prose, Task 5):
- Strong: `explain-rlhf`, `explain-vector-db`, `python-gil-tradeoffs`, `tcp-congestion-control`, `transformer-attention`
- Borderline: `k8s-configmap-vs-secret`, `jwt-vs-session`
- Should-not-fire: `sql-injection-vulnerability`, `asyncio-yield-deadlock`, `redis-cluster-failover`

Compute hits / misses by hand or:

```bash
python3 - <<'PY'
STRONG = {"explain-rlhf","explain-vector-db","python-gil-tradeoffs","tcp-congestion-control","transformer-attention"}
BORDER = {"k8s-configmap-vs-secret","jwt-vs-session"}
SAFE   = {"sql-injection-vulnerability","asyncio-yield-deadlock","redis-cluster-failover"}

import json
from pathlib import Path
from evalith.diff import diff_runs
from evalith.models import Run
a1 = Run.model_validate_json(Path('docs/blog/article2/raw/a1.json').read_text())
b  = Run.model_validate_json(Path('docs/blog/article2/raw/b.json').read_text())
flagged = {c.case_id for c in diff_runs(a1, b).cases if c.status == 'regressed'}

print("strong hits:", flagged & STRONG, "missed:", STRONG - flagged)
print("border hits:", flagged & BORDER)
print("safe false-positives:", flagged & SAFE)
PY
```
Expected: most STRONG appear in `flagged`; SAFE should have 0 or 1 false-positive.

- [ ] **Step 3: Paste the table + honest analysis into §4**

Replace `<TODO §4 table + post-data prose — Task 8>` with:

```markdown
跑完 B(注入了上面那句 prompt 偏差)后,得到:

<PASTE THE TABLE FROM STEP 1>

对照上文预先承诺的假设:

- **强预期被命中: <填写: 例如 5 中的 X 个>**(`<list>`)
- **强预期没被命中: <list 或 "无">** — <如果有, 诚实写为什么 LLM 这次没被这句指令影响>
- **边界 case 命中: <list 或 "无">**
- **应不被命中却被误报: <list 或 "无">** — <如果有, 诚实承认这是 bootstrap 的局限或 dataset 的瑕疵>

正面 case:观察到 `<具体 case_id>` 的 mean pass rate 从 <X.XX> 降到 <Y.YY>,bootstrap CI 完全在零下 — 这是经典的真回退。

反例:`<某不应被命中却被命中的 case 或某应被命中却没被命中的 case>` 提醒我们:**bootstrap 不是真相检测器,只是噪声过滤器。** 当模型对一个 case 的"基础-非基础"边界拿捏与人不同时,judge 会跟着抖,CI 区间内的判定就还是噪声决定的。

如果你只看上篇文章里的"理论上 bootstrap 抗噪"是不够的;这张表才是真正能拿出来对照的证据。
```

(Fill in the angle-bracket placeholders from the actual data. Do not skip filling them. If outcomes contradict the hypothesis, write the contradiction up frankly — honesty IS the article's brand.)

- [ ] **Step 4: Commit**

```bash
git add docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md
git commit -m "docs(blog/article2): §4 regression injection results + hypothesis check"
```

---

## Phase 3: Horizontal comparison

### Task 9: promptfoo install, config, and run

**Files:**
- Create: `docs/blog/article2/configs/promptfoo.yaml`

- [ ] **Step 1: Install promptfoo (npm)**

Run: `npm install -g promptfoo && promptfoo --version`
Expected: prints a version (e.g., `0.x.y`). If npm is unavailable, install via the user's preferred path; record the exact install line in the article.

- [ ] **Step 2: Write the promptfoo config**

`docs/blog/article2/configs/promptfoo.yaml`:
```yaml
# Mirrors the Evalith run as closely as promptfoo's schema allows.
# Same model, temp, prompt template variants, judge criteria.
prompts:
  - "{{input}}"
providers:
  - id: deepseek:deepseek-chat
    config:
      temperature: 1.0
      apiKeyEnvar: DEEPSEEK_API_KEY
tests:
  - vars:
      input: "用 4-6 句话解释 RLHF(从人类反馈中强化学习)是什么、为什么需要它,以及典型流程的主要环节。"
    assert:
      - type: llm-rubric
        provider: deepseek:deepseek-chat
        value: "回答是否充分覆盖了核心概念(SFT, 奖励模型, PPO 或 DPO, 对齐目标)、解释清楚、且没有跑题或显著省略"
  # ... (repeat block for each of the 10 cases; promptfoo doesn't support an
  # external dataset YAML in the same way, so we expand inline. Use the same
  # criteria wording as Evalith config — only the rubric form differs.)
defaultTest:
  options:
    repeat: 5  # 5 samples per case to mirror Evalith
```

(**Write out all 10 test blocks** explicitly — promptfoo doesn't read Evalith's dataset YAML. Yes, this is verbose; that's the cost of fair comparison.)

- [ ] **Step 3: Run baseline (A1-equivalent)**

```bash
promptfoo eval -c docs/blog/article2/configs/promptfoo.yaml \
    -o docs/blog/article2/raw/promptfoo_a1.json
```
Expected: completes; writes output JSON.

- [ ] **Step 4: Make a "broken" copy with the prompt-bias injection and run it**

```bash
sed 's/"{{input}}"/"Assume the user is a senior engineer who knows the basics. Skip foundational explanations and focus on the non-obvious parts. Be concise: {{input}}"/' \
    docs/blog/article2/configs/promptfoo.yaml \
    > docs/blog/article2/configs/promptfoo.broken.yaml

promptfoo eval -c docs/blog/article2/configs/promptfoo.broken.yaml \
    -o docs/blog/article2/raw/promptfoo_b.json
```

- [ ] **Step 5: Capture promptfoo's per-case pass/fail verdict**

```bash
python3 - <<'PY'
import json
for tag in ['a1','b']:
    r = json.load(open(f'docs/blog/article2/raw/promptfoo_{tag}.json'))
    # promptfoo result shape: r['results'][i]['success'] etc. Adapt to actual schema.
    summary = [(t.get('vars',{}).get('input','')[:40], t.get('success')) for t in r.get('results',{}).get('results',[])]
    print(tag, summary[:5], '...' if len(summary)>5 else '')
PY
```
(Adapt parsing to whatever promptfoo's actual output schema is in the version installed.)
Expected: prints first few case verdicts for both runs.

- [ ] **Step 6: Commit promptfoo configs + raw outputs**

```bash
git add docs/blog/article2/configs/promptfoo.yaml docs/blog/article2/configs/promptfoo.broken.yaml \
        docs/blog/article2/raw/promptfoo_a1.json docs/blog/article2/raw/promptfoo_b.json
git commit -m "data(blog/article2): promptfoo baseline + regression-injected runs"
```

### Task 10: DeepEval install, harness, and run

**Files:**
- Create: `docs/blog/article2/configs/deepeval_compare.py`

- [ ] **Step 1: Install DeepEval**

Run: `pip install deepeval && deepeval --version`
Expected: prints a version.

- [ ] **Step 2: Write the DeepEval harness**

`docs/blog/article2/configs/deepeval_compare.py`:
```python
"""Run the same 10-case dataset through DeepEval's GEval scorer, 5 samples per
case, both baseline prompt and regression-injected prompt. Output a per-case
pass rate JSON in the same shape Evalith uses, so the comparison script
downstream can be tool-agnostic."""
import os, json, sys, yaml
from pathlib import Path

# DeepEval setup
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

import litellm

DATASET = yaml.safe_load(open("docs/blog/article2/qa.high-temp.yaml"))
SAMPLES = 5
TEMP = 1.0
MODEL = "deepseek/deepseek-chat"

BASELINE_PREFIX = ""
BROKEN_PREFIX = "Assume the user is a senior engineer who knows the basics. Skip foundational explanations and focus on the non-obvious parts. Be concise: "

def model_call(prompt):
    r = litellm.completion(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=TEMP)
    return r.choices[0].message.content

def run(prefix, tag):
    out = []
    for case in DATASET["cases"]:
        passes = 0
        for _ in range(SAMPLES):
            answer = model_call(prefix + case["input"])
            metric = GEval(
                name="Concept coverage",
                criteria=f"Does the answer cover the key concepts: {', '.join(case['expected_concepts'])}? Penalize if any are missing or unclear.",
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                model=MODEL.split("/",1)[1],  # deepseek-chat
                threshold=0.5,
            )
            tc = LLMTestCase(input=case["input"], actual_output=answer)
            metric.measure(tc)
            if metric.score >= 0.5:
                passes += 1
        out.append({"case_id": case["id"], "pass_rate": passes / SAMPLES, "samples": SAMPLES})
    Path(f"docs/blog/article2/raw/deepeval_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote deepeval_{tag}.json")

if __name__ == "__main__":
    run(BASELINE_PREFIX, "a1")
    run(BROKEN_PREFIX, "b")
```

(If the installed DeepEval version's API differs, adapt — but the contract this script produces is: a JSON list of `{case_id, pass_rate, samples}` per run.)

- [ ] **Step 3: Run the DeepEval harness**

Run: `python3 docs/blog/article2/configs/deepeval_compare.py`
Expected: prints `wrote deepeval_a1.json` and `wrote deepeval_b.json`, takes 5-10 min.

- [ ] **Step 4: Sanity-check output shape**

```bash
python3 - <<'PY'
import json
for tag in ['a1','b']:
    r = json.load(open(f'docs/blog/article2/raw/deepeval_{tag}.json'))
    assert len(r) == 10
    assert all('pass_rate' in c for c in r)
print("DeepEval output shape OK")
PY
```

- [ ] **Step 5: Commit**

```bash
git add docs/blog/article2/configs/deepeval_compare.py \
        docs/blog/article2/raw/deepeval_a1.json docs/blog/article2/raw/deepeval_b.json
git commit -m "data(blog/article2): DeepEval baseline + regression-injected runs"
```

### Task 11: Build the cross-tool comparison metrics + §5 table

**Files:**
- Create: `docs/blog/article2/compare.py`

- [ ] **Step 1: Write the comparison script**

`docs/blog/article2/compare.py`:
```python
"""Produce the §5 cross-tool comparison table.
For each tool x each comparison (A1-vs-A2 noise-floor, A1-vs-B regression):
- per-case verdict: regressed / unchanged / improved
- aggregate: false-alarm rate (noise-floor), capture rate (regression), capture precision.

Ground truth for capture rate: hypothesis from article §4 (STRONG ∪ BORDER cases
were predicted to be affected; SAFE cases were not).
"""
import json
from pathlib import Path

STRONG = {"explain-rlhf","explain-vector-db","python-gil-tradeoffs","tcp-congestion-control","transformer-attention"}
BORDER = {"k8s-configmap-vs-secret","jwt-vs-session"}
SAFE   = {"sql-injection-vulnerability","asyncio-yield-deadlock","redis-cluster-failover"}
PREDICTED_AFFECTED = STRONG | BORDER

def load_evalith(path):
    from evalith.models import Run
    r = Run.model_validate_json(Path(path).read_text())
    return {c.case_id: c.pass_rate_samples or [] for c in r.results}

def load_deepeval(path):
    return {c['case_id']: [c['pass_rate']] for c in json.load(open(path))}

def load_promptfoo(path):
    # Promptfoo per-sample pass shape may differ; adapt at run time.
    # Returns {case_id: [0/1, 0/1, ...]} based on the success bool per repeat.
    raw = json.load(open(path))
    results = raw.get('results', {}).get('results', raw if isinstance(raw, list) else [])
    out = {}
    for t in results:
        cid_key = (t.get('vars') or {}).get('input','')[:40]
        out.setdefault(cid_key, []).append(1 if t.get('success') else 0)
    return out  # NOTE: case_id keying differs; mapping done by index below.

def evalith_verdicts(before_path, after_path):
    from evalith.diff import diff_runs
    from evalith.models import Run
    b = Run.model_validate_json(Path(before_path).read_text())
    a = Run.model_validate_json(Path(after_path).read_text())
    return {c.case_id: c.status for c in diff_runs(b, a).cases}

def naive_verdict(before_rate, after_rate, tol=1e-9):
    """Used for promptfoo and DeepEval — most non-Evalith tools only point-compare."""
    if after_rate < before_rate - tol: return "regressed"
    if after_rate > before_rate + tol: return "improved"
    return "unchanged"

def aggregate(verdicts):
    flagged = {cid for cid,v in verdicts.items() if v == 'regressed'}
    return flagged

def metrics(verdicts, predicted_affected, all_cases):
    flagged = aggregate(verdicts)
    safe = set(all_cases) - predicted_affected
    false_alarm = len(flagged & safe) / max(1, len(safe))
    capture_rate = len(flagged & predicted_affected) / max(1, len(predicted_affected))
    capture_precision = len(flagged & predicted_affected) / max(1, len(flagged)) if flagged else 0.0
    return {"flagged": sorted(flagged), "false_alarm": false_alarm,
            "capture_rate": capture_rate, "capture_precision": capture_precision}

# Build verdicts
ev_a1a2 = evalith_verdicts('docs/blog/article2/raw/a1.json', 'docs/blog/article2/raw/a2.json')
ev_a1b  = evalith_verdicts('docs/blog/article2/raw/a1.json', 'docs/blog/article2/raw/b.json')

# DeepEval: only point-rates, so naive verdict
de_a1 = load_deepeval('docs/blog/article2/raw/deepeval_a1.json')
de_b  = load_deepeval('docs/blog/article2/raw/deepeval_b.json')
de_a1b = {cid: naive_verdict(de_a1[cid][0], de_b[cid][0]) for cid in de_a1}

# Promptfoo: parse-dependent — print, then humans format §5
pf_a1 = load_promptfoo('docs/blog/article2/raw/promptfoo_a1.json')
pf_b  = load_promptfoo('docs/blog/article2/raw/promptfoo_b.json')

all_cases = list(ev_a1b.keys())

print("\n=== §5 Table: Cross-tool comparison ===\n")
print("| case | Evalith A1-vs-B | DeepEval A1-vs-B | promptfoo A1-vs-B |")
print("|---|---|---|---|")
for cid in all_cases:
    print(f"| `{cid}` | {ev_a1b.get(cid,'?')} | {de_a1b.get(cid,'?')} | <fill from promptfoo output> |")

print("\n=== Aggregate metrics (A1 vs A2 — noise-floor false-alarm rate, lower is better) ===\n")
print("Evalith:", metrics(ev_a1a2, PREDICTED_AFFECTED, all_cases))
# DeepEval/promptfoo noise-floor: A1 vs another A1 would need a second DeepEval run; if not done, mark N/A and note in article.
print("DeepEval: N/A (only one DeepEval baseline collected; future run will fill this)")
print("promptfoo: N/A or fill from promptfoo's own re-run")

print("\n=== Aggregate metrics (A1 vs B — capture of injected regression) ===\n")
print("Evalith:", metrics(ev_a1b, PREDICTED_AFFECTED, all_cases))
print("DeepEval:", metrics(de_a1b, PREDICTED_AFFECTED, all_cases))
print("promptfoo: <fill from promptfoo per-case verdicts above>")
```

- [ ] **Step 2: Run it and capture the output**

Run: `python3 docs/blog/article2/compare.py | tee /tmp/article2_compare.txt`
Expected: prints the cross-tool table + aggregate metrics.

- [ ] **Step 3: If promptfoo's output shape doesn't match `load_promptfoo`'s assumption, fix the loader and rerun**

This is the most likely place to need adaptation. Look at the raw JSON shape (`python3 -c "import json; print(list(json.load(open('docs/blog/article2/raw/promptfoo_a1.json')).keys()))"`) and adjust `load_promptfoo`. Recommit if needed.

- [ ] **Step 4: Commit the compare script (even if you'll re-run it later)**

```bash
git add docs/blog/article2/compare.py
git commit -m "feat(blog/article2): cross-tool comparison script

Computes per-case verdicts and aggregate metrics (false-alarm rate,
capture rate, precision) for Evalith / DeepEval / promptfoo against
the pre-committed hypothesis from article §4."
```

### Task 12: Write §5 cross-tool prose + "where they beat us" para

**Files:**
- Modify: `docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md`

- [ ] **Step 1: Replace `<TODO §5 …>` with the cross-tool prose**

Use the output from `compare.py` in Task 11. Template:

```markdown
我们把同一份 dataset、同一份 prompt 偏差注入,跑在 promptfoo 和 DeepEval 上,各自用它们的官方推荐姿势配置(配置文件全部公开在 `docs/blog/article2/configs/`)。然后看三个工具对**实验 B 的 10 个 case** 各自的判定:

<PASTE THE PER-CASE TABLE FROM compare.py>

汇总指标:

| 工具 | A1-vs-A2 假阳率 | A1-vs-B 捕获率 | A1-vs-B 精度 |
|---|---|---|---|
| Evalith (bootstrap CI) | <X>/<N> | <Y>/<M> | <Z>% |
| DeepEval | <fill> | <fill> | <fill> |
| promptfoo | <fill> | <fill> | <fill> |

观察:
1. **<具体观察 1,基于实际数据>** — 例如 "DeepEval 在 A1-vs-B 上漏报了 N 个 case,因为它默认按 pass/fail 阈值做点对点比较,采样里 1-2 次抖动会把 borderline case 判成 unchanged"。
2. **<具体观察 2>** — promptfoo 在 noise 上的表现如何。
3. **<具体观察 3>** — Evalith 在哪些 case 上反而漏报或误报,**坦白讲**。

---

### promptfoo 和 DeepEval 在哪些方面强于 Evalith

诚实声明:即便这场对比里 Evalith 在"统计显著性 gate"这个维度赢了,这两个工具在其它非常重要的维度上仍然遥遥领先:

- **promptfoo**: 安装即用的 web UI、覆盖广得多的 provider、CI 集成示例丰富、社区 plugin 生态成熟。如果你的诉求是"快速建立 eval 流水线",promptfoo 仍然是最实用的起点。
- **DeepEval**: pytest-native 接入、内建 metrics(faithfulness / answer relevance / hallucination)选择广、RAG 工作流支持完善、企业版本提供数据集托管和 UI。当你需要细粒度 metric 拆分时它更顺手。

Evalith 不是"取代它们",而是补上一块它们普遍欠缺的能力 —— **基于统计显著性的 CI gate**。最现实的姿势可能是「用 promptfoo 起步 + 用 Evalith 给 CI 把关」,而不是非此即彼。
```

- [ ] **Step 2: Fill the angle-bracketed numbers from `compare.py`'s actual output**

This is a manual step. Do not commit until every `<X>` is replaced with a real number. **Do not fabricate numbers** — if a metric couldn't be computed (e.g., promptfoo schema parse incomplete), say so explicitly.

- [ ] **Step 3: Commit**

```bash
git add docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md
git commit -m "docs(blog/article2): §5 cross-tool comparison + 'where they're better' para"
```

---

## Phase 4: Article completion

### Task 13: Write §1 hook + §2 experiment redesign rationale

**Files:**
- Modify: `docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md`

- [ ] **Step 1: Write §1**

Replace `<TODO §1 prose — Task 13>` with prose covering these beats (mirror article 1's tone — direct, problem-first, no emoji):

1. Quote the article 1 limitation paragraph verbatim (already in the top blockquote — make §1 itself open with re-framing it).
2. State the question: "如果 LLM 真的开始抖, bootstrap 究竟是真在抗噪, 还是只是在没噪声的地方画 CI?"
3. Promise three things this article will do: (a) 重新设计一个有噪声的实验, (b) 跑 A 组噪声基线 + B 组真实回退两组对照, (c) 用同一份数据让 promptfoo 和 DeepEval 同台跑一遍。
4. ~300 字 total.

- [ ] **Step 2: Write §2**

Replace `<TODO §2 prose — Task 13>` with prose covering:

1. 为什么换 dataset: 上篇是事实题,新版改为 10 道中等技术深度的开放问答(简介题目类型不展开列出 case id —— 让表格自己说)。
2. 为什么换 scorer: `contains` 不行了 → 必须 llm_judge → 引入了"第二个噪声源"(judge 也是 LLM)。
3. 为什么 temp 调到 1.0 而不是 1.5: 1.5 会触发 hallucination 让 judge 给 0 分变成主流, "拖动" 被 "糟" 盖过。
4. 为什么 5 samples: 与上篇一致, 便于读者类比上篇的 CI 计算。
5. 总成本: ~300 个 model 调用, 不到 $0.20, 任何人都能本机复现。
6. ~500 字 total.

- [ ] **Step 3: Commit**

```bash
git add docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md
git commit -m "docs(blog/article2): §1 hook + §2 experiment redesign rationale"
```

### Task 14: Write §6 synthesis + §7 limitations (with article 3 promise)

**Files:**
- Modify: `docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md`

- [ ] **Step 1: Write §6**

Replace `<TODO §6 synthesis — Task 14>` with two distinct points, each 1-2 段:

1. **"统计显著性不是可选项"** — 在 llm_judge 评分体系下,单样本 point-compare 几乎不可能稳定;A 组数据直接证明了这点。任何不加 CI 的 gate 在生产里只能是"过严则烦,过松则瞎"。
2. **"LLM-as-judge 不是中立观察者"** — judge 本身在边界 case 上的判定也会抖。这意味着 bootstrap 实际上是在为"模型噪声 × judge 噪声"两层抖动同时画 CI。这是上篇没强调出来的核心机制。

- [ ] **Step 2: Write §7**

Replace `<TODO §7 limitations + article 3 promise — Task 14>` with:

```markdown
## 七、局限和下一步

老规矩,把诚实的话写在最后:

1. **n=10 case 仍是小样本** —— 横评结论别过度泛化。这只是 lower-bound 演示。
2. **prompt 偏差注入是人为的** —— 真实业务回退可能比这隐蔽得多。bootstrap 的优势在更隐蔽的场景下应该更显著,但本文没继续做。
3. **judge 与被评模型同源** —— 同一个 deepseek-chat 既出答案又评分,有"亲缘偏差"的风险。下一篇会用 GPT-4o-mini 做第三方 judge 旁证。
4. **统计工具栈仍是最朴素的 percentile bootstrap** —— BCa(偏置纠正)、paired bootstrap(减方差)、FDR(多重比较校正)这些都还没上。**第三篇会专门把这些一一加进 Evalith 并对照本篇的同一份 raw 数据重跑,看哪些指标实际改变。**

如果你也在做 LLM eval 或 AI CI 集成,欢迎到 [github.com/dominciyue/Evalith_MingJing](https://github.com/dominciyue/Evalith_MingJing) 提 issue / PR。

```bash
pip install evalith
```

即装即用,Apache-2.0。
```

- [ ] **Step 3: Commit**

```bash
git add docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md
git commit -m "docs(blog/article2): §6 synthesis + §7 limitations + article 3 promise"
```

### Task 15: Write `experiment.sh` for one-line reproduction

**Files:**
- Create: `docs/blog/article2/experiment.sh`

- [ ] **Step 1: Write the orchestration script**

`docs/blog/article2/experiment.sh`:
```bash
#!/usr/bin/env bash
# Reproduce article-2 experiments end-to-end. Outputs Markdown tables for §3,
# §4, §5 plus saves raw JSONs.
#
# Usage: DEEPSEEK_API_KEY=sk-... bash docs/blog/article2/experiment.sh
#
# What it does:
#   Phase 1 (Evalith): runs A1, A2, B → raw/a1.json, a2.json, b.json
#                       prints §3 noise-floor table and §4 regression table
#   Phase 2 (DeepEval): runs A1-equivalent and B-equivalent
#   Phase 3 (promptfoo): runs A1-equivalent and B-equivalent
#   Phase 4 (Synthesis): runs compare.py for the §5 cross-tool table

set -euo pipefail

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: set DEEPSEEK_API_KEY" >&2; exit 1
fi

cd "$(dirname "$0")/../../.."  # repo root

mkdir -p docs/blog/article2/raw

echo "=== Phase 1: Evalith A1, A2, B ==="
python3 -m evalith.cli run docs/blog/article2/configs/eval.high-temp.yaml \
    --samples 5 --out docs/blog/article2/raw/a1.json
python3 -m evalith.cli run docs/blog/article2/configs/eval.high-temp.yaml \
    --samples 5 --out docs/blog/article2/raw/a2.json
python3 -m evalith.cli run docs/blog/article2/configs/eval.high-temp.broken.yaml \
    --samples 5 --out docs/blog/article2/raw/b.json

echo "=== Phase 2: DeepEval ==="
python3 docs/blog/article2/configs/deepeval_compare.py

echo "=== Phase 3: promptfoo ==="
promptfoo eval -c docs/blog/article2/configs/promptfoo.yaml \
    -o docs/blog/article2/raw/promptfoo_a1.json
promptfoo eval -c docs/blog/article2/configs/promptfoo.broken.yaml \
    -o docs/blog/article2/raw/promptfoo_b.json

echo "=== Phase 4: Cross-tool comparison ==="
python3 docs/blog/article2/compare.py

echo ""
echo "Done. Raw outputs in docs/blog/article2/raw/."
```

- [ ] **Step 2: chmod + verify it can at least parse**

```bash
chmod +x docs/blog/article2/experiment.sh
bash -n docs/blog/article2/experiment.sh && echo "syntax OK"
```

- [ ] **Step 3: Commit**

```bash
git add docs/blog/article2/experiment.sh
git commit -m "feat(blog/article2): one-line reproduction script for the full experiment"
```

---

## Phase 5: Finalize

### Task 16: Article self-review (factual accuracy, no fabrication)

**Files:** Read-only at this step (modify if issues found).

- [ ] **Step 1: Skim the entire article cold and check each table against raw JSON**

For every number that appears in §3, §4, §5 tables, find the corresponding JSON file under `docs/blog/article2/raw/` and verify the number actually matches. If ANY number was made up (even rounded incorrectly), fix it.

- [ ] **Step 2: Check no `<TODO>` or angle-bracket placeholders remain**

Run: `grep -n "TODO\|<fill\|<paste\|<X>\|<Y>" docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md`
Expected: no matches. If any, fill them with real content before committing.

- [ ] **Step 3: Check the §4 hypothesis-vs-results comparison is honestly written**

Verify that:
- If a STRONG case was NOT regressed, the article names the case and explains why (don't quietly drop it).
- If a SAFE case WAS regressed, the article names it as a false positive and discusses.
- The hypothesis-check numbers match Task 8 Step 2's output.

- [ ] **Step 4: Verify total length is in 3500-4500 字 range**

```bash
python3 -c "import re; t = open('docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md').read(); print('char count:', len(re.sub(r'\s','',t)))"
```
Expected: ~3500-4500. Significantly under → some section is thin; over → trim §5 or §6.

- [ ] **Step 5: If any fix was made, commit**

```bash
git add docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md
git commit -m "docs(blog/article2): self-review pass — factual + length checks"
```

### Task 17: Rename article file to publication date, push everything

**Files:**
- Rename: `docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md` → `docs/blog/<actual-date>-llm-judge-noise-bootstrap.zh.md`

- [ ] **Step 1: Pick the publication date (today, when actually publishing)**

Use `date +%Y-%m-%d` for the live filename.

- [ ] **Step 2: Rename and commit**

```bash
PUB=$(date +%Y-%m-%d)
git mv docs/blog/2026-05-31-llm-judge-noise-bootstrap.zh.md docs/blog/${PUB}-llm-judge-noise-bootstrap.zh.md
git commit -m "docs(blog/article2): stamp publication date ${PUB}"
```

- [ ] **Step 3: Update `docs/README.md` to link the new article**

Add a bullet in `docs/README.md` under `blog/`:
```markdown
- [`<date>-llm-judge-noise-bootstrap.zh.md`](blog/<date>-llm-judge-noise-bootstrap.zh.md) — 续:LLM 当 judge 自己也在抖 —— bootstrap CI 抗噪与三个 OSS eval 工具的同台对照 (中文)
```

- [ ] **Step 4: Update top-level README "Read more" sections to link article 2 alongside article 1**

In both `README.md` and `README.zh-CN.md`, add a bullet for article 2 next to article 1's link.

- [ ] **Step 5: Run pytest one last time to confirm nothing was accidentally broken**

Run: `pytest -q`
Expected: `63 passed` (or whatever current count).

- [ ] **Step 6: Push to GitHub**

```bash
git push origin main
```
Expected: success.

- [ ] **Step 7: Verify the article renders on GitHub**

Open `https://github.com/dominciyue/Evalith_MingJing/blob/main/docs/blog/<PUB>-llm-judge-noise-bootstrap.zh.md` in a browser and skim it for rendering issues (tables, code blocks, headings).

---

## Self-review (executed before saving this plan)

**Spec coverage:**
- ✅ §1 (background) — implicit context for Task 5/13.
- ✅ §2 (goals 1/2/3) — Tasks 6-12 deliver all three (gap-fill, horizontal compare, credibility rails).
- ✅ §3 (non-goals) — no Evalith feature work; no RAG/creative/English/extra-tools. Plan explicitly does none.
- ✅ §4 (article structure 7 sections) — Tasks 5/7/8/12/13/14 cover §1-§7.
- ✅ §5.1 (dataset 10 cases + expected_concepts) — Task 2.
- ✅ §5.2 (run config) — Task 3.
- ✅ §5.3 (regression injection) — Task 3 (broken config).
- ✅ §5.4 (pre-committed hypothesis) — Task 5 Step 1 + 2 ensures hypothesis lands in git BEFORE experiment runs.
- ✅ §6.1 (fairness invariants) — Tasks 9/10 hold model/temp constant; Task 11 keeps comparison logic as the only varying axis.
- ✅ §6.2 (shared metrics) — Task 11 computes all four.
- ✅ §6.3 (three credibility rails) — raw outputs committed (Tasks 6/9/10), configs side-by-side committed (Tasks 3/9/10), "where they're better" paragraph (Task 12 Step 1).
- ✅ §7 (repo deliverables) — Tasks 2/3/9/10/11/15 produce the full file tree from §7.
- ✅ §8 (honest disclosures) — Task 14 §7 prose covers all 4. Task 16 verifies they made it in.
- ✅ §9 (success criteria) — Task 16's checks map 1:1 to success criteria 1-6.
- ✅ §10 (out-of-scope follow-ups) — article 3 promise is in Task 14 Step 2; English version is genuinely out.
- ✅ Appendix A — judge-temperature is verified resolved in Task 1 Step 2; ground-truth pre-commit is Task 5; criteria wording is in Task 3 Step 1/2 (not deferred — actual string written).

**Placeholder scan:**
- The article filename uses `2026-05-31` placeholder during writing; Task 17 renames it. This is intentional and documented in the plan header.
- Angle-bracket `<X>` / `<list>` placeholders inside Task 8 Step 3 and Task 12 Step 1 prose templates exist — but Task 8 Step 3 / Task 12 Step 2 explicitly require filling them with real numbers before commit, and Task 16 Step 2 greps for any leftover. So these are tracked, not silent.
- No "TBD" / "TODO" / "appropriate error handling" / vague-implement-later strings.

**Type consistency:**
- Function names referenced (`diff_runs`, `Run.model_validate_json`, `bootstrap_diff_ci`) match the codebase as verified in pre-plan exploration.
- Field names (`pass_rate_samples`, `case_id`, `ci`, `before`, `after`, `status`) match `src/evalith/diff.py` and `src/evalith/models.py`.
- `--samples` flag and `--out` flag match the CLI (verified via `python3 -m evalith.cli run --help` in Task 1).

Plan is internally consistent and spec-complete.
