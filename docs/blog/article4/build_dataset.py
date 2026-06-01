"""Build the expanded article 4 dataset by subsetting public benchmarks +
adding self-made open-ended questions. Output is a single YAML file consumed
by Evalith.

Subsets (deterministic seed=42 for reproducibility):
  - HumanEval: 15 of 164 Python coding problems (id format: he-<task_id>)
  - GSM8K: 15 of 8500 grade-school math problems (id format: gsm-<idx>)
  - TruthfulQA: 10 of 817 misconception-prone questions (id format: tq-<idx>)
  - MMLU: 10 across 2-3 subjects (id format: mmlu-<subj>-<idx>)
  - Self-made: 15-20 open-ended concept-explanation questions (id format: self-<topic>)

For each case, fields:
  - id: stable id
  - source: benchmark origin
  - domain: code | math | safety | knowledge | concept-explanation
  - input: the prompt the model sees
  - expected_concepts: list of strings the judge can check coverage on
                       (for code/math: includes the correct answer/output;
                        for open-ended: 2-4 key concepts)
  - notes: optional human-readable annotation (e.g. answer for benchmark cases)
"""
import json
import random
from pathlib import Path

import yaml
from datasets import load_dataset

random.seed(42)

OUT_PATH = Path("docs/blog/article4/qa.large.yaml")
ALL_CASES = []

# ---------- HumanEval ----------
print("Loading HumanEval...")
he = load_dataset("openai/openai_humaneval", split="test")
he_indices = random.sample(range(len(he)), 15)
for i in he_indices:
    row = he[i]
    task_id = row["task_id"].replace("/", "-").lower()
    prompt_text = row["prompt"].strip()
    canonical = row["canonical_solution"].strip()
    # Extract function name from prompt for expected_concepts
    func_name = row["entry_point"]
    ALL_CASES.append({
        "id": f"he-{task_id}",
        "source": "HumanEval",
        "domain": "code",
        "input": f"补全下面的 Python 函数。只输出完整的函数体(从 def 开始,可读、能跑通)。\n\n```python\n{prompt_text}```",
        "expected_concepts": [
            f"函数 {func_name} 的正确实现",
            "返回值类型正确",
            "处理 docstring 中描述的所有情形",
        ],
        "notes": f"canonical_solution (truncated 240 chars): {canonical[:240]}",
    })

# ---------- GSM8K ----------
print("Loading GSM8K...")
gsm = load_dataset("openai/gsm8k", "main", split="test")
gsm_indices = random.sample(range(len(gsm)), 15)
for i in gsm_indices:
    row = gsm[i]
    question = row["question"].strip()
    answer = row["answer"].strip()
    # Extract final numeric answer (after ####)
    final = answer.split("####")[-1].strip() if "####" in answer else answer
    ALL_CASES.append({
        "id": f"gsm-{i}",
        "source": "GSM8K",
        "domain": "math",
        "input": f"解下面的应用题,最后给出数字答案。\n\n{question}",
        "expected_concepts": [
            f"最终数值答案是 {final}",
            "推导过程合理(列出关键计算步骤)",
        ],
        "notes": f"reference reasoning (truncated 200 chars): {answer[:200]}",
    })

# ---------- TruthfulQA ----------
print("Loading TruthfulQA...")
tq = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
tq_indices = random.sample(range(len(tq)), 10)
for i in tq_indices:
    row = tq[i]
    question = row["question"]
    best = row.get("best_answer", "") or (row.get("correct_answers", [""])[:1] or [""])[0]
    ALL_CASES.append({
        "id": f"tq-{i}",
        "source": "TruthfulQA",
        "domain": "safety",
        "input": f"请如实回答下面这个问题(注意常见的错误流传说法或迷思,要给出真实事实):\n\n{question}",
        "expected_concepts": [
            "回答符合科学/事实共识",
            "明确拒绝或纠正常见的错误流传说法",
        ],
        "notes": f"reference best_answer (truncated 200 chars): {best[:200]}",
    })

# ---------- MMLU (2 subjects, 5 each) ----------
print("Loading MMLU (computer_security + machine_learning)...")
mmlu_subjects = ["computer_security", "machine_learning"]
for subj in mmlu_subjects:
    try:
        mmlu = load_dataset("cais/mmlu", subj, split="test")
    except Exception as e:
        print(f"  MMLU {subj} failed: {e}, trying alternate path")
        mmlu = load_dataset("hails/mmlu_no_train", subj, split="test")
    indices = random.sample(range(len(mmlu)), 5)
    for i in indices:
        row = mmlu[i]
        q = row["question"]
        choices = row["choices"]
        answer_idx = row["answer"]
        answer_letter = "ABCD"[answer_idx]
        answer_text = choices[answer_idx]
        choices_block = "\n".join(f"{c}. {ch}" for c, ch in zip("ABCD", choices))
        ALL_CASES.append({
            "id": f"mmlu-{subj}-{i}",
            "source": f"MMLU/{subj}",
            "domain": "knowledge",
            "input": f"以下是一道选择题,选出正确选项并简短说明理由。\n\n{q}\n\n{choices_block}",
            "expected_concepts": [
                f"正确答案是 {answer_letter}",
                f"理由提到关键概念: {answer_text[:60]}",
            ],
            "notes": f"correct: {answer_letter}. {answer_text[:140]}",
        })

# ---------- Self-made (15 cases — Chinese open-ended concept explanations) ----------
print("Adding 15 self-made cases...")
SELF_MADE = [
    # ML / AI
    {"id": "self-attention-head", "domain": "concept-explanation",
     "input": "Transformer 中的多头注意力(multi-head attention)是什么?为什么用多头而不是单头?用 4-6 句话回答。",
     "expected_concepts": ["多头并行", "不同子空间", "增加表示容量", "Q/K/V 拆分到多个头"]},
    {"id": "self-lora", "domain": "concept-explanation",
     "input": "LoRA 微调原理是什么?为什么参数效率高?用 4-6 句话回答。",
     "expected_concepts": ["低秩矩阵分解", "冻结原参数", "只训练低秩矩阵", "参数量大幅减少"]},
    {"id": "self-quantization", "domain": "concept-explanation",
     "input": "INT8 量化与 FP16 推理相比,主要的精度损失来自哪里?用 4-6 句话回答。",
     "expected_concepts": ["量化粒度", "激活值动态范围", "outlier 处理", "校准 calibration"]},
    # 系统设计
    {"id": "self-eventual-consistency", "domain": "concept-explanation",
     "input": "最终一致性(eventual consistency)与强一致性的核心权衡是什么?用 4-6 句话回答。",
     "expected_concepts": ["CAP 取舍", "可用性 vs 一致性", "副本同步延迟", "应用层处理冲突"]},
    {"id": "self-circuit-breaker", "domain": "concept-explanation",
     "input": "微服务里熔断器(circuit breaker)的作用 + 三态(closed/open/half-open)各自代表什么?用 4-6 句话回答。",
     "expected_concepts": ["故障级联防护", "closed 正常调用", "open 短路降级", "half-open 探测恢复"]},
    # 系统/网络
    {"id": "self-tls-handshake", "domain": "concept-explanation",
     "input": "TLS 1.3 与 TLS 1.2 在握手流程上的主要区别是什么?为什么 1.3 更快?用 4-6 句话回答。",
     "expected_concepts": ["1-RTT 握手", "去除旧加密套件", "0-RTT resume", "ephemeral key exchange 默认"]},
    {"id": "self-tcp-fastopen", "domain": "concept-explanation",
     "input": "TCP Fast Open 解决什么问题?为什么有部署阻碍?用 4-6 句话回答。",
     "expected_concepts": ["首次 RTT 携带数据", "cookie 防护", "中间盒子兼容", "需要 OS + 应用支持"]},
    # 数据库
    {"id": "self-mvcc", "domain": "concept-explanation",
     "input": "MVCC(多版本并发控制)的核心思想是什么?相比锁定有什么优势?用 4-6 句话回答。",
     "expected_concepts": ["版本快照", "读不阻塞写", "事务 ID 排序", "GC 回收旧版本"]},
    {"id": "self-bloom-filter", "domain": "concept-explanation",
     "input": "布隆过滤器(Bloom filter)的工作原理 + 假阳率怎么控制?用 4-6 句话回答。",
     "expected_concepts": ["多个哈希函数", "位数组", "不会假阴只会假阳", "bit数 / hash数 / 元素数 三者权衡"]},
    # 安全
    {"id": "self-csrf-vs-xss", "domain": "concept-explanation",
     "input": "CSRF 与 XSS 攻击的本质区别是什么?对应的防御手段?用 4-6 句话回答。",
     "expected_concepts": ["CSRF 利用已认证状态", "XSS 注入恶意脚本", "CSRF token 防御", "CSP / 转义 防御 XSS"]},
    {"id": "self-jwt-revoke", "domain": "concept-explanation",
     "input": "JWT 撤销难做的根本原因是什么?有哪些常见折中方案?用 4-6 句话回答。",
     "expected_concepts": ["无状态特性", "服务端不存 token", "黑名单需引入状态", "短 expiry + refresh token"]},
    # AI eval / 统计
    {"id": "self-perplexity", "domain": "concept-explanation",
     "input": "Perplexity 作为 LLM 评测指标的优势和局限是什么?用 4-6 句话回答。",
     "expected_concepts": ["几何平均概率", "对训练分布敏感", "不反映任务能力", "比较跨 tokenizer 不公平"]},
    {"id": "self-cohen-kappa", "domain": "concept-explanation",
     "input": "Cohen's kappa 度量两个评分者一致性时为什么比简单一致率好?用 4-6 句话回答。",
     "expected_concepts": ["扣除随机一致部分", "范围 [-1, 1]", "kappa<0.4 一致性差", "依赖类别分布"]},
    # 经典算法
    {"id": "self-consistent-hashing", "domain": "concept-explanation",
     "input": "一致性哈希(consistent hashing)解决传统哈希的什么问题?虚拟节点的作用?用 4-6 句话回答。",
     "expected_concepts": ["节点变化时重新映射代价小", "环形空间", "虚拟节点均衡负载", "降低数据倾斜"]},
    {"id": "self-paxos-vs-raft", "domain": "concept-explanation",
     "input": "Paxos 和 Raft 共识算法的本质相同点和工程区别是什么?用 4-6 句话回答。",
     "expected_concepts": ["都基于 majority quorum", "Raft 强 leader 简化理解", "Paxos 论文晦涩", "Raft 日志连续"]},
]

for c in SELF_MADE:
    ALL_CASES.append({
        "id": c["id"],
        "source": "self-made",
        "domain": c["domain"],
        "input": c["input"],
        "expected_concepts": c["expected_concepts"],
    })

# ---------- Write ----------
print(f"\nTotal cases: {len(ALL_CASES)}")
domain_counts = {}
source_counts = {}
for c in ALL_CASES:
    domain_counts[c["domain"]] = domain_counts.get(c["domain"], 0) + 1
    source_counts[c["source"]] = source_counts.get(c["source"], 0) + 1
print("By domain:", domain_counts)
print("By source:", source_counts)

# Write as Evalith-compatible dataset YAML
ds = {
    "name": "qa-large-mixed-v1",
    "cases": ALL_CASES,
}
OUT_PATH.write_text(yaml.safe_dump(ds, allow_unicode=True, sort_keys=False, width=120))
print(f"\nwrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")
