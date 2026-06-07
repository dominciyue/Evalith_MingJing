"""Build examples/code.humaneval.yaml from the canonical HumanEval set.

Reuses the same problem ids as article 4's he-* cases so the execution
ground truth can be contrasted directly against the judge panel.

Run: python3 docs/blog/article4/build_code_exec_dataset.py
"""
from __future__ import annotations

from pathlib import Path

import yaml
from datasets import load_dataset

IDS = [151, 28, 163, 108, 62, 70]
PROMPT = ("补全下面的 Python 函数。只输出完整的函数体(从 def 开始,可读、能跑通)。"
          "\n\n```python\n{sig}```")

OUT = Path(__file__).resolve().parents[3] / "examples" / "code.humaneval.yaml"


def main() -> None:
    rows = {r["task_id"]: r for r in load_dataset("openai_humaneval", split="test")}
    cases = []
    for n in IDS:
        r = rows[f"HumanEval/{n}"]
        cases.append({
            "id": f"he-humaneval-{n}",
            "source": "HumanEval",
            "domain": "code",
            "input": PROMPT.format(sig=r["prompt"]),
            "metadata": {"entry_point": r["entry_point"], "test": r["test"]},
        })
    data = {"name": "code-humaneval-exec-v1", "cases": cases}
    OUT.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    print(f"wrote {OUT} with {len(cases)} cases")


if __name__ == "__main__":
    main()
