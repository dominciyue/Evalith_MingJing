from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from .models import Dataset, TestCase


def load_dataset(path: str | Path) -> Dataset:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return Dataset.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
    if suffix == ".json":
        return Dataset.model_validate(json.loads(p.read_text(encoding="utf-8")))
    if suffix == ".csv":
        with p.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        cases = [
            TestCase(
                id=row.get("id") or str(i),
                input=row["input"],
                expected=row.get("expected") or None,
            )
            for i, row in enumerate(rows)
        ]
        return Dataset(name=p.stem, cases=cases)
    if suffix == ".jsonl":
        cases = []
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cases.append(TestCase(
                id=str(row.get("id") or i),
                input=row["input"],
                expected=row.get("expected"),
                metadata=row.get("metadata") or {},
            ))
        return Dataset(name=p.stem, cases=cases)
    raise ValueError(f"Unsupported dataset format: {suffix}")
