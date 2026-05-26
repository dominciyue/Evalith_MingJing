from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_action_yml_is_valid_composite():
    data = yaml.safe_load((ROOT / "action.yml").read_text(encoding="utf-8"))
    assert data["runs"]["using"] == "composite"
    assert "config" in data["inputs"]
    assert "fail-under" in data["inputs"]


def test_example_workflow_valid():
    data = yaml.safe_load((ROOT / ".github/workflows/eval-example.yml").read_text(encoding="utf-8"))
    assert "jobs" in data
    job = next(iter(data["jobs"].values()))
    assert any("uses" in step or "run" in step for step in job["steps"])
