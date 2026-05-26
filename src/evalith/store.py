from __future__ import annotations

from pathlib import Path

from .models import Run


class RunStore:
    """Persists Runs as JSON files under <root>/runs/."""

    def __init__(self, root: str | Path = ".evalith"):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def save(self, run: Run) -> Path:
        path = self.runs_dir / f"{run.id}.json"
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, run_id: str) -> Run:
        path = self.runs_dir / f"{run_id}.json"
        return Run.model_validate_json(path.read_text(encoding="utf-8"))

    def list_runs(self) -> list[Run]:
        runs = [Run.model_validate_json(p.read_text(encoding="utf-8"))
                for p in self.runs_dir.glob("*.json")]
        return sorted(runs, key=lambda r: r.created_at, reverse=True)
