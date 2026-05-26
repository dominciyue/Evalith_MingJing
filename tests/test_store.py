from datetime import datetime, timezone

from evalith.models import CaseResult, Run
from evalith.store import RunStore


def _run(rid: str) -> Run:
    return Run(id=rid, name="t", created_at=datetime(2026, 5, 25, tzinfo=timezone.utc),
               model="echo", results=[CaseResult(case_id="1", input="q", output="a")])


def test_store_round_trip(tmp_path):
    store = RunStore(tmp_path / "data")
    store.save(_run("r1"))
    loaded = store.load("r1")
    assert loaded.id == "r1"
    assert loaded.results[0].output == "a"


def test_store_list_runs(tmp_path):
    store = RunStore(tmp_path / "data")
    store.save(_run("r1"))
    store.save(_run("r2"))
    ids = {r.id for r in store.list_runs()}
    assert ids == {"r1", "r2"}
