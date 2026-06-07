from evalith.scorers.sandbox import run_program


def test_passing_program():
    ok, detail = run_program("assert 1 + 1 == 2")
    assert ok is True
    assert detail == "ok"


def test_failing_assert():
    ok, detail = run_program("assert 1 == 2")
    assert ok is False
    assert "AssertionError" in detail or "assert" in detail.lower()


def test_timeout():
    ok, detail = run_program("while True:\n    pass", timeout=1)
    assert ok is False
    assert "timeout" in detail.lower()


def test_memory_limit():
    ok, detail = run_program("x = [0] * (10 ** 9)", memory_mb=128)
    assert ok is False


def test_dangerous_call_blocked():
    ok, detail = run_program("import os\nos.system('echo hi')")
    assert ok is False
