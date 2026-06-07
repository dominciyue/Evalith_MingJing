from evalith.models import TestCase
from evalith.scorers.hard import CodeExec, extract_code


def test_extract_fenced_python():
    out = "Here you go:\n```python\ndef f():\n    return 1\n```\nDone"
    assert extract_code(out) == "def f():\n    return 1"


def test_extract_bare_fence():
    out = "```\nx = 2\n```"
    assert extract_code(out) == "x = 2"


def test_extract_no_fence_returns_whole():
    assert extract_code("def f(): return 1") == "def f(): return 1"


def test_extract_first_block_when_multiple():
    out = "```python\na = 1\n```\nthen\n```python\nb = 2\n```"
    assert extract_code(out) == "a = 1"


def test_extract_empty_returns_none():
    assert extract_code("   \n  ") is None


_TEST = "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n"


def _case(**meta):
    return TestCase(id="c1", input="add a and b", metadata=meta)


def test_code_exec_passes_correct_solution():
    case = _case(entry_point="add", test=_TEST)
    out = "```python\ndef add(a, b):\n    return a + b\n```"
    score = CodeExec().score(case, out)
    assert score.passed is True
    assert score.value == 1.0


def test_code_exec_fails_wrong_solution():
    case = _case(entry_point="add", test=_TEST)
    out = "```python\ndef add(a, b):\n    return a - b\n```"
    score = CodeExec().score(case, out)
    assert score.passed is False
    assert "assert" in score.detail.lower() or "AssertionError" in score.detail


def test_code_exec_missing_metadata():
    score = CodeExec().score(_case(), "def add(a, b): return a + b")
    assert score.passed is False
    assert "missing" in score.detail


def test_code_exec_no_code_in_output():
    case = _case(entry_point="add", test=_TEST)
    score = CodeExec().score(case, "   ")
    assert score.passed is False
    assert "no code" in score.detail


from evalith.scorers.hard import NumericMatch


def _ncase(expected):
    return TestCase(id="n1", input="compute", expected=expected)


def test_numeric_exact():
    s = NumericMatch().score(_ncase("42"), "the answer is 42")
    assert s.passed is True


def test_numeric_within_tol():
    s = NumericMatch(rel_tol=1e-3).score(_ncase("3.14159"), "result is 3.1416")
    assert s.passed is True


def test_numeric_outside_tol():
    s = NumericMatch(rel_tol=1e-3).score(_ncase("3.14159"), "about 3.0")
    assert s.passed is False


def test_numeric_no_number():
    s = NumericMatch().score(_ncase("42"), "no digits here")
    assert s.passed is False
    assert "no number" in s.detail


def test_numeric_expected_not_numeric():
    s = NumericMatch().score(_ncase("cat"), "42")
    assert s.passed is False
    assert "not numeric" in s.detail
