from evalith.scorers.hard import extract_code


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
