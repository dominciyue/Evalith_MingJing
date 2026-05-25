import pytest

from mingjing.dataset import load_dataset


def test_load_yaml_dataset(tmp_path):
    p = tmp_path / "ds.yaml"
    p.write_text(
        "name: d\ncases:\n  - id: '1'\n    input: hello\n    expected: hi\n",
        encoding="utf-8",
    )
    ds = load_dataset(p)
    assert ds.name == "d"
    assert ds.cases[0].input == "hello"
    assert ds.cases[0].expected == "hi"


def test_load_csv_dataset(tmp_path):
    p = tmp_path / "ds.csv"
    p.write_text("id,input,expected\n1,hello,hi\n", encoding="utf-8")
    ds = load_dataset(p)
    assert ds.cases[0].id == "1"
    assert ds.cases[0].expected == "hi"


def test_unsupported_format_raises(tmp_path):
    p = tmp_path / "ds.txt"
    p.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(p)
