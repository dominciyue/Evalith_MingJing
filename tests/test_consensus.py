import json
import math
from datetime import datetime, timezone
from pathlib import Path

from evalith.consensus import (
    MISSING,
    case_spread,
    cohen_kappa,
    consensus_summary,
    domain_agreement,
    judge_means,
    pairwise_kappa,
    threshold_from_config,
)
from evalith.models import CaseResult, Run


def _case(cid, prs, panel=None, domain=None):
    return CaseResult(case_id=cid, input="i", output="o",
                      pass_rate_samples=prs, panel_samples=panel or {},
                      domain=domain)


def _run(cases):
    return Run(id="r", name="n", created_at=datetime(2026, 6, 6, tzinfo=timezone.utc),
               model="m", results=cases)


# ---------------------------------------------------------------- kappa

def test_cohen_kappa_hand_computed():
    # po=0.5, pa1=pb1=0.5 -> pe=0.5 -> kappa=0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 1]) == 0.0
    # perfect agreement, mixed labels
    assert cohen_kappa([1, 0, 1], [1, 0, 1]) == 1.0
    # degenerate: both constant-and-equal -> pe=1 -> return 1.0
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0
    # empty / mismatched -> nan
    assert math.isnan(cohen_kappa([], []))
    assert math.isnan(cohen_kappa([1], [1, 0]))


def test_cohen_kappa_golden_article4():
    """Golden 回归:必须复现 multi_compare.py 在文章 4 raw 数据上的 κ。"""
    raw = Path("docs/blog/article4/raw")
    if not raw.exists():
        import pytest
        pytest.skip("article4 raw data not present")

    def labels(name):
        run = json.loads((raw / name).read_text(encoding="utf-8"))
        out = []
        for c in run["results"]:
            for v in (c.get("pass_rate_samples") or []):
                out.append(1 if v >= 0.5 else 0)
        return out

    assert abs(cohen_kappa(labels("j_ds_by_ds_a1.json"),
                           labels("j_ds_by_qw_a1.json")) - 0.253920) < 1e-4
    assert abs(cohen_kappa(labels("j_ds_by_ds_a1.json"),
                           labels("j_ds_by_glm_a1.json")) - 0.568138) < 1e-4
    assert abs(cohen_kappa(labels("j_ds_by_qw_a1.json"),
                           labels("j_ds_by_glm_a1.json")) - 0.252408) < 1e-4


# ---------------------------------------------------------------- spread

def test_judge_means_and_spread():
    c = _case("c1", [1.0, 1.0], panel={"qw": [0.0, 0.0], "glm": [1.0, MISSING]})
    m = judge_means(c)
    assert m["primary"] == 1.0
    assert m["qw"] == 0.0
    assert m["glm"] == 1.0          # MISSING trial skipped
    assert case_spread(c) == 1.0


def test_spread_zero_without_panel():
    assert case_spread(_case("c", [1.0, 0.0])) == 0.0


# ---------------------------------------------------------------- pairwise

def test_pairwise_kappa_pairwise_deletion():
    run = _run([
        _case("c1", [1.0, 1.0], panel={"qw": [1.0, MISSING]}),
        _case("c2", [0.0, 1.0], panel={"qw": [0.0, 0.0]}),
    ])
    k = pairwise_kappa(run)
    # labels after dropping the MISSING trial: primary=[1,0,1], qw=[1,0,0]
    # po=2/3, pa1=2/3, pb1=1/3 -> pe=2/9+2/9=4/9 -> kappa=(6/9-4/9)/(5/9)=0.4
    assert abs(k[("primary", "qw")] - 0.4) < 1e-9


# ---------------------------------------------------------------- domain

def test_domain_agreement_groups_and_flags():
    run = _run([
        _case("a", [1.0], panel={"qw": [0.0]}, domain="code"),     # spread 1.0 -> low
        _case("b", [1.0], panel={"qw": [1.0]}, domain="safety"),   # spread 0
        _case("c", [1.0], panel={"qw": [1.0]}),                    # untagged -> "?"
    ])
    d = domain_agreement(run, threshold=0.5)
    assert d["code"]["n"] == 1 and d["code"]["low_consensus"] == 1
    assert d["safety"]["low_consensus"] == 0
    assert "?" in d


# ---------------------------------------------------------------- summary

def test_consensus_summary_none_without_panel():
    assert consensus_summary(_run([_case("c", [1.0])])) is None


def test_consensus_summary_counts():
    run = _run([
        _case("a", [1.0, 1.0], panel={"qw": [0.0, 0.0]}),   # spread 1.0
        _case("b", [1.0, 1.0], panel={"qw": [1.0, 1.0]}),   # spread 0
    ])
    s = consensus_summary(run, threshold=0.5)
    assert s["judges"] == ["primary", "qw"]
    assert s["low_consensus_cases"] == ["a"]
    assert s["n_cases"] == 2
    assert ("primary", "qw") in s["kappa"]


def test_threshold_from_config():
    cfg = {"scorers": [{"type": "llm_judge",
                        "params": {"consensus_threshold": 0.3}}]}
    assert threshold_from_config(cfg) == 0.3
    assert threshold_from_config({}) == 0.5
