from evalith.config import EvalConfig, ScorerConfig
from evalith.engine import run_eval
from evalith.providers.base import FakeProvider


def test_run_eval_with_fake_provider(tmp_path):
    ds = tmp_path / "ds.yaml"
    ds.write_text(
        "name: d\ncases:\n  - id: '1'\n    input: hello\n    expected: HELLO\n",
        encoding="utf-8",
    )
    cfg = EvalConfig(
        name="t", dataset=str(ds), model="fake", prompt_template="{{input}}",
        scorers=[ScorerConfig(type="contains", params={"text": "HELLO"})],
    )
    provider = FakeProvider(responses={"hello": "HELLO world"})
    run = run_eval(cfg, provider)
    assert len(run.results) == 1
    assert run.results[0].output == "HELLO world"
    assert run.results[0].scores[0].passed is True
    assert run.model == "fake"
    assert run.config["name"] == "t"


def test_run_eval_renders_template(tmp_path):
    ds = tmp_path / "ds.yaml"
    ds.write_text("name: d\ncases:\n  - id: '1'\n    input: world\n", encoding="utf-8")
    cfg = EvalConfig(name="t", dataset=str(ds), model="echo",
                     prompt_template="Hello {{input}}")
    from evalith.providers.base import EchoProvider

    run = run_eval(cfg, EchoProvider())
    assert run.results[0].output == "Hello world"


def test_run_eval_concurrent_preserves_order(tmp_path):
    ds = tmp_path / "ds.yaml"
    cases = "".join(f"  - id: '{i}'\n    input: q{i}\n" for i in range(20))
    ds.write_text(f"name: d\ncases:\n{cases}", encoding="utf-8")
    cfg = EvalConfig(name="t", dataset=str(ds), model="echo", prompt_template="{{input}}",
                     scorers=[ScorerConfig(type="contains", params={"text": "q"})])
    run = run_eval(cfg, FakeProvider(default="q-default"), concurrency=8)
    assert [r.case_id for r in run.results] == [str(i) for i in range(20)]
    assert len(run.results) == 20


def test_run_eval_with_samples_records_per_trial_pass_rates(tmp_path):
    """samples=N -> each CaseResult.pass_rate_samples has N values, one per trial."""
    from evalith.providers.base import Response

    class FlipProvider:
        """Alternates between matching and missing the expected text."""
        model = "flip"

        def __init__(self):
            self.calls = 0

        def complete(self, prompt, *, system=None, temperature=0.0):
            self.calls += 1
            text = "ok" if self.calls % 2 == 1 else "miss"
            return Response(text=text)

    ds = tmp_path / "ds.yaml"
    ds.write_text("name: d\ncases:\n  - id: c1\n    input: hi\n", encoding="utf-8")
    cfg = EvalConfig(name="t", dataset=str(ds), model="flip", prompt_template="{{input}}",
                     scorers=[ScorerConfig(type="contains", params={"text": "ok"})],
                     samples=4)
    run = run_eval(cfg, FlipProvider())
    assert len(run.results) == 1                              # one case, still one result
    cr = run.results[0]
    assert len(cr.pass_rate_samples) == 4                     # one pass-rate per trial
    # alternating ok/miss -> 1,0,1,0 -> mean 0.5
    assert cr.pass_rate_samples == [1.0, 0.0, 1.0, 0.0]


def test_run_eval_survives_provider_error(tmp_path):
    from evalith.providers.base import Response

    class FlakyProvider:
        model = "flaky"

        def complete(self, prompt, *, system=None, temperature=0.0):
            if "boom" in prompt:
                raise RuntimeError("rate limited")
            return Response(text="ok")

    ds = tmp_path / "ds.yaml"
    ds.write_text("name: d\ncases:\n  - id: good\n    input: hi\n"
                  "  - id: bad\n    input: boom\n", encoding="utf-8")
    cfg = EvalConfig(name="t", dataset=str(ds), model="flaky", prompt_template="{{input}}",
                     scorers=[ScorerConfig(type="contains", params={"text": "ok"})])
    run = run_eval(cfg, FlakyProvider(), concurrency=4)
    assert len(run.results) == 2                       # one bad case did NOT kill the run
    by_id = {r.case_id: r for r in run.results}
    assert by_id["good"].scores[0].passed is True
    assert by_id["bad"].scores and all(not s.passed for s in by_id["bad"].scores)


# ---------------------------------------------------------------------------
# Adaptive sampling tests (P5)
# ---------------------------------------------------------------------------

def test_run_eval_adaptive_stops_early_when_stable():
    """Adaptive mode: when all pass_rate_samples are 1.0, stop at min_samples."""
    from evalith.config import load_config
    from evalith.engine import run_eval
    from evalith.providers.base import EchoProvider

    # examples/eval.yaml uses a `contains` scorer on echo model — always passes
    cfg = load_config("examples/eval.yaml")
    cfg = cfg.model_copy(update={
        "adaptive": True,
        "min_samples": 2,
        "max_samples": 10,
        "ci_tolerance": 0.05,
    })
    provider = EchoProvider()
    run = run_eval(cfg, provider)
    for case in run.results:
        assert len(case.pass_rate_samples) == 2, (
            f"{case.case_id}: expected stable case to stop at min_samples=2, "
            f"got {len(case.pass_rate_samples)}"
        )


def test_run_eval_adaptive_runs_to_max_when_noisy(tmp_path):
    """Adaptive mode: when pass-rate samples oscillate, run to max_samples."""
    from evalith.config import EvalConfig, ScorerConfig
    from evalith.engine import run_eval
    from evalith.providers.base import Response

    class FlipProvider:
        """Alternates between matching and missing the expected text."""
        model = "flip"

        def __init__(self):
            self.calls = 0

        def complete(self, prompt, *, system=None, temperature=0.0):
            self.calls += 1
            text = "ok" if self.calls % 2 == 1 else "miss"
            return Response(text=text)

    ds = tmp_path / "ds.yaml"
    ds.write_text(
        "name: d\ncases:\n  - id: flipcase\n    input: x\n    expected: ok\n",
        encoding="utf-8",
    )
    cfg = EvalConfig(
        name="flip-noisy",
        dataset=str(ds),
        model="flip",
        prompt_template="{{input}}",
        scorers=[ScorerConfig(type="contains", params={"text": "ok"})],
        adaptive=True,
        min_samples=2,
        max_samples=6,
        ci_tolerance=0.01,
    )
    run = run_eval(cfg, FlipProvider())
    case = run.results[0]
    assert len(case.pass_rate_samples) == 6, (
        f"expected adaptive to run to max=6 on noisy data, got {len(case.pass_rate_samples)}"
    )


def test_run_eval_fixed_samples_unchanged():
    """Without adaptive flag, fixed --samples N path is identical to v0.5."""
    from evalith.config import load_config
    from evalith.engine import run_eval
    from evalith.providers.base import EchoProvider

    cfg = load_config("examples/eval.yaml")
    cfg = cfg.model_copy(update={"samples": 3})
    provider = EchoProvider()
    run = run_eval(cfg, provider)
    for case in run.results:
        assert len(case.pass_rate_samples) == 3
