# Contributing to Evalith

Thanks for considering a contribution. This repo aims to stay small, well-tested,
and easy to reason about — please keep that in mind when proposing changes.

## Dev setup

```bash
git clone git@github.com:dominciyue/Evalith_MingJing.git
cd Evalith_MingJing
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,litellm]"
pytest -q
```

All tests should pass with zero warnings before you start.

## Workflow

1. **Open an issue first** for anything beyond a typo or small docs tweak.
   It saves you time if the proposal isn't a fit for the project's scope.
2. **Branch from `main`** — e.g. `feat/scorer-rougeL`, `fix/diff-empty-baseline`.
3. **Write the test first.** Evalith is TDD-built; new behavior comes with a
   failing test that turns green. See `docs/design/plans/` for examples of how
   tasks are decomposed.
4. **Keep PRs focused.** One concern per PR. If you find unrelated cleanups,
   open a separate PR for them.
5. **Conventional commits** — `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`,
   `test:`. Scope optional (`feat(diff): …`).

## Running the full check locally

```bash
pytest -q                                  # unit tests
python -m evalith.cli run examples/eval.yaml          # offline smoke
# Optional real-model smoke (needs DEEPSEEK_API_KEY + pip install ".[litellm]"):
DEEPSEEK_API_KEY=sk-... python -m evalith.cli run examples/eval.deepseek.yaml
```

## Project layout

```
src/evalith/        # core package: cli, engine, diff, models, report, scorers, providers
tests/              # unit tests (pytest)
examples/           # runnable sample configs + datasets
docs/blog/          # long-form articles (public)
docs/design/        # historical spec + TDD plans (v0.1, v0.2)
.github/workflows/  # CI (pytest matrix) + example eval-gate action
```

## Releasing (maintainers)

1. Bump `version` in `pyproject.toml`.
2. `python -m build`
3. `twine check dist/*`
4. `twine upload dist/*`
5. Tag the commit `vX.Y.Z` and push.

## Questions

Open an issue or start a discussion. For background on the design choices
(bootstrap CI, per-case isolation, file-based baselines), the Chinese deep-dive
article is the best reference: <https://zhuanlan.zhihu.com/p/2043351926964848178>.
