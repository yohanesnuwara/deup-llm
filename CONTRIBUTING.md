# Contributing to deup

Thank you for helping improve the canonical open-source implementation of
[DEUP](https://openreview.net/forum?id=eGLdVRvvfQ) (Lahlou et al., 2023).

## Quick start

```bash
git clone https://github.com/ursinasanderink/deup.git
cd deup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,gbm,finance,docs]"
pre-commit install   # optional but recommended
pytest
ruff check . && ruff format --check .
mypy
mkdocs build --strict
```

## What we welcome

- Bug fixes with a regression test
- Documentation improvements (tutorials, API clarity, benchmarks)
- New **optional** extras (e.g. additional GBM or domain presets) that follow existing patterns
- Benchmarks that reproduce on CPU in CI (or are marked `@pytest.mark.integration`)

## Design principles

1. **Leakage-correct OOF errors** — never train `g` on in-sample residuals
2. **Sklearn-compatible estimators** — `fit` / `predict` / `get_params`
3. **Thin domain presets** — wire axes; don't fork core logic
4. **Honest uncertainty** — document when aggregation guards apply

See `ARCHITECTURE.md` for the five-axis mental model.

## Pull request checklist

- [ ] Tests pass locally (`pytest`)
- [ ] Lint + types pass (`ruff`, `mypy`)
- [ ] Docs build (`mkdocs build --strict`) if you changed docs or public API
- [ ] `CHANGELOG.md` updated under `[Unreleased]` for user-visible changes
- [ ] New public symbols have NumPy-style docstrings

## Reporting issues

Use [GitHub Issues](https://github.com/ursinasanderink/deup/issues). Include:

- Python version, OS, `pip show deup`
- Minimal reproducible example
- Expected vs actual behavior

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## Attribution

DEUP the **method** is due to Lahlou et al. (2023). You are contributing to the
**library** — please do not claim authorship of the method in docs or marketing copy.

## Release process

Maintainers: see [`RELEASING.md`](RELEASING.md).
