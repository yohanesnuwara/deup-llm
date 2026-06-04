# Releasing

## PyPI (one-time setup)

The release workflow (`.github/workflows/release.yml`) uses **PyPI trusted publishing**.
Configure once at [pypi.org](https://pypi.org/manage/account/publishing/):

| Field | Value |
|---|---|
| PyPI project name | `deup` |
| Owner | `ursinasanderink` |
| Repository | `deup` |
| Workflow | `release.yml` |
| Environment | (leave blank unless using GitHub Environment `pypi`) |

Then push a tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

**Alternative:** add a repository secret `PYPI_API_TOKEN` and the workflow will fall
back to token upload (see workflow file).

## Manual upload (if CI publish fails)

```bash
python -m pip install build twine
python -m build
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-XXXX twine upload dist/deup-0.1.0*
```

## GitHub Pages

Enable at **Settings → Pages → Build and deployment → GitHub Actions**.

Docs deploy automatically on push to `main` via `.github/workflows/docs.yml`.
