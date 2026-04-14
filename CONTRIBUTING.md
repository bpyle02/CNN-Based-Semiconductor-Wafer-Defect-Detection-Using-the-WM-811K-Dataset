# Contributing

Guidelines for the four of us (and anyone we hand the project off to).

## Setup once

```bash
conda env create -f environment.yml        # creates py313 env
conda activate py313
pip install -e ".[dev]"                    # editable install
pytest -q                                  # should print "188 passed"
```

If `conda` isn't your thing: `pip install -r requirements.txt` works too.

## Everyday workflow

1. **Branch off `main`** — never commit directly to `main`.
   - Naming: `feat/<short-thing>`, `fix/<bug>`, `docs/<what>`, `experiment/<idea>`.
   - Example: `feat/grad-cam-overlay`, `fix/colab-cell-5b-hang`.

2. **Commit often, small, in imperative mood**
   - Good: `Add temperature scaling to classification report`
   - Bad:  `updated stuff`, `wip`, `fixing the thing Brett mentioned`
   - One concern per commit. If you catch yourself typing "and", split it.

3. **Run tests before pushing**
   ```bash
   pytest -q                 # unit + integration
   python -c "from src.models import WaferCNN; print('imports ok')"
   ```

4. **Push to your branch, open a PR**
   - Title: same style as commits.
   - Description: what + why + how you tested.
   - Tag a teammate for review. One approval is enough; CI must be green.

5. **Merge via "Squash and merge"** on GitHub, unless the PR has
   meaningfully distinct commits worth preserving (rare).

## What not to commit

- `data/LSWMD_new.pkl`, `data/LSWMD_cache.npz` (big binary; `.gitignore` covers it)
- `checkpoints/*.pth` (checkpoints; use W&B/MLflow or upload to Drive)
- `wafer_runs/`, `results/*.png`, `outputs/` (run artifacts)
- `.env`, `*.key`, anything with credentials
- IDE settings (`.idea/`, `.vscode/` beyond the shared `settings.json`)

## Style

- **Python**: type hints on public functions, f-strings, no `print()` in
  library code — use `logging`. Run `black src train.py` before pushing
  if you changed formatting a lot.
- **Commit messages**: imperative, <72-char first line, body for the *why*.
- **No emoji** in code, commits, or docstrings.
- **No AI assistant attribution** in commits or docs. Author yourself.

## Tests

- Unit tests live in `tests/unit/`, mirror `src/` layout.
- Integration tests in `tests/integration/` — these may be slower.
- When adding a feature, add a test. If you can't, note why in the PR.
- If a test is flaky or slow, tag it `@pytest.mark.slow` and skip by default.

## When something breaks

- **CI red on `main`**: open a PR reverting the offending commit, then
  investigate. Don't leave `main` broken overnight.
- **Colab notebook drift**: re-run end-to-end in Colab before merging
  anything that touches cells 4–7 of `docs/colab_quickstart.ipynb`.
- **Merge conflicts**: rebase your branch onto `main`, resolve locally,
  force-push to your own branch only. Never force-push to `main`.

## Signing commits (optional but encouraged)

GitHub shows a **Verified** badge on signed commits. To set up SSH signing:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/github_signing -C "you@example.com"
gh auth refresh -s admin:ssh_signing_key
gh api -X POST user/ssh_signing_keys \
  --field title="commit signing" \
  --field key="$(cat ~/.ssh/github_signing.pub)"
git config --local gpg.format ssh
git config --local user.signingkey ~/.ssh/github_signing.pub
git config --local commit.gpgsign true
```

## Questions

Ping the team in whatever channel you use. If it's architectural
(model choice, dataset split, eval metric), raise it as a GitHub
Discussion or PR comment so the answer is written down.
