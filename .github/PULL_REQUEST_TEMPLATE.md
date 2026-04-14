<!--
Thanks for opening a PR.

Keep the description short — the diff speaks for itself. Fill in "what" and
"why", then tick the checkboxes you actually did. Unchecked boxes are fine
if they genuinely don't apply; just say so in a comment.
-->

## What

<!-- One or two sentences: what this PR changes. -->

## Why

<!-- One or two sentences: the motivation. Link the issue if there is one: "Closes #42". -->

## How I tested

<!-- pytest output, screenshot, smoke test result, manual steps, "only touches docs" -- whatever applies. -->

## Checklist

- [ ] Linked to an issue (or explained in "Why" why there isn't one)
- [ ] `make check-all` passes locally (lint + tests + doctor)
- [ ] Tests added or updated to cover the change
- [ ] `--seed 42` preserved in any new training invocations
- [ ] If `results/metrics.json` changed: regression noted in the PR description
      and `results/metrics.baseline.json` + `CHANGELOG.md` updated
- [ ] Docs updated (README, CHANGELOG, docstrings) where behavior changed
- [ ] Notebook outputs stripped (`nbstripout` is in pre-commit; run it if skipped)
- [ ] No secrets, credentials, or >1 MB binaries added

## Reviewer notes

<!-- Anything a reviewer should look at first, or tradeoffs you want feedback on. Delete if nothing. -->
