# TODO — Brandon (repo admin only)

Items below require admin rights on `bpyle02/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset`, or Kaggle account ownership for `brandonpyle/...`. The rest of the team has push but not admin, so only you can do these.

Delete this file once everything is checked off.

---

## 0. Publish the Kaggle notebook as a Kernel (one-time, ~2 minutes)

**Why:** makes the "Open in Kaggle" badge in the README functional. Currently the badge points at `kaggle.com/code/brandonpyle/wm811k-wafer-defect-quickstart` which 404s until you publish. After this step, anyone who clicks the badge lands on a published kernel they can Copy & Edit with GPU P100, Internet, and the WM-811K dataset all pre-attached — no manual setup.

**Prerequisite:** install and authenticate the Kaggle CLI once:
```bash
pip install kaggle
# Download kaggle.json from https://www.kaggle.com/settings/account ("Create API Token")
# Save it to ~/.kaggle/kaggle.json (macOS/Linux) or %USERPROFILE%\.kaggle\kaggle.json (Windows)
chmod 600 ~/.kaggle/kaggle.json
```

**Publish:**
```bash
cd /path/to/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset
kaggle kernels push -p .
```

The `kernel-metadata.json` at the repo root tells the Kaggle CLI to:
- use `docs/kaggle_quickstart.ipynb` as the code file
- enable GPU P100 and Internet
- attach `brandonpyle/wm-811k-wafer-map` as the input dataset
- publish at `kaggle.com/code/brandonpyle/wm811k-wafer-defect-quickstart`
- keep it public

**Re-publish after notebook changes:**
```bash
kaggle kernels push -p .
```
Same command — it creates a new version and keeps the URL stable, so the README badge keeps working.

**Verify it worked:**
```bash
gh browse  # or just open https://www.kaggle.com/code/brandonpyle/wm811k-wafer-defect-quickstart
```
The kernel page should show GPU P100 / Internet On / dataset attached in the right sidebar. Click **Copy & Edit → Run All** to confirm it trains end-to-end for an uninvolved user.

---

## 1. Enable branch protection on `main`

**Why:** prevents anyone (including you on a bad day) from force-pushing or merging unreviewed changes to `main`. Pairs with the `CODEOWNERS` file already committed.

### Option A — one-shot `gh` command (fastest)

```bash
gh api -X PUT repos/bpyle02/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset/branches/main/protection \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["test", "lint"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF
```

**Before running:** confirm the CI job names by checking `.github/workflows/ci.yml`. If the job keys there are not literally `test` and `lint`, replace the `contexts` array above with whatever the actual job names are. Run `gh run list --limit 1 --json name,conclusion` to see recent check names.

### Option B — web UI (if `gh` is acting up)

1. Go to **Settings → Branches → Add branch protection rule**
2. Branch name pattern: `main`
3. Check these boxes:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (1)
   - ✅ Dismiss stale pull request approvals when new commits are pushed
   - ✅ Require review from Code Owners
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
     - Search for and select: `test`, `lint` (or whatever your CI job names are)
   - ✅ Do not allow bypassing the above settings
   - ❌ Allow force pushes (leave unchecked)
   - ❌ Allow deletions (leave unchecked)
4. Click **Create**

### Verify it worked

```bash
gh api repos/bpyle02/CNN-Based-Semiconductor-Wafer-Defect-Detection-Using-the-WM-811K-Dataset/branches/main/protection \
  --jq '{pr_required: .required_pull_request_reviews.required_approving_review_count, codeowner_review: .required_pull_request_reviews.require_code_owner_reviews, force_push: .allow_force_pushes.enabled, status_checks: .required_status_checks.contexts}'
```

Expected:
```
{"pr_required": 1, "codeowner_review": true, "force_push": false, "status_checks": ["test", "lint"]}
```

---

## 2. Decide and commit a permanent license (team vote)

**Current state:** `LICENSE` says "All Rights Reserved" — we can't legally accept outside contributions or let classmates reuse the code until we pick something.

**Recommended options** (5-minute team discussion):

| License | What it means |
|---------|---------------|
| **MIT** | "Do anything, just credit us, no warranty." Most permissive. Most common for academic/research code. |
| **Apache-2.0** | Same as MIT + explicit patent grant. Slightly more legal armor. |
| **BSD-3-Clause** | Like MIT but also forbids using our names to endorse derivatives. |
| **Keep private** | Leave "All Rights Reserved." Viewable on GitHub but nobody can legally fork. |

**My recommendation:** MIT. It's the default for coursework and keeps things simple.

Once agreed, drop the corresponding SPDX-standard text into `LICENSE`, update the README badge (`license-All%20Rights%20Reserved-lightgrey` → `license-MIT-green` or similar), and bump `CITATION.cff` if needed. Any of us can do the PR once the team picks.

---

## 3. Tag a release

**Why:** gives the course grader (and your future self) an immutable reference point. Tags appear in the GitHub sidebar and show up on the "Releases" page.

```bash
git tag -a v0.1.0 -m "Initial release — end-of-course submission baseline"
git push origin v0.1.0
gh release create v0.1.0 --title "v0.1.0 — Course submission" --notes "Reported metrics: CNN accuracy 0.9611, macro F1 0.7988 (see results/metrics.json)."
```

Do this right before final submission so the grader can cite a stable SHA.

---

## 4. (Optional) Fill in CODEOWNERS

`.github/CODEOWNERS` currently routes everything to `@bpyle02`. The commented lines below show how each teammate can claim specific paths (`src/models/` → Anindita, etc.). Ping the team for their GitHub handles, uncomment, commit.

This isn't urgent but reduces the volume of review pings you get once branch protection + `require_code_owner_reviews` is live.

---

## 5. (Optional) Dependabot

Auto-PRs for dependency updates. One file:

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "monthly"
    open-pull-requests-limit: 3
```

Skip this if you don't want the noise during the course.

---

_Last updated: 2026-04-14 by ddisqq. Delete this file once items 1–3 are done._
