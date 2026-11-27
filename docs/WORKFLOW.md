# Git workflow (PowerShell) — and the state of `origin` as found

## What is on the remote today
`main` = GitHub's initial README commit, then PR #1 (`phase1/initial-scaffold`, the first AI-assisted scaffold) merged and
immediately reverted by PR #2. The scaffold branch `phase1/initial-scaffold` and the auto-created
`revert-1-phase1/initial-scaffold` still exist on `origin`.

**Do not merge `phase1/initial-scaffold` again.** Its commits are already ancestors of `main` (via PR #1), so merging it
brings nothing back — the revert wins. Re-apply the work with `git cherry-pick` onto a fresh branch (this is how
`wip/phase1-audit` was built: the scaffold commit is cherry-picked with its original author, audit commits follow).

**Branch names.** `phase1` cannot be created while `phase1/initial-scaffold` exists (Git refs cannot be both a file and a
directory; verified: `cannot lock ref 'refs/heads/phase1'`). Stage branches are therefore `stage/phase1`, `stage/phase2`.
The old scaffold branches can be deleted on GitHub once the audit branch is published.

## Policy
Nothing reaches `main` until the stage is implemented, tested, documented and audited. Local commits on private branches are
allowed and expected (recoverable checkpoints, and each member's authorship stays visible — the course grades individual
contribution). See CONTRIBUTING.md for commit/attribution rules.

## Receive the audited work (from the bundle or the zip with `.git`)
```powershell
git clone https://github.com/Ahmed-Tariq-197/multi-token-prediction-research.git
Set-Location multi-token-prediction-research
git fetch ..\mtp-phase1-audit.bundle wip/phase1-audit:wip/phase1-audit     # path to the bundle you were given
git log --oneline origin/main..wip/phase1-audit                            # review every commit before trusting it
git switch wip/phase1-audit
pip install -e ".[dev]"; pytest -q; ruff check .
```

## Day-to-day (per member)
```powershell
git switch -c feat/<area>-<topic> wip/phase1-audit        # later: stage/phase1
# ...work...
git add <files>; git commit -m "<area>: <what and why>"   # your own git config user.name/user.email
```

## Audit gate and publishing (only after docs/AUDIT_PHASE1.md is fully ticked)
```powershell
python -m pytest -q; ruff check .; git status --short      # all clean
git fetch origin
git switch -c stage/phase1 wip/phase1-audit
git push -u origin stage/phase1                            # open a PR into main; CI must be green; 1 reviewer
# after approval, on GitHub: merge commit (not squash — keeps per-member history). Then:
git switch main; git pull --ff-only
git tag -a phase1-v1.0 -m "Phase 1 submission"; git push origin phase1-v1.0
```
Never `git push --force` to `main` or a shared branch. Large artifacts (`data/`, `runs/`, `*.pt`) are git-ignored; commit
small result JSON/CSV under `results/` and attach checkpoints to a GitHub Release.
