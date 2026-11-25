# Git workflow (PowerShell)

Policy: nothing is pushed to `origin` or merged to `main` until the stage is implemented, tested, documented and
audited. Local commits on private `wip/*` branches are allowed and expected (recoverable checkpoints; they carry each
member's authorship, which the course asks to be visible). `main` currently contains GitHub's initial commit (README) — keep it.

```powershell
# one-time
git clone https://github.com/Ahmed-Tariq-197/multi-token-prediction-research.git
Set-Location multi-token-prediction-research
git config user.name  "Your Name"
git config user.email "you@example.com"
git switch -c wip/phase1            # shared local integration branch (never pushed)

# per member, per task
git switch -c wip/phase1-<member>-<topic> wip/phase1
git add <files>; git commit -m "<area>: <what and why>"
git switch wip/phase1; git merge --no-ff wip/phase1-<member>-<topic>

# audit gate (all must be true before publishing)
python -m pytest -q
python scripts/calibrate.py --help | Out-Null
git status --short                  # must be clean
# checklist: docs/AUDIT_PHASE1.md all boxes ticked, a second member re-ran the tests on a clean clone

# publish (only after the audit)
git fetch origin
git switch -c phase1 wip/phase1
git push -u origin phase1
git switch main; git merge --no-ff phase1 -m "Phase 1: Base NTP + Standard MTP"
git tag -a phase1-v1.0 -m "Phase 1 submission"
git push origin main --follow-tags
```
Never use `git push --force` on `main`. Large artifacts (`data/`, `runs/`, `*.pt`) are git-ignored; publish small result
JSON/CSV under `results/` and attach checkpoints to a GitHub Release if needed.
