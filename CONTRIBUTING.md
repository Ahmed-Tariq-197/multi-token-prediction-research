# Contributing

Four people, one repository, graded on individual understanding. The rules below keep history honest and reviewable.

## Setup (Windows PowerShell or Linux/macOS)
```powershell
git clone https://github.com/Ahmed-Tariq-197/multi-token-prediction-research.git
cd multi-token-prediction-research
python -m venv .venv
.venv\Scripts\Activate.ps1                 # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU is enough for the test-suite
pip install -e ".[dev]"
pytest -q ; ruff check .
```
On Colab/Kaggle torch is preinstalled: `pip install -r requirements.txt`.

## Branches
| Branch | Purpose | Rules |
|---|---|---|
| `main` | audited stages only | no direct pushes; PR + CI green + one reviewer |
| `stage/phase1`, `stage/phase2` | stage integration, merged to `main` (merge commit) at the audit gate | created from `main`; **not** `phase1` — it collides with the existing `phase1/initial-scaffold` ref |
| `feat/<area>-<topic>`, `fix/<topic>` | day-to-day work, one author | branch from the stage branch, PR into it |

Never force-push a shared branch. Never merge a branch that was already merged and then reverted: Git considers its
commits applied, so the merge silently brings nothing back (this happened with `phase1/initial-scaffold`; see
docs/WORKFLOW.md). Re-apply the work with `git cherry-pick` or open a new branch.

## Commits and attribution
* One logical change per commit, imperative subject: `area: what and why`, areas = `model`, `data`, `train`, `eval`, `tools`, `docs`, `ci`, `test`.
* Commit under your own name/e-mail (`git config user.name/user.email`): the course grades individual contribution.
* AI assistance must be disclosed (course rule): add a trailer `Assisted-by: <tool/model>` to commits where it
  contributed materially, and `Co-authored-by: Name <email>` for pair work. Do not commit AI output you cannot explain.
* The scaffold history (`feat(phase1): add reproducible MTP research scaffold`) and the audit commits that follow it are
  AI-assisted; the audit commits carry the author "Claude (AI-assisted authoring)" so provenance is visible in `git log`.

## Module ownership (fill in; every module also has a different reviewer)
| Module | Owner | Reviewer |
|---|---|---|
| data pipeline + tokenizer (`scripts/prepare_data.py`, `src/mtp/data.py`) | _TBD_ | _TBD_ |
| model + base NTP (`src/mtp/model.py`, `train.py`) | _TBD_ | _TBD_ |
| MTP heads, targets, loss (`targets.py`, `loss.py`) | _TBD_ | _TBD_ |
| evaluation + calibration (`evaluate.py`, `scripts/calibrate.py`, `budget.py`) | _TBD_ | _TBD_ |

## Repository settings to enable on GitHub (cannot be done from code)
Branch protection on `main`: require pull request, 1 approval, status checks `tests (...)` from CI, no force pushes,
dismiss stale approvals. Keep the repository public (course requirement).

## Definition of done for a change
Tests added/updated and passing, `ruff check .` clean, docs/ADRs updated when behaviour or numbers change, any number in
the docs is either measured (with the run that produced it) or labelled `[estimate]` / `[assumption]`.
