## What and why
<!-- One paragraph. Link the issue / ADR this implements. -->

## Verification (state what you actually ran)
- [ ] `ruff check .` clean
- [ ] `pytest -q` passes locally (paste the summary line): 
- [ ] New/changed behaviour has a test (or explain why not)
- [ ] GPU-only code path touched? If yes: ran on a T4, GPU model + torch/CUDA versions: 
- [ ] Docs updated (README / docs/DECISIONS.md / configs) where behaviour or numbers changed

## Honesty checklist
- [ ] No result, number or claim in this PR is unmeasured, or it is labelled `[estimate]` / `[assumption]`
- [ ] AI assistance used? Disclosed in commit trailer (`Assisted-by:`) and, if substantial, in the PR text
- [ ] I can explain every changed line without notes (course rule: each member defends the whole project)

## Review
Reviewer must be a different member from the author and must re-run the tests on a clean clone.
