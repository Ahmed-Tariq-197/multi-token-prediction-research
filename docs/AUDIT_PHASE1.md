# Phase 1 audit checklist (tick before any push to `main`)
Legend: [x] done and verified in the authoring environment (CPU sandbox, torch 2.15) · [~] partly verified · [ ] not done.
Test counts are deliberately not hard-coded here: CI output is the source of truth.

## Engineering (code)
- [x] Model, targets, loss, data, train/eval written; MTP tests for shifted targets, valid-position/cross-document masking,
      per-head gradient flow, no future leakage, KV-cache equivalence, paired-seed init
- [x] Reference parity with `transformers.LlamaForCausalLM` (fp64), two rope_theta values, plus a negative control
- [x] Resume exactness (CPU), clean logs after crash, finished-run immutability, config-drift refusal, manifest/data checks,
      non-finite step handling, versioned self-describing checkpoints, eval-overwrite protection
- [x] Shared evaluator verified against an independent per-token brute-force implementation; exact eval-window counts
- [x] Packaging: editable install, `mtp-train` / `mtp-eval` entry points, honest dependency bounds, `ruff check .` clean
- [x] CI workflow written (Ubuntu + Windows × Py 3.10/3.12, HF parity required) — **[ ] not yet executed on GitHub**
- [ ] Windows execution of the test-suite (fixes made for paths/encodings/line endings; only CI can confirm)
- [ ] GPU: full test-suite re-run on a T4; fp16/CUDA/GradScaler/torch.compile paths are UNVERIFIED
- [ ] `scripts/prepare_data.py --source hf` executed on a small real subset (unexecuted)

## Experiments / science
- [ ] Calibration sweep done, budget derived, configs updated (docs/CALIBRATION.md) — all budget figures are `[estimate]` until then
- [ ] LR sweep per arm; seed-variance pilot; Base NTP and Standard MTP trained (≥3 seeds), logs/configs/checkpoints kept
- [ ] One test-split evaluation per arm, at the end
- [ ] Draft paper (setup, MTP implementation, Base vs MTP results, limitations, improvement proposal with ≥3 sources)

## Team / process
- [ ] Every member ran the pipeline end to end and can explain targets/loss/masking without notes
- [ ] Module owners/reviewers filled in (CONTRIBUTING.md); branch protection enabled on `main`
- [ ] A second member reproduced the README commands on a clean clone
- [ ] Old remote branches `phase1/initial-scaffold`, `revert-1-phase1/initial-scaffold` deleted after the audit branch is published
