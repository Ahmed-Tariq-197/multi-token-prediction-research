# Phase 1 audit checklist (tick before any push)
Status legend: [x] done & verified · [~] partly verified · [ ] not done. State as of authoring (CPU sandbox only).

- [x] Model, targets, loss, data, train/eval code written; 34 CPU tests pass (see README)
- [x] MTP tests: shifted targets, valid-position/cross-document masking, gradient flow per head, no-leak, KV-cache, HF parity
- [ ] GPU: full test-suite re-run on a T4 (fp16 / CUDA paths are unverified)
- [ ] Calibration sweep done, budget derived, configs updated (docs/CALIBRATION.md)
- [ ] Data prepared with the real source (hf path unexecuted); manifest + dataset revision pinned
- [ ] Base NTP and Standard MTP trained, ≥3 seeds each (pilot variance), logs/configs/checkpoints saved
- [ ] One shared eval run on test split (once, at the end): loss, PPL, top-1, train time, peak memory per arm
- [ ] Draft paper: setup, MTP implementation, Base vs MTP results, limitations, improvement proposal with ≥3 sources
- [ ] Every member has run the pipeline end to end and can explain targets/loss/masking without notes
- [ ] README reproduction commands executed verbatim on a clean clone by a second member
