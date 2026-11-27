# AIE417 — Multi-Token Prediction: Base NTP, Standard MTP, Improved MTP

Controlled study of next-token prediction (NTP), multi-token prediction (MTP) and one research-informed MTP
improvement on a ~30M-parameter Llama-style model, sized for a free Colab/Kaggle T4.

**Status — Phase 1 engineering code.** Implemented and CPU-tested: model with k parallel heads, shifted/masked targets,
weighted loss, deterministic data sampling, resumable training loop, shared evaluator, data + tokenizer pipeline,
calibration / budget / decode-profiling tools, CI. **Not done yet:** any GPU run, real-data preparation, trained results,
the Phase 2 improvement. **No result in this repository is a trained-model result.** What is and is not verified is
tracked in [`docs/AUDIT_PHASE1.md`](docs/AUDIT_PHASE1.md); design rationale and open assumptions in
[`docs/DECISIONS.md`](docs/DECISIONS.md).

## Install and test (Windows PowerShell or Linux/macOS)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1                   # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU build is enough for the tests
pip install -e ".[dev]"                      # Colab/Kaggle: torch is preinstalled -> pip install -r requirements.txt
ruff check .
pytest -q                                    # set REQUIRE_HF_PARITY=1 to make a missing `transformers` a failure
```
Python >= 3.10. Declared minimum torch is 2.4 (API availability); only torch 2.15 on CPU has been exercised so far, CI
covers Ubuntu and Windows on Python 3.10/3.12.

## Smoke run (CPU, synthetic toy data — not a result)
```powershell
python scripts/make_synthetic_data.py --out data/synthetic
mtp-train --config configs/smoke.yaml
mtp-eval  --config configs/smoke.yaml --ckpt runs/smoke/final.pt --split val
```

## Real pipeline (GPU machine; steps marked * have NOT been executed yet)
```powershell
python scripts/prepare_data.py --source hf --out data/fineweb_edu_8k --revision <dataset-commit-hash> --gpt2-token-budget 700000000 --vocab-size 8192   # *
python scripts/calibrate.py --config configs/mtp_k4.yaml --k 1 2 4 --micro-batch 8 16 32 --out calib_t4.json                                       # *
python scripts/budget.py --calib calib_t4.json --arm ntp:1 --arm mtp_k4:4 --seeds 3 --weekly-quota <your meter> --accounts 4                          # *
mtp-train --config configs/base_ntp.yaml --set seed=1 train.total_tokens=<from budget.py>                                                           # *
mtp-eval  --config configs/base_ntp.yaml --ckpt runs/base_ntp_s1/final.pt --split val                                                                # *
```
`out_dir` is templated (`runs/{name}_s{seed}`), so changing `seed` or the config name never reuses a run directory.
Details and the full protocol: [`docs/CALIBRATION.md`](docs/CALIBRATION.md).

## Notation
`k` = total number of prediction heads, including the main head. Head `i` predicts `x_{t+i}`; head 1 is the main NTP
head; `k = 1` is Base NTP. `L = L_1 + λ/(k-1) · Σ_{i≥2} L_i`. Valid-position rule: a target is used only if no EOS lies
between the input position and the target (the EOS that ends the document is a legal target). See ADR-004.

## Reproducibility and result integrity (all covered by tests unless marked)
* Seeded per-parameter init: the trunk of a k=4 model is bit-identical to the k=1 model with the same seed.
* Stateless data order `(data_seed, step, micro)` and identical input windows for every k.
* Resume restores model/optimizer/scaler/step, truncates `log.jsonl` to the checkpoint step, and **refuses** a changed
  trajectory configuration (seed, model, data protocol, LR schedule, budget); cosmetic changes (paths, logging cadence,
  device) are allowed so a run can move between Colab and Kaggle.
* A finished run is immutable: re-running returns the stored `results.json`; `mtp-eval --out` refuses to overwrite.
* Data/manifest consistency (vocab, EOS id, token counts, token-id range) is checked before training; manifest records
  dataset revision, tokenizer hash and file SHA-256.
* Non-finite steps are skipped and counted; 10 in a row abort the run.
* **Not guaranteed:** bitwise-identical GPU trajectories (CUDA kernels differ across GPUs/library versions) — compare GPU
  runs statistically. Exact resume equivalence is verified on CPU only. Test-split numbers must be produced once, at the end.

## Layout
`src/mtp/` model, targets, loss, data, train, evaluate · `scripts/` data prep, calibration, budget, decode profiling ·
`tests/` correctness suite · `configs/` per-arm YAML · `docs/` decisions, calibration protocol, workflow, audit record,
validation artifacts · `.github/` CI and PR template · [`CONTRIBUTING.md`](CONTRIBUTING.md) branches, attribution, ownership.

## Provenance
The scaffold and the subsequent engineering audit were written with AI assistance (disclosure required by the course; see
commit authors/trailers in `git log`). Soft-MTP (MIT, github.com/JaydenTeoh/Soft-MTP) was read and its identity tests
run for validation only; none of its code is copied.
