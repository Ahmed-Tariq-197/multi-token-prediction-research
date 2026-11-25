# AIE417 — Multi-Token Prediction: Base NTP, Standard MTP, Improved MTP

Controlled study of next-token prediction (NTP), multi-token prediction (MTP) and one research-informed MTP
improvement on a ~30M-parameter Llama-style model, sized for a free Colab/Kaggle T4.

**Status (Phase 1 code).** Implemented and CPU-tested: model with k parallel heads, shifted/masked targets, weighted
loss, deterministic data, resumable training loop, shared evaluator, data+tokenizer pipeline, calibration/budget/decode-profiling
tools. **Not yet done:** GPU runs, real-data preparation, trained results, the Phase 2 improvement. No result in this repo
is a trained-model result yet. See `docs/AUDIT_PHASE1.md` and `docs/DECISIONS.md`.

## Notation
`k` = total number of heads, including the main head. Head `i` predicts `x_{t+i}`; head 1 is the main NTP head; `k=1` is Base NTP.
`L = L_1 + λ/(k-1) · Σ_{i≥2} L_i`. Details and the document-masking rule: `docs/DECISIONS.md` (ADR-004).

## Quick start
```bash
pip install -r requirements.txt
pytest -q                                   # 34 tests, CPU is enough
python scripts/make_synthetic_data.py --out data/synthetic
PYTHONPATH=src python -m mtp.train --config configs/smoke.yaml        # CPU smoke run (toy data, not a result)
```
Real pipeline (on the GPU machine):
```bash
python scripts/prepare_data.py --source hf --out data/fineweb_edu_8k --gpt2-token-budget 700000000 --vocab-size 8192
python scripts/calibrate.py --config configs/mtp_k4.yaml --k 1 2 4 --micro-batch 8 16 32 --out calib_t4.json
python scripts/budget.py --calib calib_t4.json --arm ntp:1 --arm mtp_k4:4 --seeds 3 --weekly-quota 30 --accounts 4
python -m mtp.train --config configs/base_ntp.yaml --set seed=1 train.total_tokens=<from budget.py>
python -m mtp.evaluate --config configs/base_ntp.yaml --ckpt runs/base_ntp_s1/final.pt --split val
```

## Layout
`src/mtp/` model, targets, loss, data, train, evaluate · `scripts/` data prep, calibration, budget, decode profiling ·
`tests/` correctness suite · `configs/` per-arm YAML · `docs/` decisions, calibration protocol, workflow, audit, validation artifacts.

## Provenance
All code in `src/`, `scripts/`, `tests/` was written for this project (AI-assisted authoring; disclosure required by the course).
Reference-only: Soft-MTP (MIT, github.com/JaydenTeoh/Soft-MTP) was read and its identity tests were run for validation; none of
its code is copied.
