# GPU calibration protocol (derives the training budget; nothing here is a model-quality result)

Run on the **same hardware class used for all reported runs** (Kaggle/Colab T4). Record the GPU name printed by the
script; mixing P100 and T4 results in one comparison is not allowed.

1. **Environment.** `pip install -r requirements.txt`; run `pytest -q` on the GPU machine (all tests must pass; the
   fp16/CUDA paths are NOT exercised by the CPU test-suite and are unverified until this step).
2. **Quota meter.** Note the account's remaining GPU hours before and after step 3; compute hours consumed per wall hour
   (does a T4×2 session count 1× or 2×?). Use only one GPU per run unless a concurrency test shows no slowdown.
3. **Throughput/memory sweep** (~15 min):
   `python scripts/calibrate.py --config configs/mtp_k4.yaml --k 1 2 4 --micro-batch 8 16 32 --compile 0 1 --steps 40 --out calib_t4.json`
   Look for: OOM boundary per k; tokens/s per k (expected ratio k=4 : k=1 ≈ 1/1.7, ADR-003); fp16 `scale` collapsing to
   tiny values or NaN losses in a 300-step real-data smoke run of k=4 (stability check); torch.compile benefit/failure on sm_75.
4. **Session survival.** Start a real run, kill the notebook mid-run, restart, confirm `latest.pt` resume continues the
   same trajectory (CPU tests prove exactness; confirm on GPU). Measure checkpoint save time/size from the sweep output.
5. **LR sweep** (equal effort per arm): {3e-4, 6e-4, 1e-3, 2e-3} × {Base NTP, MTP k=4} at ~20% of the final token budget,
   one seed, select by validation loss of the main head. Different optimal LRs → each arm uses its own tuned LR and the
   paper says so.
6. **Seed-variance pilot** (Phase 1): 3 seeds × {Base NTP, MTP k=4} at the final budget → σ for main-head val loss;
   feeds ADR-006 (δ and number of seeds).
7. **Derive the budget:**
   `python scripts/budget.py --calib calib_t4.json --arm ntp:1 --arm mtp_k4:4 --arm soft_k4:4:1.05 --arm kl_k4:4:1.10 --arm mtp_k2:2 --arm pm_ntp:1:1.25 --arm curr_k4:4 --seeds 3 --weekly-quota <meter> --accounts <n>`
   Binding constraint, tokens/param, hours per run and campaign hours are printed. If it warns about undertraining, shrink the
   arm list or model before launching anything. Copy the resulting `total_tokens` into every config in one commit.
8. Freeze: record GPU, torch/CUDA versions, dataset revision, tokenizer hash, final budget, LR per arm in `docs/RUN_PROTOCOL.md`.

Assumed (not measured) numbers in ADR-001/003/008 are replaced by the measured ones at this point.
