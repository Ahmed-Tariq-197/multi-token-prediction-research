"""GPU calibration: measure steady-state throughput and peak memory per (k, micro-batch, compile) trial.

  python scripts/calibrate.py --config configs/mtp_k4.yaml --k 1 4 --micro-batch 8 16 32 --steps 40 --out calib_t4.json

Protocol (docs/CALIBRATION.md): synthetic random token windows (throughput does not depend on token values),
the same loss/optimizer code path as training, warm-up steps discarded, torch.cuda.synchronize around the timed
region, OOM caught and recorded. Output is consumed by scripts/budget.py. Nothing here is a model-quality result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mtp.config import load_config  # noqa: E402
from mtp.evaluate import autocast_ctx  # noqa: E402
from mtp.loss import multi_head_loss  # noqa: E402
from mtp.model import MTPModel  # noqa: E402
from mtp.targets import build_head_targets  # noqa: E402
from mtp.train import build_optimizer  # noqa: E402
from mtp.utils import atomic_save, env_info  # noqa: E402


def trial(cfg, k, micro, compile_, steps, warmup, device, precision):
    cfg.model.num_pred_heads, cfg.train.micro_batch_size = k, micro
    cfg.data.max_horizon = max(cfg.data.max_horizon, k)
    d = cfg.data
    model = MTPModel(cfg.model, init_seed=0).to(device)
    opt = build_optimizer(model, cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and precision == "fp16"))
    fwd = torch.compile(model.heads_forward) if compile_ else model.heads_forward
    V = cfg.model.vocab_size
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def step():
        window = torch.randint(1, V, (micro, d.seq_len + d.max_horizon), device=device)
        tg = build_head_targets(window, d.seq_len, k, cfg.model.eos_id)
        with autocast_ctx(device, precision):
            hs = fwd(window[:, :d.seq_len])
            loss, _ = multi_head_loss(hs, model.embed_tokens.weight, tg, cfg.loss.lam)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

    for _ in range(warmup):
        step()
    sync(); t0 = time.perf_counter()
    for _ in range(steps):
        step()
    sync(); dt = time.perf_counter() - t0
    tok = steps * micro * d.seq_len
    # checkpoint round trip
    p = "/tmp/_calib_ckpt.pt"
    t1 = time.perf_counter(); atomic_save({"model": model.state_dict(), "opt": opt.state_dict()}, p)
    save_s = time.perf_counter() - t1
    size_mb = os.path.getsize(p) / 2**20; os.remove(p)
    return {"k": k, "micro_batch": micro, "compile": compile_, "precision": precision,
            "tokens_per_s": tok / dt, "step_ms": 1000 * dt / steps,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None,
            "ckpt_save_s": save_s, "ckpt_mb": size_mb, "oom": False,
            "params_main": model.num_params(False), "params_total": model.num_params(True)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--micro-batch", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--compile", type=int, nargs="+", default=[0], help="0/1 values to try")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default="calibration.json")
    a = ap.parse_args()
    results = []
    for k in a.k:
        for mb in a.micro_batch:
            for comp in a.compile:
                cfg = load_config(a.config, a.set + [f"train.global_batch_tokens={mb * load_config(a.config, a.set).data.seq_len}"])
                device = cfg.train.device or ("cuda" if torch.cuda.is_available() else "cpu")
                precision = cfg.train.precision if device == "cuda" else "fp32"
                try:
                    r = trial(cfg, k, mb, bool(comp), a.steps, a.warmup, device, precision)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    r = {"k": k, "micro_batch": mb, "compile": bool(comp), "oom": True}
                print(json.dumps(r)); results.append(r)
    out = {"env": env_info(), "config": a.config, "overrides": a.set, "trials": results}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
