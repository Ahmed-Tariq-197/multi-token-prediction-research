"""Training loop for Base NTP (k=1) and Standard MTP (k>=2). Same code path for every arm.

Reproducibility: seeded init (paired across arms), stateless data order, atomic resumable checkpoints,
JSONL logs, config + environment snapshot, train/eval time accounted separately, peak GPU memory recorded.
The test split is never touched here (use `python -m mtp.evaluate --split test` once, at the end).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from .config import RunConfig, load_config
from .data import TokenFile, load_manifest
from .evaluate import autocast_ctx, evaluate
from .loss import multi_head_loss
from .model import MTPModel
from .targets import build_head_targets
from .utils import JsonlLogger, atomic_save, env_info, set_seed


def lr_at(step: int, total: int, cfg: RunConfig) -> float:
    t = cfg.train
    if step < t.warmup_steps:
        return t.lr * (step + 1) / t.warmup_steps
    prog = (step - t.warmup_steps) / max(1, total - t.warmup_steps)
    return t.lr * (t.min_lr_ratio + (1 - t.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


def build_optimizer(model: MTPModel, cfg: RunConfig, device: str):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": cfg.train.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.train.lr, betas=cfg.train.betas,
                             fused=(device == "cuda"))


def train(cfg: RunConfig, max_steps: int | None = None) -> dict:
    t, d = cfg.train, cfg.data
    device = t.device or ("cuda" if torch.cuda.is_available() else "cpu")
    precision = t.precision if device == "cuda" else "fp32"
    cfg.train.precision = precision
    os.makedirs(t.out_dir, exist_ok=True)
    set_seed(cfg.seed)
    data_seed = cfg.seed if d.data_seed is None else d.data_seed

    model = MTPModel(cfg.model, init_seed=cfg.seed).to(device)
    opt = build_optimizer(model, cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and precision == "fp16"))
    fwd = torch.compile(model.heads_forward) if t.compile else model.heads_forward

    train_data, val_data = TokenFile(d.train_bin), TokenFile(d.val_bin)
    bpt = load_manifest(d.manifest).get("bytes_per_token", {}).get("val")

    accum = t.global_batch_tokens // (t.micro_batch_size * d.seq_len)
    total_steps = math.ceil(t.total_tokens / t.global_batch_tokens)
    k = model.k

    state = {"step": 0, "train_time_s": 0.0, "eval_time_s": 0.0}
    ckpt_path = os.path.join(t.out_dir, "latest.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        print(f"[resume] step {state['step']}/{total_steps}")

    log = JsonlLogger(os.path.join(t.out_dir, "log.jsonl"))
    with open(os.path.join(t.out_dir, "config.json"), "w") as f:
        json.dump({"config": cfg.to_dict(), "fingerprint": cfg.fingerprint(), "env": env_info(),
                   "params_main_path": model.num_params(False), "params_total": model.num_params(True),
                   "total_steps": total_steps, "grad_accum": accum}, f, indent=2)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    model.train()
    stop = total_steps if max_steps is None else min(total_steps, max_steps)
    last_eval = None
    sync(); t_seg = time.perf_counter()

    for step in range(state["step"], stop):
        lr = lr_at(step, total_steps, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        acc_total, acc_heads = 0.0, torch.zeros(k)
        for micro in range(accum):
            window = train_data.train_batch(data_seed, step, micro, t.micro_batch_size, d.seq_len, d.max_horizon).to(device)
            ids = window[:, :d.seq_len]
            targets = build_head_targets(window, d.seq_len, k, cfg.model.eos_id, cfg.loss.mask_cross_document)
            with autocast_ctx(device, precision):
                hs = fwd(ids)
                loss, parts = multi_head_loss(hs, model.embed_tokens.weight, targets, cfg.loss.lam)
            scaler.scale(loss / accum).backward()
            acc_total += loss.item() / accum
            acc_heads += torch.stack(parts["per_head"]).cpu() / accum
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip).item()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        state["step"] = step + 1

        if state["step"] % t.log_every == 0 or state["step"] == stop:
            sync(); now = time.perf_counter()
            dt = now - t_seg; state["train_time_s"] += dt
            n_steps = t.log_every if state["step"] % t.log_every == 0 else (state["step"] % t.log_every)
            log.log(kind="train", step=state["step"], tokens=state["step"] * t.global_batch_tokens, lr=lr,
                    loss=acc_total, head_losses=acc_heads.tolist(), grad_norm=gnorm,
                    tokens_per_s=n_steps * t.global_batch_tokens / max(dt, 1e-9),
                    train_time_s=state["train_time_s"],
                    peak_mem_gb=(torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None),
                    scale=scaler.get_scale() if scaler.is_enabled() else None)
            t_seg = time.perf_counter()

        do_eval = state["step"] % t.eval_every == 0 or state["step"] == total_steps
        do_ckpt = state["step"] % t.ckpt_every == 0 or state["step"] == total_steps
        if do_eval or do_ckpt:
            sync(); t_pause = time.perf_counter()
            if do_eval:
                last_eval = evaluate(model, val_data, cfg, device, t.eval_windows, bytes_per_token=bpt)
                state["eval_time_s"] += last_eval["eval_time_s"]
                log.log(kind="eval", step=state["step"], tokens=state["step"] * t.global_batch_tokens, **last_eval)
            if do_ckpt:
                atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                             "scaler": scaler.state_dict(), "state": state, "fingerprint": cfg.fingerprint()}, ckpt_path)
            sync(); t_seg += time.perf_counter() - t_pause      # exclude eval/ckpt from training time

    result = {"name": cfg.name, "seed": cfg.seed, "k": k, "steps_done": state["step"], "total_steps": total_steps,
              "tokens_seen": state["step"] * t.global_batch_tokens, "train_time_s": state["train_time_s"],
              "eval_time_s": state["eval_time_s"], "last_eval": last_eval,
              "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None,
              "params_main_path": model.num_params(False), "params_total": model.num_params(True),
              "fingerprint": cfg.fingerprint(), "env": env_info()}
    if state["step"] >= total_steps:
        atomic_save({"model": model.state_dict()}, os.path.join(t.out_dir, "final.pt"))
        with open(os.path.join(t.out_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. model.num_pred_heads=4 seed=2")
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    print(json.dumps(train(cfg, a.max_steps), indent=2, default=str))


if __name__ == "__main__":
    main()
