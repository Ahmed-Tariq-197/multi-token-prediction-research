"""Training loop for Base NTP (k=1) and Standard MTP (k>=2). Same code path for every arm.

Reproducibility and result-integrity guarantees (each covered by tests/):
  * seeded, per-parameter init (paired across arms) and a stateless data order (see data.py);
  * atomic checkpoints; resume restores model/optimizer/scaler/step and TRUNCATES log.jsonl to the checkpoint step, so a
    crash never leaves duplicated or contradictory records;
  * resume refuses a changed trajectory config (seed, model, data protocol, LR schedule, budget) unless forced;
  * re-running a finished run returns its stored results and never overwrites them;
  * data/manifest consistency (vocab, EOS id, token counts) is verified before the first step;
  * training time excludes eval/checkpoint pauses and is accumulated at every pause, so it survives resume;
  * the test split is never touched here (use `python -m mtp.evaluate --split test` once, at the end).

Determinism caveat: exact trajectory equality is verified on CPU only. CUDA kernels (SDPA, atomics) may differ across
GPU models / library versions; compare GPU runs statistically, not bitwise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from .config import RunConfig, config_diff, load_config
from .data import TokenFile, check_data_consistency, load_manifest
from .evaluate import CKPT_FORMAT_VERSION, autocast_ctx, effective_precision, evaluate
from .loss import multi_head_loss
from .model import MTPModel
from .targets import build_head_targets
from .utils import JsonlLogger, atomic_save, env_info, set_seed, truncate_jsonl

MAX_CONSECUTIVE_NONFINITE = 10


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
    return torch.optim.AdamW(groups, lr=cfg.train.lr, betas=cfg.train.betas, fused=(device == "cuda"))


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def train(cfg: RunConfig, max_steps: int | None = None, force_resume: bool = False) -> dict:
    t, d = cfg.train, cfg.data
    device = t.device or ("cuda" if torch.cuda.is_available() else "cpu")
    precision = effective_precision(cfg, device)
    os.makedirs(t.out_dir, exist_ok=True)
    p_ckpt, p_log = os.path.join(t.out_dir, "latest.pt"), os.path.join(t.out_dir, "log.jsonl")
    p_results, p_config = os.path.join(t.out_dir, "results.json"), os.path.join(t.out_dir, "config.json")
    traj = cfg.trajectory_fingerprint()

    # ---- a finished run is immutable: return its record instead of recomputing/overwriting it
    if os.path.exists(p_results):
        with open(p_results, encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("trajectory_fingerprint") == traj or force_resume:
            print(f"[done] {t.out_dir} already completed; returning stored results (nothing overwritten)")
            return prev
        raise RuntimeError(f"{t.out_dir} holds a finished run with a different configuration "
                           f"({prev.get('trajectory_fingerprint')} != {traj}). Use another out_dir.")

    provenance = check_data_consistency(cfg)
    train_data, val_data = TokenFile(d.train_bin), TokenFile(d.val_bin)
    bpt = load_manifest(d.manifest).get("bytes_per_token", {}).get("val")
    epochs = t.total_tokens / len(train_data)
    provenance["train_epochs_estimate"] = epochs
    if epochs > 0.5:
        print(f"[warn] budget is {epochs:.2f}x the training split; random-window sampling with replacement "
              f"leaves ~{math.exp(-epochs):.0%} of tokens unseen and repeats others")

    set_seed(cfg.seed)
    data_seed = cfg.seed if d.data_seed is None else d.data_seed
    model = MTPModel(cfg.model, init_seed=cfg.seed).to(device)
    opt = build_optimizer(model, cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and precision == "fp16"))
    fwd = torch.compile(model.heads_forward) if t.compile else model.heads_forward

    accum = t.global_batch_tokens // (t.micro_batch_size * d.seq_len)
    total_steps = math.ceil(t.total_tokens / t.global_batch_tokens)
    k = model.k
    if t.warmup_steps >= total_steps:
        print(f"[warn] warmup_steps ({t.warmup_steps}) >= total steps ({total_steps}): LR never reaches its peak")

    state = {"step": 0, "train_time_s": 0.0, "eval_time_s": 0.0, "nonfinite_steps": 0}
    log = JsonlLogger(p_log)
    resumed = os.path.exists(p_ckpt)
    if resumed:
        ck = torch.load(p_ckpt, map_location=device, weights_only=True)
        if ck.get("trajectory_fingerprint") != traj:
            diff = config_diff(ck.get("config", {}), cfg.to_dict())
            msg = "checkpoint was produced under a different training configuration:\n  " + "\n  ".join(diff[:12])
            if not force_resume:
                raise RuntimeError(msg + "\nRefusing to resume (use force_resume only if this is intentional).")
            print("[warn] force-resuming across a configuration change:\n  " + "\n  ".join(diff[:12]))
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        state.update(ck["state"])
        dropped = truncate_jsonl(p_log, state["step"])
        log.log(kind="resume", step=state["step"], dropped_records=dropped, env=env_info(),
                forced=bool(ck.get("trajectory_fingerprint") != traj))
        print(f"[resume] step {state['step']}/{total_steps}; dropped {dropped} stale log record(s)")

    record = {"config": cfg.to_dict(), "fingerprint": cfg.fingerprint(), "trajectory_fingerprint": traj,
              "effective_precision": precision, "env": env_info(), "data_provenance": provenance,
              "params_main_path": model.num_params(False), "params_total": model.num_params(True),
              "total_steps": total_steps, "grad_accum": accum}
    cfg_path = p_config
    if os.path.exists(p_config):            # keep the original record; later (re)starts are stored beside it
        cfg_path = os.path.join(t.out_dir, f"config.resume{state['step']}.json")
    _write_json(cfg_path, record)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    model.train()
    stop = total_steps if max_steps is None else min(total_steps, max_steps)
    last_eval, bad = None, 0

    # ---- training-time accounting: time since t_mark is training time unless a pause (eval/ckpt) intervenes
    t_mark = time.perf_counter()
    win = {"steps": 0, "time": 0.0}

    def account() -> None:
        nonlocal t_mark
        sync()
        now = time.perf_counter()
        dt = now - t_mark
        state["train_time_s"] += dt
        win["time"] += dt
        t_mark = now

    for step in range(state["step"], stop):
        lr = lr_at(step, total_steps, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = torch.zeros((), device=device)
        heads_acc = torch.zeros(k, device=device)
        for micro in range(accum):
            window = train_data.train_batch(data_seed, step, micro, t.micro_batch_size, d.seq_len,
                                            d.max_horizon).to(device)
            ids = window[:, :d.seq_len]
            targets = build_head_targets(window, d.seq_len, k, cfg.model.eos_id, cfg.loss.mask_cross_document)
            with autocast_ctx(device, precision):
                hs = fwd(ids)
                loss, parts = multi_head_loss(hs, model.embed_tokens.weight, targets, cfg.loss.lam)
            scaler.scale(loss / accum).backward()
            loss_acc += loss.detach() / accum
            heads_acc += torch.stack(parts["per_head"]).to(device) / accum
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        stats = torch.cat([loss_acc.reshape(1), gnorm.detach().float().reshape(1), heads_acc]).tolist()  # one host sync
        loss_v, gnorm_v, head_v = stats[0], stats[1], stats[2:]
        # fp16: GradScaler itself skips steps with inf/NaN grads. fp32/bf16 has no scaler, so a non-finite step would
        # be applied and poison every weight; skip it explicitly. Either way it is counted in `nonfinite_steps`.
        if scaler.is_enabled() or (math.isfinite(loss_v) and math.isfinite(gnorm_v)):
            scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        state["step"] = step + 1
        win["steps"] += 1

        if not math.isfinite(loss_v):
            bad += 1
            state["nonfinite_steps"] += 1
            if bad >= MAX_CONSECUTIVE_NONFINITE:
                raise FloatingPointError(
                    f"loss was non-finite for {bad} consecutive steps (step {state['step']}); training diverged. "
                    "Lower the learning rate / check fp16 loss scaling; the last good checkpoint is latest.pt.")
        else:
            bad = 0

        if state["step"] % t.log_every == 0 or state["step"] == stop:
            account()
            log.log(kind="train", step=state["step"], tokens=state["step"] * t.global_batch_tokens, lr=lr,
                    loss=loss_v, head_losses=head_v, grad_norm=gnorm_v,
                    tokens_per_s=win["steps"] * t.global_batch_tokens / max(win["time"], 1e-9),
                    train_time_s=state["train_time_s"],
                    peak_mem_gb=(torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None),
                    scale=scaler.get_scale() if scaler.is_enabled() else None)
            win["steps"], win["time"] = 0, 0.0

        do_eval = state["step"] % t.eval_every == 0 or state["step"] == total_steps
        do_ckpt = state["step"] % t.ckpt_every == 0 or state["step"] == total_steps
        if do_eval or do_ckpt:
            account()                                   # training time up to here is banked before the pause
            if do_eval:
                last_eval = evaluate(model, val_data, cfg, device, t.eval_windows, bytes_per_token=bpt,
                                     precision=precision)
                state["eval_time_s"] += last_eval["eval_time_s"]
                log.log(kind="eval", step=state["step"], tokens=state["step"] * t.global_batch_tokens, **last_eval)
            if do_ckpt:
                atomic_save({"format_version": CKPT_FORMAT_VERSION, "model": model.state_dict(),
                             "opt": opt.state_dict(), "scaler": scaler.state_dict(), "state": dict(state),
                             "fingerprint": cfg.fingerprint(), "trajectory_fingerprint": traj,
                             "config": cfg.to_dict()}, p_ckpt)
            sync()
            t_mark = time.perf_counter()                # exclude the pause from training time
    account()

    completed = state["step"] >= total_steps
    if completed and last_eval is None:                 # e.g. resumed from the final checkpoint after a late crash
        last_eval = evaluate(model, val_data, cfg, device, t.eval_windows, bytes_per_token=bpt, precision=precision)
    result = {"name": cfg.name, "seed": cfg.seed, "k": k, "completed": completed, "steps_done": state["step"],
              "total_steps": total_steps, "tokens_seen": state["step"] * t.global_batch_tokens,
              "train_time_s": state["train_time_s"], "eval_time_s": state["eval_time_s"],
              "nonfinite_steps": state["nonfinite_steps"], "last_eval": last_eval,
              "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None,
              "params_main_path": model.num_params(False), "params_total": model.num_params(True),
              "fingerprint": cfg.fingerprint(), "trajectory_fingerprint": traj, "effective_precision": precision,
              "data_provenance": provenance, "resumed": resumed, "env": env_info()}
    if completed:
        atomic_save({"format_version": CKPT_FORMAT_VERSION, "model": model.state_dict(),
                     "model_config": record["config"]["model"], "step": state["step"],
                     "tokens_seen": result["tokens_seen"], "fingerprint": cfg.fingerprint(),
                     "trajectory_fingerprint": traj}, os.path.join(t.out_dir, "final.pt"))
        _write_json(p_results, result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. model.num_pred_heads=4 seed=2")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--force-resume", action="store_true",
                    help="resume even though the training configuration changed (scientifically a different run)")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    print(json.dumps(train(cfg, a.max_steps, a.force_resume), indent=2, default=str))


if __name__ == "__main__":
    main()
