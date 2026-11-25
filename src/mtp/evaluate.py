"""One shared evaluation routine used for every arm (Base NTP, Standard MTP, later Improved MTP).

Reported per split:
  head{i}_loss / head{i}_top1     mean CE (nats) and top-1 accuracy of head i over its valid targets
  loss, ppl, top1                 MAIN head (head 1) -- the directly comparable numbers across arms
  bpb                             bits per byte of the main head, if manifest provides bytes_per_token
Evaluation never uses the auxiliary heads for the headline metrics.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from .config import RunConfig, load_config
from .data import TokenFile, load_manifest
from .model import MTPModel
from .targets import IGNORE_INDEX, build_head_targets


def autocast_ctx(device: str, precision: str):
    if device == "cuda" and precision in ("fp16", "bf16"):
        return torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def evaluate(model: MTPModel, data: TokenFile, cfg: RunConfig, device: str, n_windows: int | None = None,
             batch_size: int = 8, bytes_per_token: float | None = None) -> dict:
    model.eval()
    d, k = cfg.data, model.k
    n = n_windows or data.n_eval_windows(d.seq_len, d.max_horizon)
    n = min(n, data.n_eval_windows(d.seq_len, d.max_horizon))
    loss_sum = torch.zeros(k, dtype=torch.float64)
    hit_sum = torch.zeros(k, dtype=torch.float64)
    cnt = torch.zeros(k, dtype=torch.float64)
    t0 = time.perf_counter()
    for s in range(0, n, batch_size):
        window = data.eval_windows(n, d.seq_len, d.max_horizon, start=s, count=batch_size).to(device)
        ids = window[:, :d.seq_len]
        targets = build_head_targets(window, d.seq_len, k, cfg.model.eos_id, cfg.loss.mask_cross_document)
        with autocast_ctx(device, cfg.train.precision):
            hs = model.heads_forward(ids)
        for i, h in enumerate(hs):
            logits = model.unembed(h).float()
            tgt = targets[:, i]
            valid = tgt != IGNORE_INDEX
            if valid.any():
                loss_sum[i] += F.cross_entropy(logits[valid], tgt[valid], reduction="sum").item()
                hit_sum[i] += (logits[valid].argmax(-1) == tgt[valid]).sum().item()
                cnt[i] += valid.sum().item()
    model.train()
    out = {"n_windows": n, "eval_time_s": time.perf_counter() - t0}
    for i in range(k):
        c = max(cnt[i].item(), 1.0)
        out[f"head{i + 1}_loss"] = loss_sum[i].item() / c
        out[f"head{i + 1}_top1"] = hit_sum[i].item() / c
        out[f"head{i + 1}_n"] = int(cnt[i].item())
    out["loss"], out["top1"] = out["head1_loss"], out["head1_top1"]
    out["ppl"] = math.exp(out["loss"])
    if bytes_per_token:
        out["bpb"] = out["loss"] / math.log(2) / bytes_per_token
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on a split with the shared eval code.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--windows", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    device = cfg.train.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MTPModel(cfg.model).to(device)
    sd = torch.load(a.ckpt, map_location=device)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    path = cfg.data.val_bin if a.split == "val" else cfg.data.test_bin
    bpt = load_manifest(cfg.data.manifest).get("bytes_per_token", {}).get(a.split)
    res = evaluate(model, TokenFile(path), cfg, device, a.windows, bytes_per_token=bpt)
    res.update(split=a.split, ckpt=a.ckpt, config_fingerprint=cfg.fingerprint())
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
