"""Profiling-first diagnosis of the autoregressive decode bottleneck (input to the Bonus decision; ADR-007).

  python scripts/profile_decode.py --config configs/base_ntp.yaml --prompt-len 128 --new-tokens 128 --out prof_ours.json
  python scripts/profile_decode.py --hf HuggingFaceTB/SmolLM2-360M ...          # a larger, memory-bound-regime model

Measures, for batch-1 greedy decoding with the standard dynamic KV cache (the BASELINE, untouched):
  ms/token, CUDA kernels launched per token, GPU-busy fraction (sum of kernel time / wall time), weight bytes per
  token vs the memory-bandwidth roofline, and the top kernels by time. It then prints which bottleneck class the
  numbers point to. It does NOT implement any optimisation. Classification thresholds are heuristics to be
  revisited with data. On CPU only wall-clock metrics are produced (smoke test).
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
from mtp.model import MTPModel  # noqa: E402


@torch.no_grad()
def decode(model_fn, prompt: torch.Tensor, n_new: int):
    logits, past = model_fn(prompt, None)
    tok = logits[:, -1:].argmax(-1)
    for _ in range(n_new):
        logits, past = model_fn(tok, past)
        tok = logits[:, -1:].argmax(-1)
    return tok


def classify(m: dict) -> str:
    if "gpu_busy_frac" not in m:
        return "no CUDA profile (CPU run): cannot classify"
    if m["gpu_busy_frac"] < 0.5:
        return ("LAUNCH/CPU-OVERHEAD-BOUND (GPU idle >50%): candidates = remove per-token launch overhead "
                "(static cache + graph capture, fusion). Quantisation would NOT help.")
    if m["bw_utilisation"] > 0.5:
        return ("MEMORY-BANDWIDTH-BOUND (weights streamed near roofline): candidates = weight/KV quantisation. "
                "Launch-overhead removal would NOT help.")
    return "GPU-BUSY BUT BELOW ROOFLINE (inefficient kernels): candidates = fusion / better attention kernels."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--hf")
    ap.add_argument("--ckpt")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--peak-bw-gbs", type=float, default=320.0, help="published T4 spec; verify with a copy benchmark")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--out", default="profile_decode.json")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if (device == "cuda" and a.dtype == "fp16") else torch.float32

    if a.hf:
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(a.hf, torch_dtype=dt).to(device).eval()
        V = hf.config.vocab_size
        def fn(ids, past):
            o = hf(ids, past_key_values=past, use_cache=True)
            return o.logits, o.past_key_values
        n_params = sum(p.numel() for p in hf.parameters())
    else:
        cfg = load_config(a.config, a.set)
        m = MTPModel(cfg.model, init_seed=0)
        if a.ckpt:
            sd = torch.load(a.ckpt, map_location="cpu"); m.load_state_dict(sd["model"] if "model" in sd else sd)
        m = m.to(device).to(dt).eval()
        V = cfg.model.vocab_size
        fn = lambda ids, past: m(ids, past_key_values=past, use_cache=True)
        n_params = m.num_params(include_aux=False)

    prompt = torch.randint(1, V, (1, a.prompt_len), device=device)
    decode(fn, prompt, 8)                                   # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter(); decode(fn, prompt, a.new_tokens)
    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    res = {"model": a.hf or a.config, "device": device, "dtype": str(dt), "n_params": n_params,
           "new_tokens": a.new_tokens, "ms_per_token": 1000 * wall / a.new_tokens,
           "tokens_per_s": a.new_tokens / wall}
    bytes_tok = n_params * (2 if dt == torch.float16 else 4)
    res["weight_bytes_per_token"] = bytes_tok
    res["roofline_ms_per_token"] = 1000 * bytes_tok / (a.peak_bw_gbs * 1e9)

    if device == "cuda":
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            decode(fn, prompt, a.new_tokens)
            torch.cuda.synchronize()
        ev = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
        kernel_us = sum(e.self_device_time_total for e in ev)
        launches = sum(e.count for e in ev)
        res.update(kernels_per_token=launches / (a.new_tokens + 1),
                   gpu_kernel_ms_per_token=kernel_us / 1000 / (a.new_tokens + 1),
                   gpu_busy_frac=min(1.0, kernel_us / 1e6 / wall),
                   bw_utilisation=res["roofline_ms_per_token"] / max(1000 * wall / a.new_tokens, 1e-9),
                   top_kernels=[(e.key, e.count, e.self_device_time_total) for e in
                                sorted(ev, key=lambda e: -e.self_device_time_total)[:8]])
    res["diagnosis"] = classify(res)
    print(json.dumps(res, indent=2, default=str))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, default=str)


if __name__ == "__main__":
    main()
