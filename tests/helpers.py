"""Shared test helpers (plain functions; imported by test modules)."""
import json

import torch
import yaml

from mtp.config import load_config
from mtp.targets import IGNORE_INDEX as IGN


def run_cfg(synth_dir, out_dir, k=2, seed=1, **extra):
    """Tiny CPU RunConfig on the synthetic corpus. Values go through repr() so that Windows paths
    (backslashes) survive ast.literal_eval; f"'{path}'" would corrupt them."""
    paths = {"train_bin": f"{synth_dir}/train.bin", "val_bin": f"{synth_dir}/val.bin",
             "test_bin": f"{synth_dir}/test.bin", "manifest": f"{synth_dir}/manifest.json"}
    ov = [f"seed={seed}"] + [f"data.{key}={str(val)!r}" for key, val in paths.items()]
    ov += ["data.seq_len=32", "data.max_horizon=4",
           "model.vocab_size=64", "model.d_model=32", "model.n_layers=2", "model.n_heads=4", "model.ffn_hidden=64",
           "model.max_seq_len=32", f"model.num_pred_heads={k}",
           "train.total_tokens=32768", "train.global_batch_tokens=2048", "train.micro_batch_size=8",
           "train.lr=3e-3", "train.warmup_steps=2", "train.log_every=2", "train.eval_every=8", "train.ckpt_every=4",
           "train.eval_windows=16", f"train.out_dir={str(out_dir)!r}", "train.precision='fp32'", "train.device='cpu'"]
    ov += [f"{a}={b}" for a, b in extra.items()]
    return load_config(None, ov)


def write_yaml(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(json.loads(json.dumps(cfg.to_dict())), f)
    return str(path)


def read_log(out_dir, kind=None):
    recs = [json.loads(line) for line in open(f"{out_dir}/log.jsonl")]
    return [r for r in recs if kind is None or r["kind"] == kind]


def reference_targets(window, T, k, eos, mask=True):
    """Slow, obviously-correct python-loop reference for head targets."""
    B = window.shape[0]
    out = torch.full((B, k, T), IGN, dtype=torch.long)
    for b in range(B):
        toks = window[b].tolist()
        for t in range(T):
            for i in range(1, k + 1):
                j = t + i
                if mask and eos in toks[t:j]:        # an EOS at positions t..j-1 closes the document before j
                    continue
                out[b, i - 1, t] = toks[j]
    return out
