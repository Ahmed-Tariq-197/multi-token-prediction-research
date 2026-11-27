"""Token streams stored as uint16 .bin files (vocab < 65536) read through np.memmap.

Determinism contract
--------------------
* Training batch (data_seed, step, micro) is a pure function: offsets come from
  np.random.default_rng([data_seed, step, micro]). Hence (a) resuming at any step reproduces the stream exactly,
  (b) two arms with the same seed see byte-identical windows (window length is seq_len + max_horizon for every k).
* Evaluation windows are the first N non-overlapping stride-seq_len windows of the split. Every token is a head-1
  target exactly once; the extra max_horizon tokens only serve as look-ahead for auxiliary-head targets.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch


class TokenFile:
    def __init__(self, path: str):
        self.path = path
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return len(self.tokens)

    def train_batch(self, data_seed: int, step: int, micro: int, batch_size: int,
                    seq_len: int, horizon: int) -> torch.Tensor:
        win = seq_len + horizon
        n_offsets = len(self.tokens) - win + 1            # every full window is a candidate, incl. the last one
        if n_offsets < 1:
            raise ValueError(f"{self.path} has only {len(self.tokens)} tokens (< one window of {win})")
        rng = np.random.default_rng([data_seed, step, micro])
        offs = rng.integers(0, n_offsets, size=batch_size)
        arr = np.stack([self.tokens[o:o + win] for o in offs]).astype(np.int64)
        return torch.from_numpy(arr)

    def n_eval_windows(self, seq_len: int, horizon: int) -> int:
        # window w reads tokens [w*T, w*T + T + H): it fits iff (w+1)*T + H <= len
        return max(0, (len(self.tokens) - horizon) // seq_len)

    def eval_windows(self, n: int, seq_len: int, horizon: int, start: int = 0,
                     count: int | None = None) -> torch.Tensor:
        total = min(n, self.n_eval_windows(seq_len, horizon))
        end = total if count is None else min(total, start + count)
        win = seq_len + horizon
        arr = np.stack([self.tokens[w * seq_len: w * seq_len + win] for w in range(start, end)]).astype(np.int64)
        return torch.from_numpy(arr)


def load_manifest(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def check_data_consistency(cfg) -> dict:
    """Refuse to train/evaluate on data that does not match the configuration; return provenance for the run record.

    A wrong eos_id silently corrupts document masking; a vocab mismatch gives out-of-range embeddings. Both would
    otherwise only show up as unexplained numbers, so they are hard errors. A missing manifest is allowed (custom
    .bin files) but is recorded as `manifest_present: False` and warned about.
    """
    d, m = cfg.data, cfg.model
    prov: dict = {"manifest_present": False}
    man = load_manifest(d.manifest)
    if man:
        prov.update(manifest_present=True, tokenizer_sha256=man.get("tokenizer_sha256"), dataset=man.get("dataset"),
                    dataset_revision=man.get("revision"), split_rule=man.get("split_rule"))
        if man.get("vocab_size") != m.vocab_size:
            raise ValueError(f"manifest vocab_size {man.get('vocab_size')} != model.vocab_size {m.vocab_size}")
        if man.get("eos_id") != m.eos_id:
            raise ValueError(f"manifest eos_id {man.get('eos_id')} != model.eos_id {m.eos_id} "
                             "(document masking would be wrong)")
    else:
        print(f"[warn] no manifest at {d.manifest}: vocab/eos/tokenizer provenance not verified")
    for split, path in (("train", d.train_bin), ("val", d.val_bin)):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{split} tokens not found: {path} (run scripts/prepare_data.py)")
        size = os.path.getsize(path)
        if size % 2:
            raise ValueError(f"{path}: size {size} is not a multiple of 2 (uint16 tokens)")
        expect = man.get("tokens", {}).get(split) if man else None
        if expect is not None and expect * 2 != size:
            raise ValueError(f"{path}: {size // 2} tokens on disk but manifest says {expect}")
        head = np.memmap(path, dtype=np.uint16, mode="r")[:2_000_000]
        if len(head) and int(head.max()) >= m.vocab_size:
            raise ValueError(f"{path}: token id {int(head.max())} >= vocab_size {m.vocab_size}")
        prov[f"{split}_tokens"] = size // 2
    return prov
