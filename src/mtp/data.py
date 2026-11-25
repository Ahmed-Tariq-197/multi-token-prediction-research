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
        hi = len(self.tokens) - win
        assert hi > 0, f"{self.path} has only {len(self.tokens)} tokens"
        rng = np.random.default_rng([data_seed, step, micro])
        offs = rng.integers(0, hi, size=batch_size)
        arr = np.stack([self.tokens[o:o + win] for o in offs]).astype(np.int64)
        return torch.from_numpy(arr)

    def n_eval_windows(self, seq_len: int, horizon: int) -> int:
        return max(0, (len(self.tokens) - horizon - 1) // seq_len)

    def eval_windows(self, n: int, seq_len: int, horizon: int, start: int = 0, count: int | None = None) -> torch.Tensor:
        total = min(n, self.n_eval_windows(seq_len, horizon))
        end = total if count is None else min(total, start + count)
        win = seq_len + horizon
        arr = np.stack([self.tokens[w * seq_len: w * seq_len + win] for w in range(start, end)]).astype(np.int64)
        return torch.from_numpy(arr)


def load_manifest(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}
