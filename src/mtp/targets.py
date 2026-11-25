"""Targets for k parallel heads from one token window.

Window layout: `window` has T + H tokens (H = data.max_horizon >= k).
  inputs         = window[:, :T]                       (positions t = 0..T-1)
  target(i, t)   = window[:, t + i]   for head i in 1..k   (head i predicts offset +i)

A target is VALID iff it lies in the same document as the input token at position t, where documents are
separated by `eos_id` and the EOS token belongs to the document it terminates:

    seg[j] = number of EOS tokens strictly before position j
    valid(i, t)  <=>  seg[t + i] == seg[t]

So the model may predict the EOS that ends the current document (any head whose offset reaches it), but is
never asked to predict a token of the next document from the previous one. The same rule is applied to head 1
in the Base NTP model, so head-1 losses are defined identically in every arm. Invalid targets get IGNORE_INDEX.
"""
from __future__ import annotations

import torch

IGNORE_INDEX = -100


def document_segments(window: torch.Tensor, eos_id: int) -> torch.Tensor:
    is_eos = (window == eos_id).long()
    return torch.cumsum(is_eos, dim=1) - is_eos          # EOS strictly before j


def build_head_targets(window: torch.Tensor, seq_len: int, k: int, eos_id: int,
                       mask_cross_document: bool = True) -> torch.Tensor:
    """Return LongTensor [B, k, T]; entry [b, i-1, t] is the target of head i at position t (or IGNORE_INDEX)."""
    B, W = window.shape
    assert W >= seq_len + k, f"window length {W} < seq_len + k = {seq_len + k}"
    seg = document_segments(window, eos_id) if mask_cross_document else None
    out = torch.full((B, k, seq_len), IGNORE_INDEX, dtype=torch.long, device=window.device)
    for i in range(1, k + 1):
        tgt = window[:, i:i + seq_len]
        if seg is not None:
            valid = seg[:, i:i + seq_len] == seg[:, :seq_len]
            tgt = torch.where(valid, tgt, torch.full_like(tgt, IGNORE_INDEX))
        out[:, i - 1] = tgt
    return out
