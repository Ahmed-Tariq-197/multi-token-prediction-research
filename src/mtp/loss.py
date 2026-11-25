"""Weighted multi-head cross-entropy.

    L = sum_i w_i * L_i,   w_1 = 1,   w_i = lam / (k - 1)  for i = 2..k
    L_i = mean over head i's VALID positions of  -log p_i(x_{t+i} | x_{<=t})

Matches the normalisation used by Soft-MTP (L_NTP + lam * mean_j L_MTP_j) so that Phase 2 changes only the gradient
routing, not the weighting. Gloeckle et al. use an unweighted sum; that corresponds to lam = k - 1 here.
k = 1 reduces exactly to the standard next-token loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .targets import IGNORE_INDEX


def head_weights(k: int, lam: float) -> list[float]:
    return [1.0] + [lam / (k - 1)] * (k - 1) if k > 1 else [1.0]


def head_ce(h: torch.Tensor, W: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean CE over valid targets (fp32 logits). Returns (loss, n_valid). Zero loss if nothing is valid."""
    logits = F.linear(h, W)
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()               # fp32 softmax/CE; never downcast fp64 (used by tests)
    n_valid = (target != IGNORE_INDEX).sum()
    if n_valid == 0:
        return logits.sum() * 0.0, n_valid
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                           ignore_index=IGNORE_INDEX, reduction="sum") / n_valid
    return loss, n_valid


def multi_head_loss(hiddens: list[torch.Tensor], W: torch.Tensor, targets: torch.Tensor,
                    lam: float = 1.0) -> tuple[torch.Tensor, dict]:
    """hiddens: k tensors [B,T,d] (post-norm); targets: [B,k,T]. Returns (total, detached per-head losses)."""
    k = len(hiddens)
    assert targets.shape[1] == k, f"targets have {targets.shape[1]} heads, model has {k}"
    w = head_weights(k, lam)
    total, per_head = 0.0, []
    for i, h in enumerate(hiddens):
        li, _ = head_ce(h, W, targets[:, i])
        per_head.append(li.detach())
        total = total + w[i] * li
    return total, {"per_head": per_head, "weights": w}
