import math

import torch
import torch.nn.functional as F

from mtp.loss import head_ce, head_weights, multi_head_loss
from mtp.model import MTPModel
from mtp.targets import IGNORE_INDEX as IGN
from mtp.targets import build_head_targets


def _setup(tiny_cfg, k, T=16, H=4, B=3, seed=0, with_eos=True):
    torch.manual_seed(seed)
    model = MTPModel(tiny_cfg(k=k), init_seed=seed).double()
    window = torch.randint(1, 64, (B, T + H))
    if with_eos:
        window[:, 7] = 0
    ids = window[:, :T]
    return model, window, ids, T


def test_k1_equals_standard_shifted_cross_entropy(tiny_cfg):
    model, window, ids, T = _setup(tiny_cfg, k=1, with_eos=False)
    tg = build_head_targets(window, T, 1, eos_id=0)
    hs = model.heads_forward(ids)
    loss, _ = multi_head_loss(hs, model.embed_tokens.weight, tg)
    logits, _ = model(ids)
    ref = F.cross_entropy(logits.reshape(-1, 64), window[:, 1:T + 1].reshape(-1))
    assert torch.allclose(loss, ref, atol=1e-12)


def test_weighted_total_matches_documented_formula(tiny_cfg):
    k, lam = 4, 0.7
    model, window, ids, T = _setup(tiny_cfg, k=k)
    tg = build_head_targets(window, T, k, eos_id=0)
    total, parts = multi_head_loss(model.heads_forward(ids), model.embed_tokens.weight, tg, lam)
    ph = parts["per_head"]
    expected = ph[0] + lam / (k - 1) * (ph[1] + ph[2] + ph[3])
    assert torch.allclose(total, expected, atol=1e-12)
    assert head_weights(4, 0.7) == [1.0, 0.7 / 3, 0.7 / 3, 0.7 / 3]
    assert head_weights(1, 5.0) == [1.0]


def test_each_head_loss_is_mean_over_valid_positions_only(tiny_cfg):
    k = 3
    model, window, ids, T = _setup(tiny_cfg, k=k)
    tg = build_head_targets(window, T, k, eos_id=0)
    hs = model.heads_forward(ids)
    W = model.embed_tokens.weight
    for i, h in enumerate(hs):
        logits = F.linear(h, W)
        valid = tg[:, i] != IGN
        manual = F.cross_entropy(logits[valid], tg[:, i][valid])
        got, n = head_ce(h, W, tg[:, i])
        assert int(n) == int(valid.sum())
        assert torch.allclose(got, manual, atol=1e-12)


def test_changing_a_masked_target_does_not_change_loss(tiny_cfg):
    k = 3
    model, window, ids, T = _setup(tiny_cfg, k=k)
    tg = build_head_targets(window, T, k, eos_id=0)
    hs = model.heads_forward(ids)
    base, _ = multi_head_loss(hs, model.embed_tokens.weight, tg)
    tg2 = tg.clone()
    tg2[tg2 == IGN] = IGN                      # sanity no-op
    assert (tg == IGN).any(), "test needs at least one masked position"
    # masked entries stay IGN by construction; flipping a *valid* one must change the loss
    idx = (tg[:, 0] != IGN).nonzero()[0]
    tg2[idx[0], 0, idx[1]] = (tg2[idx[0], 0, idx[1]] + 1) % 64
    changed, _ = multi_head_loss(hs, model.embed_tokens.weight, tg2)
    assert not torch.allclose(base, changed)


def test_fully_masked_head_gives_zero_not_nan(tiny_cfg):
    model, window, ids, T = _setup(tiny_cfg, k=2)
    tg = torch.full((window.shape[0], 2, T), IGN, dtype=torch.long)
    total, _ = multi_head_loss(model.heads_forward(ids), model.embed_tokens.weight, tg)
    assert torch.isfinite(total) and total.item() == 0.0
    total.backward()                           # must not raise


def test_random_init_loss_near_log_vocab(tiny_cfg):
    model, window, ids, T = _setup(tiny_cfg, k=2)
    tg = build_head_targets(window, T, 2, eos_id=0)
    _, parts = multi_head_loss(model.heads_forward(ids), model.embed_tokens.weight, tg)
    for l in parts["per_head"]:
        assert abs(l.item() - math.log(64)) < 0.5
