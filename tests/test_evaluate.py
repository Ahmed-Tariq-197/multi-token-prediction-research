"""The shared evaluator against an independent brute-force implementation, plus window-boundary behaviour."""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from helpers import reference_targets, run_cfg

from mtp.data import TokenFile
from mtp.evaluate import evaluate
from mtp.model import MTPModel
from mtp.targets import IGNORE_INDEX as IGN


def test_evaluate_matches_independent_per_token_loop(synth_dir, tmp_path):
    cfg = run_cfg(synth_dir, tmp_path / "e", 3)
    d, T, H = cfg.data, cfg.data.seq_len, cfg.data.max_horizon
    model = MTPModel(cfg.model, init_seed=11).eval()
    data = TokenFile(d.val_bin)
    got = evaluate(model, data, cfg, "cpu", n_windows=5, batch_size=2)

    nll, hit, n = [0.0] * 3, [0] * 3, [0] * 3
    for w in range(5):
        window = torch.from_numpy(np.asarray(data.tokens[w * T: w * T + T + H]).astype(np.int64))[None]
        tg = reference_targets(window, T, 3, cfg.model.eos_id)
        with torch.no_grad():
            hs = model.heads_forward(window[:, :T])
            for i in range(3):
                logp = F.log_softmax(model.unembed(hs[i])[0], -1)
                for t in range(T):
                    y = int(tg[0, i, t])
                    if y == IGN:
                        continue
                    nll[i] -= float(logp[t, y])
                    hit[i] += int(logp[t].argmax() == y)
                    n[i] += 1
    for i in range(3):
        assert got[f"head{i + 1}_n"] == n[i]
        assert math.isclose(got[f"head{i + 1}_loss"], nll[i] / n[i], rel_tol=1e-5)
        assert math.isclose(got[f"head{i + 1}_top1"], hit[i] / n[i], abs_tol=1e-9)
    assert math.isclose(got["ppl"], math.exp(got["head1_loss"]), rel_tol=1e-9)
    assert got["loss"] == got["head1_loss"]


def test_evaluation_is_batch_size_invariant_and_restores_train_mode(synth_dir, tmp_path):
    cfg = run_cfg(synth_dir, tmp_path / "e", 2)
    model = MTPModel(cfg.model, init_seed=2).train()
    data = TokenFile(cfg.data.val_bin)
    a, b = evaluate(model, data, cfg, "cpu", 6, batch_size=1), evaluate(model, data, cfg, "cpu", 6, batch_size=4)
    assert model.training
    for key in ("loss", "top1", "head2_loss", "head2_n"):
        assert math.isclose(a[key], b[key], rel_tol=1e-6), key


def test_main_head_metrics_are_identical_in_definition_across_arms(synth_dir, tmp_path):
    """Head-1 token counts must not depend on k (same valid-position rule in every arm)."""
    res = {}
    for k in (1, 2, 4):
        cfg = run_cfg(synth_dir, tmp_path / f"e{k}", k)
        res[k] = evaluate(MTPModel(cfg.model, init_seed=1), TokenFile(cfg.data.val_bin), cfg, "cpu", 8)
    assert res[1]["head1_n"] == res[2]["head1_n"] == res[4]["head1_n"]
    assert math.isclose(res[1]["loss"], res[4]["loss"], rel_tol=1e-6), "same init + same data => same head-1 loss"


def test_eval_window_count_is_exact_at_the_boundary(tmp_path):
    T, H = 8, 4
    for n_tokens, expect in ((T + H - 1, 0), (T + H, 1), (2 * T + H - 1, 1), (2 * T + H, 2)):
        p = os.path.join(tmp_path, f"b{n_tokens}.bin")
        np.arange(n_tokens, dtype=np.uint16).tofile(p)
        tf = TokenFile(p)
        assert tf.n_eval_windows(T, H) == expect, n_tokens
        if expect:
            last = tf.eval_windows(expect, T, H)[-1]
            assert last.shape == (T + H,) and int(last[-1]) <= n_tokens - 1


def test_training_sampler_can_reach_the_last_window(tmp_path):
    win = 12
    p = os.path.join(tmp_path, "tiny.bin")
    np.arange(win + 2, dtype=np.uint16).tofile(p)                    # offsets 0, 1, 2 are valid
    tf = TokenFile(p)
    starts = {int(tf.train_batch(0, s, 0, 8, 8, 4)[i, 0]) for s in range(60) for i in range(8)}
    assert starts == {0, 1, 2}
