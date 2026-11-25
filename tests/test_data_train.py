import json
import os

import numpy as np
import torch

from mtp.config import load_config
from mtp.data import TokenFile
from mtp.train import train


def _cfg(synth_dir, out_dir, k, seed=1, **extra):
    ov = [f"seed={seed}", f"data.train_bin='{synth_dir}/train.bin'", f"data.val_bin='{synth_dir}/val.bin'",
          f"data.test_bin='{synth_dir}/test.bin'", f"data.manifest='{synth_dir}/manifest.json'",
          "data.seq_len=32", "data.max_horizon=4",
          "model.vocab_size=64", "model.d_model=32", "model.n_layers=2", "model.n_heads=4", "model.ffn_hidden=64",
          "model.max_seq_len=32", f"model.num_pred_heads={k}",
          "train.total_tokens=32768", "train.global_batch_tokens=2048", "train.micro_batch_size=8",
          "train.lr=3e-3", "train.warmup_steps=2", "train.log_every=2", "train.eval_every=8", "train.ckpt_every=4",
          "train.eval_windows=16", f"train.out_dir='{out_dir}'", "train.precision='fp32'", "train.device='cpu'"]
    ov += [f"{a}={b}" for a, b in extra.items()]
    return load_config(None, ov)


def test_train_batches_are_pure_functions_of_seed_step_micro(synth_dir):
    tf = TokenFile(os.path.join(synth_dir, "train.bin"))
    a = tf.train_batch(5, 10, 0, 4, 32, 8)
    assert torch.equal(a, tf.train_batch(5, 10, 0, 4, 32, 8))
    assert not torch.equal(a, tf.train_batch(5, 11, 0, 4, 32, 8))
    assert not torch.equal(a, tf.train_batch(6, 10, 0, 4, 32, 8))
    assert not torch.equal(a, tf.train_batch(5, 10, 1, 4, 32, 8))


def test_every_arm_sees_identical_inputs_regardless_of_k(synth_dir):
    """Window length is seq_len+max_horizon for all k, so input ids are identical across arms."""
    tf = TokenFile(os.path.join(synth_dir, "train.bin"))
    w = tf.train_batch(3, 7, 0, 4, 32, 8)
    assert w.shape == (4, 40)
    assert torch.equal(w[:, :32], tf.train_batch(3, 7, 0, 4, 32, 8)[:, :32])


def test_eval_windows_cover_each_token_once_and_are_fixed(synth_dir):
    tf = TokenFile(os.path.join(synth_dir, "val.bin"))
    ws = tf.eval_windows(10, 32, 4)
    flat = ws[:, :32].reshape(-1)
    assert torch.equal(flat, torch.from_numpy(np.asarray(tf.tokens[:320]).astype(np.int64)))
    assert torch.equal(ws, tf.eval_windows(10, 32, 4))


def test_smoke_training_learns_and_logs(synth_dir, tmp_path):
    cfg = _cfg(synth_dir, tmp_path / "run", k=2)
    res = train(cfg)
    assert res["steps_done"] == res["total_steps"] == 16
    recs = [json.loads(l) for l in open(tmp_path / "run" / "log.jsonl")]
    tr = [r for r in recs if r["kind"] == "train"]
    assert tr[-1]["loss"] < tr[0]["loss"] - 0.2, "loss should drop on the learnable toy language"
    assert len(tr[-1]["head_losses"]) == 2
    ev = [r for r in recs if r["kind"] == "eval"][-1]
    assert {"loss", "ppl", "top1", "head2_loss", "bpb"} <= set(ev)
    for f in ("config.json", "results.json", "final.pt", "latest.pt"):
        assert os.path.exists(tmp_path / "run" / f)


def test_same_seed_same_trajectory_and_resume_is_exact(synth_dir, tmp_path):
    a = train(_cfg(synth_dir, tmp_path / "a", k=2, seed=4))
    b = train(_cfg(synth_dir, tmp_path / "b", k=2, seed=4))
    la = [json.loads(l)["loss"] for l in open(tmp_path / "a" / "log.jsonl") if '"train"' in l]
    lb = [json.loads(l)["loss"] for l in open(tmp_path / "b" / "log.jsonl") if '"train"' in l]
    assert la == lb, "two runs with the same seed must be identical on CPU"

    # interrupted run: stop after 8 steps (a checkpoint exists at step 8), then resume to the end
    cfg = _cfg(synth_dir, tmp_path / "c", k=2, seed=4)
    train(cfg, max_steps=8)
    train(cfg)
    lc = [json.loads(l) for l in open(tmp_path / "c" / "log.jsonl") if '"train"' in l]
    tail_c = [r["loss"] for r in lc if r["step"] > 8]
    tail_a = [json.loads(l)["loss"] for l in open(tmp_path / "a" / "log.jsonl")
              if '"train"' in l and json.loads(l)["step"] > 8]
    assert tail_c == tail_a, "resume must reproduce the uninterrupted trajectory exactly"


def test_base_and_mtp_share_trunk_init_and_batches(synth_dir, tmp_path):
    train(_cfg(synth_dir, tmp_path / "k1", k=1, seed=9), max_steps=1)
    train(_cfg(synth_dir, tmp_path / "k4", k=4, seed=9), max_steps=1)
    r1 = [json.loads(l) for l in open(tmp_path / "k1" / "log.jsonl")][0]
    r4 = [json.loads(l) for l in open(tmp_path / "k4" / "log.jsonl")][0]
    # first-step head-1 loss is computed from identical weights on identical inputs
    assert abs(r1["head_losses"][0] - r4["head_losses"][0]) < 1e-6
