"""Run-integrity guarantees: resume, idempotence, config drift, data/manifest consistency, divergence, checkpoints."""
import json
import os
import subprocess
import sys

import pytest
import torch
from helpers import read_log, run_cfg, write_yaml

import mtp.train as train_mod
from mtp.evaluate import load_model_from_checkpoint
from mtp.model import MTPModel
from mtp.train import train

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_crash_then_resume_leaves_a_clean_log(synth_dir, tmp_path):
    out = tmp_path / "r"
    train(run_cfg(synth_dir, out, 2, seed=3, **{"train.log_every": 1}), max_steps=6)   # ckpt exists at step 4 only
    train(run_cfg(synth_dir, out, 2, seed=3, **{"train.log_every": 1}))
    steps = [r["step"] for r in read_log(out, "train")]
    assert steps == list(range(1, 17)), steps                  # no duplicates, no gaps
    resume = read_log(out, "resume")
    assert len(resume) == 1 and resume[0]["step"] == 4 and resume[0]["dropped_records"] >= 2


def test_resumed_run_matches_uninterrupted_run_exactly(synth_dir, tmp_path):
    kw = {"train.log_every": 1}
    train(run_cfg(synth_dir, tmp_path / "full", 2, seed=5, **kw))
    train(run_cfg(synth_dir, tmp_path / "part", 2, seed=5, **kw), max_steps=6)
    train(run_cfg(synth_dir, tmp_path / "part", 2, seed=5, **kw))
    a = {r["step"]: r["loss"] for r in read_log(tmp_path / "full", "train")}
    b = {r["step"]: r["loss"] for r in read_log(tmp_path / "part", "train")}
    assert a == b
    sd_a = torch.load(tmp_path / "full" / "final.pt", weights_only=True)["model"]
    sd_b = torch.load(tmp_path / "part" / "final.pt", weights_only=True)["model"]
    assert all(torch.equal(sd_a[n], sd_b[n]) for n in sd_a)


def test_finished_run_is_immutable(synth_dir, tmp_path):
    out = tmp_path / "done"
    first = train(run_cfg(synth_dir, out, 2))
    raw = (out / "results.json").read_bytes()
    final = (out / "final.pt").read_bytes()
    again = train(run_cfg(synth_dir, out, 2))
    assert again["last_eval"] == first["last_eval"] and again["last_eval"] is not None
    assert (out / "results.json").read_bytes() == raw and (out / "final.pt").read_bytes() == final


def test_finished_run_refuses_a_different_configuration(synth_dir, tmp_path):
    out = tmp_path / "occupied"
    train(run_cfg(synth_dir, out, 2))
    with pytest.raises(RuntimeError, match="different configuration"):
        train(run_cfg(synth_dir, out, 2, **{"train.lr": 0.01}))


def test_resume_refuses_config_drift_unless_forced(synth_dir, tmp_path):
    out = tmp_path / "drift"
    train(run_cfg(synth_dir, out, 2), max_steps=4)
    with pytest.raises(RuntimeError, match=r"train\.lr"):
        train(run_cfg(synth_dir, out, 2, **{"train.lr": 0.01}))
    res = train(run_cfg(synth_dir, out, 2, **{"train.lr": 0.01}), force_resume=True)
    assert res["completed"]
    assert any(r["forced"] for r in read_log(out, "resume"))      # the forced resume is recorded in the log


def test_cosmetic_config_changes_do_not_block_resume(synth_dir, tmp_path):
    out = tmp_path / "cosmetic"
    train(run_cfg(synth_dir, out, 2), max_steps=4)
    res = train(run_cfg(synth_dir, out, 2, **{"train.log_every": 4, "train.eval_every": 16}))
    assert res["completed"] and res["resumed"]


def test_training_time_accumulates_across_pauses_and_resume(synth_dir, tmp_path):
    out = tmp_path / "t"
    kw = {"train.log_every": 3, "train.ckpt_every": 4}          # ckpt cadence not aligned with log cadence
    train(run_cfg(synth_dir, out, 2, **kw), max_steps=8)
    ck = torch.load(out / "latest.pt", weights_only=True)
    logged = [r["train_time_s"] for r in read_log(out, "train")]
    assert logged == sorted(logged)
    assert ck["state"]["train_time_s"] >= max(logged), "time since the last log must be banked in the checkpoint"
    res = train(run_cfg(synth_dir, out, 2, **kw))
    assert res["train_time_s"] >= ck["state"]["train_time_s"]
    assert res["eval_time_s"] > 0


def test_manifest_mismatches_are_hard_errors(synth_dir, tmp_path):
    import shutil
    d = tmp_path / "data"
    shutil.copytree(synth_dir, d)
    man = json.loads((d / "manifest.json").read_text())

    def attempt(**changes):
        (d / "manifest.json").write_text(json.dumps({**man, **changes}))
        return train(run_cfg(str(d), tmp_path / f"o{len(changes)}{next(iter(changes))}", 2), max_steps=1)

    with pytest.raises(ValueError, match="vocab_size"):
        attempt(vocab_size=999)
    with pytest.raises(ValueError, match="eos_id"):
        attempt(eos_id=7)
    with pytest.raises(ValueError, match="manifest says"):
        attempt(tokens={"train": 12345, "val": 1})


def test_token_ids_outside_vocab_are_rejected(synth_dir, tmp_path):
    import shutil
    d = tmp_path / "data2"
    shutil.copytree(synth_dir, d)
    import numpy as np
    arr = np.fromfile(d / "train.bin", dtype=np.uint16)
    arr[10] = 64                                                     # == vocab_size -> out of range
    arr.tofile(d / "train.bin")
    man = json.loads((d / "manifest.json").read_text())
    man.pop("tokens", None)
    (d / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match=">= vocab_size"):
        train(run_cfg(str(d), tmp_path / "o", 2), max_steps=1)


def test_missing_manifest_warns_and_is_recorded(synth_dir, tmp_path, capsys):
    import shutil
    d = tmp_path / "data3"
    shutil.copytree(synth_dir, d)
    os.remove(d / "manifest.json")
    res = train(run_cfg(str(d), tmp_path / "o", 2), max_steps=1)
    assert "no manifest" in capsys.readouterr().out
    assert res["data_provenance"]["manifest_present"] is False


def test_divergence_aborts_but_a_transient_nan_does_not(synth_dir, tmp_path, monkeypatch):
    real = train_mod.multi_head_loss
    calls = {"n": 0}

    def nan_once(*a, **kw):
        calls["n"] += 1
        loss, parts = real(*a, **kw)
        return (loss * float("nan"), parts) if calls["n"] == 3 else (loss, parts)

    monkeypatch.setattr(train_mod, "multi_head_loss", nan_once)
    res = train(run_cfg(synth_dir, tmp_path / "transient", 2), max_steps=6)
    assert res["nonfinite_steps"] == 1

    def always_nan(*a, **kw):
        loss, parts = real(*a, **kw)
        return loss * float("nan"), parts

    monkeypatch.setattr(train_mod, "multi_head_loss", always_nan)
    with pytest.raises(FloatingPointError, match="non-finite"):
        train(run_cfg(synth_dir, tmp_path / "diverged", 2))


def test_final_checkpoint_is_self_describing_and_validated(synth_dir, tmp_path):
    out = tmp_path / "ck"
    cfg = run_cfg(synth_dir, out, 4)
    train(cfg)
    ck = torch.load(out / "final.pt", weights_only=True)
    assert ck["format_version"] == 1 and ck["model_config"]["num_pred_heads"] == 4 and ck["tokens_seen"] == 32768
    model = load_model_from_checkpoint(str(out / "final.pt"), cfg.model)
    ref = MTPModel(cfg.model)
    ref.load_state_dict(ck["model"])
    ids = torch.randint(1, 64, (1, 12))
    assert torch.equal(model(ids)[0], ref(ids)[0])
    wrong = run_cfg(synth_dir, out, 2).model
    with pytest.raises(ValueError, match="num_pred_heads"):
        load_model_from_checkpoint(str(out / "final.pt"), wrong)


def test_evaluate_cli_end_to_end_and_never_overwrites_results(synth_dir, tmp_path):
    out = tmp_path / "cli"
    cfg = run_cfg(synth_dir, out, 2)
    train(cfg)
    yml, res = write_yaml(cfg, tmp_path / "cfg.yaml"), tmp_path / "test_result.json"
    cmd = [sys.executable, "-m", "mtp.evaluate", "--config", yml, "--ckpt", str(out / "final.pt"),
           "--split", "test", "--out", str(res)]
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}
    ok = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert ok.returncode == 0, ok.stderr
    first = json.loads(res.read_text())
    assert first["split"] == "test" and first["ppl"] > 1 and first["head2_n"] > 0
    again = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert again.returncode != 0 and "refusing to overwrite" in (again.stdout + again.stderr)
    assert json.loads(res.read_text()) == first
    assert subprocess.run(cmd + ["--force"], capture_output=True, env=env).returncode == 0


def test_budget_larger_than_corpus_warns(synth_dir, tmp_path, capsys):
    train(run_cfg(synth_dir, tmp_path / "w", 1, **{"train.total_tokens": 131072}), max_steps=1)
    assert "random-window sampling" in capsys.readouterr().out
