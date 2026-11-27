import glob
import os

import pytest

from mtp.config import config_diff, load_config

ROOT = os.path.join(os.path.dirname(__file__), "..")
CONFIGS = sorted(glob.glob(os.path.join(ROOT, "configs", "*.yaml")))


@pytest.mark.parametrize("path", CONFIGS, ids=os.path.basename)
def test_every_shipped_config_loads_and_validates(path):
    cfg = load_config(path)
    assert cfg.train.out_dir and "{" not in cfg.train.out_dir, "out_dir template must be resolved"


def test_experiment_arms_differ_only_in_the_intended_fields():
    """Fairness guard: Base NTP, MTP-k4 and MTP-k2 must share model size, data protocol, optimiser and budget."""
    base, k4, k2 = (load_config(os.path.join(ROOT, "configs", f"{n}.yaml")) for n in ("base_ntp", "mtp_k4", "mtp_k2"))
    for other in (k4, k2):
        diff = config_diff(base.to_dict(), other.to_dict())
        assert {d.split(":")[0] for d in diff} <= {"name", "model.num_pred_heads", "train.out_dir"}, diff
    assert (base.model.num_pred_heads, k2.model.num_pred_heads, k4.model.num_pred_heads) == (1, 2, 4)


def test_out_dir_template_and_overrides():
    cfg = load_config(os.path.join(ROOT, "configs", "mtp_k4.yaml"), ["seed=7", "train.lr=0.002"])
    assert cfg.train.out_dir.endswith("mtp_k4_s7") and cfg.train.lr == 0.002 and cfg.seed == 7


def test_windows_paths_survive_override_parsing():
    for raw in (r"C:\Users\me\data\train.bin", r"D:\new\table\x.bin"):
        assert load_config(None, [f"data.train_bin={raw}"]).data.train_bin == raw
        assert load_config(None, [f"data.train_bin={raw!r}"]).data.train_bin == raw


def test_unknown_keys_are_rejected():
    with pytest.raises(KeyError):
        load_config(None, ["model.num_heads_typo=3"])
    with pytest.raises(KeyError):
        load_config(None, ["trian.lr=1"])


@pytest.mark.parametrize("override", [
    "model.num_pred_heads=9",                    # > data.max_horizon
    "model.vocab_size=70000",                    # does not fit uint16
    "model.eos_id=99999",
    "model.d_model=30",                          # not divisible by n_heads
    "train.global_batch_tokens=1000",            # not a multiple of micro_batch*seq_len
    "train.total_tokens=10",                     # less than one batch
    "train.precision='int8'",
    "train.eval_every=0",
    "train.lr=0",
    "loss.lam=-1",
    "data.seq_len=2048",                         # > max_seq_len
])
def test_invalid_configs_fail_fast(override):
    with pytest.raises(ValueError):
        load_config(None, [override])


def test_trajectory_fingerprint_ignores_cosmetics_but_not_science():
    a = load_config(None, ["seed=1"])
    cosmetic = load_config(None, ["seed=1", "name='other'", "train.log_every=7", "train.out_dir='elsewhere'",
                                  "train.device='cpu'", "data.train_bin='/other/machine/train.bin'"])
    assert a.trajectory_fingerprint() == cosmetic.trajectory_fingerprint()
    assert a.fingerprint() != cosmetic.fingerprint()
    for override in ("seed=2", "train.lr=0.5", "model.num_pred_heads=2", "loss.lam=2.0", "train.total_tokens=999999999",
                     "data.max_horizon=6", "train.micro_batch_size=8", "train.warmup_steps=9"):
        changed = load_config(None, ["seed=1", override])
        assert changed.trajectory_fingerprint() != a.trajectory_fingerprint(), override


def test_default_data_seed_follows_run_seed_in_fingerprint():
    a = load_config(None, ["seed=3"])
    b = load_config(None, ["seed=3", "data.data_seed=3"])
    assert a.trajectory_fingerprint() == b.trajectory_fingerprint()
