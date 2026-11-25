import os, sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from make_synthetic_data import make  # noqa: E402
from mtp.config import ModelConfig  # noqa: E402


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    make(str(d), vocab=64, n_train=120_000, n_val=12_000)
    return str(d)


@pytest.fixture
def tiny_cfg():
    def _make(k=1, **kw):
        base = dict(vocab_size=64, d_model=32, n_layers=2, n_heads=4, ffn_hidden=64, max_seq_len=64,
                    num_pred_heads=k, eos_id=0)
        base.update(kw)
        return ModelConfig(**base)
    return _make
