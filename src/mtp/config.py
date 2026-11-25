"""Typed run configuration (YAML -> dataclasses) with dotted command-line overrides.

Notation used everywhere in this code base (see docs/DECISIONS.md, ADR-004):

  k            total number of prediction heads, k >= 1. Head i in {1..k} predicts x_{t+i} from x_{<=t}.
               Head 1 is the MAIN next-token head. k = 1 is the Base NTP model.
  aux heads    heads 2..k (k-1 of them): one transformer block + RMSNorm each, shared tied unembedding.
  max_horizon  fixed look-ahead appended to every training window (>= k), so every arm sees
               byte-identical input batches regardless of k.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    ffn_hidden: int = 1408          # SwiGLU hidden size (~8/3 * d_model, multiple of 64)
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    num_pred_heads: int = 1         # k
    eos_id: int = 0                 # document separator id; also drives cross-document masking


@dataclass
class LossConfig:
    # L = L_1 + lam/(k-1) * sum_{i=2..k} L_i ; each L_i is a mean over head i's valid positions.
    lam: float = 1.0
    mask_cross_document: bool = True


@dataclass
class DataConfig:
    train_bin: str = "data/train.bin"
    val_bin: str = "data/val.bin"
    test_bin: str = "data/test.bin"
    manifest: str = "data/manifest.json"
    seq_len: int = 1024
    max_horizon: int = 8            # window = seq_len + max_horizon tokens
    data_seed: int | None = None    # defaults to the run seed (paired seeds across arms)


@dataclass
class TrainConfig:
    total_tokens: int = 100_000_000
    global_batch_tokens: int = 131072
    micro_batch_size: int = 16
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    precision: str = "fp16"         # fp16 | bf16 | fp32 (T4 has no bf16; CPU always runs fp32)
    compile: bool = False
    device: str | None = None
    log_every: int = 20
    eval_every: int = 500
    eval_windows: int = 256
    ckpt_every: int = 500
    out_dir: str = "runs/debug"


@dataclass
class RunConfig:
    name: str = "debug"
    seed: int = 1337
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def _build(cls, d: dict):
    default = cls()
    kwargs = {}
    names = {f.name for f in dataclasses.fields(cls)}
    for key, val in (d or {}).items():
        if key not in names:
            raise KeyError(f"unknown config key '{key}' for {cls.__name__}")
        cur = getattr(default, key)
        if dataclasses.is_dataclass(cur):
            kwargs[key] = _build(type(cur), val)
        else:
            kwargs[key] = tuple(val) if isinstance(val, list) else val
    return cls(**kwargs)


def load_config(path: str | None, overrides: list[str] | None = None) -> RunConfig:
    raw: dict = {}
    if path:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    cfg = _build(RunConfig, raw)
    validate(cfg)
    return cfg


def validate(cfg: RunConfig) -> None:
    m, d, t = cfg.model, cfg.data, cfg.train
    assert m.d_model % m.n_heads == 0, "d_model must be divisible by n_heads"
    assert (m.d_model // m.n_heads) % 2 == 0, "head_dim must be even for RoPE"
    assert 1 <= m.num_pred_heads <= d.max_horizon, "need 1 <= k <= data.max_horizon"
    assert d.seq_len <= m.max_seq_len, "seq_len exceeds model max_seq_len"
    per_micro = t.micro_batch_size * d.seq_len
    assert t.global_batch_tokens % per_micro == 0, (
        f"global_batch_tokens ({t.global_batch_tokens}) must be a multiple of "
        f"micro_batch_size*seq_len ({per_micro})")
    assert t.precision in ("fp16", "bf16", "fp32")
