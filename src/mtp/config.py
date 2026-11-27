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
import copy
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
        """Hash of the complete configuration (provenance record)."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def trajectory_dict(self) -> dict:
        """Only the settings that determine the training trajectory. Machine-specific paths, logging/eval/checkpoint
        cadence, run name, device and torch.compile are excluded, so a run may legitimately resume on another
        machine (Colab -> Kaggle) but not under a different seed, model, data protocol, LR schedule or budget."""
        d = copy.deepcopy(self.to_dict())
        d.pop("name", None)
        for k in ("out_dir", "device", "compile", "log_every", "eval_every", "eval_windows", "ckpt_every"):
            d["train"].pop(k, None)
        for k in ("train_bin", "val_bin", "test_bin", "manifest"):
            d["data"].pop(k, None)
        d["data"]["data_seed"] = self.seed if self.data.data_seed is None else self.data.data_seed
        return d

    def trajectory_fingerprint(self) -> str:
        blob = json.dumps(self.trajectory_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def config_diff(a: dict, b: dict, prefix: str = "") -> list[str]:
    """Human-readable list of differing leaves between two (nested) config dicts."""
    out = []
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key, "<missing>"), b.get(key, "<missing>")
        if isinstance(va, dict) and isinstance(vb, dict):
            out += config_diff(va, vb, f"{prefix}{key}.")
        elif va != vb:
            out.append(f"{prefix}{key}: {va!r} -> {vb!r}")
    return out


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
        with open(path, encoding="utf-8") as f:
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
    cfg.train.out_dir = cfg.train.out_dir.format_map(
        {"seed": cfg.seed, "name": cfg.name, "k": cfg.model.num_pred_heads})   # e.g. runs/{name}_s{seed}
    validate(cfg)
    return cfg


def validate(cfg: RunConfig) -> None:
    """Fail early (before any GPU time is spent) on settings that would silently invalidate a run."""
    m, d, t, lo = cfg.model, cfg.data, cfg.train, cfg.loss
    if m.d_model % m.n_heads:
        raise ValueError("model.d_model must be divisible by model.n_heads")
    if (m.d_model // m.n_heads) % 2:
        raise ValueError("head_dim must be even for RoPE")
    if not 1 <= m.num_pred_heads <= d.max_horizon:
        raise ValueError(f"need 1 <= model.num_pred_heads ({m.num_pred_heads}) <= data.max_horizon ({d.max_horizon})")
    if d.seq_len > m.max_seq_len:
        raise ValueError("data.seq_len exceeds model.max_seq_len")
    if not 1 < m.vocab_size <= 65536:
        raise ValueError("model.vocab_size must be in (1, 65536]: token files are stored as uint16")
    if not 0 <= m.eos_id < m.vocab_size:
        raise ValueError("model.eos_id must be a valid token id")
    per_micro = t.micro_batch_size * d.seq_len
    if t.global_batch_tokens % per_micro:
        raise ValueError(f"train.global_batch_tokens ({t.global_batch_tokens}) must be a multiple of "
                         f"micro_batch_size*seq_len ({per_micro})")
    if t.total_tokens < t.global_batch_tokens:
        raise ValueError("train.total_tokens must be at least one global batch")
    if t.precision not in ("fp16", "bf16", "fp32"):
        raise ValueError("train.precision must be fp16, bf16 or fp32")
    if min(t.log_every, t.eval_every, t.ckpt_every, t.eval_windows, t.micro_batch_size) < 1:
        raise ValueError("log_every, eval_every, ckpt_every, eval_windows, micro_batch_size must be >= 1")
    if t.lr <= 0 or not 0 <= t.min_lr_ratio <= 1 or t.warmup_steps < 0:
        raise ValueError("need lr > 0, 0 <= min_lr_ratio <= 1, warmup_steps >= 0")
    if lo.lam < 0:
        raise ValueError("loss.lam must be >= 0")
