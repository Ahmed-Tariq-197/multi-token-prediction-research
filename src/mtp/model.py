"""Llama-style decoder with k parallel prediction heads (Gloeckle et al. 2024 style).

Layout
------
  tokens -> embed -> [Block x n_layers] -> z                    (shared trunk, pre-norm hidden)
  head 1 (MAIN, = Base NTP head): RMSNorm(z)                    -> tied unembedding W -> x_{t+1}
  head i (2..k, auxiliary)      : RMSNorm_i(Block_i(z))         -> tied unembedding W -> x_{t+i}

The main head has exactly the Base NTP architecture, so "evaluate with the main head" is a like-for-like
comparison. Heads are independent given z (parallel), which makes single-pass drafting possible later.

Module names intentionally mirror Hugging Face's Llama (embed_tokens, layers.N.self_attn.q_proj, mlp.gate_proj, ...)
so that tests/test_model.py can verify numerical parity against transformers.LlamaForCausalLM.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .utils import stable_hash


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(positions.float(), inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.d_model // cfg.n_heads
        d = cfg.d_model
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin, past=None):
        B, S, _ = x.shape
        sh = lambda t: t.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = sh(self.q_proj(x)), sh(self.k_proj(x)), sh(self.v_proj(x))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        past_len = 0
        if past is not None:
            past_len = past[0].shape[2]
            k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        new_past = (k, v)
        if past_len == 0:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif S == 1:
            out = F.scaled_dot_product_attention(q, k, v)
        else:  # multi-token continuation (used later for draft verification)
            kv_len = k.shape[2]
            mask = torch.arange(kv_len, device=x.device)[None, :] <= (past_len + torch.arange(S, device=x.device))[:, None]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, -1)), new_past


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, past=None):
        a, new_past = self.self_attn(self.input_layernorm(x), cos, sin, past)
        x = x + a
        return x + self.mlp(self.post_attention_layernorm(x)), new_past


class MTPModel(nn.Module):
    """k = cfg.num_pred_heads. k = 1 reproduces the Base NTP architecture exactly."""

    def __init__(self, cfg: ModelConfig, init_seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)                       # main-head final norm
        self.aux_blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.num_pred_heads - 1))
        self.aux_norms = nn.ModuleList(RMSNorm(cfg.d_model, cfg.norm_eps) for _ in range(cfg.num_pred_heads - 1))
        self.init_weights(init_seed)

    # ---------------------------------------------------------------- init
    @torch.no_grad()
    def init_weights(self, seed: int) -> None:
        """Per-parameter generators keyed by (seed, name): the trunk/embedding of a k=4 model is
        bit-identical to that of the k=1 model with the same seed (paired-seed design)."""
        std = 0.02
        resid_std = std / math.sqrt(2 * self.cfg.n_layers)
        for name, p in self.named_parameters():
            g = torch.Generator().manual_seed((seed * 1_000_003 + stable_hash(name)) % (2**31 - 1))
            if "norm" in name:
                p.fill_(1.0)
                continue
            s = resid_std if name.endswith(("o_proj.weight", "down_proj.weight")) else std
            p.copy_(torch.randn(p.shape, generator=g) * s)

    # ------------------------------------------------------------- forward
    @property
    def k(self) -> int:
        return self.cfg.num_pred_heads

    def unembed(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.embed_tokens.weight)

    def _rope(self, S: int, offset: int, device, dtype):
        cos, sin = rope_cos_sin(torch.arange(offset, offset + S, device=device),
                                self.cfg.d_model // self.cfg.n_heads, self.cfg.rope_theta, dtype)
        return cos[None, None], sin[None, None]

    def trunk(self, input_ids, past_key_values=None, use_cache=False):
        x = self.embed_tokens(input_ids)
        offset = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        cos, sin = self._rope(input_ids.shape[1], offset, x.device, x.dtype)
        new_past = []
        for i, layer in enumerate(self.layers):
            x, kv = layer(x, cos, sin, None if past_key_values is None else past_key_values[i])
            new_past.append(kv)
        return x, (new_past if use_cache else None)

    def head_hiddens(self, z: torch.Tensor) -> list[torch.Tensor]:
        """Post-norm hidden state of every head, [head1, ..., headk], each [B, T, d]."""
        hs = [self.norm(z)]
        if self.aux_blocks:
            cos, sin = self._rope(z.shape[1], 0, z.device, z.dtype)
            for blk, nrm in zip(self.aux_blocks, self.aux_norms):
                hs.append(nrm(blk(z, cos, sin)[0]))
        return hs

    def heads_forward(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        z, _ = self.trunk(input_ids)
        return self.head_hiddens(z)

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        """Main-head logits [B, S, V] (+ KV cache). This is the Base-NTP-comparable path."""
        z, past = self.trunk(input_ids, past_key_values, use_cache)
        return self.unembed(self.norm(z)), past

    # ------------------------------------------------------------- helpers
    def num_params(self, include_aux: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if not include_aux:
            n -= sum(p.numel() for m in (self.aux_blocks, self.aux_norms) for p in m.parameters())
        return n
