import os

import pytest
import torch

from mtp.loss import head_ce, multi_head_loss
from mtp.model import MTPModel
from mtp.targets import build_head_targets


def _model(tiny_cfg, k, seed=0):
    return MTPModel(tiny_cfg(k=k), init_seed=seed).double().eval()


@pytest.mark.parametrize("k", [1, 2, 4])
def test_no_future_leakage_in_any_head(tiny_cfg, k):
    """Perturbing tokens after position t0 must not change any head's hidden state at positions <= t0."""
    m = _model(tiny_cfg, k)
    ids = torch.randint(1, 64, (2, 20))
    ids2 = ids.clone()
    t0 = 9
    ids2[:, t0 + 1 :] = torch.randint(1, 64, (2, 20 - t0 - 1))
    with torch.no_grad():
        a, b = m.heads_forward(ids), m.heads_forward(ids2)
    for ha, hb in zip(a, b, strict=False):
        assert torch.allclose(ha[:, : t0 + 1], hb[:, : t0 + 1], atol=1e-12)
        assert not torch.allclose(ha[:, t0 + 1 :], hb[:, t0 + 1 :])  # ...and the test can actually detect change


def test_kv_cache_decode_matches_full_forward(tiny_cfg):
    m = _model(tiny_cfg, 1)
    ids = torch.randint(1, 64, (2, 24))
    with torch.no_grad():
        full, _ = m(ids)
        logits, past = m(ids[:, :10], use_cache=True)
        outs = [logits]
        for t in range(10, 24):
            lg, past = m(ids[:, t : t + 1], past_key_values=past, use_cache=True)
            outs.append(lg)
        # multi-token continuation (needed later for draft verification)
        lg_a, past2 = m(ids[:, :10], use_cache=True)
        lg_b, _ = m(ids[:, 10:17], past_key_values=past2, use_cache=True)
    assert torch.allclose(torch.cat(outs, 1), full, atol=1e-10)
    assert torch.allclose(lg_b, full[:, 10:17], atol=1e-10)


def _grads_of(m, loss):
    m.zero_grad(set_to_none=True)
    loss.backward()
    return {n: p.grad for n, p in m.named_parameters()}


def test_gradient_flow_per_head(tiny_cfg):
    k = 3
    m = MTPModel(tiny_cfg(k=k), init_seed=0).double().train()
    window = torch.randint(1, 64, (2, 16 + 4))
    tg = build_head_targets(window, 16, k, eos_id=0)
    W = m.embed_tokens.weight

    def head_only(i):
        hs = m.heads_forward(window[:, :16])
        return head_ce(hs[i], W, tg[:, i])[0]

    def nz(g):
        return g is not None and g.abs().sum().item() > 0

    # Head 2 (index 1) only: trunk + its own block receive gradient, the other aux heads receive none.
    g = _grads_of(m, head_only(1))
    assert all(nz(g[n]) for n in g if n.startswith(("layers.", "embed_tokens", "aux_blocks.0", "aux_norms.0")))
    assert all(not nz(g[n]) for n in g if n.startswith(("aux_blocks.1", "aux_norms.1", "norm.")))
    # Head 1 only: no aux parameter gets any gradient.
    g = _grads_of(m, head_only(0))
    assert all(not nz(g[n]) for n in g if n.startswith("aux_"))
    assert all(nz(g[n]) for n in g if n.startswith(("layers.", "norm.")))
    # Full objective: every parameter is trained.
    hs = m.heads_forward(window[:, :16])
    total, _ = multi_head_loss(hs, W, tg, lam=1.0)
    g = _grads_of(m, total)
    assert all(nz(v) for v in g.values()), [n for n, v in g.items() if not nz(v)]


def test_trunk_receives_gradient_from_every_head(tiny_cfg):
    k = 3
    m = MTPModel(tiny_cfg(k=k), init_seed=0).double().train()
    window = torch.randint(1, 64, (2, 16 + 4))
    tg = build_head_targets(window, 16, k, eos_id=0)
    first = m.layers[0].self_attn.q_proj.weight
    gs = []
    for i in range(k):
        hs = m.heads_forward(window[:, :16])
        l, _ = head_ce(hs[i], m.embed_tokens.weight, tg[:, i])
        m.zero_grad(set_to_none=True)
        l.backward()
        gs.append(first.grad.clone())
    assert all(g.abs().sum() > 0 for g in gs)
    assert not torch.allclose(gs[0], gs[1])  # heads pull the trunk in different directions


def test_paired_seed_trunk_identical_across_arms(tiny_cfg):
    a, b = MTPModel(tiny_cfg(k=1), init_seed=7), MTPModel(tiny_cfg(k=4), init_seed=7)
    sa, sb = a.state_dict(), b.state_dict()
    assert all(torch.equal(sa[n], sb[n]) for n in sa), "k=1 weights must be a bit-exact subset of k=4"
    c = MTPModel(tiny_cfg(k=1), init_seed=8)
    assert not torch.equal(sa["embed_tokens.weight"], c.state_dict()["embed_tokens.weight"])


def test_param_counts(tiny_cfg):
    m1, m4 = MTPModel(tiny_cfg(k=1)), MTPModel(tiny_cfg(k=4))
    block = sum(p.numel() for p in m1.layers[0].parameters())
    assert m4.num_params(False) == m1.num_params(True)
    assert m4.num_params(True) - m1.num_params(True) == 3 * (block + 32)


def _transformers():
    """Dependency-sensitive test guard. Locally a missing `transformers` skips; in CI (REQUIRE_HF_PARITY=1) it FAILS,
    so the reference-parity check can never be silently dropped from the pipeline."""
    try:
        import transformers

        return transformers
    except ImportError:
        if os.environ.get("REQUIRE_HF_PARITY"):
            pytest.fail("REQUIRE_HF_PARITY is set but transformers is not installed")
        pytest.skip("transformers not installed (pip install -e '.[dev]')")


def _hf_model(tr, cfg, theta):
    kwargs = dict(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        intermediate_size=cfg.ffn_hidden,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_heads,
        max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=cfg.norm_eps,
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        attn_implementation="eager",
    )
    try:
        hf_cfg = tr.LlamaConfig(rope_theta=theta, **kwargs)
    except TypeError:
        hf_cfg = tr.LlamaConfig(rope_parameters={"rope_type": "default", "rope_theta": theta}, **kwargs)
    return tr.LlamaForCausalLM(hf_cfg).double().eval()


def _load_into_hf(hf, ours):
    sd = {"model.embed_tokens.weight": ours.embed_tokens.weight, "model.norm.weight": ours.norm.weight}
    for n, p in ours.layers.named_parameters():
        sd[f"model.layers.{n}"] = p
    res = hf.load_state_dict({k: v.detach().clone() for k, v in sd.items()}, strict=False)
    assert not [m for m in res.missing_keys if "lm_head" not in m], res
    assert not res.unexpected_keys, res


@pytest.mark.parametrize("theta", [10000.0, 500000.0])
def test_hf_llama_parity(tiny_cfg, theta):
    """Our k=1 model equals transformers' LlamaForCausalLM given identical weights (RoPE, RMSNorm, SwiGLU, tying).
    Two thetas: a non-default value proves rope_theta actually reaches both implementations."""
    tr = _transformers()
    cfg = tiny_cfg(k=1, rope_theta=theta)
    ours = MTPModel(cfg, init_seed=3).double().eval()
    hf = _hf_model(tr, cfg, theta)
    _load_into_hf(hf, ours)
    ids = torch.randint(0, 64, (2, 21))
    with torch.no_grad():
        ref, (got, _) = hf(ids).logits, ours(ids)
    # Residual ~1e-8 comes from RoPE cos/sin being built in fp32 on both sides (fp32 eps ~6e-8); a structural
    # mismatch (rotate_half, norm eps, SwiGLU order, tying, theta) shows up at >= 1e-3. Tolerance sits in between.
    err = (ref - got).abs().max().item()
    assert err < 1e-6, err


def test_hf_parity_has_teeth_negative_control(tiny_cfg):
    """If our theta differs from the reference's, the parity check must fail; otherwise it proves nothing."""
    tr = _transformers()
    ours = MTPModel(tiny_cfg(k=1, rope_theta=500000.0), init_seed=3).double().eval()
    hf = _hf_model(tr, tiny_cfg(k=1), 10000.0)
    _load_into_hf(hf, ours)
    ids = torch.randint(0, 64, (2, 21))
    with torch.no_grad():
        err = (hf(ids).logits - ours(ids)[0]).abs().max().item()
    assert err > 1e-4, f"parity check cannot distinguish rope_theta (err={err})"


def test_mixed_precision_path_keeps_one_dtype_inside_attention(tiny_cfg):
    """CPU bf16 autocast emulates the CUDA fp16 path: q, k, v must enter attention in the same dtype
    (previously fp32 q/k met bf16 v and only autocast's SDPA cast made it work), and results stay close to fp32."""
    m = MTPModel(tiny_cfg(k=3), init_seed=0)
    seen, orig = [], torch.nn.functional.scaled_dot_product_attention

    def spy(q, k, v, *a, **kw):
        seen.append((q.dtype, k.dtype, v.dtype))
        return orig(q, k, v, *a, **kw)

    ids = torch.randint(1, 64, (2, 16))
    try:
        torch.nn.functional.scaled_dot_product_attention = spy
        with torch.autocast("cpu", dtype=torch.bfloat16):
            low = m.heads_forward(ids)
    finally:
        torch.nn.functional.scaled_dot_product_attention = orig
    assert seen and all(len(set(t)) == 1 and t[0] == torch.bfloat16 for t in seen), seen
    full = m.heads_forward(ids)
    for a, b in zip(low, full, strict=True):
        assert torch.isfinite(a.float()).all()
        assert (a.float() - b).abs().max().item() < 0.05


def test_mixed_precision_backward_is_finite(tiny_cfg):
    m = MTPModel(tiny_cfg(k=3), init_seed=0)
    window = torch.randint(1, 64, (2, 16 + 4))
    tg = build_head_targets(window, 16, 3, eos_id=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = multi_head_loss(m.heads_forward(window[:, :16]), m.embed_tokens.weight, tg)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters())
