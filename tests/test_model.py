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
    ids2[:, t0 + 1:] = torch.randint(1, 64, (2, 20 - t0 - 1))
    with torch.no_grad():
        a, b = m.heads_forward(ids), m.heads_forward(ids2)
    for ha, hb in zip(a, b):
        assert torch.allclose(ha[:, :t0 + 1], hb[:, :t0 + 1], atol=1e-12)
        assert not torch.allclose(ha[:, t0 + 1:], hb[:, t0 + 1:])       # ...and the test can actually detect change


def test_kv_cache_decode_matches_full_forward(tiny_cfg):
    m = _model(tiny_cfg, 1)
    ids = torch.randint(1, 64, (2, 24))
    with torch.no_grad():
        full, _ = m(ids)
        logits, past = m(ids[:, :10], use_cache=True)
        outs = [logits]
        for t in range(10, 24):
            lg, past = m(ids[:, t:t + 1], past_key_values=past, use_cache=True)
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

    nz = lambda g: g is not None and g.abs().sum().item() > 0
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
        m.zero_grad(set_to_none=True); l.backward()
        gs.append(first.grad.clone())
    assert all(g.abs().sum() > 0 for g in gs)
    assert not torch.allclose(gs[0], gs[1])                 # heads pull the trunk in different directions


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


def test_hf_llama_parity(tiny_cfg):
    """Our k=1 model must equal transformers' LlamaForCausalLM given identical weights (validates RoPE, RMSNorm, SwiGLU)."""
    tr = pytest.importorskip("transformers")
    cfg = tiny_cfg(k=1)
    kwargs = dict(vocab_size=cfg.vocab_size, hidden_size=cfg.d_model, intermediate_size=cfg.ffn_hidden,
                  num_hidden_layers=cfg.n_layers, num_attention_heads=cfg.n_heads,
                  num_key_value_heads=cfg.n_heads, max_position_embeddings=cfg.max_seq_len,
                  rms_norm_eps=cfg.norm_eps, tie_word_embeddings=True, attention_bias=False,
                  mlp_bias=False, attn_implementation="eager")
    try:
        hf_cfg = tr.LlamaConfig(rope_theta=cfg.rope_theta, **kwargs)
    except TypeError:
        hf_cfg = tr.LlamaConfig(rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_theta}, **kwargs)
    hf = tr.LlamaForCausalLM(hf_cfg).double().eval()
    ours = MTPModel(cfg, init_seed=3).double().eval()
    sd = {"model.embed_tokens.weight": ours.embed_tokens.weight, "model.norm.weight": ours.norm.weight}
    for n, p in ours.layers.named_parameters():
        sd[f"model.layers.{n}"] = p
    missing = hf.load_state_dict({k: v.detach().clone() for k, v in sd.items()}, strict=False)
    assert not [m for m in missing.missing_keys if "lm_head" not in m], missing
    ids = torch.randint(0, 64, (2, 21))
    with torch.no_grad():
        ref = hf(ids).logits
        got, _ = ours(ids)
    # Residual ~1e-8 comes from RoPE cos/sin being built in fp32 on both sides (fp32 eps ~6e-8); a structural
    # mismatch (wrong rotate_half, norm eps, SwiGLU order, tying) shows up at >=1e-3. Tolerance sits in between.
    err = (ref - got).abs().max().item()
    assert err < 1e-6, err
