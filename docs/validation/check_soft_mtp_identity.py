"""Independent check of the Soft-MTP gradient identity (no code shared with the authors' repo).

Sum-reduced CE, parallel (Meta-style) heads, shared unembedding W, float64.
Run as a script for a printed report; tests/test_validation_artifacts.py imports `run_checks()` and asserts the
tolerances, so the claim in docs/DECISIONS.md (ADR-005) is enforced by CI rather than remembered.
"""
import torch
import torch.nn.functional as F


def run_checks(seed: int = 0, B: int = 2, T: int = 9, d: int = 6, V: int = 13, K: int = 3, lam: float = 0.7) -> dict:
    """K = number of auxiliary heads (our k = K + 1). Returns max absolute errors / booleans."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(seed)
        W = torch.randn(V, d)
        Z = torch.randn(B, T, d)
        A = [torch.randn(d, d) * 0.5 for _ in range(K)]
        y = torch.randint(0, V, (B, T))            # y[:, t] = target of NTP position t

        def run(mode):
            Zc = Z.clone().requires_grad_(True)
            Wc = W.clone().requires_grad_(True)
            heads = [torch.tanh(Zc @ A[k]) for k in range(K)]
            for h in heads:
                h.retain_grad()
            h0 = Zc.clone()
            if mode == "soft":                     # straight-through routing: value unchanged, gradient re-routed
                for k, h in enumerate(heads, start=1):
                    h0 = torch.cat([h0[:, :k], h0[:, k:] + lam * (h[:, :T - k].detach() - h[:, :T - k])], 1)
            ntp = F.cross_entropy((h0 @ Wc.T).reshape(-1, V), y.reshape(-1), reduction="sum")
            mtp = sum(F.cross_entropy((h[:, :T - k] @ Wc.T).reshape(-1, V), y[:, k:].reshape(-1), reduction="sum")
                      for k, h in enumerate(heads, start=1))
            if mode == "kl":                       # explicit forward KL(q || p) to a detached NTP teacher
                q = F.softmax(Zc @ Wc.T, -1).detach()
                kl = 0
                for k, h in enumerate(heads, start=1):
                    logp = F.log_softmax(h[:, :T - k] @ Wc.T, -1)
                    qk = q[:, k:]
                    kl = kl + (qk * (qk.log() - logp)).sum()
                loss = ntp + lam * kl
            else:
                loss = ntp + lam * mtp
            loss.backward()
            return Zc, Wc, heads, loss.item()

        Zs, Ws, Hs, Ls = run("soft")
        Zh, Wh, _, Lh = run("hard")
        Zk, Wk, Hk, _ = run("kl")
        q = F.softmax(Zs.detach() @ Ws.detach().T, -1)
        closed, vs_kl = 0.0, 0.0
        for k, (hs, hk) in enumerate(zip(Hs, Hk, strict=True), start=1):
            p = F.softmax(hs[:, :T - k].detach() @ Ws.detach().T, -1)
            want = lam * (p - q[:, k:]) @ Ws.detach()
            closed = max(closed, (hs.grad[:, :T - k] - want).abs().max().item())
            vs_kl = max(vs_kl, (hs.grad - hk.grad).abs().max().item())
        return {
            "forward_value_equal": Ls == Lh,
            "head_grad_vs_closed_form": closed,
            "head_grad_vs_explicit_kl": vs_kl,
            "trunk_grad_soft_vs_kl": (Zs.grad - Zk.grad).abs().max().item(),
            "trunk_grad_soft_vs_hard": (Zs.grad - Zh.grad).abs().max().item(),      # must be LARGE (routing changes it)
            "W_grad_soft_vs_hard": (Ws.grad - Wh.grad).abs().max().item(),           # must be ~0
            "W_grad_soft_vs_kl": (Ws.grad - Wk.grad).abs().max().item(),             # must be LARGE (KL differs on W)
        }
    finally:
        torch.set_default_dtype(old)


if __name__ == "__main__":
    for key, val in run_checks().items():
        print(f"{key}: {val}")
