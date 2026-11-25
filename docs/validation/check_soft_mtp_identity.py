"""Independent check of the Soft-MTP gradient identity (no code shared with the authors' repo).
Sum-reduced CE, parallel (Meta-style) heads, shared unembedding W, float64."""
import torch, torch.nn.functional as F
torch.manual_seed(0); torch.set_default_dtype(torch.float64)
B,T,d,V,K = 2,9,6,13,3           # K = number of EXTRA heads (offsets +2..+K+1 in our k-notation)
lam = 0.7
W  = torch.randn(V,d, requires_grad=True)
Z  = torch.randn(B,T,d, requires_grad=True)             # trunk output
A  = [torch.randn(d,d)*0.5 for _ in range(K)]
y  = torch.randint(0,V,(B,T))                            # y[:,t] = target of position t (pre-shifted)

def build():
    Zc = Z.detach().clone().requires_grad_(True); Wc = W.detach().clone().requires_grad_(True)
    heads = [torch.tanh(Zc@A[k]) for k in range(K)]      # head hiddens, post-"norm"
    for h in heads: h.retain_grad()
    return Zc, Wc, heads

def run(mode):
    Zc, Wc, heads = build()
    h0 = Zc.clone()
    if mode == "soft":
        for k,h in enumerate(heads, start=1):
            h0 = torch.cat([h0[:,:k], h0[:,k:] + lam*(h[:,:T-k].detach() - h[:,:T-k])], 1)
    ntp = F.cross_entropy((h0@Wc.T).reshape(-1,V), y.reshape(-1), reduction="sum")
    mtp = sum(F.cross_entropy((h[:,:T-k]@Wc.T).reshape(-1,V), y[:,k:].reshape(-1), reduction="sum")
              for k,h in enumerate(heads, start=1))
    loss = ntp + lam*mtp if mode != "kl" else None
    if mode == "kl":                                    # explicit forward KL to detached NTP teacher
        q = F.softmax(Zc@Wc.T, -1).detach()
        kl = 0
        for k,h in enumerate(heads, start=1):
            logp = F.log_softmax(h[:,:T-k]@Wc.T, -1); qk = q[:,k:]
            kl = kl + (qk*(qk.log()-logp)).sum()
        loss = ntp + lam*kl
    loss.backward()
    return Zc, Wc, heads, loss.item(), ntp.item()+lam*mtp.item()

Zs,Ws,Hs,Ls,Ls_hard = run("soft"); Zh,Wh,Hh,Lh,_ = run("hard"); Zk,Wk,Hk,Lk,_ = run("kl")

# (1) forward value of soft == hard (coupling is exactly x.detach()-x)
print("forward value soft==hard:", Ls == Lh, f"({Ls:.12f} vs {Lh:.12f})")
# (2) head-hidden gradients: soft == explicit KL (closed form W^T(p_k - q_{t+k}) * lam)
q = F.softmax(Zs.detach()@Ws.detach().T, -1)
for k,(hs,hk_) in enumerate(zip(Hs,Hk), start=1):
    p = F.softmax(hs[:,:T-k].detach()@Ws.detach().T, -1)
    want = lam*(p - q[:,k:])@Ws.detach()
    print(f"head+{k+1}: |soft-closedform|={ (hs.grad[:,:T-k]-want).abs().max():.2e}  |soft-explicitKL|={(hs.grad-hk_.grad).abs().max():.2e}")
# (3) trunk gradient: soft == explicit KL ; hard differs
print("trunk grad soft vs explicit-KL:", f"{(Zs.grad-Zk.grad).abs().max():.2e}", "| soft vs hard:", f"{(Zs.grad-Zh.grad).abs().max():.2e}")
# (4) W gradient: soft == hard (CE push kept on W); explicit-KL differs
print("W grad soft vs hard:", f"{(Ws.grad-Wh.grad).abs().max():.2e}", "| soft vs explicit-KL:", f"{(Ws.grad-Wk.grad).abs().max():.2e}")
