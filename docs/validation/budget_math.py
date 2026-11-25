"""Back-of-envelope FLOP/param accounting. All throughput/MFU numbers are SCENARIOS, not measurements."""
import itertools
T4_PEAK = 65e12   # NVIDIA-published fp16 tensor peak for T4 (spec value; sustained is far lower)

def ffn_hidden(d):                       # SwiGLU, ~8/3 d rounded to multiple of 64
    return int(round(8*d/3/64))*64

def counts(d, L, V, k, ctx=1024):
    h = ffn_hidden(d)
    block = 4*d*d + 3*d*h                # attn (MHA) + SwiGLU
    emb = V*d
    trunk = L*block
    aux = (k-1)*block
    n_infer = trunk + emb                # what Base NTP / the main head path contains (tied unembedding)
    n_train = n_infer + aux
    # per-token fwd+bwd FLOPs: 6 * (matmul weights actually multiplied) ; unembedding matmul runs once per head
    mm = trunk + emb + (k-1)*(block + emb)        # emb counted once per head (lookup is free, unembed matmul is not)
    attn = 6*ctx*d*(L + (k-1))                    # causal score+value matmuls, fwd+bwd, one attn per block
    flops_tok = 6*mm + attn
    return dict(block=block, n_infer=n_infer, n_train=n_train, flops_tok=flops_tok, mm=mm)

print("== per-token cost and parameter counts (d=512, L=8, ctx=1024) ==")
base = {}
for V in (8192, 16384):
    for k in (1,2,4):
        c = counts(512, 8, V, k)
        if k==1: base[V]=c
        print(f"V={V:5d} k={k}: params(infer path)={c['n_infer']/1e6:6.1f}M  params(train)={c['n_train']/1e6:6.1f}M  "
              f"FLOPs/token={c['flops_tok']/1e6:7.1f}M  vs NTP x{c['flops_tok']/base[V]['flops_tok']:.2f}")

print("\n== scratch plan: tokens/param and GPU-hours per run (V=8192, d=512, L=8) ==")
n_p = counts(512,8,8192,1)['n_infer']
print(f"Base NTP N={n_p/1e6:.1f}M params")
hdr = "tok/param  tokens   | " + " | ".join(f"MFU {m:>4.0%}: NTP / MTP-k4 h" for m in (0.08,0.15,0.25))
print(hdr)
for tpp in (5, 10, 20):
    D = tpp*n_p
    row=[]
    for mfu in (0.08,0.15,0.25):
        eff = T4_PEAK*mfu
        h1 = counts(512,8,8192,1)['flops_tok']*D/eff/3600
        h4 = counts(512,8,8192,4)['flops_tok']*D/eff/3600
        row.append(f"{h1:5.1f} / {h4:5.1f}")
    print(f"{tpp:>5}     {D/1e6:6.0f}M | " + " | ".join(f"{r:>24}" for r in row))

print("\n== continued pretraining alternative: SmolLM2-135M (134.5M params, V=49152, 2T pretraining tokens) ==")
d,L,V=576,30,49152
pre=134.5e6
for D in (0.1e9,0.3e9,1.0e9):
    for mfu in (0.08,0.15,0.25):
        fl = 6*pre*D + 6*2048*d*L*D/2048*1.0  # attn term approx with ctx 2048
        # add k=4 heads: 3 extra (1 block + unembed matmul)
        block = 4*d*d + 3*int(round(8*d/3/64))*64*d
        fl4 = fl + 6*3*(block+V*d)*D
        print(f"D={D/1e9:.1f}B ({D/pre:5.1f} tok/param; {D/2e12:.4%} of its pretraining) MFU {mfu:.0%}: NTP {fl/(T4_PEAK*mfu)/3600:6.1f} h | MTP-k4 {fl4/(T4_PEAK*mfu)/3600:6.1f} h")

print("\n== how many runs fit? (scratch, V=8192, 10 tok/param, MFU scenario 15%) ==")
D = 10*n_p; eff=T4_PEAK*0.15
h_ntp = counts(512,8,8192,1)['flops_tok']*D/eff/3600
h_mtp = counts(512,8,8192,4)['flops_tok']*D/eff/3600
arms = {"NTP":h_ntp,"MTP k4":h_mtp,"Soft-MTP k4":h_mtp,"param-matched NTP":h_ntp*1.25,"MTP k2":(h_ntp+h_mtp)/2.0,"KL-MTP k4":h_mtp*1.1,"MTP k4+curriculum":h_mtp}
seeds=3
tot=sum(arms.values())*seeds
print({a:round(v,2) for a,v in arms.items()}, f"\nTotal for {seeds} seeds x {len(arms)} arms = {tot:.0f} GPU-h")
print(f"Sustained weekly capacity if 4 accounts x 30h/week x ~1 concurrent run = {4*30} h (optimistic; quota is floating, T4x2 concurrency unverified)")
