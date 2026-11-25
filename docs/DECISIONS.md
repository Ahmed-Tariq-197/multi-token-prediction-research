# Decision log (ADRs) — AIE417 MTP project

Evidence labels: **[verified]** read in the source / executed here · **[reported]** published claim not reproduced by us ·
**[assumption]** must be validated experimentally · **[estimate]** arithmetic from stated assumptions, not a measurement.

## ADR-001 Training regime: from scratch (primary)

| | From scratch (chosen) | Continued pretraining of a released checkpoint |
|---|---|---|
| Example | 29.9M-param main path, V=8192, d=512, L=8 | SmolLM2-135M (134.5M params, 2T pretraining tokens [verified, HF model card]) |
| Tokens / param | 10–20 (final value from calibration) | 0.3B tokens = 2.2 tok/param, i.e. 0.015% of its pretraining |
| GPU-h per run [estimate, 15% MFU] | ~1.7 (NTP) / ~2.8 (k=4) at 10 tok/param | ~6.9 (NTP) / ~11.9 (k=4) at 0.3B tokens |
| Runs affordable (7 arms × 3 seeds) | ~53 GPU-h total | ~250 GPU-h total → effectively 1 seed |
| What it tests | Effect of the objective on a trunk trained *under* it | Effect of adapting heads/trunk of a trunk already shaped by 2T NTP tokens |

Decision: from scratch is the primary controlled study because MTP is a pretraining objective, identical random
init is trivial, and ≥3 paired seeds are affordable. Continued pretraining is **not** adopted for the main
comparison; at ~2 tokens/param the trunk barely moves, so it would mostly measure head fitting. It stays an optional
single-seed external-validity check in Phase 2 if budget remains. Threat to validity: absolute quality of a 30M model is
low and conclusions may not transfer to ≥1B models (the regime where Gloeckle et al. report MTP gains).

## ADR-002 Data and tokenizer
FineWeb-Edu `sample-10BT` (ODC-By, has `id` and `token_count` fields [verified via HF Hub]); split by
`sha256(id) mod 1000` (0.5% val, 0.5% test, rest train), no index files needed; stream until a GPT-2-token budget is
reached; byte-level BPE **V=8192 trained on train text only**, EOS = id 0; uint16 bins; manifest records dataset
revision (must be pinned), tokenizer hash, counts, bytes/token (for BPB). V=8192 rather than 16k/32k because at d=512
the unembedding matmul dominates head cost (ADR-003). Open: V changes how hard MTP is (the curriculum paper argues
byte-level inputs make MTP easier for small models [reported]); we do not ablate V — listed as a limitation.
The `hf` ingestion path is **unexecuted** in the authoring environment; the offline path is tested.

## ADR-003 Architecture
Llama-style (RMSNorm, RoPE, SwiGLU, tied unembedding), own implementation, validated against
`transformers.LlamaForCausalLM` in fp64 (max logit error < 1e-6, test passes [verified]). Parallel heads
(Gloeckle et al.): trunk output `z`; head 1 = `RMSNorm(z)` (identical to Base NTP), head i≥2 = `RMSNorm_i(Block_i(z))`;
all share the tied unembedding W. Not DeepSeek-style sequential modules: the spec phrase "head i predicts offset +i"
maps directly onto parallel heads, drafting needs one trunk pass, and tests are unambiguous. Sequential heads remain an
optional Phase 2 ablation.
**Compute fact [estimate, verify in calibration]:** at this scale the unembedding matmul (V·d = 4.2M MACs/token) is larger
than a transformer block (3.2M), so each extra head adds ~24% to the per-token FLOPs of a base step; k=4 costs ≈1.7× FLOPs/token of NTP
(≈2.0× with V=16k). At 7B this overhead is ~0 because V·d is ~2% of the model. Consequence: *equal-token* comparisons give
MTP more compute; we report FLOPs and wall-clock for every arm and add a compute-matched NTP control in Phase 2.

## ADR-004 Notation and objective (resolves the head-count ambiguity)
Spec: "k ≥ 2 heads: head i predicts the token at offset +i … use the main next-token head for comparable evaluation."
- **k = total number of heads, including the main head.** Head i ∈ {1..k} predicts x_{t+i} from x_{≤t}. Head 1 is the main
  NTP head. k=1 is Base NTP. Standard MTP primary k=4; k=2 is the minimum the spec allows (pilot + ablation).
- Soft-MTP paper mapping: its K auxiliary heads ↔ our k−1; its head j (predicting x_{t+j+1}) ↔ our head i=j+1.
- Loss: `L = L_1 + λ/(k−1) · Σ_{i=2..k} L_i`, λ=1; `L_i` = mean CE over head i's valid positions. Gloeckle's unweighted sum
  = λ=k−1. Chosen to match Soft-MTP's normalisation so Phase 2 changes only gradient routing.
- Window of `T + H` tokens (H=8 for all arms): inputs `w[:, :T]`; target(i,t) = `w[:, t+i]`. Valid iff no EOS lies in
  positions t..t+i−1 (same document; the EOS ending the document is a legal target). Same rule for head 1 in every arm.
  Attention is not masked across packed documents (limitation).
- Residual ambiguity: if the instructor intends "main head + k extra heads", set `num_pred_heads = k+1`; nothing else
  changes. Confirm at the Phase 1 checkpoint.

## ADR-005 Phase 2 improvement: Soft-MTP (gradient-routed self-distillation), with a published comparator
Mechanism: each aux head is trained toward the model's own next-token distribution at the aligned position instead of the
one-hot realised token, implemented by straight-through routing of the NTP loss gradient (forward value unchanged).
Evidence:
- Identity verified **by us** in fp64 with code independent of the authors': head and trunk gradients equal the closed form
  `λ/(K·n)·(p_k − q)ᵀW` and an explicit KL(q‖p) objective to ~1e-15; forward loss unchanged; gradient on shared W
  differs from explicit KL, equals hard MTP [verified; script and output in docs/validation/]. The authors' own
  17 identity tests also pass on our machine [verified]. Official code (MIT) exists: github.com/JaydenTeoh/Soft-MTP.
- Their results [reported, not reproduced]: 167M–1.2B models, 31.5B tokens, Muon optimizer, bf16, sequential heads,
  untied embeddings. 167M FineWeb-Edu PPL: NTP 15.23, MTP 15.98, Soft-MTP 15.95 — i.e. ~0.2% over MTP and still 4.7% behind
  NTP; acceptance length +1.8–2.2%. In the portions we read (through Appendix B.3) no seed counts or confidence intervals
  are reported; LAMBADA PPL is *worse* for Soft-MTP than MTP at 167M and 332M. It is an arXiv preprint (28 Oct 2026);
  peer review not confirmed.
- Their own theory says the bias term is the NTP verifier's error, which shrinks with scale and training time → **our regime
  (30M, ~10–20 tok/param) is the least favourable one for the NTP-loss benefit**. For *agreement with the verifier* (the
  quantity speculative decoding needs) the bias is not a cost. Hence the primary target is inference-side (ADR-006).
- Transfer to our setting is an **[assumption]**: different optimizer (AdamW), precision (fp16), head type (parallel),
  tokens/param, vocab. Appendix on Meta-style parallel heads was not read — read before Phase 2.

Comparator: forward MTP curriculum (Aynetdinov & Akbik, ACL 2025) [verified from the paper's tables]. Models 1.3B/3B on
MiniPile (≈1.7B tokens, ≈1.3 tok/param) plus a 10B-token FineWeb-Edu 1.3B check. Forward curriculum improves on *static*
MTP but never reaches NTP for subword models; at 10B tokens it equals static MTP (PPL 57 vs 57; LAMBADA 101 vs 100); reverse
curriculum approaches NTP quality but yields no speculative-decoding benefit. Single runs, no variance reported. Evidence is
at larger scale than ours, in a data-starved regime unlike ours.

Fair comparison summary: neither method has evidence at ≤50M parameters; both have small, unreplicated effects; both are
near-free to run. Soft-MTP is selected because it is a genuine objective change with a falsifiable mechanism (gradient
SNR / agreement with verifier) and a unit-testable identity; the curriculum is a schedule hyperparameter. To avoid
cherry-picking, the curriculum is run in Phase 2 as an **external comparison arm** (not as "the improvement"). The choice is
frozen now; it changes only if the Phase 2 identity test fails on our implementation, never after seeing results.

## ADR-006 Phase 2 evaluation protocol (pre-registered before any Phase 2 run)
Principle: a metric the method directly optimises cannot, alone, show benefit.
- **Tier A (confirmatory):**
  A1. Mean accepted tokens per verifier step τ in *real* lossless self-speculative decoding (Leviathan-style rejection
  sampling, T=1.0; greedy run used as a losslessness test) on held-out prompts. This is end-to-end behaviour, not the KL we
  train on. A2. Main-head validation loss as a **non-inferiority** check vs Standard MTP, margin δ = 2 × seed-std measured in
  the Phase 1 pilot, frozen before Phase 2.
- **Tier B (mechanistic, optimisation-aligned — explicitly *not* evidence of benefit):** per-head KL/TV to the verifier
  (teacher-forced), per-head loss/top-1, gradient SNR decomposition (hard vs KL vs residual).
- **Tier C (generalisation / downstream, exploratory):** WikiText-103 test PPL (out of domain), BLiMP via main head
  (the curriculum paper's choice for small models), LAMBADA PPL, repetition/distinct-n of samples to detect degenerate
  gains, measured wall-clock speed-up and forward-pass-count speed-up. Zero-shot QA benchmarks are at chance for ~30M models;
  reported with CIs, not relied upon.
- Statistics: ≥3 paired seeds/arm (same init for the trunk, same batches); report mean ± std, paired differences with
  bootstrap CIs; no best-seed selection. Seed count revised from the pilot variance: n ≈ 2(1.96σ/δ)² per comparison.
- Controls: param-matched deeper NTP (answers "extra parameters?"), compute-matched NTP (equal FLOPs), hard vs soft vs
  explicit-KL (ablation 1), k=2 vs k=4 (ablation 2), forward curriculum comparator.
- Honest expectation **[assumption]**: Standard MTP loses to Base NTP on main-head PPL at this scale; Soft-MTP's effect
  may be below seed noise. A null result will be reported as such.

## ADR-007 Bonus: profiling first, no technique committed
`scripts/profile_decode.py` measures ms/token, kernels/token, GPU-busy fraction, achieved vs roofline bandwidth for batch-1
cached decoding (the untouched baseline). Decision table:
| Measured | Bottleneck | Candidates |
|---|---|---|
| GPU-busy < 50% | launch/CPU overhead | static KV cache + graph capture, fusion |
| busy, bandwidth utilisation > 50% | weight/KV traffic | weight-only or KV-cache quantisation |
| busy, below roofline | kernel inefficiency | fusion, better attention kernel |
Hypothesis **[estimate, untested]**: our 30M model is launch-bound (weights 60 MB → ~0.2 ms at 320 GB/s vs several ms of
eager launches), while a ~0.4–1B pretrained model is closer to memory-bound; we profile both and pick the technique and the
model on evidence. The speculative-decoding family is excluded from the Bonus: it shares verification machinery with the
Improved-MTP target metric and would blur the "distinct method" requirement.

## ADR-008 Training budget = derived, not chosen
See docs/CALIBRATION.md. Session facts [reported by third-party sources; Kaggle's own docs give no quota number]: Kaggle
≈30 GPU-h/week (floating), 12 h session cap, 2×T4 or P100; Colab free has no published quota and ~12 h cap. Quota is read from
each account's meter. Budget constraint set: slowest arm ≤ 5 h per run, ≤ 20 tokens/param, campaign ≤ 60% of team quota.

## ADR-009 Repository workflow
Local work on `wip/*` branches with member-attributed commits, never pushed. At the audit gate: clean history preserving
authorship, push `phase1`, merge to `main` with `--no-ff`, tag `phase1-v1.0`. `main` already holds GitHub's initial README
commit; it is preserved. See docs/WORKFLOW.md.

## Unresolved uncertainties (to be closed by experiment)
1. Achieved T4 throughput, memory, fp16 stability of MTP training (calibration).
2. Whether the unembedding-dominated head cost is as predicted (calibration: ratio of k=4 to k=1 tokens/s).
3. Seed-to-seed σ of main-head loss and of τ (Phase 1 pilot) → seed count and δ.
4. LR for each arm (short sweep) — tuning effort must be equal across arms.
5. Whether Soft-MTP transfers to parallel heads, AdamW, fp16, ≤50M params, ≤20 tok/param.
6. Whether the T4×2 session counts quota per GPU or per wall-hour, and whether two concurrent runs are practical.
7. HF streaming ingestion path and dataset revision pinning (unexecuted here).

## References
Gloeckle et al. 2024, arXiv:2404.19737 · Aynetdinov & Akbik, ACL 2025, arXiv:2505.22757 · Teoh 2026, "Multi-Token Prediction
Done Right", arXiv:2610.38331 · DeepSeek-AI 2024, arXiv:2412.19437 · Cai et al. 2024 (Medusa), arXiv:2401.10774 ·
Stern et al. 2018 (blockwise parallel decoding) · Leviathan et al. 2023 (speculative decoding; not re-read this session) ·
FastMTP, arXiv:2509.18362 (seen in search results only).
