# SKEPTIC FINDINGS — Qwen-Image pure-Mojo port (2026-05-26)

Reviewer: fresh-eyes skeptic. Method: line-by-line vs the Rust oracle
(`inference-flame/src/models/qwenimage_dit.rs`, `vae/wan21_vae.rs` +
`vae/qwenimage_decoder.rs`, `models/qwen25vl_encoder.rs`) plus the flame-core
foundation kernels each path calls. CODE-ONLY: compiled, NOT executed.

Files reviewed:
- `serenitymojo/models/dit/qwenimage_dit.mojo`
- `serenitymojo/models/vae/qwenimage_decoder.mojo`
- `serenitymojo/models/text_encoder/qwen25vl_encoder.mojo`
- `serenitymojo/pipeline/qwenimage_pipeline_smoke.mojo`
- foundation ops touched: `ops/rope.mojo`, `ops/attention.mojo`,
  `ops/norm.mojo`, `ops/linear.mojo`, `ops/activations.mojo`,
  `ops/embeddings.mojo`, `ops/tensor_algebra.mojo`, `ops/layout.mojo`

## Compile honesty — VERIFIED

`pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/qwenimage_pipeline_smoke.mojo -o /tmp/sk_smoke`
→ **EXIT=0**, binary produced (1.9 MB). The smoke imports all three model
modules (DiT + encoder + VAE decoder), so this compile-checks every reviewed
model file together. Only warnings are unused-var (`p0/p1/p2` at
qwenimage_dit.mojo:171-173, `d` at :511, `di` at decoder:254 — STYLE). Without
`-lm` the link fails on `sinf@@GLIBC` (known, unrelated to this code). The
builder's "compile-verified, not executed" claim is **honest**.

No old syntax: zero `alias` (all `comptime`), zero `inout`/`let`. `def`s
without `raises` are pure constructors that call nothing raising.

---

## The two highest-risk items — BOTH CORRECT

### 1. VAE channel-axis RMS reduction — CORRECT (not the wrong dim)
`qwenimage_decoder.mojo:294-300` `_rms_norm5d` flattens gamma to `[C]` and
calls the foundation `rms_norm`, which reduces over the **last dim**. The
entire decoder runs in **NDHWC `[N,D,H,W,C]` with channel LAST** (decode:411
reshapes the latent to NDHWC; every conv3d is QRSCF/NDHWC-native). So last-dim
rms_norm == channel-dim rms_norm. The reduction axis is right. Confirmed
against Rust `RmsNorm5d::forward` (wan21_vae.rs:166-179) which does
`F.normalize(x, dim=1)` over channel on a `[B,C,T,H,W]` (channel-first) tensor —
the Mojo layout choice (channel-last) makes the foundation last-dim op the
correct equivalent. **Not a bug.**

Minor eps-position difference (FRAGILE, see below) is the only deviation.

### 2. Encoder RoPE is HALF-SPLIT, DiT RoPE is INTERLEAVED — CORRECT, not swapped
- DiT `qwenimage_dit.mojo:425-426` calls `rope_interleaved`; matches Rust
  `rope_fused_bf16` (flame-core bf16_ops.rs:766-797: "Interleaved (complex)
  RoPE: pairs adjacent elements (2d,2d+1)"). Identical math
  `out[2i]=x0c-x1s, out[2i+1]=x0s+x1c`.
- Encoder `qwen25vl_encoder.mojo:597-598` calls `rope_halfsplit`; matches Rust
  `rope_halfsplit_bf16` / HF `rotate_half` (qwen25vl_encoder.rs:254-256).
  Mojo halfsplit kernel (rope.mojo:113-118) is exact HF `rotate_half`.

The builder did NOT accidentally use interleaved in the encoder. **Not a bug.**

Layout-adaptation subtlety (verified correct): the Rust tables are
`[1,1,N,half]` and the kernel broadcasts the angle across heads
(`cos_idx = seq*half + d`). The Mojo `rope_*` ops flatten BSHD `[1,S,H,Dh]` to
rows `r = s*H + head` and consume cos/sin as `[rows, half]`. The Mojo table
builders (DiT build_qwenimage_rope_tables:188-200; encoder _build_rope_tables:
378-387) **replicate the per-token angle across `heads` rows** in exactly that
`(token, head)` row order. Net effect identical: each (head,token,pair) gets
the angle that depends only on (token,pair). Correct.

---

## DiT (qwenimage_dit.mojo) — self-flagged items, all CORRECT

| Claim | Verdict | Evidence |
|---|---|---|
| concat TXT-then-IMG | ✅ | mojo:420-422 `concat(1, ctx, txt_q, img_q)`; Rust:1155 `cat([txt_q,img_q],2)` |
| RoPE interleaved, axes (16,56,56), θ=10000 | ✅ | config:106-108; freq tables mojo:144-155 == Rust:353-364 |
| AdaLN-Continuous scale-first | ✅ | mojo:567 `out_scale=mods[0:dim]`, `out_shift=mods[dim:2dim]`; Rust:685-686 same |
| per-block 6-way {shift,scale,gate}×2 | ✅ | mojo:363-375 == Rust:1076-1088 (shift1,scale1,gate1,shift2,scale2,gate2) |
| LayerNorm no-affine then `x*(1+scale)+shift` | ✅ | mojo:298-303 `_modulate`; Rust:1093-1095. Biased var matches torch (norm.mojo:272) |
| QK-RMSNorm on both img+txt | ✅ | mojo:410-417 norm_q/k + norm_added_q/k; Rust:1149-1152 |
| FFN GELU-tanh | ✅ | activations.mojo:36-38 == flame gelu_bf16 (bf16_ops.rs:44-47), same consts |
| to_out.0 / to_add_out split | ✅ | mojo:438-443; Rust:1168-1169 |
| timestep `t*1000` + cos-first (flip_sin_to_cos) | ✅ | mojo:528 pre-scales `t*1000`; embeddings.mojo:74-76 cos[0:half],sin[half:]; Rust time_proj:283 `cat([cos,sin])` |

GQA / fused-QKV note: the Rust DiT has a fused `to_qkv` turbo path; the Mojo
uses only the **split** to_q/to_k/to_v path (img) and add_q/k/v_proj (txt).
Standard 2512 checkpoints use the split path (Rust:1117-1126 fallback), so this
is the correct target. ✅

---

## VAE decoder (qwenimage_decoder.mojo) — self-flagged items, all CORRECT

| Claim | Verdict | Evidence |
|---|---|---|
| 3D causal, ZERO left-temporal pad (not replicate) | ✅ | mojo:258-267 prepends zero frames; Rust PadMode::Zero (qwenimage_decoder.rs:39, wan21:128-133) |
| image-mode T=1 skips temporal doubling | ✅ | mojo:_resample:384-400 spatial-only; Rust image_mode branch wan21:474-475 |
| per-channel unnormalize `z/inv_std + mean` | ✅ | mojo:414; Rust:754 |
| MEAN/STD constants | ✅ | all 16 values match wan21:42-50 exactly |
| clamp [-1,1] | ✅ | mojo:_clamp_unit:558; Rust:778 |
| channel flow 384→192→96 + remap | ✅ | mojo decode:433-450 == Rust block_spec wan21:632-655; remap == qwenimage_decoder.rs:89-92 |
| ResBlock order (norm0/silu/conv2/norm3/silu/conv6 + shortcut) | ✅ | mojo:311-332 == Rust:282-294 + load:234-257 |
| mid AttnBlock single-head Dh=C=384 | ✅ | mojo:338-363 `sdpa[1,SEQ,1,384]`; Rust:334-379 `[B*T,1,N,C]` |

---

## Encoder (qwen25vl_encoder.mojo) — self-flagged items, mostly CORRECT

| Claim | Verdict | Evidence |
|---|---|---|
| 28 layers + final RMSNorm, EXTRACT_LAYER=27 | ✅ | smoke:74,147-150 extract 27 then final_norm; Rust encode_with_intermediates:469-476 |
| Q/K/V biases present, o_proj bias-free | ✅ | mojo:569-571 (+bias), :614 o_proj None; Rust:361-362 etc |
| GQA n_rep=7 (28/4) | ✅ | repeat_kv `kvh=head//n_rep` (mojo:191) == Rust stack-then-reshape head ordering (rust:276-278): head h ← kv h//7 |
| RoPE half-split, θ=1e6 (NOT interleaved) | ✅ | see item 2 above; freq `exp(-lnθ·2i/Dh)` mojo:381-384 == Rust:228-230 |
| NO per-head q_norm/k_norm | ✅ | mojo:593 comment + no call; Rust has none |

---

## BLOCKERS

**0 (zero).** Every self-flagged correctness item checks out against the
oracle, and the smoke compiles + links clean.

---

## FRAGILE (parity-time risk; not wrong, but watch in the cos-comparison)

1. **Encoder pad-token RoPE positions differ.**
   `qwen25vl_encoder.mojo:378` `_build_rope_tables` assigns position `t` to
   ALL tokens including padding. Rust `build_rope_cache`
   (qwen25vl_encoder.rs:219-221) assigns position `1.0` to padding tokens
   (`if i < real_len { i } else { 1.0 }`).
   Impact: only the pad tokens' own q/k get a different angle. The causal mask
   blocks `j >= real_len`, so pad KEYS don't affect real-token outputs; pad
   token outputs are discarded downstream. So real-token last_hidden_state is
   unaffected **as long as** the parity driver compares only the real-token
   rows. If a parity ref dumps the full padded `[1,N_TXT,H]` and compares
   pad rows too, those rows WILL mismatch.
   Minimal fix (for exactness): in `_build_rope_tables` accept `real_len` and
   emit `pos = (t if t < real_len else 1.0)`. file:line
   qwen25vl_encoder.mojo:371-391 (+ thread real_len from encode_layer_states,
   which already computes it at :646-650).

2. **Text token-dropping (`drop_idx`) not applied.** smoke header lines 22-24
   self-flag this. diffusers `pipeline_qwenimage` drops the chat-template
   prefix from the hidden states before the DiT (`split_hidden_states` /
   `drop_idx`). The smoke feeds the full padded `[1,256,3584]` as
   `encoder_hidden_states`. Real-output parity vs diffusers will differ until
   the template-prefix tokens are sliced off (and N_TXT updated to the kept
   length). Wiring/memory proof only as-is. Parity-time fix, not a compile bug.

3. **Timestep freq precision: f32 vs f64.** Rust precomputes the sinusoidal
   freq table in f64 then casts to f32 (qwenimage_dit.rs:264-266; encoder
   uses device arange/exp in f32). Mojo computes `exp(-ln(mp)·i/half)` directly
   in f32 (embeddings.mojo:72; encoder mojo:381-384). Sub-ULP difference in the
   angle; negligible but will show as a tiny per-element delta in a strict
   bit-exact compare. Acceptable.

4. **VAE RMS eps position.** Rust adds `1e-12` to the L2 norm AFTER sqrt
   (`norm = sqrt(sum_sq) + 1e-12`, wan21:174). Mojo puts eps INSIDE the sqrt as
   a mean form (`1/sqrt(mean+1e-12)`, norm.mojo:103 via _VAE_EPS). Algebraically
   equal only at eps=0; the 1e-12 placement differs but is negligible vs
   signal ~O(1). Self-flagged at decoder:24-31. Acceptable.

5. **DiT per-block full-dense mask allocation.** `_zeros_mask[S]`
   (qwenimage_dit.mojo:474-486) builds a host `List[Float32]` of
   `heads*S*S = 24*1280*1280 ≈ 39.3M` floats (~157 MB host) and uploads a
   `[1,24,S,S]` zeros mask EVERY block (×60). Plus the math-mode sdpa allocates
   `BH*S*S` F32 scores per call. Correct, but a large memory/time cost and a
   likely OOM/slowness risk at 512² (S=1280) and worse at 1024² (S≈4352). The
   mask is constant (all zeros) — build it ONCE in `forward` and pass it in, or
   give the foundation sdpa a "no-mask" overload. Perf/robustness, not parity.
   file:line qwenimage_dit.mojo:429, 474-486.

---

## STYLE

- Unused vars: qwenimage_dit.mojo:171-173 (`p0/p1/p2` shadowed immediately),
  :511 (`d`), qwenimage_decoder.mojo:254 (`di`). Compiler warns; assign to `_`.
- `_clone` of every bias on every linear call (qwenimage_dit.mojo:257-261,
  encoder _clone) — an extra D2D copy + synchronize per projection. Correct
  (Tensor is move-only so Optional needs an owned value) but a per-call
  allocation; not load-bearing.
- VAE `_ATTN_DH=384` is a module constant matching the single mid-block attn
  call site (dim=384). Correct today; fragile if the decoder ever gains a
  second attention block at a different width. sdpa validates at runtime, so
  it would raise rather than silently corrupt.

---

## VERDICT

**BLOCKERS: 0 / clean.**

All four self-flagged "most likely silent bug" candidates (concat order,
interleaved-vs-halfsplit RoPE in DiT, halfsplit RoPE in encoder, VAE
channel-axis reduction) are implemented correctly against the Rust oracle.
The two deepest-scrutinized items — VAE channel-axis reduction and
encoder-vs-DiT RoPE layout — are both correct. Compile + link verified EXIT=0.

Recommended before parity run: fix FRAGILE #1 (pad-token RoPE) if the parity
harness compares full padded sequences, and plan FRAGILE #2 (drop_idx) +
FRAGILE #5 (mask alloc) before any real end-to-end run.
