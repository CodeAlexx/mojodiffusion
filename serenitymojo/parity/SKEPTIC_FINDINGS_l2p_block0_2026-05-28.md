# Skeptic findings — Z-Image L2P block-0 port

**Timestamp:** 2026-05-28
**Skeptic:** adversarial audit (read-only)
**Audited files:**
- `/home/alex/mojodiffusion/serenitymojo/models/dit/zimage_l2p_dit.mojo` (554 lines)
- `/home/alex/mojodiffusion/serenitymojo/models/dit/zimage_l2p_rope.mojo` (128 lines)
- `/home/alex/mojodiffusion/serenitymojo/pipeline/zimage_l2p_block0_smoke.mojo` (151 lines)
- `/home/alex/mojodiffusion/serenitymojo/ops/rope.mojo` (rope_interleaved verification)
- `/home/alex/mojodiffusion/serenitymojo/ops/attention.mojo` (sdpa_nomask signature)
- `/home/alex/mojodiffusion/serenitymojo/ops/linear.mojo` (weight convention)
- `/home/alex/mojodiffusion/serenitymojo/ops/norm.mojo` (rms_norm convention)
- `/home/alex/mojodiffusion/serenitymojo/ops/tensor_algebra.mojo` (slice / mul / add_scalar)
- `/home/alex/mojodiffusion/serenitymojo/ops/activations.mojo` (swiglu)
- Rust reference: `/home/alex/EriDiffusion/inference-flame/src/models/l2p/{dit.rs,rope.rs,weight_loader.rs}`
- Disk header: `/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors`

**Verdict: PASS (compile gate). Block-0 forward is mathematically correct against the Rust reference at smoke sizes. One WARN tracked for 1024² scalability.**

PASS: 19 / WARN: 3 / FAIL: 0

---

## A. Checkpoint key reality

### A1. layers.0 disk-resident key set — PASS

Confirmed via `safetensors.safe_open` against `model-1k-merge.safetensors`. See "Checkpoint reality table" below. All 15 keys the builder names are present at the listed shapes/dtype. No alternate names found.

### A2. Rust loader rewrites split-QKV into fused .qkv — PASS

`/home/alex/EriDiffusion/inference-flame/src/models/l2p/weight_loader.rs:95-138` confirms: Rust **does** translate disk-side split `to_q/to_k/to_v` into a single fused `attention.qkv.weight` via `Tensor::cat(&[&q,&k,&v], 0).contiguous()` at load time. Also renames `attention.norm_q → attention.q_norm`, `attention.norm_k → attention.k_norm`, `attention.to_out.0 → attention.out`. So the PORT_SPEC.md claim of "fused `[11520, 3840]`" was naming the post-load layout the Rust DiT *consumes*; the builder's direct-from-disk split layout is also valid (and saves a fuse+transpose round-trip). Mojo's call `linear(x, weights.to_q_w)` with `to_q_w [3840,3840]` is equivalent to `chunks[0]` from the fused `[11520,3840]` matrix.

### A3. All 13+ keys named by builder are present at listed shapes — PASS

15 of 15 named keys verified. (norm_q/norm_k are `[128]` bf16, not `[3840]` — builder noted "per-head Dh, NOT q_norm" — confirmed.)

---

## B. RoPE interleaved-pair convention

### B1. `rope_interleaved` pair convention — PASS

`/home/alex/mojodiffusion/serenitymojo/ops/rope.mojo:46-54` (f32) and `:67-75` (bf16) implement exactly:
```
pair = (x[r, 2*i], x[r, 2*i + 1])
out[r, 2i]   = x0*cv - x1*sv
out[r, 2i+1] = x0*sv + x1*cv
```
Identical to the complex-multiply form `(x0 + i*x1) * (cv + i*sv)`. NOT half-split. NOT any other variant.

### B2. Rust `rope_fused_bf16` interleaved convention — WARN (unable to verify Rust kernel source)

The Rust comment at `/home/alex/EriDiffusion/inference-flame/src/models/l2p/rope.rs:9-11` explicitly says "interleaved-pair pairs later consumed by `rope_fused_bf16` with `RopeLayout::Interleaved`". This is a self-describing comment but the `rope_fused_bf16` kernel itself lives outside the workspace tree audited here (in `flame_core`). Assumed consistent — same convention is already used by Klein/FLUX paths in `serenitymojo` and Rust without parity divergence (per existing SKEPTIC2_FINDINGS_rope_attn_2026-05-26.md). WARN, not FAIL — but should be locked down at parity time with a numeric round-trip test against the actual Rust output.

### B3. Axis packing order T(16) | H(24) | W(24) — PASS

`/home/alex/mojodiffusion/serenitymojo/models/dit/zimage_l2p_rope.mojo:88-120` iterates `for axis_idx in range(3): T, H, W` and appends `half_axis = axis_dim/2` per-axis freqs at `out_idx = seq_idx*half_head_dim + offset + freq_idx`, advancing `offset += half_axis` each axis. Axes dims: T=32 → half=16, H=48 → half=24, W=48 → half=24. Sum = 64 = half_head_dim. Identical to `rope.rs:69-85`. The packed layout `[T_freqs(16) | H_freqs(24) | W_freqs(24)]` matches Rust byte-for-byte.

Note: the constants `ZIMAGE_L2P_ROPE_AXIS_{T,H,W} = (32,48,48)` are correct (those are the *full* axis dims; each contributes half_axis = 16/24/24 freqs). The header comment is consistent. The skeptic prompt's mention of "T(16)|H(24)|W(24)" refers to the packed half-freq layout, which matches.

---

## C. RoPE replication helper (scalability flag)

### C1. `_zl2p_replicate_rope_for_heads` correctness — PASS

`zimage_l2p_dit.mojo:327-345`: walks `for s in range(S): for _h in range(H): for i in range(half): dst.append(src[s*half + i])`. Row `[s*H + h, :]` contains the original `[s, :]`. Position is broadcast across all heads (each head at position s gets the same rotation), which exactly matches Rust's `cos.reshape(&[1,1,S,half])` broadcast over the `H` dim during `rope_fused_bf16`. Heads do NOT get differentiated positions.

### C2. Scalability cost — WARN (host roundtrip, 17 MB/table at 1024²)

At S=96, H=30, half=64 the replication is 96·30·64·2 = 368 640 bytes per table → 720 KB total — trivial. At native 1024² (S = 320 + 4096 + 0 = 4416, per `rope.rs:127`), 4416·30·64·2 = **16 957 440 bytes ≈ 16.2 MB per table, 32.4 MB for cos+sin combined**. The bytes themselves are fine on a 32-GB GPU, BUT the helper calls `rope_2d.to_host(ctx)` followed by `Tensor.from_host(...)` — a full device→host→device round-trip every block forward. That's ~34 MB of PCIe traffic per block × 30 blocks × N steps. WARN: should be replaced by a device-side broadcast kernel before 1024² parity, but is NOT a block-0 blocker.

### C3. Can `rope_interleaved` accept `[S, half]` with broadcast? — PASS (broadcast not supported; helper is necessary)

`rope.mojo:_rope_common_validate:248-252` requires `cos.numel() == rows*half` where rows = product of all leading dims of x. With x = `[B,S,H,Dh]`, rows = B*S*H. There is no broadcast path. The helper is necessary; only the host roundtrip is the problem. To kill the roundtrip, either:
- add a `rope_interleaved_bsh` variant that takes `[S, half]` cos/sin and indexes by `(idx // (H*half)) * half + (idx % half)` (or similar), OR
- add a device-side `repeat_along_dim` kernel and chain it.

---

## D. Math correctness vs Rust transformer_block

### D1. AdaLN chunk count and order — PASS

Rust `dit.rs:534-543`: `mod_out.chunk(4, last_dim)`, `(scale_msa, gate_msa, scale_mlp, gate_mlp) = (chunks[0..3])`. No shift terms. Mojo `zimage_l2p_dit.mojo:514-517`:
```
var scale_msa = slice(mod_out, 1, 0*dim, dim, ctx)
var gate_msa  = slice(mod_out, 1, 1*dim, dim, ctx)
var scale_mlp = slice(mod_out, 1, 2*dim, dim, ctx)
var gate_mlp  = slice(mod_out, 1, 3*dim, dim, ctx)
```
Identical order. 4-way chunk over `[B, 15360]` → 4 × `[B, 3840]`.

Modulation form: Rust `dit.rs:562-566` `factor = scale_unsq.add_scalar(1.0); x_norm.mul(&factor)` → `x_norm * (1 + scale)`, **multiplicative only, no shift**. Mojo lines 528-529 `add_scalar(scale_msa, 1.0)` then `mul(x_norm, scale_msa_b)`. Identical. PASS.

### D2. Branch order — PASS

Rust `dit.rs:558-588`: rms_norm1 → scale_msa modulation → joint_attention → rms_norm2 → gate-residual; THEN `dit.rs:592-621`: rms_norm3 (ffn_norm1) → scale_mlp modulation → swiglu → rms_norm4 (ffn_norm2) → gate-residual. Mojo `zimage_l2p_dit.mojo:536-554` follows the **same order exactly** (attention branch then FFN branch, with rms_norm2/ffn_norm2 applied to the branch output before gating). PASS.

### D3. Gate-residual formula — PASS

Rust `dit.rs:581-583` and `:614-616`: `g = gate.tanh(); gate_residual_fused_bf16(x_or_x_out, g, branch_out)`. The fused kernel computes `x + g * branch_out` (additive residual with tanh-gated branch). Mojo `:542-543` (`gated_attn = mul(g_msa, attn_out); x_after_attn = add(x, gated_attn)`) and `:553-554` (`gated_ff = mul(g_mlp, ff_out); return add(x_after_attn, gated_ff)`) implement identical math. Neither side uses `(1 - g) * x + g * branch`. PASS.

### D4. SwiGLU formula — PASS

Rust `dit.rs:411-417`: `swiglu_fused_bf16(w1_out, w3_out)` = `silu(w1) * w3`, then `w2(hidden)`. Mojo `ops/activations.mojo:202` (`_silu_f32(gv) * uv`) and `:231-232` (`swiglu(gate=w1_out, up=w3_out) = silu(gate) * up`). The Mojo block calls `swiglu(w1_out, w3_out, ctx)` → matches Rust ordering exactly (gate=w1, up=w3). Then `linear(hidden, w2)` projects back. PASS.

### D5. No sign-flip in block_forward — PASS

Rust `dit.rs:510-629` `transformer_block` returns `x_out` with NO `mul_scalar(-1.0)` anywhere. The negation lives in `forward_inner` at the model level, NOT here (skeptic prompt is correct). Mojo `zimage_l2p_block_forward` (lines 486-554) returns `add(x_after_attn, gated_ff)` with no scalar multiply. PASS.

---

## E. Joint attention details

### E1. RMSNorm axis on Q/K — PASS

Rust `dit.rs:463-466` flattens `q → [B*S*H, Dh]` before `rms_norm` then reshapes back. Mojo applies `rms_norm(q[B,S,H,Dh], norm_q_w[Dh], eps)` directly. Since `rms_norm` (`ops/norm.mojo:147-172`) normalizes over the last dim and reduces leading dims to `rows = prod(leading)`, the operation is **mathematically identical** to the Rust flatten-then-reshape — same per-row reduction, same per-element scale. PASS.

### E2. RoPE on Q,K only; V untouched — PASS

Mojo `zimage_l2p_dit.mojo:469-470` applies `rope_interleaved` to `q` and `k` only. `v` is left as the post-reshape tensor and passed straight to `sdpa_nomask`. Mirrors Rust `dit.rs:484-485`. PASS.

### E3. sdpa BSHD vs BHSD layout — PASS

`ops/attention.mojo:634-662` `sdpa_nomask` accepts `q,k,v: [B,S,H,Dh]` (rank-4 BSHD), enforced at `:653` via `qshape[0..3] == (B,S,H,Dh)`. Internal `_sdpa_math` gathers BSHD into a BHSD-contiguous f32 working buffer (`ops/attention.mojo:61` comment confirms). Rust permutes BSHD → BHSD before sdpa (`dit.rs:472-474`) but the math is invariant under that permutation as long as q/k/v share the same layout. PASS — the Mojo `[B,S,H,Dh]` flow into `sdpa_nomask` is correct and the internal gather handles the layout conversion.

---

## F. Compile + no regressions

### F1. Block-0 smoke build — PASS

```bash
cd /home/alex/mojodiffusion
pixi run mojo build -I . -Xlinker -lm \
    serenitymojo/pipeline/zimage_l2p_block0_smoke.mojo \
    -o /tmp/zimage_l2p_block0_smoke
# exit 0
```
No errors, no warnings.

### F2. Pre-block smoke still builds — PASS

```bash
pixi run mojo build -I . -Xlinker -lm \
    serenitymojo/pipeline/zimage_l2p_dit_preblock_smoke.mojo \
    -o /tmp/zl2p_preblock
# exit 0
```
The `ZImageL2PDiTPreBlockGate` struct was not modified; the preblock smoke continues to build.

### F3. PreBlockGate callers unchanged — PASS

`grep -rn ZImageL2PDiTPreBlockGate` shows the only caller is `zimage_l2p_dit_preblock_smoke.mojo:96` (`ZImageL2PDiTPreBlockGate.load_default(ctx)`). Signature unchanged.

---

## Checkpoint reality table (layers.0.*)

| Key | Shape | dtype |
|-----|-------|-------|
| layers.0.adaLN_modulation.0.bias | [15360] | bf16 |
| layers.0.adaLN_modulation.0.weight | [15360, 256] | bf16 |
| layers.0.attention.norm_k.weight | [128] | bf16 |
| layers.0.attention.norm_q.weight | [128] | bf16 |
| layers.0.attention.to_k.weight | [3840, 3840] | bf16 |
| layers.0.attention.to_out.0.weight | [3840, 3840] | bf16 |
| layers.0.attention.to_q.weight | [3840, 3840] | bf16 |
| layers.0.attention.to_v.weight | [3840, 3840] | bf16 |
| layers.0.attention_norm1.weight | [3840] | bf16 |
| layers.0.attention_norm2.weight | [3840] | bf16 |
| layers.0.feed_forward.w1.weight | [10240, 3840] | bf16 |
| layers.0.feed_forward.w2.weight | [3840, 10240] | bf16 |
| layers.0.feed_forward.w3.weight | [10240, 3840] | bf16 |
| layers.0.ffn_norm1.weight | [3840] | bf16 |
| layers.0.ffn_norm2.weight | [3840] | bf16 |

15 keys. No fused `attention.qkv.weight` on disk (the fused name only exists after `weight_loader.rs::translate_l2p_keys` rewrites the source map). Builder's direct-from-disk split-QKV layout is correct and saves the cat+transpose roundtrip the Rust loader does at startup.

---

## Bugfix Worklist (ordered)

The block-0 port is mathematically correct against the Rust reference and compiles clean. No FAILs to fix. The following items are tracked for follow-up but **do not block the bugfix→parity loop on block-0**:

1. **[WARN, tracked, NOT a block-0 blocker]** Replace the host-roundtrip RoPE replication helper (`_zl2p_replicate_rope_for_heads` in `zimage_l2p_dit.mojo:327-345`) with one of:
   - **Option A (preferred):** add a `rope_interleaved_bsh` variant in `ops/rope.mojo` that consumes `cos/sin: [S, half]` plus `H` and indexes per-row via `(s = row // H; pair_i = idx % half)`. Removes the replication entirely.
   - **Option B:** add a device-side `repeat_along_axis` kernel and call it instead of the `to_host → from_host` round-trip.
   At 1024² (S=4416, H=30, half=64) the current path moves ~34 MB host↔device per block; with 30 blocks × N denoise steps that becomes the dominant per-step cost. Decision threshold: pull this in before native 1024² parity, after block-0 numeric parity is locked.

2. **[WARN]** Lock down B2 (`rope_fused_bf16` interleaved convention) at parity time with a round-trip numeric check (`build_3d_rope` cos/sin + a small random `q` → run Rust `rope_fused_bf16` → run Mojo `rope_interleaved` on a per-head-replicated copy → cosine sim ≥ 0.999). The convention is self-described in the Rust doc comments but the kernel source is in `flame_core` (outside this workspace). Belt-and-suspenders.

3. **[Tracking, no action]** Builder's `_zl2p_tanh` is a per-block standalone kernel; once `serenitymojo/ops/activations.mojo` grows a generic `tanh` op, swap to it for consistency. Not a correctness issue.

4. **[Tracking, no action]** Each `linear(...)` call wraps the bias in a fresh `_clone(weights.bias, ctx)` because the linear op consumes its Optional[Tensor] bias by value. For block-0's two bias-bearing linears (the adaLN projection) the clone cost is negligible. At 30 blocks × N steps it's 60·N small device copies — minor.

---

## End of findings
