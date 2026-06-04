# Ops Gap Audit — Phase 2-4 DiT op coverage (2026-06-03)

Scope: scan the inference-flame model `.rs` sources for the op primitives the
upcoming **inference-only** Mojo ports will need, and check each against what
`serenitymojo/ops/*` (+ `models/vae/conv3d.mojo`) already provides. Models in
scope (all under `/home/alex/EriDiffusion/inference-flame/src/models/`):
`wan22_dit.rs`, `wan_vace_dit.rs`, `hunyuan15_dit.rs`, `kandinsky5_dit.rs`,
`cosmos_predict25_dit.rs`, `magihuman_dit.rs`, `acestep_dit.rs`, `nava_av.rs`,
`hidream_i1/`, `asymflux2.rs`.

Reference is **read-only architecture** (Rust/flame-core). Shipped code stays
pure Mojo + MAX, GPU-only, inference-only.

---

## 0. Method

For each model I grepped op categories (rope/conv/norm/activation/attention/moe/
adaln) and read the concrete apply bodies for the unusual cases (3-axis complex
RoPE in `wan22_dit.rs:364`, partial RoPE in `magihuman_dit.rs:391` and
`nava_av.rs:464`, learnable abs-pos in `cosmos_predict25_dit.rs:577`, sorted-MoE
in `hidream_i1/moe.rs`). I then matched each against the public API in
`docs/SERENITYMOJO_MODULES.md` and the actual op sources.

**Key structural fact established by the read:** every "3D RoPE" /
"complex 3-axis RoPE" in these models reduces to ONE existing apply kernel
(`rope_interleaved` *or* `rope_halfsplit`) fed a **per-token cos/sin table of
shape `[rows, head_dim/2]`** whose row is the *concatenation of the three axes'
angle vectors* at that token's `(frame, height, width)` position. This is
exactly the layout `models/dit/zimage_dit.mojo::_build_rope` already produces
inline (host loop) and feeds to `rope_interleaved`. Cosmos's own docstring
(`cosmos_predict25_dit.rs:26-42`) and wan22 (`:364-415`) both confirm the
`cat([t,h,w])` table → standard rope-apply structure. So the *apply kernel is
present and proven*; the **table builder** is the duplicated/missing piece.

Likewise every "partial RoPE" (`magihuman`, `nava` audio) =
`slice(0..ro_dim) → rope_* → concat(rotated, passthrough)` — pure composition
of existing `slice` / `rope_*` / `concat`. No new kernel.

Conv3d in these *DiTs* is always `kernel == stride == patch_size` (a per-patch
linear "patch embed", e.g. `wan22_dit.rs:454`, `nava_av.rs:150`), NOT a sliding
conv. It is `patchify3d + linear`, not `models/vae/conv3d.mojo` (that is a true
sliding conv for VAEs and is fine for VAE use).

---

## 1. RANKED gap list

Ranking = (# models needing) × (frequency / centrality). `missing` = genuinely
absent from `serenitymojo/ops/*`; `present` = already provided (possibly needs
composition); `compose` = present-but-callers must compose existing ops.

| Rank | Op | Models needing | Freq | Status | Notes |
|---|---|---|---|---|---|
| 1 | **multi-axis (3D) RoPE table builder** → `[rows, Dh/2]` cos/sin | wan22, wan_vace, hunyuan15, kandinsky5, cosmos, magihuman, nava(video) | per-forward, 7 models | **MISSING (reusable)** | apply kernel present (`rope_interleaved`/`rope_halfsplit`); each model currently re-implements the host concat loop (see `zimage_dit::_build_rope`). PORTED THIS PASS. |
| 2 | layer_norm_no_affine (AdaLN normalize, no γ/β) | kandinsky5, cosmos, hunyuan15, wan22, magihuman, nava, hidream_final | per-block ×N | **present (added 2026-06-03 by prior pass; VERIFIED)** | `ops/norm.layer_norm_no_affine`. Probe `parity/phase24_gap_ops_probe.mojo`: f32 cos 0.99999999, bf16 cos 0.9999973 PASS. Kept. |
| 3 | gelu_exact (erf form, `approximate="none"`) | cosmos, kandinsky5(FFN GELU no-bias), asymflux2 | per-FFN | **present (added 2026-06-03 by prior pass; VERIFIED)** | `ops/activations.gelu_exact`. Same probe: f32 cos 1.0, bf16 cos 0.9999981 PASS. Tanh `gelu` already existed for tanh-approx callers. Kept. |
| 4 | partial RoPE (rotate first ro_dim, pass-through tail) | magihuman, nava(audio) | per-attn | **compose** | `slice → rope_halfsplit/rope_interleaved → concat`. No kernel. Documented here so porters don't write a new one. |
| 5 | 3D patch-embed (`patchify3d`, cube channel-major) | wan22, wan_vace, hunyuan15, cosmos, nava(video) | once/forward | **MISSING (minor)** | conv3d-as-linear with k==s==patch. Currently expressible via `permute`+`reshape`+`linear` but verbose; candidate next pass. NOT ported this pass (lower freq, composable). |
| 6 | sorted-routing MoE (shared expert + per-expert SwiGLU) | hidream_i1, (acestep none) | per-MoE block | **present** | `ops/moe.top_k_router` + `grouped_expert_ffn` + `gated_scatter_add`; shared expert = `linear`+`swiglu`. Matches `hidream_i1/moe.rs` sorted dispatch. |
| 7 | rms_norm per-head (QK norm) | acestep, nava, hunyuan15, kandinsky5, wan22, cosmos, hidream | per-attn ×2 | **present** | `ops/norm.rms_norm` over last dim == head_dim. `rms_norm_per_head` in Rust is just rms_norm on `[...,Dh]`. |
| 8 | modulate / AdaLN (1+scale)·x+shift | all 10 | per-block ×6 | **present** | `ops/elementwise.modulate`, `residual_gate`. |
| 9 | sdpa (full, GQA-expanded, any Dh) | all 10 | per-attn | **present** | `ops/attention.sdpa` / `sdpa_nomask`; Dh=128 uses math-mode (flash unsupported sm_86). |
| 10 | silu / swiglu | acestep, wan22, hunyuan15, nava, magihuman, hidream | per-FFN | **present** | `ops/activations`. |
| 11 | conv1d (k=7,p=3; k=2,s=2 patch) | nava(audio), acestep | once/forward | **present** | `ops/conv1d.conv1d` (stride/pad/dil/groups). nava ChannelLastConv1d = permute+conv1d+permute. |
| 12 | learnable per-axis additive abs-pos emb | cosmos (`extra_per_block_abs_pos_emb`) | per-block | **compose** | sliced learned table + `tensor_algebra.add` broadcast. cosmos-only; no kernel. |
| 13 | conv3d (true sliding) | (VAEs only; not these DiTs) | n/a | **present** | `models/vae/conv3d.mojo`. DiT "conv3d" is patch-embed (rank 5 above), not this. |
| 14 | timestep sinusoidal embed (cos-first / sin-first) | all | once | **present** | `ops/embeddings.timestep_embedding{,_sin_first}`. wan22 sinusoidal = cos-first half/half (`:334`). |
| 15 | adaLN-LoRA branch (Linear+SiLU+Linear on temb) | cosmos, magihuman | per-block | **present** | composition of `linear`+`silu`; not an op. |

### Truly-missing (drives this pass)
1. **multi-axis RoPE table builder** (rank 1) — highest frequency, 7 models,
   currently duplicated host-loop. → PORTED.
2. 3D patch-embed `patchify3d` (rank 5) — flagged, deferred (composable, 1×/fwd).

Everything else is present or pure composition of present ops.

---

## 2. Foundation issues / notes

- **No foundation bug found.** `rope_interleaved`/`rope_halfsplit` correctly
  consume `[rows, Dh/2]` tables; the new builder targets exactly that contract.
- inv_freq convention differs across refs and the builder must expose it:
  - zimage: `inv_freq_i = theta^(-2i/axis_dim)` (`zimage_dit.rs` / `_build_rope`).
  - wan22 / cosmos per-axis: `inv_freq_i = theta^(-i/axis_half)` (i.e. `2i/axis_dim`
    is replaced by `i/axis_half` — **same value** since `axis_half = axis_dim/2`).
  These are identical (`2i/axis_dim == i/(axis_dim/2)`). The builder uses the
  `theta^(-i/axis_half)` form per axis → matches both.
- The prior interrupted pass added `layer_norm_no_affine` (norm.mojo) and
  `gelu_exact` (activations.mojo). Both follow the kernel-triple + dispatcher
  convention, F32-accumulate, BF16-store, and compile. **Kept as-is.** The
  staged `models/ernie/ernie_stack_lora.mojo` change is unrelated user training
  work (not part of this op sweep) and was left untouched.

---

## 3. Ported this pass

- **`ops/rope_tables.mojo`** — `build_multiaxis_rope_tables(positions_flat,
  axes_dims, theta, ctx) -> (cos, sin)` producing `[rows, Dh/2]` F32 tables in
  the layout `rope_interleaved`/`rope_halfsplit` consume. positions is flat F32
  `[rows*num_axes]`, token-major. inv_freq `theta^(-i/half_a)` per axis,
  concatenated over axes. One GPU thread per (row, col); axis dims uploaded as a
  small I32 device buffer. num_axes bounded ≤ 4.
  - Probe `ops/rope_tables_probe.mojo`: 3-axis [4,4,2], theta 100, 3 tokens →
    `max_err 1.27e-7`, **exit 0 PASS** (`pixi run mojo run -I .`).
  - Oracle `parity/gen_rope_tables_reference.py`: numpy F64 + optional torch GPU
    bf16 round-trip; numbers match the probe's host recompute.

### Verification commands
```
pixi run mojo run -I . serenitymojo/ops/rope_tables_probe.mojo          # PASS exit 0
pixi run mojo run -I . serenitymojo/ops/parity/phase24_gap_ops_probe.mojo  # PASS exit 0 (ops 2,3)
pixi run python serenitymojo/ops/parity/gen_rope_tables_reference.py
```
