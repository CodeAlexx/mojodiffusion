# SD3.5-Large device-compute port — execution map (MJ-1069, unfrozen 2026-07-06)

> **STATUS 2026-07-06 EOD: PORT COMPLETE — MJ-1069 FIXED.** All rungs landed:
> block gates 33/33 max_abs=0.0 (standard + pre_only), device conductors with
> recompute-in-backward (backward measured FLAT across 38 blocks), PER-HEAD
> chunked math SDPA (bit-identical; H/2 was not enough — measured), device-
> direct weight builders (the 780ms/block host round trip WAS the 54s step;
> now 14.4 s/step BYTE-IDENTICAL), fp8-resident arm fits (22.65GiB). FIRST
> training steps in the model's existence: loss 7.217→7.094 DECREASING.
> Open follow-ups: MJ-1085 fused-AdamW segv (unfused arm active), flash Dh=64
> SDPA, reduced-depth host-vs-device e2e parity, webui preset.

Basis: MJ-1069 — Large has NEVER completed a step; the stack is host-F32-list
based (`_block_host_f32` casts every block weight to F32 HOST per visit) and
the block runs EVERY op as a host round trip (upload→op→download), with
modulate/gated-residual/qkv-split/joint-concat as HOST SCALAR LOOPS
(sd35_block.mojo:190-262, 846-868, 1000-1018). ~55GB host appetite measured.
Unfreeze (Alex 2026-07-06) explicitly commissions the chroma/flux-class port.

## Verified block math (READ 2026-07-06; sd35_block.mojo)

Joint MMDiT block, ALL-F32 (no bf16 anywhere, NO rope):
- `_stream_pre` (819): ln1 (no-affine, eps) -> modulate(scale_msa, shift_msa)
  -> FUSED qkv linear+bias [N,3D] -> LoRA delta ADDED IN F32 (plain adds)
  -> split q/k/v -> qk RMS with SEPARATE qk_eps.
- joint attention: concat CTX FIRST then IMG (host loop; device = concat op),
  `sdpa_nomask[B,S,H,Dh]` F32, split back.
- `_stream_post` (882): proj+LoRA -> gated_residual(gate_msa) -> ln2 ->
  modulate(scale_mlp, shift_mlp) -> fc1+LoRA -> GELU -> fc2+LoRA ->
  gated_residual(gate_mlp).
- LoRA: 8 Optional[LoraAdapter] args per block (ctx/x × qkv,proj,fc1,fc2);
  `x_qkv_lora_delta`/`ctx_qkv_lora_delta` alternate injection path exists —
  device port supports the adapter args; delta-arg path can fail loud.
- ModVecs: shift/scale/gate_msa + shift/scale/gate_mlp per stream (F32 host
  lists -> QwenModVecsDevice-style struct, F32 tensors, uploaded per block).

## Device-port numerics contract

F32 device chain end-to-end == bit-comparable to the host chain: same shared
GPU kernels for linear/ln/rms/sdpa/gelu; host scalar loops (modulate, gated
residual, adds) are elementwise F32 — the device ops compute the same
expression per element. bf16->F32 weight cast on device == the host's cast
(exact upcast). Gate bar: chroma/qwen-class (cos=1.0 / max_abs=0 expected).

## Deltas vs the qwen template (qwenimage_block_device.mojo, freshest)

1. F32 everywhere (qwen was bf16): no `_bf16` casts, no bf16 rounding mirrors.
2. FUSED qkv (like chroma) + biases; LoRA on `norm` at the qkv OUTPUT [N,3D]
   (ONE adapter for the fused 3D output — in_f=D, out_f=3D).
3. NO rope (drop rope/rope_backward entirely).
4. qk RMS uses qk_eps (separate from ln eps).
5. 8 slots/block: ctx_qkv, ctx_proj, ctx_fc1, ctx_fc2, x_qkv, x_proj, x_fc1,
   x_fc2 (define SD35_D_* comptime constants).
6. pre_only LAST block (Large joint_blocks.37 = qkv-only ctx stream, header-
   verified MJ-1065 wave): stack passes `last_ctx_preonly`; port needs a
   pre_only fwd/bwd variant (ctx stream: _stream_pre only; no ctx post; no
   ctx output). READ the stack's pre_only arm before writing
   (sd35_stack_lora.mojo — the fixed fwd+bwd call sites from MJ-1069).
7. RECOMPUTE-IN-BACKWARD IS MANDATORY (flux 238c089 pattern): full per-block
   F32 tapes at Large scale ≈ 15GB+ (38 blocks × ~410MB at 1024², S≈4250,
   D=2432?) — save ONLY block inputs (ctx_in/x_in TArc per block ≈ 1.6GB),
   rebuild tapes per block in the reverse loop. (Do NOT copy qwen's
   all-tapes-saved stack design here.)

## Remaining reads for the next stint (exact anchors)

- `_sd35_lora_fwd` (123) + `_sd35_lora_bwd` (157): LoRA GEMM dtypes/order.
- backward: `_stream_post_backward` (1255), `_stream_pre_backward` (1438),
  `sd35_joint_block_backward` (1561) — mirror exactly (F32; frozen-base
  dx-only in the device port; host computes d_w — drop).
- stack: sd35_stack_lora.mojo conductors + `last_ctx_preonly` arm + LoRA
  slot/grad-set layout + loader/weights source (`_block_host_f32` — device
  port casts bf16->F32 ON DEVICE instead).
- trainer: serenity-trainer train_sd35_real.mojo seam (argv, cadence, save).
- dims: Large D/H/Dh/MLP/S from configs (Dh=64 would enable the SDK flash
  path later; correctness first with sdpa_nomask F32 = the oracle).

## Gates (qwen-port bars)

1. Block gate at real D (small N): fwd ctx/x out, d_x both streams, all 16
   LoRA d_A/d_B — max_abs=0.0 expected (F32 chain). Include a pre_only case.
2. E2E: first-ever sd35-Large steps — loss finite + LoRA-B growth (no prior
   baseline exists; the HOST arm cannot run at scale, so e2e parity is
   host-vs-device at REDUCED depth (e.g. 4 blocks) + full-depth device run).
3. VRAM/RAM watch: 16GB bf16 weights streamed + F32 acts; recompute keeps the
   tape ~1.6GB. RAM guard per MJ-1066.

## LoRA dtype nuance (READ: _sd35_lora_fwd 123 / _sd35_lora_bwd 157)

Inside the F32 chain, the LoRA fwd/bwd are BF16 GEMMs (klein_lora pattern):
- fwd: F32 x -> BF16 upload (ROUNDS) -> bf16 t -> host F32 -> BF16 re-upload
  (identity) -> bf16 dy -> F32 scale -> F32 add into the F32 base.
  Device mirror: `_bf16(x_f32)` -> bf16 GEMMs -> `mul_scalar(_f32(dy), scale)`
  -> F32 add. (qwen `_lora_apply_device` with an extra _bf16 on x and NO
  final bf16 round — the base stays F32.)
- bwd: d_contrib F32 -> scale in F32 -> BF16 upload (ROUNDS) -> bf16
  linear_backward chain (d_t/d_b then d_x/d_a) on bf16 x/a/b -> F32 host
  outputs. Device mirror: qwen `_lora_bwd_device` with `_bf16(x_f32)` inputs
  and d_dy = `_bf16(mul_scalar(d_contrib_f32, scale))`; d_x_lo upcast F32 to
  fold into the F32 base dx (host folds in F32, no re-round — F32 add).
