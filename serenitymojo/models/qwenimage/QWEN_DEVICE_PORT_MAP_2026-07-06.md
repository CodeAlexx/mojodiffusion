# QwenImage device-compute port — execution map (MJ-1084)

Measured basis (2026-07-06): qwen fp8_e4m3_host = 173-176 s/step; dmon 535s
window: 0s at sm>=50, 295s idle (<5%), fine-grained (longest idle run 9s) —
host-churn-bound across the whole step, NOT optimizer-block-bound. 95
to_host/from_host sites in qwenimage_block.mojo (mod vecs uploaded from host
lists PER BLOCK CALL, block outs to_host per block, host activation tape).

## Template: chroma_block_device.mojo (verified line-level fit)

qwen block math == chroma/flux double-block class (READ 2026-07-06):
per-stream adaLN (shift/scale/gate x2 via modulate/residual_gate), QK
rms_norm, joint attention concat TXT FIRST (qwen double_block_forward:499),
rope_interleaved on joint, sdpa_nomask[1,S,H,Dh], slice back txt/img,
per-stream out-proj + GELU MLP (NOT gated GLU — klein template excluded,
chroma/flux included). LoRA slots per stream: to_q,to_k,to_v,proj(out),
mlp0(up),mlp2(dn) — EXACT chroma DoubleBlockLoraDevice/StreamLoraDevice match.

## Deltas from chroma template (the ONLY changes)

1. QKV: chroma fused `wqkv`[3D,D]+`bqkv`+slice -> qwen THREE separate biased
   linears: wq/bq, wk/bk, wv/bv (qwenimage_block.mojo _stream_pre:401-404).
   LoRA q/k/v applies on `norm` exactly as chroma does post-slice.
2. Field renames: wproj/bproj->wout/bout, wmlp0/bmlp0->wup/bup,
   wmlp2/bmlp2->wdn/bdn (StreamWeights at qwenimage_block.mojo:156).
3. ModVecs: qwen's are HOST List[Float32] (shift1/scale1/gate1/shift2/scale2/
   gate2) — add qwen ModVecsDevice + modvecs_to_device converter (chroma
   pattern verbatim, chroma_block_device.mojo:202-226), built ONCE per step.
4. No single blocks: qwen = 60 doubles only; port ONLY the double sections
   (chroma_block_device.mojo 388-688 + the shared structs/helpers 78-386).
5. dtype: mirror chroma bf16 device activations + boundary casts (_bf16/_f32).

## Files to create/edit

- NEW `serenitymojo/models/qwenimage/qwenimage_block_device.mojo` — the
  template port above (fwd + bwd + device structs + converters).
- EDIT `qwenimage_stack_lora.mojo` — device stack driver pair (fwd +
  recompute-in-backward), mirroring `chroma_stack_lora.mojo`'s device arms
  (import site chroma_stack_lora.mojo:122; drivers
  chroma_stack_*_device_offload per MJ-1068 close-out). Awaits/prefetch via
  TurboPlannedLoader unchanged; fp8_e4m3_host base arm stays the weight
  source (H2D+dequant per await — that part is NOT the disease; the
  activation round-trips are).
- OPTIMIZER inside the port: device-resident klein resident-set pattern —
  grads stay device, P/M/V persistent device, NO pinned staging (retires the
  MJ-1070 segv class). QWEN_GPU_ADAMW stays False until this lands.
- Trainer routing: device arm DEFAULT + `QWEN_HOST_STACK=1` env oracle
  (chroma/flux convention, CHROMA_HOST_STACK/FLUX_HOST_STACK precedent).

## Gates (chroma 476d947 / flux 238c089 bars)

1. Block gate: device fwd+bwd vs host block at REAL dims — cos=1.0/max_abs=0
   class (chroma achieved BIT-IDENTICAL; flux 107/107 stack slots).
2. End-to-end A/B: loss/grads/LoRA-B EVERY-DIGIT vs the byte-untouched host
   oracle over >=3 steps (same seeds; fp8h base identical both arms).
3. Speed: 2nd-consecutive-run discipline (MJ-1082); expect the chroma/flux
   class (~10-35x from 173-176 s/step; existence proof: chroma 3.6-4.0 at 19+38
   blocks, flux 5.6-6.2 at 19+38 with 504 adapters — qwen 60 doubles, 720
   adapters, D/H per configs/qwenimage.json).
4. VRAM watch: fp8h host-pinned 20GB + device acts for 60 blocks — recompute-
   in-backward keeps the tape off-host (flux autopsy: the 9.7GB HOST tape was
   the fp8-arm OOM).

## Evidence trail
- MJ-1084 (dmon probe), MJ-1070 (fp8h arm + segv autopsy), MJ-1075/1068
  (fleet doctrine: host-activation round-trips, check-existing-arms-first —
  checked: qwen has NO device block arms, only _direct_proj_{fwd,bwd}_device).

## Backward template deltas (chroma_block_device.mojo 518-688, READ)

- `_stream_post_backward_lora_device`: VERBATIM with renames (recompute mlp_y
  and proj_out WITH LoRA so gate_residual_backward y matches fwd; frozen
  mod/ln -> _dx only; F32 gate vecs `gate1_f32/gate2_f32` on ModVecsDevice;
  bf16-native chain with F32 at gate residuals).
- `_stream_pre_backward_lora_device`: chroma's fused
  `linear_backward_dx(d_qkv, wqkv)` becomes THREE separate
  `linear_backward_dx(d_q, wq) + (d_k, wk) + (d_v, wv)` SUMMED into d_norm
  (bias grads dropped — frozen base). LoRA q/k/v d_x_lo adds unchanged.
- Slot constants: define qwen D_SQ/D_SK/D_SV/D_OUT/D_UP/D_DN per stream
  (12 slots/block total) mirroring chroma's D_* layout; grads returned as
  `d_a_slots/d_b_slots: List[Optional[TArc]]` exactly as chroma.
- Double backward driver (chroma 621-688): cat_backward of joint d_att into
  per-stream d_q_rms/d_k_rms/d_v via slice + rope_backward on joint — mirror
  with qwen txt-first order (same as chroma).

NEXT STINT: write qwenimage_block_device.mojo (est. ~750 lines), block parity
gate vs host block (real dims from configs/qwenimage.json), then stack driver
+ trainer routing + e2e/speed gates per the plan above.
