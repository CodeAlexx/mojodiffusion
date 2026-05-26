# Foundation-op blocker: `ops/attention.sdpa` fails to compile at head_dim=128

**Found by:** Team-Text-Encoder (qwen3_encoder chunk), 2026-05-25.
**Severity:** blocks the Qwen3 text encoder end-to-end (Qwen3 head_dim = 128).
**Owner:** foundation/ops team (do NOT fix in models/ — sdpa is a foundation op).

## Symptom
`serenitymojo/ops/attention.sdpa[B, S, H, Dh]` (the SDK
`nn.attention.gpu.mha.flash_attention` LayoutTensor variant) fails to
**compile** (instantiate) whenever `Dh == 128`, for ANY head count and ANY
input dtype (BF16 and F32 both tried). The compiler error:

```
oss/.../std/gpu/compute/mma.mojo:87:5: note: constraint failed:
  no valid implementation of mma for a=8xfloat16, b=4xfloat16,
  c=4xfloat32, and d=4xfloat32
max/kernels/src/nn/attention/gpu/mha.mojo:3173:5: ...
  ("BM": 16, "BN": 128, "BK": 32, "WM": 16, "WN": 32, "depth": 128,
   "num_heads": 32, ... "group": 1, ...)
```

i.e. the flash-attention kernel's MMA tiling for `depth=128` selects an
`m16n8k8` (or similar) f16 tensor-core op that has no implementation on this
GPU (RTX 3090 Ti, sm_86).

## Minimal isolation (verified)
| B | S | H | Dh | result |
|---|---|---|----|--------|
| 1 | 4 | 2 | 8   | OK (foundation ops_smoke2 case) |
| 1 | 8 | 32 | 64  | **OK** |
| 1 | 8 | 32 | 128 | **FAIL** (mma constraint) |
| 1 | 8 | 8  | 128 | **FAIL** |
| 1 | 8 | 2  | 128 | **FAIL** |

→ The failure is purely **`Dh == 128`**, independent of head count or seq.
`Dh == 64` compiles for all head counts. `attention.mojo`'s header claims
"head_dim 64/128", but Dh=128 was never actually instantiated (the foundation
smoke only ran Dh=8).

## What IS proven correct
The qwen3 encoder's entire non-attention path is parity-clean vs the numpy
oracle at the real Z-Image text_encoder weights (see qwen3_preattn_probe.mojo):
```
embed          cos=1.0000000
l0_input_norm  cos=0.9999986   (rms_norm)
l0_q_rope      cos=0.9999956   (linear q_proj -> per-head qk-norm -> rope_halfsplit)
l0_k_rope      cos=0.9999979
```
So embedding-gather, rms_norm, linear, per-head qk-norm, and rope_halfsplit
are all wired correctly. ONLY sdpa@Dh=128 blocks the full forward.

## Suggested central fix (foundation team, in ops/attention.mojo)
Options, in rough order of preference:
1. Pass a flash_attention config that uses a Dh=128-valid MMA shape for sm_86
   (e.g. force `BK`/`WN` that map to a supported f16 tensor-core op, or the
   `m16n8k16` path), if the SDK exposes config knobs on this overload.
2. Fall back to a non-tensor-core / naive softmax-attention SDK entry for
   head_dim 128 on pre-sm_90 GPUs.
3. Hand-roll a tiled SDPA kernel for Dh=128 in ops/attention.mojo (last resort;
   keeps it a foundation op, not duplicated per model).

Once sdpa@Dh=128 compiles, `qwen3_parity.mojo` should pass per-layer +
final last_hidden_state with no encoder changes (the dispatch + forward are
already written and compile up to the sdpa call).
```
pixi run mojo run -I . \
  serenitymojo/models/text_encoder/parity/qwen3_parity.mojo
```
