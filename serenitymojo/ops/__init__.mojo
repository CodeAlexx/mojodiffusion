# serenitymojo.ops — the GPU op layer over the MAX SDK + hand-rolled kernels.
#
# Phase A foundation (PHASE_AB_PLAN.md). Each op takes/returns the shared
# `Tensor` type (serenitymojo/tensor.mojo) and runs its compute on the GPU.
# F32 accumulation where it matters (matmul, rms_norm) even for BF16 storage.
#
# Modules (chunk A1 — this chunk):
#   linear  — y = x @ wᵀ + b  via `linalg.matmul.vendor.blas.matmul`
#             (transpose_b=True, c_row_major=True). Weight is PyTorch [out,in].
#   norm    — rms_norm(x, weight, eps). Composed from a hand-rolled GPU kernel:
#             the SDK `nn.normalization.rms_norm_gpu` has a `gamma: TileTensor`
#             parameter whose `gamma.origin.mut` cannot be inferred from a plain
#             `LayoutTensor` in Mojo 1.0.0b1 (see norm.mojo header). The kernel
#             does F32 reduction + cast-back, numerically matching torch.
#
# NOT in this chunk (chunk A2): group_norm / layer_norm / rope / sdpa / conv2d /
# silu / swiglu / modulate / residual_gate / random.
#
# LTX2 P0 (LTX2_PORT_PLAN_2026-05-28 §P0):
#   activations — silu, sigmoid, gelu, swiglu  (sigmoid exported here per P0).
#   unary       — sin_op, exp_op, sqrt_op, rsqrt_op, tanh_op, reciprocal_op
#                 (elementwise unary math; f32/bf16/f16, bf16/f16 upcast to f32).
#                 Import directly, e.g. `from serenitymojo.ops.unary import sin_op`.
#
# LTX2 P-reduce (LTX2_PORT_PLAN_2026-05-28 §P-reduce):
#   reduce      — reduce_sum / reduce_mean / reduce_var / reduce_std over an
#                 arbitrary set of dims (keepdim), F32-accumulated. sum/mean
#                 /var/std preserve storage dtype; *_f32 variants emit F32.
#                 Unblocks AdaIN per-(B,C)-over-(F,H,W) + general PixelNorm.
#                 Import directly, e.g. `from serenitymojo.ops.reduce import reduce_mean`.

# Cross-model diffusion parity helpers. These preserve PyTorch eager F32
# operation boundaries before final BF16 stores, avoiding fused-kernel BF16 tie
# drift in scheduler/noiser math. Public API:
#   from serenitymojo.ops import torch_bf16_eager_add_scaled
from serenitymojo.ops.torch_bf16 import (
    torch_bf16_eager_add_scaled,
    torch_bf16_eager_blend_with_f32_mask,
    torch_bf16_eager_velocity_from_x0,
    torch_f32_to_bf16_rne,
)
