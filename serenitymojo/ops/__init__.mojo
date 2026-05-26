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
# silu / swiglu / modulate / residual_gate.
