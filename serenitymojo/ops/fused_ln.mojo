# ops/fused_ln.mojo — two fused LayerNorm ops ported from flame-core
# (EriDiffusion/flame-core/src/fused_kernels.rs):
#
#   layernorm_linear (rs ~line 85):
#       norm = (x - mean) * rsqrt(var + eps) * gamma + beta   # affine LN, last dim
#       y    = norm @ weightᵀ + bias                          # weight [out, in] row-major
#     var = E[x²] - E[x]²  (biased, /hidden_size). eps as passed. bias optional.
#
#   residual_layernorm (rs ~line 353):
#       s = x + residual                                       # add order: x + residual
#       y = (s - mean) * rsqrt(var + eps) * gamma + beta       # affine LN, last dim
#     The Rust public fn delegates to `x.add(residual)` then a LayerNorm with
#     elementwise_affine=true, weight=gamma, bias=beta. We match that exactly.
#
# These are NOT reimplementations of LN stats or matmul — they compose the
# existing foundation ops (ops/norm.layer_norm + ops/linear.linear +
# ops/tensor_algebra.add), which already do biased-variance affine LN over the
# last dim and F32-accumulated GEMM, matching the flame-core math. The "fusion"
# is the LN prologue around the matmul / the residual-add before LN.
#
# bf16 and f32 paths both flow through the same composition (the foundation ops
# dispatch on dtype internally; F32-accumulate, store dtype).
#
# Mojo 1.0.0b1, NVIDIA GPU, inference-only.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import add


def layernorm_linear(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Fused LayerNorm (affine, last dim, biased var) then Linear.

    x:      [..., hidden]        (compute dtype; leading dims flattened to rows)
    gamma:  [hidden]             (LN scale; same dtype as x)
    beta:   [hidden]             (LN shift; same dtype as x)
    weight: [out, hidden]        (PyTorch row-major; same dtype as x)
    bias:   [out] or None        (same dtype as x)
    eps:    LayerNorm epsilon (added to variance before rsqrt)
    returns [..., out]           (x's dtype; F32-accumulated LN + GEMM).

    Math (matches flame-core layernorm_linear):
        norm = (x - mean) * rsqrt(var + eps) * gamma + beta
        y    = norm @ weightᵀ + bias
    """
    var normed = layer_norm(x, gamma, beta, eps, ctx)
    return linear(normed, weight, bias, ctx)


def residual_layernorm(
    x: Tensor,
    residual: Tensor,
    gamma: Tensor,
    beta: Tensor,
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Fused residual-add then LayerNorm (affine, last dim, biased var).

    x:        [..., hidden]      (compute dtype)
    residual: [..., hidden]      (same shape/dtype as x)
    gamma:    [hidden]           (LN scale)
    beta:     [hidden]           (LN shift)
    eps:      LayerNorm epsilon
    returns   [..., hidden]      (x's dtype; F32-accumulated add + LN).

    Math (matches flame-core residual_layernorm public fn):
        s = x + residual
        y = (s - mean) * rsqrt(var + eps) * gamma + beta
    """
    var summed = add(x, residual, ctx)
    return layer_norm(summed, gamma, beta, eps, ctx)
