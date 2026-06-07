# Host-resident Klein LoRA adapter and OneTrainer-style AdamW.
#
# This is the mojodiffusion core counterpart of the OneTrainer-named Klein port
# helper. It intentionally keeps BF16 adapter storage at tensor boundaries while
# doing optimizer math in F32 registers, then writes back with stochastic BF16
# rounding to match OneTrainer's optimizer behavior.

from std.collections import List, Optional
from std.builtin.dtype import DType
from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import mul_scalar
from serenitymojo.ops.torch_bf16 import torch_bf16_rne_value
from serenitymojo.tensor import Tensor
from serenitymojo.training.train_step import LoraAdapter, LoraGrads
from serenitymojo.util.bf16_stochastic_rounding import _sr_bf16, sr_uniform


def _adamw_host_list(
    mut p: List[BFloat16],
    g: List[Float32],
    mut m: List[Float32],
    mut v: List[Float32],
    t: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    stochastic_rounding: Bool,
) raises:
    if t < 1:
        raise Error("_adamw_host_list: t must be >= 1")

    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p
    var bc2_sqrt = sqrt(bc2)
    var step_size = lr / bc1
    var seed = UInt32(t)
    _adamw_host_list_precomputed(
        p, g, m, v, step_size, bc2_sqrt,
        Float32(1.0) - lr * weight_decay,
        Float32(1.0) - beta1, beta2, Float32(1.0) - beta2,
        eps, seed, stochastic_rounding,
    )


def _adamw_host_list_precomputed(
    mut p: List[BFloat16],
    g: List[Float32],
    mut m: List[Float32],
    mut v: List[Float32],
    step_size: Float32,
    bc2_sqrt: Float32,
    decay: Float32,
    one_minus_beta1: Float32,
    beta2: Float32,
    one_minus_beta2: Float32,
    eps: Float32,
    seed: UInt32,
    stochastic_rounding: Bool,
) raises:
    var n = len(p)
    if len(g) != n or len(m) != n or len(v) != n:
        raise Error("_adamw_host_list: param/grad/m/v len mismatch")

    for i in range(n):
        var pf = p[i].cast[DType.float32]()
        var mf = m[i]
        var vf = v[i]
        var gv = g[i]

        pf = pf * decay
        mf = mf + one_minus_beta1 * (gv - mf)
        vf = beta2 * vf + one_minus_beta2 * gv * gv

        var m_q = torch_bf16_rne_value(mf)
        var v_q = torch_bf16_rne_value(vf)
        m[i] = m_q.cast[DType.float32]()
        v[i] = v_q.cast[DType.float32]()
        var mfq = m_q.cast[DType.float32]()
        var vfq = v_q.cast[DType.float32]()

        var denom = sqrt(vfq) / bc2_sqrt + eps
        var newp = pf - step_size * mfq / denom

        if stochastic_rounding:
            p[i] = _sr_bf16(newp, sr_uniform(seed, i))
        else:
            p[i] = torch_bf16_rne_value(newp)


def _lora_adamw_precomputed(
    mut lo: LoraAdapter,
    d_a: List[Float32],
    d_b: List[Float32],
    step_size: Float32,
    bc2_sqrt: Float32,
    decay: Float32,
    one_minus_beta1: Float32,
    beta2: Float32,
    one_minus_beta2: Float32,
    eps: Float32,
    seed: UInt32,
    stochastic_rounding: Bool = True,
) raises:
    _adamw_host_list_precomputed(
        lo.a, d_a, lo.ma, lo.va, step_size, bc2_sqrt, decay,
        one_minus_beta1, beta2, one_minus_beta2, eps, seed,
        stochastic_rounding,
    )
    _adamw_host_list_precomputed(
        lo.b, d_b, lo.mb, lo.vb, step_size, bc2_sqrt, decay,
        one_minus_beta1, beta2, one_minus_beta2, eps, seed,
        stochastic_rounding,
    )


def _lora_adamw(
    mut lo: LoraAdapter,
    g: LoraGrads,
    t: Int,
    lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9),
    beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8),
    weight_decay: Float32 = Float32(0.01),
    stochastic_rounding: Bool = True,
) raises:
    _ = ctx
    if t < 1:
        raise Error("_lora_adamw: t must be >= 1")

    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p
    _lora_adamw_precomputed(
        lo, g.d_a, g.d_b, lr / bc1, sqrt(bc2),
        Float32(1.0) - lr * weight_decay,
        Float32(1.0) - beta1, beta2, Float32(1.0) - beta2,
        eps, UInt32(t), stochastic_rounding,
    )


def lora_forward(
    x: Tensor,
    lo: LoraAdapter,
    M: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var a = Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx)
    var b = Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx)
    var nb1 = Optional[Tensor](None)
    var xa = linear(x, a, nb1^, ctx)
    var nb2 = Optional[Tensor](None)
    var xb = linear(xa, b, nb2^, ctx)
    return mul_scalar(xb, lo.scale, ctx)


def lora_backward(
    x: Tensor,
    d_y: Tensor,
    lo: LoraAdapter,
    M: Int,
    ctx: DeviceContext,
) raises -> LoraGrads:
    var a = Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx)
    var b = Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx)
    var nb1 = Optional[Tensor](None)
    var xa = linear(x, a, nb1^, ctx)
    var d_scaled = mul_scalar(d_y, lo.scale, ctx)
    var db = linear_backward(d_scaled, xa, b, M, lo.rank, lo.out_f, ctx)
    var da = linear_backward(db.d_x^, x, a, M, lo.in_f, lo.rank, ctx)
    return LoraGrads(da.d_w.to_host(ctx), db.d_w.to_host(ctx))


def make_lora_adapter(
    rank: Int,
    alpha: Float32,
    in_f: Int,
    out_f: Int,
    seed: UInt64,
) -> LoraAdapter:
    var a = List[Float32]()
    var b = List[Float32]()
    var s = seed
    for _ in range(rank * in_f):
        s = s * UInt64(6364136223846793005) + UInt64(1)
        var u = Float32((s >> UInt64(40)) & UInt64(0xFFFF)) / Float32(65536.0)
        a.append((u - Float32(0.5)) * Float32(0.02))
    for _ in range(out_f * rank):
        b.append(Float32(0.0))
    var z_a = List[Float32]()
    var z_b = List[Float32]()
    for _ in range(rank * in_f):
        z_a.append(Float32(0.0))
    for _ in range(out_f * rank):
        z_b.append(Float32(0.0))
    return LoraAdapter(
        a^, b^, rank, in_f, out_f, alpha / Float32(rank),
        z_a.copy(), z_a^, z_b.copy(), z_b^,
    )
