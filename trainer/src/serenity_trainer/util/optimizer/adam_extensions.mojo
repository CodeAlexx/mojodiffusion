# adam_extensions.mojo — pure-Mojo port of Serenity's plain Adam
# (modules/util/optimizer/adam_extensions.py + bf16_stochastic_rounding.py).
#
# This is the L2-regularized (COUPLED) Adam, distinct from AdamW. MEASURED
# semantics from adam_extensions.py (non-capturable default path):
#   state exp_avg, exp_avg_sq = zeros_like(p)  -> SAME dtype as param (BF16).
#   per step (step_adam_parameter, :60-146):
#     step += 1                                          (:72)
#     if weight_decay != 0: grad = grad + wd*p           (:74-75  COUPLED L2)
#     exp_avg    = exp_avg + (1-b1)*(grad - exp_avg)      (:86 lerp_)
#     exp_avg_sq = b2*exp_avg_sq + (1-b2)*grad*grad       (:87 mul_+addcmul_)
#     bc1 = 1 - b1**step ; bc2 = 1 - b2**step             (:124-125 HOST scalars)
#     step_size = lr / bc1                                (:127)
#     denom = sqrt(exp_avg_sq)/sqrt(bc2) + eps            (:139-141)
#     if p is BF16 and stochastic_rounding:
#         addcdiv_stochastic_(p, exp_avg, denom, value=-step_size)   (:144)
#     else:
#         p.addcdiv_(exp_avg, denom, value=-step_size)               (:146)
#
# DIFFERENCE vs AdamW (adamw_extensions.py): Adam folds weight decay INTO the
# gradient as coupled L2 (grad += wd*p) and has NO decoupled `p *= (1-lr*wd)`
# stepweight-decay term. Everything else is identical.
#
# DTYPE POLICY (matches Serenity + adamw.mojo): p, exp_avg, exp_avg_sq are ALL
# BF16 STORAGE. No F32 master, no F32 moment state. Per-element compute in F32
# registers, written back to BF16 — the f32 `result` in copy_stochastic_ is a
# throwaway register, never stored. bf16 in/out, f32 compute, zero persistent F32.
#
# Stochastic rounding mirrors copy_stochastic_/addcdiv_stochastic_ via the SHARED
# helpers in util/bf16_stochastic_rounding.mojo (_sr_bf16 + _pcg_hash).

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.builtin.dtype import DType
from std.math import sqrt, floor, log, pow
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from std.utils.index import IndexList
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.torch_bf16 import torch_bf16_rne_value

# Shared SR helpers (1:1 of bf16_stochastic_rounding.py): _sr_bf16 is the
# copy_stochastic_ equivalent, _pcg_hash the random-16-bit source.
from serenity_trainer.util.bf16_stochastic_rounding import _sr_bf16, _pcg_hash

comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _U24 = Float32(1.0) / Float32(16777216.0)  # 1/2^24 → uniform [0,1)


def _bf16_lt(t: Tensor, n: Int) -> LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin]:
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    return LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        t.buf.unsafe_ptr().bitcast[BFloat16](), rl
    )


def _adam_sr_kernel(
    p: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    m: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    v: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    g: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    wd: Float32,
    bc1: Float32,
    bc2_sqrt: Float32,
    seed: UInt32,
    stochastic: Int,
):
    var i = Int(global_idx.x)
    if i >= n:
        return

    var pf = rebind[Scalar[DType.bfloat16]](p[i]).cast[DType.float32]()
    var mf = rebind[Scalar[DType.bfloat16]](m[i]).cast[DType.float32]()
    var vf = rebind[Scalar[DType.bfloat16]](v[i]).cast[DType.float32]()
    var gf = rebind[Scalar[DType.bfloat16]](g[i]).cast[DType.float32]()

    # COUPLED L2 weight decay: grad = grad + wd*p  (adam_extensions.py:74-75).
    # (AdamW differs here: it does decoupled p *= (1-lr*wd) instead.)
    if wd != Float32(0.0):
        gf = gf + wd * pf
    # exp_avg.lerp_(grad, 1-beta1)                  (:86)
    mf = mf + (Float32(1.0) - beta1) * (gf - mf)
    # exp_avg_sq = beta2*v + (1-beta2)*g*g          (:87)
    vf = beta2 * vf + (Float32(1.0) - beta2) * gf * gf

    # Moments written back BF16 via the PyTorch/CUDA-parity RNE helper, then
    # RE-READ as the bf16-quantized value for the update — Serenity's exp_avg /
    # exp_avg_sq are bf16 tensors written in place (:86-87) BEFORE denom/p read
    # them (:139,:144), so the update consumes bf16-quantized moments.
    var m_q = torch_bf16_rne_value(mf)
    var v_q = torch_bf16_rne_value(vf)
    m[i] = rebind[m.element_type](m_q)
    v[i] = rebind[v.element_type](v_q)
    var mfq = m_q.cast[DType.float32]()
    var vfq = v_q.cast[DType.float32]()

    # step_size = lr/bc1 ; denom = sqrt(v)/sqrt(bc2) + eps        (:127,:139-141)
    var step_size = lr / bc1
    var denom = sqrt(vfq) / bc2_sqrt + eps
    # p.addcdiv_(exp_avg, denom, value=-step_size)                (:144,:146)
    var newp = pf - step_size * mfq / denom

    if stochastic == 1:
        # addcdiv_stochastic_ equivalent: unbiased round of the f32 update to bf16
        # using a per-element uniform from a PCG hash of (seed, index).
        var rnd = _pcg_hash(seed ^ UInt32(i))
        var u = Float32(Int(rnd >> UInt32(8))) * _U24   # uniform [0,1)
        p[i] = rebind[p.element_type](_sr_bf16(newp, u))
    else:
        # Non-SR param write-back: route through the PyTorch/CUDA-parity RNE
        # helper, matching the moment write-back above and CAME/Adafactor —
        # Mojo's native cast differs from torch's bf16 store by one BF16 quantum
        # on some values, which would drift the param trajectory.
        p[i] = rebind[p.element_type](torch_bf16_rne_value(newp))


# Host-side single-tensor Adam step. p, m(exp_avg), v(exp_avg_sq), g all BF16,
# mutated IN PLACE. `step` is the 1-based counter (host Int). bc1/bc2 computed on
# host (matches `bias_correction = 1 - beta**step`, adam_extensions.py:124-125).
def adam_step(
    p: Tensor,
    m: Tensor,
    v: Tensor,
    g: Tensor,
    step: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    stochastic_rounding: Bool,
    seed: UInt32,
    ctx: DeviceContext,
) raises:
    if p.dtype() != STDtype.BF16:
        raise Error("adam_step: param must be BF16 (port dtype policy)")
    if m.dtype() != STDtype.BF16 or v.dtype() != STDtype.BF16 or g.dtype() != STDtype.BF16:
        raise Error("adam_step: m/v/grad must be BF16 (no F32 optimizer state)")
    var n = p.numel()
    if m.numel() != n or v.numel() != n or g.numel() != n:
        raise Error("adam_step: numel mismatch")
    if step < 1:
        raise Error("adam_step: step must be >= 1 (1-based)")

    # HOST scalar bias corrections — pow via repeated mul to avoid a libm dep.
    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(step):
        b1p = b1p * beta1
        b2p = b2p * beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p
    var bc2_sqrt = sqrt(bc2)

    var sr = 1 if stochastic_rounding else 0
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_adam_sr_kernel, _adam_sr_kernel](
        _bf16_lt(p, n),
        _bf16_lt(m, n),
        _bf16_lt(v, n),
        _bf16_lt(g, n),
        n,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bc1,
        bc2_sqrt,
        seed,
        sr,
        grid_dim=grid,
        block_dim=_BLOCK,
    )
