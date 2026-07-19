# adamw.mojo — pure-Mojo port of Serenity's AdamW (adamw_extensions.py +
# bf16_stochastic_rounding.py). MEASURED semantics from Serenity source:
#
#   state exp_avg, exp_avg_sq = zeros_like(p)  -> SAME dtype as param (BF16).
#   per step (non-capturable default path):
#     step += 1
#     p        *= (1 - lr*wd)                       # decoupled weight decay
#     exp_avg   = exp_avg + (1-b1)*(grad - exp_avg) # lerp_
#     exp_avg_sq= b2*exp_avg_sq + (1-b2)*grad*grad  # mul_ + addcmul_
#     bc1 = 1 - b1**step ;  bc2 = 1 - b2**step       (HOST scalars)
#     step_size = lr / bc1 ;  denom = sqrt(exp_avg_sq)/sqrt(bc2) + eps
#     if p is BF16 and stochastic_rounding:
#         addcdiv_stochastic_(p, exp_avg, denom, value=-step_size)
#     else:
#         p -= step_size * exp_avg / denom
#
# DTYPE POLICY (matches Serenity): p, exp_avg, exp_avg_sq are ALL
# BF16 STORAGE. There is NO F32 master and NO F32 moment state. Every per-element
# computation is done in F32 REGISTERS and written back to BF16 — the f32 `result`
# in copy_stochastic_ is a throwaway register, never stored. So: bf16 in/out,
# f32 compute, zero persistent F32.
#
# Stochastic rounding (copy_stochastic_): for a freshly computed f32 value, add a
# uniform 16-bit integer to the int32 bit pattern, mask off the low 16 mantissa
# bits, and keep the high 16 bits as the bf16. We reproduce this in-register; the
# RNG is a per-element PCG hash of (seed, index) — Serenity seeds a torch
# Generator per step (bf16_stochastic_rounding.set_seed(step_seed)); any unbiased
# 16-bit source preserves the stochastic-rounding guarantee.

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

# Shared stochastic-rounding helpers (1:1 of bf16_stochastic_rounding.py). The SR
# kernel below uses _sr_bf16 (the copy_stochastic_ equivalent) and _pcg_hash (the
# random-16-bit source), now owned by util/bf16_stochastic_rounding.mojo so adam /
# adamw / came / adafactor all share ONE implementation.
from serenity_trainer.util.bf16_stochastic_rounding import _sr_bf16, _pcg_hash

comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _U24 = Float32(1.0) / Float32(16777216.0)  # 1/2^24 → uniform [0,1)


def _bf16_lt(t: Tensor, n: Int) -> LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin]:
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    return LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        t.buf.unsafe_ptr().bitcast[BFloat16](), rl
    )


def _adamw_sr_kernel(
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

    # decoupled weight decay
    pf = pf * (Float32(1.0) - lr * wd)
    # exp_avg.lerp_(grad, 1-beta1)
    mf = mf + (Float32(1.0) - beta1) * (gf - mf)
    # exp_avg_sq = beta2*v + (1-beta2)*g*g
    vf = beta2 * vf + (Float32(1.0) - beta2) * gf * gf

    # Moments written back BF16 via the PyTorch/CUDA-parity RNE helper, then
    # RE-READ as the bf16-quantized value for the update. Serenity's exp_avg /
    # exp_avg_sq are bf16 tensors written in place (adamw_extensions.py:86-87)
    # BEFORE denom/p read them (:141,:146), so the update consumes bf16-quantized
    # moments — not the full-precision f32 accumulators. Matching that here.
    var m_q = torch_bf16_rne_value(mf)
    var v_q = torch_bf16_rne_value(vf)
    m[i] = rebind[m.element_type](m_q)
    v[i] = rebind[v.element_type](v_q)
    var mfq = m_q.cast[DType.float32]()
    var vfq = v_q.cast[DType.float32]()

    var step_size = lr / bc1
    var denom = sqrt(vfq) / bc2_sqrt + eps
    var newp = pf - step_size * mfq / denom

    if stochastic == 1:
        # Serenity copy_stochastic_ equivalent: unbiased round of the f32 update
        # to bf16, using a per-element uniform from a PCG hash of (seed, index).
        var rnd = _pcg_hash(seed ^ UInt32(i))
        var u = Float32(Int(rnd >> UInt32(8))) * _U24   # uniform [0,1)
        p[i] = rebind[p.element_type](_sr_bf16(newp, u))
    else:
        # Non-SR param write-back: route through the PyTorch/CUDA-parity RNE
        # helper, matching the moment write-back above and CAME/Adafactor —
        # Mojo's native cast differs from torch's bf16 store by one BF16 quantum
        # on some values, which would drift the param trajectory.
        p[i] = rebind[p.element_type](torch_bf16_rne_value(newp))


# Host-side single-tensor AdamW step. p, m(exp_avg), v(exp_avg_sq), g all BF16,
# mutated IN PLACE. `step` is the 1-based counter (host Int). bc1/bc2 computed on
# host (matches Serenity's `bias_correction = 1 - beta**step`).
def adamw_step(
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
        raise Error("adamw_step: param must be BF16 (port dtype policy)")
    if m.dtype() != STDtype.BF16 or v.dtype() != STDtype.BF16 or g.dtype() != STDtype.BF16:
        raise Error("adamw_step: m/v/grad must be BF16 (no F32 optimizer state)")
    var n = p.numel()
    if m.numel() != n or v.numel() != n or g.numel() != n:
        raise Error("adamw_step: numel mismatch")
    if step < 1:
        raise Error("adamw_step: step must be >= 1 (1-based)")

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
    ctx.enqueue_function[_adamw_sr_kernel, _adamw_sr_kernel](
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
