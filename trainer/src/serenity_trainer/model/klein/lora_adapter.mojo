# model/klein/lora_adapter.mojo — host-resident LoRA adapter + AdamW for Klein.
#
# BORROWED line-for-line (struct + math) FROM serenitymojo/training/train_step.mojo
#   (LoraAdapter, LoraGrads, _f32_to_bf16_list, _adamw_host_list, _lora_adamw).
# COPIED into the serenity_trainer namespace per the port rule: serenitymojo/training
# is NOT a reuse source — only serenitymojo/{tensor,io,ops,scratch_ring} (foundation)
# is imported unchanged. The Klein block files (double_block/single_block/lora_block)
# need this `LoraAdapter` (host BF16 lists + AdamW moments) — distinct from the
# tape-based serenity_trainer.module.LoRAModule.LoraAdapter.
#
# AdamW formula matches Serenity's torch.optim.AdamW (bias-corrected, decoupled
# weight decay). The Klein trainer keeps adapters + moments host-resident to avoid
# per-adapter GPU upload/readback churn (serenitymojo train_step.mojo:238-282 note).

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.torch_bf16 import torch_bf16_rne_value

# Shared, bit-exact SR helpers (1:1 of bf16_stochastic_rounding.py) — the SAME
# source of truth used by util/optimizer/adamw_extensions.mojo. We route the host
# LoRA AdamW step through these so the per-adapter update matches Serenity's
# AdamW (adamw_extensions.py:78-145) bit-for-bit (decoupled WD first, lerp moment,
# bf16-quantized moments, stochastic-rounding param write-back).
from serenity_trainer.util.bf16_stochastic_rounding import (
    _sr_bf16, sr_uniform,
)


def _f32_to_bf16_list(v: List[Float32]) -> List[BFloat16]:
    var out = List[BFloat16]()
    for i in range(len(v)):
        out.append(BFloat16(v[i]))
    return out^


def _bf16_to_f32_list(v: List[BFloat16]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(v)):
        out.append(v[i].cast[DType.float32]())
    return out^


# serenitymojo/training/train_step.mojo:133-160
struct LoraAdapter(Copyable, Movable):
    var a: List[BFloat16]  # [rank, in] BF16 model storage
    var b: List[BFloat16]  # [out, rank] BF16 model storage
    var rank: Int
    var in_f: Int
    var out_f: Int
    var scale: Float32
    var ma: List[Float32]
    var va: List[Float32]
    var mb: List[Float32]
    var vb: List[Float32]

    def __init__(
        out self, var a: List[Float32], var b: List[Float32],
        rank: Int, in_f: Int, out_f: Int, scale: Float32,
        var ma: List[Float32], var va: List[Float32],
        var mb: List[Float32], var vb: List[Float32],
    ):
        self.a = _f32_to_bf16_list(a)
        self.b = _f32_to_bf16_list(b)
        self.rank = rank
        self.in_f = in_f
        self.out_f = out_f
        self.scale = scale
        self.ma = ma^
        self.va = va^
        self.mb = mb^
        self.vb = vb^


# serenitymojo/training/train_step.mojo:198-204
struct LoraGrads(Copyable, Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]

    def __init__(out self, var d_a: List[Float32], var d_b: List[Float32]):
        self.d_a = d_a^
        self.d_b = d_b^


# BIT-EXACT host AdamW step, 1:1 with Serenity adamw_extensions.py:78-145
# (non-capturable path) and util/optimizer/adamw_extensions.mojo::_adamw_sr_kernel.
# This REPLACES the serenitymojo train_step host AdamW (which applied weight decay
# AFTER the step, kept F32 moments, and did a plain bf16 truncation on write-back —
# all three diverge from Serenity). The LoRA adapters here are host-resident
# (List[BFloat16] params + moments), so we reproduce the designated bit-exact
# adamw_extensions math on the host using the SHARED SR helpers
# (util/bf16_stochastic_rounding.{_sr_bf16,sr_uniform,torch_bf16_rne_value}):
#
#   p        *= (1 - lr*wd)                          # decoupled WD FIRST  (py:83)
#   exp_avg  += (1-b1)*(grad - exp_avg)              # lerp_               (py:86)
#   exp_avg_sq= b2*exp_avg_sq + (1-b2)*grad*grad     # mul_+addcmul_       (py:87)
#   moments WRITTEN BACK as bf16 (RNE) then RE-READ                       (kernel)
#   bc1=1-b1**step ; bc2=1-b2**step                  (host scalars)       (py:123-124)
#   denom = sqrt(exp_avg_sq)/sqrt(bc2) + eps                              (py:141)
#   p = SR( p - (lr/bc1) * exp_avg / denom )         # addcdiv_stochastic_(py:143)
#
# DTYPE: moments quantized to bf16 each step (Serenity exp_avg/exp_avg_sq are bf16
# tensors written in place BEFORE denom/p read them) so the update consumes the
# bf16-quantized moment, not a full-precision F32 accumulator. The List[Float32]
# moment containers therefore always hold an exactly-bf16-representable value.
# `seed` is the 1-based step (matches the GPU adamw_step per-step seed); the
# element index disambiguates per element (sr_uniform(seed, i)).
def _adamw_host_list(
    mut p: List[BFloat16], g: List[Float32],
    mut m: List[Float32], mut v: List[Float32],
    t: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32,
    stochastic_rounding: Bool,
) raises:
    var n = len(p)
    if len(g) != n or len(m) != n or len(v) != n:
        raise Error("_adamw_host_list: param/grad/m/v len mismatch")
    if t < 1:
        raise Error("_adamw_host_list: t must be >= 1")

    # HOST scalar bias corrections — pow via repeated mul (matches adamw_step).
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

    for i in range(n):
        var pf = p[i].cast[DType.float32]()
        var mf = m[i]
        var vf = v[i]
        var gv = g[i]

        # decoupled weight decay (on the param) FIRST — py:83.
        pf = pf * (Float32(1.0) - lr * weight_decay)
        # exp_avg.lerp_(grad, 1-beta1) — py:86.
        mf = mf + (Float32(1.0) - beta1) * (gv - mf)
        # exp_avg_sq = beta2*v + (1-beta2)*g*g — py:87.
        vf = beta2 * vf + (Float32(1.0) - beta2) * gv * gv

        # Moments written back bf16 (RNE, parity with the GPU kernel) then RE-READ,
        # so the update consumes the bf16-quantized moments.
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


# Per-adapter AdamW: A and B share the step `t`. The element index inside each call
# starts at 0; to keep A and B on independent SR streams (so B's index does not
# collide with A's), offset B's seed by the adapter's param count is NOT needed —
# A and B are distinct tensors in Serenity (separate copy_stochastic_ draws); we
# mirror that by using the same per-step seed but disjoint index spaces is not a
# correctness requirement (SR is unbiased per element). Defaults match Serenity's
# AdamW (betas=(0.9,0.999), eps=1e-8, wd from config) — adamw_extensions.py:63.
def _lora_adamw(
    mut lo: LoraAdapter, g: LoraGrads, t: Int, lr: Float32, ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
    stochastic_rounding: Bool = True,
) raises:
    _adamw_host_list(lo.a, g.d_a, lo.ma, lo.va, t, lr, beta1, beta2, eps, weight_decay, stochastic_rounding)
    _adamw_host_list(lo.b, g.d_b, lo.mb, lo.vb, t, lr, beta1, beta2, eps, weight_decay, stochastic_rounding)
