# module/LensLoRAModule.mojo — Lens LoRA adapter: hand-chained host-resident
# down/up forward + backward (grad wrt A,B) + Serenity-style AdamW update.
#
# BORROWED IMPLEMENTATION MATERIAL (copied + namespaced into serenity_trainer) FROM
#   serenitymojo/models/klein/lora_adapter.mojo  (lora_forward / lora_backward /
#   _lora_adamw / _adamw_host_list_precomputed), and the LoraAdapter / LoraGrads
#   host types from serenitymojo/training/train_step.mojo. Per the borrow boundary
#   only serenitymojo.{tensor,io,ops,autograd,util} foundation is IMPORTED
#   unchanged; serenitymojo.{models,training} are reuse SOURCES that get COPIED
#   here, so LoraAdapter/LoraGrads are RE-DEFINED locally (NOT imported from
#   serenitymojo.training) to keep this unit inside the boundary.
#
# ── VERIFIED 1:1 vs Serenity pr-1510 modules/module/LoRAModule.py ───────────
# LoRAModule.forward (LoRAModule.py:558-564):
#     ld = self.lora_up(self.dropout(self.lora_down(x)))
#     return self.orig_forward(x) + ld * (self.alpha / self.rank)
#   → lora_forward returns ONLY the delta `ld*(alpha/rank)` (the caller adds the
#     frozen base orig_forward(x) separately). dropout defaults to Dropout(0)
#     (LoRAModule.py:537) → no-op, so this matches the default/inference path.
# LoRAModule.initialize_weights (LoRAModule.py:547-551):
#     lora_down, lora_up = create_layer()
#     nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
#     nn.init.zeros_(lora_up.weight)
#   → make_lora_adapter: A = small-uniform (kaiming-ish, see note), B = 0
#     (PEFT identity at step 0). Save format (PEFT/ai-toolkit):
#     <prefix>.lora_down.weight (=A) / <prefix>.lora_up.weight (=B).
# scale = alpha / rank (LoRAModule.py:564).
# AdamW: Serenity uses torch AdamW over the trainable A/B params. The host AdamW
# below mirrors torch's AdamW (decoupled weight decay, bias-corrected moments,
# eps-after-sqrt) with BF16 moment storage + stochastic rounding to match the
# trainer's bf16_stochastic_rounding optimizer behavior.
#
# DTYPE: BF16 adapter storage (a/b), F32 optimizer math in registers, BF16
# write-back via RNE / stochastic rounding. No persistent F32 tensors.

from std.collections import List, Optional
from std.builtin.dtype import DType
from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import mul_scalar
from serenitymojo.ops.torch_bf16 import torch_bf16_rne_value
from serenitymojo.util.bf16_stochastic_rounding import _sr_bf16, sr_uniform


# ── host List[Float32] → List[BFloat16] (copied from train_step._f32_to_bf16_list)
def _f32_to_bf16_list(x: List[Float32]) -> List[BFloat16]:
    var out = List[BFloat16]()
    for i in range(len(x)):
        out.append(x[i].cast[DType.bfloat16]())
    return out^


# ── LoraAdapter (copied from serenitymojo.training.train_step:133-160) ────────
# A = lora_down.weight [rank, in]; B = lora_up.weight [out, rank]. BF16 model
# storage; ma/va/mb/vb are the F32 AdamW moment buffers for A and B.
struct LoraAdapter(Copyable, Movable):
    var a: List[BFloat16]   # [rank, in] BF16
    var b: List[BFloat16]   # [out, rank] BF16
    var rank: Int
    var in_f: Int
    var out_f: Int
    var scale: Float32      # alpha / rank  (LoRAModule.py:564)
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


# ── LoraGrads (copied from serenitymojo.training.train_step:198-204) ──────────
struct LoraGrads(Copyable, Movable):
    var d_a: List[Float32]   # [rank, in]
    var d_b: List[Float32]   # [out, rank]

    def __init__(out self, var d_a: List[Float32], var d_b: List[Float32]):
        self.d_a = d_a^
        self.d_b = d_b^


# ── make_lora_adapter (copied from serenitymojo/models/klein/lora_adapter.mojo:190)
# A ~ small-uniform PCG draw (LoRAModule.py:550 nn.init.kaiming_uniform_(a=sqrt(5));
# the magnitude is small for a LoRA-down so the ×0.02 PCG-uniform is the practical
# init the working Klein adapter uses — NOTE: not the exact torch kaiming
# distribution; the gate at B=0 makes step-0 identity regardless of A's exact
# init). B = 0 (LoRAModule.py:551, PEFT identity at init). scale = alpha/rank.
def make_lora_adapter(
    rank: Int, alpha: Float32, in_f: Int, out_f: Int, seed: UInt64,
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


# ── lora_forward (copied from serenitymojo/models/klein/lora_adapter.mojo:158-170)
# Returns the LoRA DELTA contribution on x [M,in] → [M,out]:
#   delta = (x @ Aᵀ @ Bᵀ) * scale            (LoRAModule.py:563-564 `ld * scale`)
# The caller adds the frozen base orig_forward(x) separately. dropout p=0 → no-op.
def lora_forward(x: Tensor, lo: LoraAdapter, M: Int, ctx: DeviceContext) raises -> Tensor:
    var a = Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx)
    var b = Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx)
    var nb1 = Optional[Tensor](None)
    var xa = linear(x, a, nb1^, ctx)         # [M,rank] = x @ Aᵀ
    var nb2 = Optional[Tensor](None)
    var xb = linear(xa, b, nb2^, ctx)        # [M,out]  = (x@Aᵀ) @ Bᵀ
    return mul_scalar(xb, lo.scale, ctx)     # * (alpha/rank)


# ── lora_backward (copied from serenitymojo/models/klein/lora_adapter.mojo:173-187)
# Given x [M,in] and d_y (grad of the loss wrt the LoRA delta output [M,out]),
# returns d_A [rank,in] and d_B [out,rank]. Chain (reverse of lora_forward):
#   d_scaled = d_y * scale                     (scale backward)
#   d_B, d_xa = linear_backward(d_scaled, xa, B)   # xb = xa @ Bᵀ
#   d_A, _    = linear_backward(d_xa,    x,  A)     # xa = x  @ Aᵀ
# (the d_x contribution into the projection input is folded by the block backward;
#  here only the factor grads are returned, matching LoraGrads.)
def lora_backward(
    x: Tensor, d_y: Tensor, lo: LoraAdapter, M: Int, ctx: DeviceContext
) raises -> LoraGrads:
    var a = Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx)
    var b = Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx)
    var nb1 = Optional[Tensor](None)
    var xa = linear(x, a, nb1^, ctx)                       # recompute [M,rank]
    var d_scaled = mul_scalar(d_y, lo.scale, ctx)          # scale backward
    var db = linear_backward(d_scaled, xa, b, M, lo.rank, lo.out_f, ctx)  # → d_B, d_xa
    var da = linear_backward(db.d_x^, x, a, M, lo.in_f, lo.rank, ctx)     # → d_A
    return LoraGrads(da.d_w.to_host(ctx), db.d_w.to_host(ctx))


# ── AdamW host step (copied from serenitymojo/models/klein/lora_adapter.mojo) ──
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

        pf = pf * decay                                    # decoupled weight decay
        mf = mf + one_minus_beta1 * (gv - mf)              # m += (1-b1)(g-m)
        vf = beta2 * vf + one_minus_beta2 * gv * gv        # v = b2 v + (1-b2)g²

        var m_q = torch_bf16_rne_value(mf)
        var v_q = torch_bf16_rne_value(vf)
        m[i] = m_q.cast[DType.float32]()
        v[i] = v_q.cast[DType.float32]()
        var mfq = m_q.cast[DType.float32]()
        var vfq = v_q.cast[DType.float32]()

        var denom = sqrt(vfq) / bc2_sqrt + eps             # eps AFTER sqrt
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
        one_minus_beta1, beta2, one_minus_beta2, eps, seed, stochastic_rounding,
    )
    _adamw_host_list_precomputed(
        lo.b, d_b, lo.mb, lo.vb, step_size, bc2_sqrt, decay,
        one_minus_beta1, beta2, one_minus_beta2, eps, seed, stochastic_rounding,
    )


# Full AdamW update for one adapter at optimizer step `t` (1-based). lr/betas/eps/
# weight_decay match Serenity's AdamW defaults; bias correction recomputed here.
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
