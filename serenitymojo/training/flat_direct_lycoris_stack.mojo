# training/flat_direct_lycoris_stack.mojo -- streamed direct DoRA/OFT slots.
#
# Product direction for DoRA/OFT on large streamed trainers: the set owns only
# trainables and optimizer moments. The frozen W_orig is supplied by the model
# stack at the exact projection call site, then discarded with the streamed
# block. This avoids the full-delta `a=I, b=W_eff-W` carrier that exceeds 24 GB
# on large models.

from std.collections import List
from std.math import sqrt

from serenitymojo.training.dora_adapter import (
    DoRAAdapter, DoRAGrads, new_dora_adapter, dora_substitution_forward,
    dora_substitution_backward, dora_adamw,
)
from serenitymojo.training.oft_onetrainer import (
    OFTOTGrads, oft_ot_forward, oft_ot_backward,
)


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _abs_f64(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _adamw_f32(
    mut p: List[Float32], g: List[Float32],
    mut m: List[Float32], mut v: List[Float32],
    t: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32,
) raises:
    if len(p) != len(g) or len(p) != len(m) or len(p) != len(v):
        raise Error("_adamw_f32: param/grad/m/v len mismatch")
    if t < 1:
        raise Error("_adamw_f32: t must be >= 1")
    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p
    for i in range(len(p)):
        var gv = g[i]
        var mi = beta1 * m[i] + (Float32(1.0) - beta1) * gv
        var vi = beta2 * v[i] + (Float32(1.0) - beta2) * gv * gv
        m[i] = mi
        v[i] = vi
        var pv = p[i]
        if weight_decay > 0.0:
            pv = pv * (Float32(1.0) - lr * weight_decay)
        p[i] = pv - lr * (mi / bc1) / (sqrt(vi / bc2) + eps)


# ── DoRA ─────────────────────────────────────────────────────────────────────
struct FlatDirectDoRASet(Copyable, Movable):
    var ad: List[DoRAAdapter]
    var active: List[Bool]
    var prefix: List[String]
    var rank: Int
    var wd_on_out: Bool

    def __init__(
        out self, var ad: List[DoRAAdapter], var active: List[Bool],
        var prefix: List[String], rank: Int, wd_on_out: Bool,
    ):
        self.ad = ad^
        self.active = active^
        self.prefix = prefix^
        self.rank = rank
        self.wd_on_out = wd_on_out


def empty_flat_direct_dora_set() -> FlatDirectDoRASet:
    return FlatDirectDoRASet(List[DoRAAdapter](), List[Bool](), List[String](), 0, False)


def flat_direct_dora_append_from_weight(
    mut set: FlatDirectDoRASet,
    w_orig: List[Float32], in_f: Int, out_f: Int,
    rank: Int, alpha: Float32, prefix: String, seed: UInt64,
    wd_on_out: Bool = False,
) raises:
    if len(w_orig) != in_f * out_f:
        raise Error("flat_direct_dora_append_from_weight: w_orig numel mismatch")
    if len(set.ad) == 0:
        set.rank = rank
        set.wd_on_out = wd_on_out
    elif set.rank != rank or set.wd_on_out != wd_on_out:
        raise Error("flat_direct_dora_append_from_weight: rank/wd_on_out mismatch")
    set.ad.append(new_dora_adapter(
        w_orig, in_f, out_f, rank, alpha, seed, Float32(1.0e-7), wd_on_out,
    ))
    set.active.append(True)
    set.prefix.append(prefix)


def flat_direct_dora_forward_slot(
    set: FlatDirectDoRASet, slot: Int,
    x_h: List[Float32], w_orig: List[Float32], M: Int,
) raises -> List[Float32]:
    if slot < 0 or slot >= len(set.ad):
        raise Error("flat_direct_dora_forward_slot: slot out of range")
    if not set.active[slot]:
        raise Error("flat_direct_dora_forward_slot: inactive slot")
    return dora_substitution_forward(x_h, w_orig, set.ad[slot], M)


def flat_direct_dora_backward_slot(
    set: FlatDirectDoRASet, slot: Int,
    d_y_h: List[Float32], x_h: List[Float32], w_orig: List[Float32], M: Int,
) raises -> DoRAGrads:
    if slot < 0 or slot >= len(set.ad):
        raise Error("flat_direct_dora_backward_slot: slot out of range")
    if not set.active[slot]:
        raise Error("flat_direct_dora_backward_slot: inactive slot")
    return dora_substitution_backward(d_y_h, x_h, w_orig, set.ad[slot], M)


struct FlatDirectDoRAGrads(Copyable, Movable):
    var g: List[DoRAGrads]

    def __init__(out self, var g: List[DoRAGrads]):
        self.g = g^


def flat_direct_dora_grad_norm(g: FlatDirectDoRAGrads) -> Float64:
    var s = Float64(0.0)
    for gi in range(len(g.g)):
        for i in range(len(g.g[gi].d_a)):
            s += Float64(g.g[gi].d_a[i]) * Float64(g.g[gi].d_a[i])
        for i in range(len(g.g[gi].d_b)):
            s += Float64(g.g[gi].d_b[i]) * Float64(g.g[gi].d_b[i])
        for i in range(len(g.g[gi].d_m)):
            s += Float64(g.g[gi].d_m[i]) * Float64(g.g[gi].d_m[i])
    return sqrt(s)


def _scale_dora(mut g: DoRAGrads, scale: Float32):
    for i in range(len(g.d_a)):
        g.d_a[i] *= scale
    for i in range(len(g.d_b)):
        g.d_b[i] *= scale
    for i in range(len(g.d_m)):
        g.d_m[i] *= scale


def flat_direct_dora_clip_grads(mut g: FlatDirectDoRAGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(g.g)):
        _scale_dora(g.g[i], clip_scale)


def flat_direct_dora_adamw_step(
    mut set: FlatDirectDoRASet, g: FlatDirectDoRAGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    if len(g.g) != len(set.ad):
        raise Error("flat_direct_dora_adamw_step: grad count mismatch")
    for i in range(len(set.ad)):
        if set.active[i]:
            dora_adamw(set.ad[i], g.g[i], t, lr, beta1, beta2, eps, weight_decay)


def flat_direct_dora_zero_leg_l1(set: FlatDirectDoRASet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref d = set.ad[i]
        for j in range(len(d.b)):
            s += _abs_f64(Float64(d.b[j].cast[DType.float32]()))
    return s


def flat_direct_dora_trainable_bytes(set: FlatDirectDoRASet) -> Int:
    var elems_bf16 = 0
    var elems_f32 = 0
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref d = set.ad[i]
        elems_bf16 += len(d.a) + len(d.b)
        elems_f32 += len(d.m) + len(d.ma) + len(d.va) + len(d.mb) + len(d.vb) + len(d.mm) + len(d.vm)
    return elems_bf16 * 2 + elems_f32 * 4


# ── OneTrainer OFT ───────────────────────────────────────────────────────────
struct FlatDirectOFTSlot(Copyable, Movable):
    var vec: List[Float32]
    var in_f: Int
    var out_f: Int
    var b: Int
    var r: Int
    var m: List[Float32]
    var v: List[Float32]

    def __init__(
        out self, var vec: List[Float32], in_f: Int, out_f: Int,
        b: Int, r: Int, var m: List[Float32], var v: List[Float32],
    ):
        self.vec = vec^
        self.in_f = in_f
        self.out_f = out_f
        self.b = b
        self.r = r
        self.m = m^
        self.v = v^


def new_flat_direct_oft_slot(in_f: Int, out_f: Int, block_size: Int) raises -> FlatDirectOFTSlot:
    if in_f % block_size != 0:
        raise Error("new_flat_direct_oft_slot: in_f must be divisible by block_size")
    var r = in_f // block_size
    var ne = block_size * (block_size - 1) // 2
    return FlatDirectOFTSlot(
        _zeros(r * ne), in_f, out_f, block_size, r, _zeros(r * ne), _zeros(r * ne),
    )


struct FlatDirectOFTSet(Copyable, Movable):
    var ad: List[FlatDirectOFTSlot]
    var active: List[Bool]
    var prefix: List[String]
    var block_size: Int

    def __init__(
        out self, var ad: List[FlatDirectOFTSlot], var active: List[Bool],
        var prefix: List[String], block_size: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.prefix = prefix^
        self.block_size = block_size


def empty_flat_direct_oft_set() -> FlatDirectOFTSet:
    return FlatDirectOFTSet(List[FlatDirectOFTSlot](), List[Bool](), List[String](), 0)


def flat_direct_oft_append(
    mut set: FlatDirectOFTSet,
    in_f: Int, out_f: Int, block_size: Int, prefix: String,
) raises:
    if len(set.ad) == 0:
        set.block_size = block_size
    elif set.block_size != block_size:
        raise Error("flat_direct_oft_append: block_size mismatch")
    set.ad.append(new_flat_direct_oft_slot(in_f, out_f, block_size))
    set.active.append(True)
    set.prefix.append(prefix)


def flat_direct_oft_forward_slot(
    set: FlatDirectOFTSet, slot: Int,
    x_h: List[Float32], w_orig: List[Float32], M: Int,
) raises -> List[Float32]:
    if slot < 0 or slot >= len(set.ad):
        raise Error("flat_direct_oft_forward_slot: slot out of range")
    if not set.active[slot]:
        raise Error("flat_direct_oft_forward_slot: inactive slot")
    ref sl = set.ad[slot]
    return oft_ot_forward(x_h, sl.vec, w_orig, M, sl.in_f, sl.out_f, sl.b, sl.r)


def flat_direct_oft_backward_slot(
    set: FlatDirectOFTSet, slot: Int,
    d_y_h: List[Float32], x_h: List[Float32], w_orig: List[Float32], M: Int,
) raises -> OFTOTGrads:
    if slot < 0 or slot >= len(set.ad):
        raise Error("flat_direct_oft_backward_slot: slot out of range")
    if not set.active[slot]:
        raise Error("flat_direct_oft_backward_slot: inactive slot")
    ref sl = set.ad[slot]
    return oft_ot_backward(d_y_h, x_h, sl.vec, w_orig, M, sl.in_f, sl.out_f, sl.b, sl.r)


struct FlatDirectOFTGrads(Copyable, Movable):
    var d_vec: List[List[Float32]]

    def __init__(out self, var d_vec: List[List[Float32]]):
        self.d_vec = d_vec^


def flat_direct_oft_grad_norm(g: FlatDirectOFTGrads) -> Float64:
    var s = Float64(0.0)
    for gi in range(len(g.d_vec)):
        for i in range(len(g.d_vec[gi])):
            s += Float64(g.d_vec[gi][i]) * Float64(g.d_vec[gi][i])
    return sqrt(s)


def flat_direct_oft_clip_grads(mut g: FlatDirectOFTGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for gi in range(len(g.d_vec)):
        for i in range(len(g.d_vec[gi])):
            g.d_vec[gi][i] *= clip_scale


def flat_direct_oft_adamw_step(
    mut set: FlatDirectOFTSet, g: FlatDirectOFTGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    if len(g.d_vec) != len(set.ad):
        raise Error("flat_direct_oft_adamw_step: grad count mismatch")
    for i in range(len(set.ad)):
        if set.active[i]:
            _adamw_f32(
                set.ad[i].vec, g.d_vec[i], set.ad[i].m, set.ad[i].v,
                t, lr, beta1, beta2, eps, weight_decay,
            )


def flat_direct_oft_vec_l1(set: FlatDirectOFTSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref sl = set.ad[i]
        for j in range(len(sl.vec)):
            s += _abs_f64(Float64(sl.vec[j]))
    return s


def flat_direct_oft_trainable_bytes(set: FlatDirectOFTSet) -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        elems += len(set.ad[i].vec) + len(set.ad[i].m) + len(set.ad[i].v)
    return elems * 4
