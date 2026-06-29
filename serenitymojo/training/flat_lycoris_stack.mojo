# training/flat_lycoris_stack.mojo -- LoKr/LoHa carrier orchestration for flat
# List[LoraAdapter] model surfaces.

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lokr_adapter import (
    LoKrAdapter, LoKrGrads, new_lokr_adapter, lokr_adamw,
)
from serenitymojo.training.lokr_stack import (
    lokr_carrier_adapter, lokr_carrier_r_eff, lokr_chain_carrier_grads,
)
from serenitymojo.training.lokr_save import NamedLoKr, save_lokr_peft
from serenitymojo.training.loha_adapter import (
    LoHaAdapter, LoHaGrads, new_loha_adapter, loha_adamw,
)
from serenitymojo.training.loha_stack import (
    loha_carrier_adapter, loha_carrier_r_eff, loha_chain_carrier_grads,
)
from serenitymojo.training.loha_save import NamedLoHa, save_loha_peft
from serenitymojo.training.dora_adapter import (
    DoRAAdapter, DoRAGrads, new_dora_adapter, dora_adamw,
)
from serenitymojo.training.dora_stack import (
    dora_carrier_adapter, dora_chain_carrier_grads,
)
from serenitymojo.training.dora_save import NamedDoRA, save_dora_onetrainer
from serenitymojo.training.oft_stack import (
    oft_ot_carrier_adapter, oft_ot_chain_carrier_grads,
)


def _inactive_flat_carrier(in_f: Int, out_f: Int) -> LoraAdapter:
    var a = List[Float32]()
    var b = List[Float32]()
    for _ in range(in_f):
        a.append(Float32(0.0))
    for _ in range(out_f):
        b.append(Float32(0.0))
    return LoraAdapter(
        a^, b^, 1, in_f, out_f, Float32(0.0),
        List[Float32](), List[Float32](), List[Float32](), List[Float32](),
    )


struct FlatLoKrSet(Copyable, Movable):
    var ad: List[LoKrAdapter]
    var active: List[Bool]
    var prefix: List[String]
    var rank: Int

    def __init__(
        out self, var ad: List[LoKrAdapter], var active: List[Bool],
        var prefix: List[String], rank: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.prefix = prefix^
        self.rank = rank


def empty_flat_lokr_set() -> FlatLoKrSet:
    return FlatLoKrSet(List[LoKrAdapter](), List[Bool](), List[String](), 0)


def build_flat_lokr_set(
    in_dims: List[Int], out_dims: List[Int], prefixes: List[String],
    rank: Int, alpha: Float32, factor: Int,
    decompose_both: Bool, full_matrix: Bool, seed: UInt64,
) raises -> FlatLoKrSet:
    if len(in_dims) != len(out_dims) or len(in_dims) != len(prefixes):
        raise Error("build_flat_lokr_set: dims/prefix count mismatch")
    var ad = List[LoKrAdapter]()
    var active = List[Bool]()
    var names = List[String]()
    var s = seed
    for i in range(len(in_dims)):
        ad.append(new_lokr_adapter(
            in_dims[i], out_dims[i], rank, alpha, factor, s,
            decompose_both, full_matrix,
        ))
        active.append(True)
        names.append(prefixes[i].copy())
        s += 1
    return FlatLoKrSet(ad^, active^, names^, rank)


def flat_lokr_carrier_list(set: FlatLoKrSet) raises -> List[LoraAdapter]:
    var out = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            out.append(lokr_carrier_adapter(set.ad[i]))
        else:
            out.append(_inactive_flat_carrier(set.ad[i].in_f, set.ad[i].out_f))
    return out^


def flat_lokr_carrier_total_bytes(set: FlatLoKrSet) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = lokr_carrier_r_eff(set.ad[i])
            elems += r * set.ad[i].in_f + set.ad[i].out_f * r
        else:
            elems += set.ad[i].in_f + set.ad[i].out_f
    return elems * 2


struct FlatLoKrGrads(Copyable, Movable):
    var g: List[LoKrGrads]

    def __init__(out self, var g: List[LoKrGrads]):
        self.g = g^


def _empty_lokr_grads_flat() -> LoKrGrads:
    return LoKrGrads(
        List[Float32](), List[Float32](), List[Float32](),
        List[Float32](), List[Float32](), List[Float32](), List[Float32](),
    )


def flat_lokr_chain_all(
    set: FlatLoKrSet, d_a: List[List[Float32]], d_b: List[List[Float32]],
) raises -> FlatLoKrGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("flat_lokr_chain_all: grad list count mismatch")
    var out = List[LoKrGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            out.append(lokr_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            out.append(_empty_lokr_grads_flat())
    return FlatLoKrGrads(out^)


def _lokr_sqsum(g: LoKrGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(g.d_w1)):
        s += Float64(g.d_w1[i]) * Float64(g.d_w1[i])
    for i in range(len(g.d_w1a)):
        s += Float64(g.d_w1a[i]) * Float64(g.d_w1a[i])
    for i in range(len(g.d_w1b)):
        s += Float64(g.d_w1b[i]) * Float64(g.d_w1b[i])
    for i in range(len(g.d_w2)):
        s += Float64(g.d_w2[i]) * Float64(g.d_w2[i])
    for i in range(len(g.d_w2a)):
        s += Float64(g.d_w2a[i]) * Float64(g.d_w2a[i])
    for i in range(len(g.d_w2b)):
        s += Float64(g.d_w2b[i]) * Float64(g.d_w2b[i])
    return s


def flat_lokr_grad_norm(g: FlatLoKrGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(g.g)):
        s += _lokr_sqsum(g.g[i])
    return sqrt(s)


def _scale_lokr(mut g: LoKrGrads, scale: Float32):
    for i in range(len(g.d_w1)):
        g.d_w1[i] *= scale
    for i in range(len(g.d_w1a)):
        g.d_w1a[i] *= scale
    for i in range(len(g.d_w1b)):
        g.d_w1b[i] *= scale
    for i in range(len(g.d_w2)):
        g.d_w2[i] *= scale
    for i in range(len(g.d_w2a)):
        g.d_w2a[i] *= scale
    for i in range(len(g.d_w2b)):
        g.d_w2b[i] *= scale


def flat_lokr_clip_grads(mut g: FlatLoKrGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(g.g)):
        _scale_lokr(g.g[i], clip_scale)


def flat_lokr_adamw_step(
    mut set: FlatLoKrSet, g: FlatLoKrGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            lokr_adamw(set.ad[i], g.g[i], t, lr, beta1, beta2, eps, weight_decay)


def flat_lokr_zero_leg_l1(set: FlatLoKrSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref lo = set.ad[i]
        if lo.w2_factored:
            for j in range(len(lo.w2b)):
                var v = Float64(lo.w2b[j].cast[DType.float32]())
                s += v if v >= 0.0 else -v
        else:
            for j in range(len(lo.w2)):
                var v = Float64(lo.w2[j].cast[DType.float32]())
                s += v if v >= 0.0 else -v
    return s


def save_flat_lokr(set: FlatLoKrSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoKr]()
    for i in range(len(set.ad)):
        if set.active[i]:
            named.append(NamedLoKr(set.prefix[i].copy(), set.ad[i].copy()))
    return save_lokr_peft(named, path, ctx)


def save_flat_lokr_pair(
    a: FlatLoKrSet, b: FlatLoKrSet, path: String, ctx: DeviceContext
) raises -> Int:
    var named = List[NamedLoKr]()
    for i in range(len(a.ad)):
        if a.active[i]:
            named.append(NamedLoKr(a.prefix[i].copy(), a.ad[i].copy()))
    for i in range(len(b.ad)):
        if b.active[i]:
            named.append(NamedLoKr(b.prefix[i].copy(), b.ad[i].copy()))
    return save_lokr_peft(named, path, ctx)


struct FlatLoHaSet(Copyable, Movable):
    var ad: List[LoHaAdapter]
    var active: List[Bool]
    var prefix: List[String]
    var rank: Int

    def __init__(
        out self, var ad: List[LoHaAdapter], var active: List[Bool],
        var prefix: List[String], rank: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.prefix = prefix^
        self.rank = rank


def empty_flat_loha_set() -> FlatLoHaSet:
    return FlatLoHaSet(List[LoHaAdapter](), List[Bool](), List[String](), 0)


def build_flat_loha_set(
    in_dims: List[Int], out_dims: List[Int], prefixes: List[String],
    rank: Int, alpha: Float32, seed: UInt64,
) raises -> FlatLoHaSet:
    if len(in_dims) != len(out_dims) or len(in_dims) != len(prefixes):
        raise Error("build_flat_loha_set: dims/prefix count mismatch")
    var ad = List[LoHaAdapter]()
    var active = List[Bool]()
    var names = List[String]()
    var s = seed
    for i in range(len(in_dims)):
        ad.append(new_loha_adapter(in_dims[i], out_dims[i], rank, alpha, s))
        active.append(True)
        names.append(prefixes[i].copy())
        s += 1
    return FlatLoHaSet(ad^, active^, names^, rank)


def flat_loha_carrier_list(set: FlatLoHaSet) raises -> List[LoraAdapter]:
    var out = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            out.append(loha_carrier_adapter(set.ad[i]))
        else:
            out.append(_inactive_flat_carrier(set.ad[i].in_f, set.ad[i].out_f))
    return out^


def flat_loha_carrier_total_bytes(set: FlatLoHaSet) -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = loha_carrier_r_eff(set.ad[i])
            elems += r * set.ad[i].in_f + set.ad[i].out_f * r
        else:
            elems += set.ad[i].in_f + set.ad[i].out_f
    return elems * 2


struct FlatLoHaGrads(Copyable, Movable):
    var g: List[LoHaGrads]

    def __init__(out self, var g: List[LoHaGrads]):
        self.g = g^


def _empty_loha_grads_flat() -> LoHaGrads:
    return LoHaGrads(
        List[Float32](), List[Float32](), List[Float32](), List[Float32](),
        List[Float32](),
    )


def flat_loha_chain_all(
    set: FlatLoHaSet, d_a: List[List[Float32]], d_b: List[List[Float32]],
) raises -> FlatLoHaGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("flat_loha_chain_all: grad list count mismatch")
    var out = List[LoHaGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            out.append(loha_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            out.append(_empty_loha_grads_flat())
    return FlatLoHaGrads(out^)


def _loha_sqsum(g: LoHaGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(g.d_w1a)):
        s += Float64(g.d_w1a[i]) * Float64(g.d_w1a[i])
    for i in range(len(g.d_w1b)):
        s += Float64(g.d_w1b[i]) * Float64(g.d_w1b[i])
    for i in range(len(g.d_w2a)):
        s += Float64(g.d_w2a[i]) * Float64(g.d_w2a[i])
    for i in range(len(g.d_w2b)):
        s += Float64(g.d_w2b[i]) * Float64(g.d_w2b[i])
    return s


def flat_loha_grad_norm(g: FlatLoHaGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(g.g)):
        s += _loha_sqsum(g.g[i])
    return sqrt(s)


def _scale_loha(mut g: LoHaGrads, scale: Float32):
    for i in range(len(g.d_w1a)):
        g.d_w1a[i] *= scale
    for i in range(len(g.d_w1b)):
        g.d_w1b[i] *= scale
    for i in range(len(g.d_w2a)):
        g.d_w2a[i] *= scale
    for i in range(len(g.d_w2b)):
        g.d_w2b[i] *= scale


def flat_loha_clip_grads(mut g: FlatLoHaGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(g.g)):
        _scale_loha(g.g[i], clip_scale)


def flat_loha_adamw_step(
    mut set: FlatLoHaSet, g: FlatLoHaGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            loha_adamw(set.ad[i], g.g[i], t, lr, beta1, beta2, eps, weight_decay)


def flat_loha_zero_leg_l1(set: FlatLoHaSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref lo = set.ad[i]
        for j in range(len(lo.w2a)):
            var v = Float64(lo.w2a[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    return s


def save_flat_loha(set: FlatLoHaSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoHa]()
    for i in range(len(set.ad)):
        if set.active[i]:
            named.append(NamedLoHa(set.prefix[i].copy(), set.ad[i].copy()))
    return save_loha_peft(named, path, ctx)


def save_flat_loha_pair(
    a: FlatLoHaSet, b: FlatLoHaSet, path: String, ctx: DeviceContext
) raises -> Int:
    var named = List[NamedLoHa]()
    for i in range(len(a.ad)):
        if a.active[i]:
            named.append(NamedLoHa(a.prefix[i].copy(), a.ad[i].copy()))
    for i in range(len(b.ad)):
        if b.active[i]:
            named.append(NamedLoHa(b.prefix[i].copy(), b.ad[i].copy()))
    return save_loha_peft(named, path, ctx)


# ── DoRA full-delta carrier for flat LoRA surfaces ───────────────────────────
struct FlatDoRASet(Copyable, Movable):
    var ad: List[DoRAAdapter]
    var w: List[List[Float32]]
    var active: List[Bool]
    var prefix: List[String]
    var in_dims: List[Int]
    var out_dims: List[Int]
    var rank: Int

    def __init__(
        out self, var ad: List[DoRAAdapter], var w: List[List[Float32]],
        var active: List[Bool], var prefix: List[String],
        var in_dims: List[Int], var out_dims: List[Int], rank: Int,
    ):
        self.ad = ad^
        self.w = w^
        self.active = active^
        self.prefix = prefix^
        self.in_dims = in_dims^
        self.out_dims = out_dims^
        self.rank = rank


def empty_flat_dora_set() -> FlatDoRASet:
    return FlatDoRASet(
        List[DoRAAdapter](), List[List[Float32]](), List[Bool](),
        List[String](), List[Int](), List[Int](), 0,
    )


def _flat_zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _dummy_flat_dora() raises -> DoRAAdapter:
    var w = List[Float32]()
    w.append(Float32(1.0))
    return new_dora_adapter(w^, 1, 1, 1, Float32(1.0), 0)


def build_flat_dora_set_from_weights(
    in_dims: List[Int], out_dims: List[Int], prefixes: List[String],
    weights: List[List[Float32]], active: List[Bool],
    rank: Int, alpha: Float32, seed: UInt64, wd_on_out: Bool = False,
) raises -> FlatDoRASet:
    if (
        len(in_dims) != len(out_dims)
        or len(in_dims) != len(prefixes)
        or len(in_dims) != len(weights)
        or len(in_dims) != len(active)
    ):
        raise Error("build_flat_dora_set_from_weights: dims/prefix/weight/active count mismatch")
    var ad = List[DoRAAdapter]()
    var w = List[List[Float32]]()
    var act = List[Bool]()
    var names = List[String]()
    var ins = List[Int]()
    var outs = List[Int]()
    var s = seed
    for i in range(len(in_dims)):
        if active[i]:
            if len(weights[i]) != in_dims[i] * out_dims[i]:
                raise Error("build_flat_dora_set_from_weights: w_orig numel mismatch")
            var wi = weights[i].copy()
            ad.append(new_dora_adapter(
                wi.copy(), in_dims[i], out_dims[i], rank, alpha, s,
                Float32(1.0e-7), wd_on_out,
            ))
            w.append(wi^)
        else:
            ad.append(_dummy_flat_dora())
            var w1 = List[Float32]()
            w1.append(Float32(1.0))
            w.append(w1^)
        act.append(active[i])
        names.append(prefixes[i].copy())
        ins.append(in_dims[i])
        outs.append(out_dims[i])
        s += 1
    return FlatDoRASet(ad^, w^, act^, names^, ins^, outs^, rank)


def flat_dora_carrier_list(set: FlatDoRASet) raises -> List[LoraAdapter]:
    var out = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            out.append(dora_carrier_adapter(set.ad[i], set.w[i]))
        else:
            out.append(_inactive_flat_carrier(set.in_dims[i], set.out_dims[i]))
    return out^


def flat_dora_carrier_total_bytes(set: FlatDoRASet) -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var inf = set.in_dims[i]
            elems += inf * inf + set.out_dims[i] * inf
        else:
            elems += set.in_dims[i] + set.out_dims[i]
    return elems * 2


def flat_dora_preflight(set: FlatDoRASet, budget_bytes: Int) raises:
    var b = flat_dora_carrier_total_bytes(set)
    if b > budget_bytes:
        raise Error(
            String("Flat DoRA carrier needs ") + String(b)
            + String(" bytes (> budget ") + String(budget_bytes)
            + String("). Use a smaller target set or direct W_eff lowering.")
        )


struct FlatDoRAGrads(Copyable, Movable):
    var g: List[DoRAGrads]

    def __init__(out self, var g: List[DoRAGrads]):
        self.g = g^


def _empty_flat_dora_grads() -> DoRAGrads:
    return DoRAGrads(List[Float32](), List[Float32](), List[Float32](), List[Float32]())


def flat_dora_chain_all(
    set: FlatDoRASet, d_a: List[List[Float32]], d_b: List[List[Float32]],
) raises -> FlatDoRAGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("flat_dora_chain_all: grad list count mismatch")
    var out = List[DoRAGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            out.append(dora_chain_carrier_grads(set.ad[i], set.w[i], d_a[i], d_b[i]))
        else:
            out.append(_empty_flat_dora_grads())
    return FlatDoRAGrads(out^)


def _flat_dora_grads_sqsum(g: DoRAGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(g.d_a)):
        s += Float64(g.d_a[i]) * Float64(g.d_a[i])
    for i in range(len(g.d_b)):
        s += Float64(g.d_b[i]) * Float64(g.d_b[i])
    for i in range(len(g.d_m)):
        s += Float64(g.d_m[i]) * Float64(g.d_m[i])
    return s


def flat_dora_grad_norm(g: FlatDoRAGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(g.g)):
        s += _flat_dora_grads_sqsum(g.g[i])
    return sqrt(s)


def _flat_scale_dora(mut g: DoRAGrads, scale: Float32):
    for i in range(len(g.d_a)):
        g.d_a[i] *= scale
    for i in range(len(g.d_b)):
        g.d_b[i] *= scale
    for i in range(len(g.d_m)):
        g.d_m[i] *= scale


def flat_dora_clip_grads(mut g: FlatDoRAGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(g.g)):
        _flat_scale_dora(g.g[i], clip_scale)


def flat_dora_adamw_step(
    mut set: FlatDoRASet, g: FlatDoRAGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            dora_adamw(set.ad[i], g.g[i], t, lr, beta1, beta2, eps, weight_decay)


def flat_dora_zero_leg_l1(set: FlatDoRASet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref d = set.ad[i]
        for j in range(len(d.b)):
            var v = Float64(d.b[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    return s


def save_flat_dora(set: FlatDoRASet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedDoRA]()
    for i in range(len(set.ad)):
        if set.active[i]:
            named.append(NamedDoRA(set.prefix[i].copy(), set.ad[i].copy()))
    return save_dora_onetrainer(named, path, ctx)


# ── OneTrainer OFT full-delta carrier for flat LoRA surfaces ─────────────────
struct FlatOFTSlot(Copyable, Movable):
    var vec: List[Float32]
    var w: List[Float32]
    var in_f: Int
    var out_f: Int
    var b: Int
    var r: Int
    var m: List[Float32]
    var v: List[Float32]

    def __init__(
        out self, var vec: List[Float32], var w: List[Float32],
        in_f: Int, out_f: Int, b: Int, r: Int,
        var m: List[Float32], var v: List[Float32],
    ):
        self.vec = vec^
        self.w = w^
        self.in_f = in_f
        self.out_f = out_f
        self.b = b
        self.r = r
        self.m = m^
        self.v = v^


struct FlatOFTSet(Copyable, Movable):
    var ad: List[FlatOFTSlot]
    var active: List[Bool]
    var prefix: List[String]
    var in_dims: List[Int]
    var out_dims: List[Int]
    var block_size: Int

    def __init__(
        out self, var ad: List[FlatOFTSlot], var active: List[Bool],
        var prefix: List[String], var in_dims: List[Int], var out_dims: List[Int],
        block_size: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.prefix = prefix^
        self.in_dims = in_dims^
        self.out_dims = out_dims^
        self.block_size = block_size


def empty_flat_oft_set() -> FlatOFTSet:
    return FlatOFTSet(
        List[FlatOFTSlot](), List[Bool](), List[String](),
        List[Int](), List[Int](), 0,
    )


def _dummy_flat_oft_slot() -> FlatOFTSlot:
    return FlatOFTSlot(_flat_zeros(1), _flat_zeros(1), 1, 1, 1, 1, _flat_zeros(1), _flat_zeros(1))


def _make_flat_oft_slot_with_w(
    var w: List[Float32], in_f: Int, out_f: Int, block_size: Int,
) raises -> FlatOFTSlot:
    if in_f % block_size != 0:
        raise Error(String("Flat OFT: in_f ") + String(in_f) + String(" not divisible by block_size ") + String(block_size))
    if len(w) != out_f * in_f:
        raise Error("Flat OFT: base weight numel mismatch")
    var b = block_size
    var r = in_f // b
    var ne = b * (b - 1) // 2
    return FlatOFTSlot(_flat_zeros(r * ne), w^, in_f, out_f, b, r, _flat_zeros(r * ne), _flat_zeros(r * ne))


def build_flat_oft_set_from_weights(
    in_dims: List[Int], out_dims: List[Int], prefixes: List[String],
    weights: List[List[Float32]], active: List[Bool], block_size: Int,
) raises -> FlatOFTSet:
    if (
        len(in_dims) != len(out_dims)
        or len(in_dims) != len(prefixes)
        or len(in_dims) != len(weights)
        or len(in_dims) != len(active)
    ):
        raise Error("build_flat_oft_set_from_weights: dims/prefix/weight/active count mismatch")
    var ad = List[FlatOFTSlot]()
    var act = List[Bool]()
    var names = List[String]()
    var ins = List[Int]()
    var outs = List[Int]()
    for i in range(len(in_dims)):
        if active[i]:
            ad.append(_make_flat_oft_slot_with_w(weights[i].copy(), in_dims[i], out_dims[i], block_size))
        else:
            ad.append(_dummy_flat_oft_slot())
        act.append(active[i])
        names.append(prefixes[i].copy())
        ins.append(in_dims[i])
        outs.append(out_dims[i])
    return FlatOFTSet(ad^, act^, names^, ins^, outs^, block_size)


def flat_oft_carrier_list(set: FlatOFTSet) raises -> List[LoraAdapter]:
    var out = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ref sl = set.ad[i]
            out.append(oft_ot_carrier_adapter(sl.vec, sl.w, sl.in_f, sl.out_f, sl.b, sl.r))
        else:
            out.append(_inactive_flat_carrier(set.in_dims[i], set.out_dims[i]))
    return out^


def flat_oft_carrier_total_bytes(set: FlatOFTSet) -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var inf = set.in_dims[i]
            elems += inf * inf + set.out_dims[i] * inf
        else:
            elems += set.in_dims[i] + set.out_dims[i]
    return elems * 2


def flat_oft_preflight(set: FlatOFTSet, budget_bytes: Int) raises:
    var b = flat_oft_carrier_total_bytes(set)
    if b > budget_bytes:
        raise Error(
            String("Flat OFT carrier needs ") + String(b)
            + String(" bytes (> budget ") + String(budget_bytes)
            + String("). Use a smaller target set or direct W_eff lowering.")
        )


struct FlatOFTGrads(Copyable, Movable):
    var g: List[List[Float32]]

    def __init__(out self, var g: List[List[Float32]]):
        self.g = g^


def flat_oft_chain_all(
    set: FlatOFTSet, d_a: List[List[Float32]], d_b: List[List[Float32]],
) raises -> FlatOFTGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("flat_oft_chain_all: grad list count mismatch")
    var out = List[List[Float32]]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ref sl = set.ad[i]
            out.append(oft_ot_chain_carrier_grads(
                sl.vec, sl.w, d_a[i], d_b[i], sl.in_f, sl.out_f, sl.b, sl.r,
            ))
        else:
            out.append(List[Float32]())
    return FlatOFTGrads(out^)


def _flat_vec_sqsum(g: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(g)):
        s += Float64(g[i]) * Float64(g[i])
    return s


def flat_oft_grad_norm(g: FlatOFTGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(g.g)):
        s += _flat_vec_sqsum(g.g[i])
    return sqrt(s)


def _flat_vec_scale(mut g: List[Float32], scale: Float32):
    for i in range(len(g)):
        g[i] *= scale


def flat_oft_clip_grads(mut g: FlatOFTGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(g.g)):
        _flat_vec_scale(g.g[i], clip_scale)


def _flat_oft_vec_adamw(
    mut vec: List[Float32], d_vec: List[Float32],
    mut m: List[Float32], mut v: List[Float32],
    t: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32,
) raises:
    if len(vec) != len(d_vec) or len(vec) != len(m) or len(vec) != len(v):
        raise Error("Flat OFT AdamW: len mismatch")
    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p
    for i in range(len(vec)):
        var gv = d_vec[i]
        var mi = beta1 * m[i] + (Float32(1.0) - beta1) * gv
        var vi = beta2 * v[i] + (Float32(1.0) - beta2) * gv * gv
        m[i] = mi
        v[i] = vi
        var pv = vec[i]
        if weight_decay > 0.0:
            pv = pv * (Float32(1.0) - lr * weight_decay)
        vec[i] = pv - lr * (mi / bc1) / (sqrt(vi / bc2) + eps)


def flat_oft_adamw_step(
    mut set: FlatOFTSet, g: FlatOFTGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            _flat_oft_vec_adamw(
                set.ad[i].vec, g.g[i], set.ad[i].m, set.ad[i].v,
                t, lr, beta1, beta2, eps, weight_decay,
            )


def flat_oft_vec_l1(set: FlatOFTSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref sl = set.ad[i]
        for j in range(len(sl.vec)):
            var v = Float64(sl.vec[j])
            s += v if v >= 0.0 else -v
    return s


def _flat_f32_2d(var values: List[Float32], rows: Int, cols: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return Tensor.from_host(values^, sh^, STDtype.F32, ctx)


def save_flat_oft(set: FlatOFTSet, path: String, ctx: DeviceContext) raises -> Int:
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    var nmods = 0
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref sl = set.ad[i]
        var ne = sl.b * (sl.b - 1) // 2
        names.append(set.prefix[i].copy() + String(".oft_R.weight"))
        tensors.append(ArcPointer(_flat_f32_2d(sl.vec.copy(), sl.r, ne, ctx)))
        nmods += 1
    if nmods == 0:
        raise Error("save_flat_oft: refusing to write an empty OFT file")
    save_safetensors(names, tensors, path, ctx)
    return nmods
