# models/ernie/ernie_lycoris_stack.mojo -- LoKr/LoHa carrier dispatch for ERNIE.
#
# ERNIE trains a flat ErnieLoraSet: num_layers * 7 adapters in lora_block slot
# order. Additive LyCORIS families can use the existing LoRA stack by
# materializing plain-LoRA carriers and chaining dA/dB back to LyCORIS masters.

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lokr_adapter import (
    LoKrAdapter, LoKrGrads, new_lokr_adapter, lokr_adamw,
)
from serenitymojo.training.lokr_stack import (
    lokr_carrier_adapter, lokr_carrier_r_eff, lokr_chain_carrier_grads,
    _inactive_carrier, _dummy_lokr, _empty_lokr_grads,
    _grads_sqsum, _grads_scale,
)
from serenitymojo.training.loha_adapter import (
    LoHaAdapter, LoHaGrads, new_loha_adapter, loha_adamw,
)
from serenitymojo.training.loha_stack import (
    loha_carrier_adapter, loha_carrier_r_eff, loha_chain_carrier_grads,
    _dummy_loha, _empty_loha_grads, _loha_grads_sqsum, _loha_grads_scale,
)
from serenitymojo.training.lokr_save import NamedLoKr, save_lokr_peft
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
from serenitymojo.training.flat_direct_lycoris_stack import (
    FlatDirectDoRASet, FlatDirectOFTSet,
    empty_flat_direct_dora_set, empty_flat_direct_oft_set,
    flat_direct_dora_append_from_weight, flat_direct_oft_append,
    flat_direct_dora_trainable_bytes, flat_direct_oft_trainable_bytes,
)
from serenitymojo.models.ernie.lora_block import (
    ERNIE_SLOTS, SLOT_Q, SLOT_K, SLOT_V, SLOT_O, SLOT_GATE, SLOT_UP, SLOT_DOWN,
)
from serenitymojo.models.ernie.weights import ErnieBlockWeights
from serenitymojo.models.ernie.ernie_stack_lora import ErnieLoraSet


comptime ERNIE_LYCORIS_TGT_ATTN = 1
comptime ERNIE_LYCORIS_TGT_ALL = 2


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def ernie_lycoris_slot_dims(slot: Int, D: Int, F: Int) raises -> Tuple[Int, Int]:
    var s = slot % ERNIE_SLOTS
    if s == SLOT_GATE or s == SLOT_UP:
        return (D, F)
    if s == SLOT_DOWN:
        return (F, D)
    return (D, D)


def _ernie_slot_is_attn(slot: Int) -> Bool:
    var s = slot % ERNIE_SLOTS
    return s == SLOT_Q or s == SLOT_K or s == SLOT_V or s == SLOT_O


def _ernie_slot_targeted(slot: Int, targets: Int) -> Bool:
    if _ernie_slot_is_attn(slot):
        return targets >= ERNIE_LYCORIS_TGT_ATTN
    return targets >= ERNIE_LYCORIS_TGT_ALL


def ernie_full_delta_carrier_bytes_estimate(
    num_layers: Int, D: Int, F: Int, targets: Int,
) raises -> Int:
    if targets < ERNIE_LYCORIS_TGT_ATTN or targets > 3:
        raise Error("ernie_full_delta_carrier_bytes_estimate: targets must be 1(attn)|2(all)|3(all)")
    var elems = 0
    for _bi in range(num_layers):
        for slot in range(ERNIE_SLOTS):
            var dims = ernie_lycoris_slot_dims(slot, D, F)
            if _ernie_slot_targeted(slot, targets):
                elems += dims[0] * dims[0] + dims[1] * dims[0]
            else:
                elems += dims[0] + dims[1]
    return elems * 2


def ernie_full_delta_preflight(
    num_layers: Int, D: Int, F: Int, targets: Int, budget_bytes: Int,
) raises -> Int:
    var b = ernie_full_delta_carrier_bytes_estimate(num_layers, D, F, targets)
    if b > budget_bytes:
        raise Error(
            String("ERNIE full-delta carrier needs ") + String(b)
            + String(" bytes (> budget ") + String(budget_bytes)
            + String("). Use attention-only targets or the direct W_eff path.")
        )
    return b


def _ernie_slot_factor(slot: Int, factor: Int, factor_attn: Int, factor_ff: Int) -> Int:
    if _ernie_slot_is_attn(slot):
        return factor_attn if factor_attn != 0 else factor
    return factor_ff if factor_ff != 0 else factor


def _ernie_prefix(block_idx: Int, slot: Int) -> String:
    var b = String("transformer.layers.") + String(block_idx)
    if slot == SLOT_Q:
        return b + ".self_attention.to_q"
    if slot == SLOT_K:
        return b + ".self_attention.to_k"
    if slot == SLOT_V:
        return b + ".self_attention.to_v"
    if slot == SLOT_O:
        return b + ".self_attention.to_out.0"
    if slot == SLOT_GATE:
        return b + ".mlp.gate_proj"
    if slot == SLOT_UP:
        return b + ".mlp.up_proj"
    return b + ".mlp.linear_fc2"


struct ErnieLoKrSet(Copyable, Movable):
    var ad: List[LoKrAdapter]
    var active: List[Bool]
    var num_layers: Int
    var rank: Int

    def __init__(
        out self, var ad: List[LoKrAdapter], var active: List[Bool],
        num_layers: Int, rank: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.num_layers = num_layers
        self.rank = rank


def empty_ernie_lokr_set() -> ErnieLoKrSet:
    return ErnieLoKrSet(List[LoKrAdapter](), List[Bool](), 0, 0)


def build_ernie_lokr_set(
    num_layers: Int, D: Int, F: Int,
    rank: Int, alpha: Float32,
    factor: Int, factor_attn: Int, factor_ff: Int,
    decompose_both: Bool, full_matrix: Bool,
    targets: Int, seed: UInt64,
) raises -> ErnieLoKrSet:
    if targets < ERNIE_LYCORIS_TGT_ATTN or targets > 3:
        raise Error("build_ernie_lokr_set: targets must be 1(attn)|2(all)|3(all)")
    var ad = List[LoKrAdapter]()
    var active = List[Bool]()
    var s = seed
    for _bi in range(num_layers):
        for slot in range(ERNIE_SLOTS):
            if _ernie_slot_targeted(slot, targets):
                var dims = ernie_lycoris_slot_dims(slot, D, F)
                var f = _ernie_slot_factor(slot, factor, factor_attn, factor_ff)
                ad.append(new_lokr_adapter(
                    dims[0], dims[1], rank, alpha, f, s,
                    decompose_both, full_matrix,
                ))
                active.append(True)
            else:
                ad.append(_dummy_lokr())
                active.append(False)
            s += 1
    return ErnieLoKrSet(ad^, active^, num_layers, rank)


def ernie_lokr_carrier_set(set: ErnieLoKrSet, D: Int, F: Int) raises -> ErnieLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ad.append(lokr_carrier_adapter(set.ad[i]))
        else:
            var dims = ernie_lycoris_slot_dims(i % ERNIE_SLOTS, D, F)
            ad.append(_inactive_carrier(dims[0], dims[1]))
    return ErnieLoraSet(ad^, set.num_layers, set.rank)


def ernie_lokr_carrier_total_bytes(set: ErnieLoKrSet, D: Int, F: Int) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = lokr_carrier_r_eff(set.ad[i])
            elems += r * set.ad[i].in_f + set.ad[i].out_f * r
        else:
            var dims = ernie_lycoris_slot_dims(i % ERNIE_SLOTS, D, F)
            elems += dims[0] + dims[1]
    return elems * 2


struct ErnieLoKrGrads(Movable):
    var g: List[LoKrGrads]

    def __init__(out self, var g: List[LoKrGrads]):
        self.g = g^


def ernie_lokr_chain_all(
    set: ErnieLoKrSet, d_a: List[List[Float32]], d_b: List[List[Float32]]
) raises -> ErnieLoKrGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("ernie_lokr_chain_all: grad list count mismatch")
    var g = List[LoKrGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            g.append(lokr_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            g.append(_empty_lokr_grads())
    return ErnieLoKrGrads(g^)


def ernie_lokr_grad_norm(grads: ErnieLoKrGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _grads_sqsum(grads.g[i])
    return sqrt(s)


def ernie_lokr_clip_grads(mut grads: ErnieLoKrGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _grads_scale(grads.g[i], clip_scale)


def ernie_lokr_adamw_step(
    mut set: ErnieLoKrSet, grads: ErnieLoKrGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            lokr_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


def ernie_lokr_zero_leg_l1(set: ErnieLoKrSet) -> Float64:
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


def save_ernie_lokr(set: ErnieLoKrSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoKr]()
    for bi in range(set.num_layers):
        for slot in range(ERNIE_SLOTS):
            var flat = bi * ERNIE_SLOTS + slot
            if set.active[flat]:
                named.append(NamedLoKr(_ernie_prefix(bi, slot), set.ad[flat].copy()))
    return save_lokr_peft(named, path, ctx)


struct ErnieLoHaSet(Copyable, Movable):
    var ad: List[LoHaAdapter]
    var active: List[Bool]
    var num_layers: Int
    var rank: Int

    def __init__(
        out self, var ad: List[LoHaAdapter], var active: List[Bool],
        num_layers: Int, rank: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.num_layers = num_layers
        self.rank = rank


def empty_ernie_loha_set() -> ErnieLoHaSet:
    return ErnieLoHaSet(List[LoHaAdapter](), List[Bool](), 0, 0)


def build_ernie_loha_set(
    num_layers: Int, D: Int, F: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
) raises -> ErnieLoHaSet:
    if targets < ERNIE_LYCORIS_TGT_ATTN or targets > 3:
        raise Error("build_ernie_loha_set: targets must be 1(attn)|2(all)|3(all)")
    var ad = List[LoHaAdapter]()
    var active = List[Bool]()
    var s = seed
    for _bi in range(num_layers):
        for slot in range(ERNIE_SLOTS):
            if _ernie_slot_targeted(slot, targets):
                var dims = ernie_lycoris_slot_dims(slot, D, F)
                ad.append(new_loha_adapter(dims[0], dims[1], rank, alpha, s))
                active.append(True)
            else:
                ad.append(_dummy_loha())
                active.append(False)
            s += 1
    return ErnieLoHaSet(ad^, active^, num_layers, rank)


def ernie_loha_carrier_set(set: ErnieLoHaSet, D: Int, F: Int) raises -> ErnieLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ad.append(loha_carrier_adapter(set.ad[i]))
        else:
            var dims = ernie_lycoris_slot_dims(i % ERNIE_SLOTS, D, F)
            ad.append(_inactive_carrier(dims[0], dims[1]))
    return ErnieLoraSet(ad^, set.num_layers, set.rank)


def ernie_loha_carrier_total_bytes(set: ErnieLoHaSet, D: Int, F: Int) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = loha_carrier_r_eff(set.ad[i])
            elems += r * set.ad[i].in_f + set.ad[i].out_f * r
        else:
            var dims = ernie_lycoris_slot_dims(i % ERNIE_SLOTS, D, F)
            elems += dims[0] + dims[1]
    return elems * 2


struct ErnieLoHaGrads(Movable):
    var g: List[LoHaGrads]

    def __init__(out self, var g: List[LoHaGrads]):
        self.g = g^


def ernie_loha_chain_all(
    set: ErnieLoHaSet, d_a: List[List[Float32]], d_b: List[List[Float32]]
) raises -> ErnieLoHaGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("ernie_loha_chain_all: grad list count mismatch")
    var g = List[LoHaGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            g.append(loha_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            g.append(_empty_loha_grads())
    return ErnieLoHaGrads(g^)


def ernie_loha_grad_norm(grads: ErnieLoHaGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _loha_grads_sqsum(grads.g[i])
    return sqrt(s)


def ernie_loha_clip_grads(mut grads: ErnieLoHaGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _loha_grads_scale(grads.g[i], clip_scale)


def ernie_loha_adamw_step(
    mut set: ErnieLoHaSet, grads: ErnieLoHaGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            loha_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


def ernie_loha_zero_leg_l1(set: ErnieLoHaSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref lo = set.ad[i]
        for j in range(len(lo.w2a)):
            var v = Float64(lo.w2a[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    return s


def save_ernie_loha(set: ErnieLoHaSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoHa]()
    for bi in range(set.num_layers):
        for slot in range(ERNIE_SLOTS):
            var flat = bi * ERNIE_SLOTS + slot
            if set.active[flat]:
                named.append(NamedLoHa(_ernie_prefix(bi, slot), set.ad[flat].copy()))
    return save_loha_peft(named, path, ctx)


# ── DoRA full-delta carrier ──────────────────────────────────────────────────
struct ErnieDoRASet(Copyable, Movable):
    var ad: List[DoRAAdapter]
    var w: List[List[Float32]]
    var active: List[Bool]
    var num_layers: Int
    var rank: Int

    def __init__(
        out self, var ad: List[DoRAAdapter], var w: List[List[Float32]],
        var active: List[Bool], num_layers: Int, rank: Int,
    ):
        self.ad = ad^
        self.w = w^
        self.active = active^
        self.num_layers = num_layers
        self.rank = rank


def empty_ernie_dora_set() raises -> ErnieDoRASet:
    return ErnieDoRASet(List[DoRAAdapter](), List[List[Float32]](), List[Bool](), 0, 0)


def _dummy_dora() raises -> DoRAAdapter:
    var w = List[Float32]()
    w.append(Float32(1.0))
    return new_dora_adapter(w^, 1, 1, 1, Float32(1.0), 0)


def _synth_w(out_f: Int, in_f: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(out_f * in_f):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * Float32(0.5))
    return out^


def _ernie_weight_key(block_idx: Int, slot: Int) raises -> String:
    var bp = String("layers.") + String(block_idx) + String(".")
    if slot == SLOT_Q:
        return bp + String("self_attention.to_q.weight")
    if slot == SLOT_K:
        return bp + String("self_attention.to_k.weight")
    if slot == SLOT_V:
        return bp + String("self_attention.to_v.weight")
    if slot == SLOT_O:
        return bp + String("self_attention.to_out.0.weight")
    if slot == SLOT_GATE:
        return bp + String("mlp.gate_proj.weight")
    if slot == SLOT_UP:
        return bp + String("mlp.up_proj.weight")
    if slot == SLOT_DOWN:
        return bp + String("mlp.linear_fc2.weight")
    raise Error(String("_ernie_weight_key: bad slot ") + String(slot))


def _read_ernie_weight_f32(
    st: ShardedSafeTensors, block_idx: Int, slot: Int, D: Int, F: Int,
) raises -> List[Float32]:
    var key = _ernie_weight_key(block_idx, slot)
    var dims = ernie_lycoris_slot_dims(slot, D, F)
    var info = st.tensor_info(key)
    if info.dtype != STDtype.BF16:
        raise Error(String("ERNIE LyCORIS base weight: expected BF16 for ") + key)
    if len(info.shape) != 2:
        raise Error(String("ERNIE LyCORIS base weight: expected 2D for ") + key)
    if Int(info.shape[0]) != dims[1] or Int(info.shape[1]) != dims[0]:
        raise Error(String("ERNIE LyCORIS base weight: shape mismatch for ") + key)
    var bytes = st.tensor_bytes(key)
    var bp = bytes.unsafe_ptr().bitcast[BFloat16]()
    var out = List[Float32]()
    for i in range(dims[0] * dims[1]):
        out.append(bp[i].cast[DType.float32]())
    return out^


def build_ernie_dora_set(
    num_layers: Int, D: Int, F: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
    wd_on_out: Bool = False,
) raises -> ErnieDoRASet:
    if targets < ERNIE_LYCORIS_TGT_ATTN or targets > 3:
        raise Error("build_ernie_dora_set: targets must be 1(attn)|2(all)|3(all)")
    var ad = List[DoRAAdapter]()
    var weights = List[List[Float32]]()
    var active = List[Bool]()
    var s = seed
    for _bi in range(num_layers):
        for slot in range(ERNIE_SLOTS):
            if _ernie_slot_targeted(slot, targets):
                var dims = ernie_lycoris_slot_dims(slot, D, F)
                var w = _synth_w(dims[1], dims[0], s * 131 + 7)
                ad.append(new_dora_adapter(
                    w.copy(), dims[0], dims[1], rank, alpha, s,
                    Float32(1.0e-7), wd_on_out,
                ))
                weights.append(w^)
                active.append(True)
            else:
                ad.append(_dummy_dora())
                var w1 = List[Float32]()
                w1.append(Float32(1.0))
                weights.append(w1^)
                active.append(False)
            s += 1
    return ErnieDoRASet(ad^, weights^, active^, num_layers, rank)


def build_ernie_dora_set_from_checkpoint(
    st: ShardedSafeTensors, num_layers: Int, D: Int, F: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
    wd_on_out: Bool = False,
) raises -> ErnieDoRASet:
    if targets < ERNIE_LYCORIS_TGT_ATTN or targets > 3:
        raise Error("build_ernie_dora_set_from_checkpoint: targets must be 1(attn)|2(all)|3(all)")
    var ad = List[DoRAAdapter]()
    var weights = List[List[Float32]]()
    var active = List[Bool]()
    var s = seed
    for bi in range(num_layers):
        for slot in range(ERNIE_SLOTS):
            if _ernie_slot_targeted(slot, targets):
                var dims = ernie_lycoris_slot_dims(slot, D, F)
                var w = _read_ernie_weight_f32(st, bi, slot, D, F)
                ad.append(new_dora_adapter(
                    w.copy(), dims[0], dims[1], rank, alpha, s,
                    Float32(1.0e-7), wd_on_out,
                ))
                weights.append(w^)
                active.append(True)
            else:
                ad.append(_dummy_dora())
                var w1 = List[Float32]()
                w1.append(Float32(1.0))
                weights.append(w1^)
                active.append(False)
            s += 1
    return ErnieDoRASet(ad^, weights^, active^, num_layers, rank)


def ernie_dora_carrier_set(set: ErnieDoRASet, D: Int, F: Int) raises -> ErnieLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ad.append(dora_carrier_adapter(set.ad[i], set.w[i]))
        else:
            var dims = ernie_lycoris_slot_dims(i % ERNIE_SLOTS, D, F)
            ad.append(_inactive_carrier(dims[0], dims[1]))
    return ErnieLoraSet(ad^, set.num_layers, set.rank)


def ernie_dora_carrier_total_bytes(set: ErnieDoRASet, D: Int, F: Int) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var inf = set.ad[i].in_f
            elems += inf * inf + set.ad[i].out_f * inf
        else:
            var dims = ernie_lycoris_slot_dims(i % ERNIE_SLOTS, D, F)
            elems += dims[0] + dims[1]
    return elems * 2


def ernie_dora_preflight(set: ErnieDoRASet, D: Int, F: Int, budget_bytes: Int) raises:
    var b = ernie_dora_carrier_total_bytes(set, D, F)
    if b > budget_bytes:
        raise Error(
            String("ERNIE DoRA carrier needs ") + String(b)
            + String(" bytes (> budget ") + String(budget_bytes)
            + String("). Use attention-only targets or the direct W_eff path.")
        )


struct ErnieDoRAGrads(Movable):
    var g: List[DoRAGrads]

    def __init__(out self, var g: List[DoRAGrads]):
        self.g = g^


def _empty_dora_grads() -> DoRAGrads:
    return DoRAGrads(List[Float32](), List[Float32](), List[Float32](), List[Float32]())


def ernie_dora_chain_all(
    set: ErnieDoRASet, d_a: List[List[Float32]], d_b: List[List[Float32]],
) raises -> ErnieDoRAGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("ernie_dora_chain_all: grad list count mismatch")
    var out = List[DoRAGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            out.append(dora_chain_carrier_grads(set.ad[i], set.w[i], d_a[i], d_b[i]))
        else:
            out.append(_empty_dora_grads())
    return ErnieDoRAGrads(out^)


def _dora_grads_sqsum(g: DoRAGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(g.d_a)):
        s += Float64(g.d_a[i]) * Float64(g.d_a[i])
    for i in range(len(g.d_b)):
        s += Float64(g.d_b[i]) * Float64(g.d_b[i])
    for i in range(len(g.d_m)):
        s += Float64(g.d_m[i]) * Float64(g.d_m[i])
    return s


def ernie_dora_grad_norm(grads: ErnieDoRAGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _dora_grads_sqsum(grads.g[i])
    return sqrt(s)


def _dora_grads_scale(mut g: DoRAGrads, scale: Float32):
    for i in range(len(g.d_a)):
        g.d_a[i] *= scale
    for i in range(len(g.d_b)):
        g.d_b[i] *= scale
    for i in range(len(g.d_m)):
        g.d_m[i] *= scale


def ernie_dora_clip_grads(mut grads: ErnieDoRAGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _dora_grads_scale(grads.g[i], clip_scale)


def ernie_dora_adamw_step(
    mut set: ErnieDoRASet, grads: ErnieDoRAGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            dora_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


def ernie_dora_zero_leg_l1(set: ErnieDoRASet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref lo = set.ad[i]
        for j in range(len(lo.b)):
            var v = Float64(lo.b[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    return s


def save_ernie_dora(set: ErnieDoRASet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedDoRA]()
    for bi in range(set.num_layers):
        for slot in range(ERNIE_SLOTS):
            var flat = bi * ERNIE_SLOTS + slot
            if set.active[flat]:
                named.append(NamedDoRA(_ernie_prefix(bi, slot), set.ad[flat].copy()))
    return save_dora_onetrainer(named, path, ctx)


# ── OneTrainer OFT full-delta carrier ────────────────────────────────────────
struct ErnieOFTSlot(Copyable, Movable):
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


struct ErnieOFTSet(Copyable, Movable):
    var ad: List[ErnieOFTSlot]
    var active: List[Bool]
    var num_layers: Int
    var block_size: Int

    def __init__(
        out self, var ad: List[ErnieOFTSlot], var active: List[Bool],
        num_layers: Int, block_size: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.num_layers = num_layers
        self.block_size = block_size


def empty_ernie_oft_set() -> ErnieOFTSet:
    return ErnieOFTSet(List[ErnieOFTSlot](), List[Bool](), 0, 0)


def _dummy_oft_slot() -> ErnieOFTSlot:
    return ErnieOFTSlot(_zeros(1), _zeros(1), 1, 1, 1, 1, _zeros(1), _zeros(1))


def _make_oft_slot_with_w(
    var w: List[Float32], in_f: Int, out_f: Int, block_size: Int,
) raises -> ErnieOFTSlot:
    if in_f % block_size != 0:
        raise Error(String("ERNIE OFT: in_f ") + String(in_f) + String(" not divisible by block_size ") + String(block_size))
    if len(w) != out_f * in_f:
        raise Error("ERNIE OFT: base weight numel mismatch")
    var b = block_size
    var r = in_f // b
    var ne = b * (b - 1) // 2
    return ErnieOFTSlot(_zeros(r * ne), w^, in_f, out_f, b, r, _zeros(r * ne), _zeros(r * ne))


def build_ernie_oft_set(
    num_layers: Int, D: Int, F: Int,
    block_size: Int, targets: Int, seed: UInt64,
) raises -> ErnieOFTSet:
    if targets < ERNIE_LYCORIS_TGT_ATTN or targets > 3:
        raise Error("build_ernie_oft_set: targets must be 1(attn)|2(all)|3(all)")
    var ad = List[ErnieOFTSlot]()
    var active = List[Bool]()
    var s = seed
    for _bi in range(num_layers):
        for slot in range(ERNIE_SLOTS):
            if _ernie_slot_targeted(slot, targets):
                var dims = ernie_lycoris_slot_dims(slot, D, F)
                var w = _synth_w(dims[1], dims[0], s * 131 + 7)
                ad.append(_make_oft_slot_with_w(w^, dims[0], dims[1], block_size))
                active.append(True)
            else:
                ad.append(_dummy_oft_slot())
                active.append(False)
            s += 1
    return ErnieOFTSet(ad^, active^, num_layers, block_size)


def build_ernie_oft_set_from_checkpoint(
    st: ShardedSafeTensors, num_layers: Int, D: Int, F: Int,
    block_size: Int, targets: Int,
) raises -> ErnieOFTSet:
    if targets < ERNIE_LYCORIS_TGT_ATTN or targets > 3:
        raise Error("build_ernie_oft_set_from_checkpoint: targets must be 1(attn)|2(all)|3(all)")
    var ad = List[ErnieOFTSlot]()
    var active = List[Bool]()
    for bi in range(num_layers):
        for slot in range(ERNIE_SLOTS):
            if _ernie_slot_targeted(slot, targets):
                var dims = ernie_lycoris_slot_dims(slot, D, F)
                var w = _read_ernie_weight_f32(st, bi, slot, D, F)
                ad.append(_make_oft_slot_with_w(w^, dims[0], dims[1], block_size))
                active.append(True)
            else:
                ad.append(_dummy_oft_slot())
                active.append(False)
    return ErnieOFTSet(ad^, active^, num_layers, block_size)


def ernie_oft_carrier_set(set: ErnieOFTSet, D: Int, F: Int) raises -> ErnieLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ref sl = set.ad[i]
            ad.append(oft_ot_carrier_adapter(sl.vec, sl.w, sl.in_f, sl.out_f, sl.b, sl.r))
        else:
            var dims = ernie_lycoris_slot_dims(i % ERNIE_SLOTS, D, F)
            ad.append(_inactive_carrier(dims[0], dims[1]))
    return ErnieLoraSet(ad^, set.num_layers, set.block_size)


def ernie_oft_carrier_total_bytes(set: ErnieOFTSet, D: Int, F: Int) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var inf = set.ad[i].in_f
            elems += inf * inf + set.ad[i].out_f * inf
        else:
            var dims = ernie_lycoris_slot_dims(i % ERNIE_SLOTS, D, F)
            elems += dims[0] + dims[1]
    return elems * 2


def ernie_oft_preflight(set: ErnieOFTSet, D: Int, F: Int, budget_bytes: Int) raises:
    var b = ernie_oft_carrier_total_bytes(set, D, F)
    if b > budget_bytes:
        raise Error(
            String("ERNIE OFT carrier needs ") + String(b)
            + String(" bytes (> budget ") + String(budget_bytes)
            + String("). Use attention-only targets or the direct W_eff path.")
        )


struct ErnieOFTGrads(Movable):
    var g: List[List[Float32]]

    def __init__(out self, var g: List[List[Float32]]):
        self.g = g^


def ernie_oft_chain_all(
    set: ErnieOFTSet, d_a: List[List[Float32]], d_b: List[List[Float32]],
) raises -> ErnieOFTGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("ernie_oft_chain_all: grad list count mismatch")
    var out = List[List[Float32]]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ref sl = set.ad[i]
            out.append(oft_ot_chain_carrier_grads(
                sl.vec, sl.w, d_a[i], d_b[i], sl.in_f, sl.out_f, sl.b, sl.r,
            ))
        else:
            out.append(List[Float32]())
    return ErnieOFTGrads(out^)


def _vec_sqsum(g: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(g)):
        s += Float64(g[i]) * Float64(g[i])
    return s


def ernie_oft_grad_norm(grads: ErnieOFTGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _vec_sqsum(grads.g[i])
    return sqrt(s)


def _vec_scale(mut g: List[Float32], scale: Float32):
    for i in range(len(g)):
        g[i] *= scale


def ernie_oft_clip_grads(mut grads: ErnieOFTGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _vec_scale(grads.g[i], clip_scale)


def _oft_vec_adamw(
    mut vec: List[Float32], d_vec: List[Float32],
    mut m: List[Float32], mut v: List[Float32],
    t: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32,
) raises:
    if len(vec) != len(d_vec) or len(vec) != len(m) or len(vec) != len(v):
        raise Error("ERNIE OFT AdamW: len mismatch")
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


def ernie_oft_adamw_step(
    mut set: ErnieOFTSet, grads: ErnieOFTGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            _oft_vec_adamw(
                set.ad[i].vec, grads.g[i], set.ad[i].m, set.ad[i].v,
                t, lr, beta1, beta2, eps, weight_decay,
            )


def ernie_oft_vec_l1(set: ErnieOFTSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref sl = set.ad[i]
        for j in range(len(sl.vec)):
            var v = Float64(sl.vec[j])
            s += v if v >= 0.0 else -v
    return s


def _f32_2d(var values: List[Float32], rows: Int, cols: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return Tensor.from_host(values^, sh^, STDtype.F32, ctx)


def save_ernie_oft(set: ErnieOFTSet, path: String, ctx: DeviceContext) raises -> Int:
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    var nmods = 0
    for bi in range(set.num_layers):
        for slot in range(ERNIE_SLOTS):
            var flat = bi * ERNIE_SLOTS + slot
            if not set.active[flat]:
                continue
            ref sl = set.ad[flat]
            var ne = sl.b * (sl.b - 1) // 2
            names.append(_ernie_prefix(bi, slot) + String(".oft_R.weight"))
            tensors.append(ArcPointer(_f32_2d(sl.vec.copy(), sl.r, ne, ctx)))
            nmods += 1
    if nmods == 0:
        raise Error("save_ernie_oft: refusing to write an empty OFT file")
    save_safetensors(names, tensors, path, ctx)
    return nmods


# ── Direct DoRA/OFT streamed W_eff state ─────────────────────────────────────
def empty_ernie_direct_dora_set() -> FlatDirectDoRASet:
    return empty_flat_direct_dora_set()


def empty_ernie_direct_oft_set() -> FlatDirectOFTSet:
    return empty_flat_direct_oft_set()


def _ernie_direct_targets(targets: Int) raises -> Int:
    if targets == ERNIE_LYCORIS_TGT_ATTN:
        return ERNIE_LYCORIS_TGT_ATTN
    if targets == ERNIE_LYCORIS_TGT_ALL or targets == 3:
        return ERNIE_LYCORIS_TGT_ALL
    raise Error("ERNIE direct LyCORIS targets must be 1(attn) or 2(all)")


def ernie_direct_dense_carrier_bytes(
    num_layers: Int, D: Int, F: Int, targets: Int,
) raises -> Int:
    return ernie_full_delta_carrier_bytes_estimate(
        num_layers, D, F, _ernie_direct_targets(targets),
    )


def ernie_direct_dora_trainable_bytes_estimate(
    num_layers: Int, D: Int, F: Int, rank: Int, targets: Int,
) raises -> Int:
    var t = _ernie_direct_targets(targets)
    var elems_bf16 = 0
    var elems_f32 = 0
    for _bi in range(num_layers):
        for slot in range(ERNIE_SLOTS):
            if not _ernie_slot_targeted(slot, t):
                continue
            var dims = ernie_lycoris_slot_dims(slot, D, F)
            var inf = dims[0]
            var outf = dims[1]
            elems_bf16 += rank * inf + outf * rank
            # magnitude + Adam moments for A/B/m. OneTrainer DoRA default uses
            # input-axis magnitude (`wd_on_out=false`), so mlen = in_f.
            elems_f32 += inf + (rank * inf) * 2 + (outf * rank) * 2 + inf * 2
    return elems_bf16 * 2 + elems_f32 * 4


def ernie_direct_oft_trainable_bytes_estimate(
    num_layers: Int, D: Int, F: Int, block_size: Int, targets: Int,
) raises -> Int:
    var t = _ernie_direct_targets(targets)
    var elems = 0
    for _bi in range(num_layers):
        for slot in range(ERNIE_SLOTS):
            if not _ernie_slot_targeted(slot, t):
                continue
            var dims = ernie_lycoris_slot_dims(slot, D, F)
            if dims[0] % block_size != 0:
                raise Error("ERNIE direct OFT: in_f not divisible by block_size")
            var r = dims[0] // block_size
            var ne = block_size * (block_size - 1) // 2
            elems += r * ne * 3  # vec + Adam m/v
    return elems * 4


def ernie_direct_dora_preflight(
    num_layers: Int, D: Int, F: Int, rank: Int, targets: Int, budget_bytes: Int,
) raises -> Int:
    var b = ernie_direct_dora_trainable_bytes_estimate(
        num_layers, D, F, rank, targets,
    )
    if b > budget_bytes:
        raise Error(
            String("ERNIE direct DoRA trainable state needs ") + String(b)
            + String(" bytes (> budget ") + String(budget_bytes) + String(")")
        )
    return b


def ernie_direct_oft_preflight(
    num_layers: Int, D: Int, F: Int, block_size: Int, targets: Int,
    budget_bytes: Int,
) raises -> Int:
    var b = ernie_direct_oft_trainable_bytes_estimate(
        num_layers, D, F, block_size, targets,
    )
    if b > budget_bytes:
        raise Error(
            String("ERNIE direct OFT trainable state needs ") + String(b)
            + String(" bytes (> budget ") + String(budget_bytes) + String(")")
        )
    return b


def _ernie_block_weight_host(
    block: ErnieBlockWeights, slot: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    if slot == SLOT_Q:
        return block.wq[].to_host(ctx)
    if slot == SLOT_K:
        return block.wk[].to_host(ctx)
    if slot == SLOT_V:
        return block.wv[].to_host(ctx)
    if slot == SLOT_O:
        return block.wo[].to_host(ctx)
    if slot == SLOT_GATE:
        return block.wgate[].to_host(ctx)
    if slot == SLOT_UP:
        return block.wup[].to_host(ctx)
    if slot == SLOT_DOWN:
        return block.wdown[].to_host(ctx)
    raise Error("ERNIE direct DoRA: bad block slot")


def build_ernie_direct_dora_set_from_blocks(
    blocks: List[ErnieBlockWeights], D: Int, F: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
    ctx: DeviceContext,
) raises -> FlatDirectDoRASet:
    var t = _ernie_direct_targets(targets)
    var set = empty_flat_direct_dora_set()
    var s = seed
    for bi in range(len(blocks)):
        for slot in range(ERNIE_SLOTS):
            if not _ernie_slot_targeted(slot, t):
                continue
            var dims = ernie_lycoris_slot_dims(slot, D, F)
            var w = _ernie_block_weight_host(blocks[bi], slot, ctx)
            flat_direct_dora_append_from_weight(
                set, w^, dims[0], dims[1], rank, alpha,
                _ernie_prefix(bi, slot), s, False,
            )
            s += 1
    return set^


def build_ernie_direct_oft_set(
    num_layers: Int, D: Int, F: Int, block_size: Int, targets: Int,
) raises -> FlatDirectOFTSet:
    var t = _ernie_direct_targets(targets)
    var set = empty_flat_direct_oft_set()
    for bi in range(num_layers):
        for slot in range(ERNIE_SLOTS):
            if not _ernie_slot_targeted(slot, t):
                continue
            var dims = ernie_lycoris_slot_dims(slot, D, F)
            flat_direct_oft_append(
                set, dims[0], dims[1], block_size, _ernie_prefix(bi, slot),
            )
    return set^


def ernie_direct_dora_trainable_bytes(set: FlatDirectDoRASet) -> Int:
    return flat_direct_dora_trainable_bytes(set)


def ernie_direct_oft_trainable_bytes(set: FlatDirectOFTSet) -> Int:
    return flat_direct_oft_trainable_bytes(set)


def save_ernie_direct_dora(
    set: FlatDirectDoRASet, path: String, ctx: DeviceContext,
) raises -> Int:
    var named = List[NamedDoRA]()
    for i in range(len(set.ad)):
        if set.active[i]:
            named.append(NamedDoRA(set.prefix[i].copy(), set.ad[i].copy()))
    return save_dora_onetrainer(named, path, ctx)


def save_ernie_direct_oft(
    set: FlatDirectOFTSet, path: String, ctx: DeviceContext,
) raises -> Int:
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    var nmods = 0
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref sl = set.ad[i]
        var ne = sl.b * (sl.b - 1) // 2
        names.append(set.prefix[i].copy() + String(".oft_R.weight"))
        tensors.append(ArcPointer(_f32_2d(sl.vec.copy(), sl.r, ne, ctx)))
        nmods += 1
    if nmods == 0:
        raise Error("save_ernie_direct_oft: refusing to write an empty OFT file")
    save_safetensors(names, tensors, path, ctx)
    return nmods
