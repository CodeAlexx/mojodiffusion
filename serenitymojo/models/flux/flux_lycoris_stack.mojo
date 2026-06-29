# models/flux/flux_lycoris_stack.mojo -- LoKr/LoHa block-projection carriers.
#
# Flux and Chroma both train the flat FluxLoraSet block-projection surface:
# double blocks first (img stream 6 slots, txt stream 6 slots), then single
# blocks (5 slots). This layer materializes LoRA carriers for LoKr/LoHa masters
# and chains returned dA/dB through the shared LyCORIS math.

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext

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
from serenitymojo.models.flux.lora_block import (
    DBL_STREAM_SLOTS, SGL_SLOTS,
    D_SQ, D_SK, D_SV, D_PROJ, D_MLP0, D_MLP2,
    S_SQ, S_SK, S_SV, S_PMLP, S_L2,
)
from serenitymojo.models.flux.flux_stack_lora import FluxLoraSet, DBL_SLOTS_PER_BLOCK


comptime FLUX_LYCORIS_TGT_ATTN = 1
comptime FLUX_LYCORIS_TGT_ALL = 2


def flux_lycoris_dbl_slot_dims(slot: Int, D: Int, F: Int) raises -> Tuple[Int, Int]:
    var s = slot % DBL_STREAM_SLOTS
    if s == D_MLP0:
        return (D, F)
    if s == D_MLP2:
        return (F, D)
    return (D, D)


def flux_lycoris_sgl_slot_dims(slot: Int, D: Int, F: Int) raises -> Tuple[Int, Int]:
    var s = slot % SGL_SLOTS
    if s == S_PMLP:
        return (D, F)
    if s == S_L2:
        return (D + F, D)
    return (D, D)


def _flux_dbl_slot_is_attn(slot: Int) -> Bool:
    var s = slot % DBL_STREAM_SLOTS
    return s == D_SQ or s == D_SK or s == D_SV or s == D_PROJ


def _flux_sgl_slot_is_attn(slot: Int) -> Bool:
    var s = slot % SGL_SLOTS
    return s == S_SQ or s == S_SK or s == S_SV


def _flux_slot_targeted(is_double: Bool, slot: Int, targets: Int) -> Bool:
    if is_double:
        if _flux_dbl_slot_is_attn(slot):
            return targets >= FLUX_LYCORIS_TGT_ATTN
        return targets >= FLUX_LYCORIS_TGT_ALL
    if _flux_sgl_slot_is_attn(slot):
        return targets >= FLUX_LYCORIS_TGT_ATTN
    return targets >= FLUX_LYCORIS_TGT_ALL


def _flux_slot_factor(
    is_double: Bool, slot: Int, factor: Int, factor_attn: Int, factor_ff: Int,
) -> Int:
    var is_attn = _flux_dbl_slot_is_attn(slot) if is_double else _flux_sgl_slot_is_attn(slot)
    if is_attn:
        return factor_attn if factor_attn != 0 else factor
    return factor_ff if factor_ff != 0 else factor


def _flux_flat_is_double(flat: Int, num_double: Int) -> Bool:
    return flat < num_double * DBL_SLOTS_PER_BLOCK


def _flux_flat_dims(flat: Int, num_double: Int, D: Int, F: Int) raises -> Tuple[Int, Int]:
    var dbl_count = num_double * DBL_SLOTS_PER_BLOCK
    if flat < dbl_count:
        return flux_lycoris_dbl_slot_dims(flat % DBL_STREAM_SLOTS, D, F)
    return flux_lycoris_sgl_slot_dims((flat - dbl_count) % SGL_SLOTS, D, F)


def _flux_dbl_prefix(bi: Int, stream_img: Bool, slot: Int) -> String:
    var b = String("lora_transformer_transformer_blocks_") + String(bi) + "_"
    if stream_img:
        if slot == D_SQ:
            return b + "attn_to_q"
        if slot == D_SK:
            return b + "attn_to_k"
        if slot == D_SV:
            return b + "attn_to_v"
        if slot == D_PROJ:
            return b + "attn_to_out_0"
        if slot == D_MLP0:
            return b + "ff_net_0_proj"
        return b + "ff_net_2"
    if slot == D_SQ:
        return b + "attn_add_q_proj"
    if slot == D_SK:
        return b + "attn_add_k_proj"
    if slot == D_SV:
        return b + "attn_add_v_proj"
    if slot == D_PROJ:
        return b + "attn_to_add_out"
    if slot == D_MLP0:
        return b + "ff_context_net_0_proj"
    return b + "ff_context_net_2"


def _flux_sgl_prefix(bi: Int, slot: Int) -> String:
    var b = String("lora_transformer_single_transformer_blocks_") + String(bi) + "_"
    if slot == S_SQ:
        return b + "attn_to_q"
    if slot == S_SK:
        return b + "attn_to_k"
    if slot == S_SV:
        return b + "attn_to_v"
    if slot == S_PMLP:
        return b + "proj_mlp"
    return b + "proj_out"


def _flux_flat_prefix(flat: Int, num_double: Int) -> String:
    var dbl_count = num_double * DBL_SLOTS_PER_BLOCK
    if flat < dbl_count:
        var bi = flat // DBL_SLOTS_PER_BLOCK
        var s12 = flat % DBL_SLOTS_PER_BLOCK
        var stream_img = s12 < DBL_STREAM_SLOTS
        return _flux_dbl_prefix(bi, stream_img, s12 % DBL_STREAM_SLOTS)
    var off = flat - dbl_count
    return _flux_sgl_prefix(off // SGL_SLOTS, off % SGL_SLOTS)


struct FluxLoKrSet(Copyable, Movable):
    var ad: List[LoKrAdapter]
    var active: List[Bool]
    var num_double: Int
    var num_single: Int
    var rank: Int

    def __init__(
        out self, var ad: List[LoKrAdapter], var active: List[Bool],
        num_double: Int, num_single: Int, rank: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.num_double = num_double
        self.num_single = num_single
        self.rank = rank


def empty_flux_lokr_set() -> FluxLoKrSet:
    return FluxLoKrSet(List[LoKrAdapter](), List[Bool](), 0, 0, 0)


def build_flux_lokr_set(
    num_double: Int, num_single: Int, D: Int, F: Int,
    rank: Int, alpha: Float32,
    factor: Int, factor_attn: Int, factor_ff: Int,
    decompose_both: Bool, full_matrix: Bool,
    targets: Int, seed: UInt64,
) raises -> FluxLoKrSet:
    if targets < FLUX_LYCORIS_TGT_ATTN or targets > FLUX_LYCORIS_TGT_ALL:
        raise Error("build_flux_lokr_set: targets must be 1(attn)|2(all)")
    var ad = List[LoKrAdapter]()
    var active = List[Bool]()
    var s = seed
    for _bi in range(num_double):
        for _ in range(2):
            for slot in range(DBL_STREAM_SLOTS):
                if _flux_slot_targeted(True, slot, targets):
                    var dims = flux_lycoris_dbl_slot_dims(slot, D, F)
                    var f = _flux_slot_factor(True, slot, factor, factor_attn, factor_ff)
                    ad.append(new_lokr_adapter(
                        dims[0], dims[1], rank, alpha, f, s,
                        decompose_both, full_matrix,
                    ))
                    active.append(True)
                else:
                    ad.append(_dummy_lokr())
                    active.append(False)
                s += 1
    for _bi in range(num_single):
        for slot in range(SGL_SLOTS):
            if _flux_slot_targeted(False, slot, targets):
                var dims = flux_lycoris_sgl_slot_dims(slot, D, F)
                var f = _flux_slot_factor(False, slot, factor, factor_attn, factor_ff)
                ad.append(new_lokr_adapter(
                    dims[0], dims[1], rank, alpha, f, s,
                    decompose_both, full_matrix,
                ))
                active.append(True)
            else:
                ad.append(_dummy_lokr())
                active.append(False)
            s += 1
    return FluxLoKrSet(ad^, active^, num_double, num_single, rank)


def flux_lokr_carrier_set(set: FluxLoKrSet, D: Int, F: Int) raises -> FluxLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ad.append(lokr_carrier_adapter(set.ad[i]))
        else:
            var dims = _flux_flat_dims(i, set.num_double, D, F)
            ad.append(_inactive_carrier(dims[0], dims[1]))
    return FluxLoraSet(ad^, set.num_double, set.num_single, set.rank)


def flux_lokr_carrier_total_bytes(set: FluxLoKrSet, D: Int, F: Int) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = lokr_carrier_r_eff(set.ad[i])
            elems += r * set.ad[i].in_f + set.ad[i].out_f * r
        else:
            var dims = _flux_flat_dims(i, set.num_double, D, F)
            elems += dims[0] + dims[1]
    return elems * 2


struct FluxLoKrGrads(Movable):
    var g: List[LoKrGrads]

    def __init__(out self, var g: List[LoKrGrads]):
        self.g = g^


def flux_lokr_chain_all(
    set: FluxLoKrSet, d_a: List[List[Float32]], d_b: List[List[Float32]]
) raises -> FluxLoKrGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("flux_lokr_chain_all: grad list count mismatch")
    var g = List[LoKrGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            g.append(lokr_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            g.append(_empty_lokr_grads())
    return FluxLoKrGrads(g^)


def flux_lokr_grad_norm(grads: FluxLoKrGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _grads_sqsum(grads.g[i])
    return sqrt(s)


def flux_lokr_clip_grads(mut grads: FluxLoKrGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _grads_scale(grads.g[i], clip_scale)


def flux_lokr_adamw_step(
    mut set: FluxLoKrSet, grads: FluxLoKrGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            lokr_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


def flux_lokr_zero_leg_l1(set: FluxLoKrSet) -> Float64:
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


def save_flux_lokr(set: FluxLoKrSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoKr]()
    for i in range(len(set.ad)):
        if set.active[i]:
            named.append(NamedLoKr(_flux_flat_prefix(i, set.num_double), set.ad[i].copy()))
    return save_lokr_peft(named, path, ctx)


struct FluxLoHaSet(Copyable, Movable):
    var ad: List[LoHaAdapter]
    var active: List[Bool]
    var num_double: Int
    var num_single: Int
    var rank: Int

    def __init__(
        out self, var ad: List[LoHaAdapter], var active: List[Bool],
        num_double: Int, num_single: Int, rank: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.num_double = num_double
        self.num_single = num_single
        self.rank = rank


def empty_flux_loha_set() -> FluxLoHaSet:
    return FluxLoHaSet(List[LoHaAdapter](), List[Bool](), 0, 0, 0)


def build_flux_loha_set(
    num_double: Int, num_single: Int, D: Int, F: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
) raises -> FluxLoHaSet:
    if targets < FLUX_LYCORIS_TGT_ATTN or targets > FLUX_LYCORIS_TGT_ALL:
        raise Error("build_flux_loha_set: targets must be 1(attn)|2(all)")
    var ad = List[LoHaAdapter]()
    var active = List[Bool]()
    var s = seed
    for _bi in range(num_double):
        for _ in range(2):
            for slot in range(DBL_STREAM_SLOTS):
                if _flux_slot_targeted(True, slot, targets):
                    var dims = flux_lycoris_dbl_slot_dims(slot, D, F)
                    ad.append(new_loha_adapter(dims[0], dims[1], rank, alpha, s))
                    active.append(True)
                else:
                    ad.append(_dummy_loha())
                    active.append(False)
                s += 1
    for _bi in range(num_single):
        for slot in range(SGL_SLOTS):
            if _flux_slot_targeted(False, slot, targets):
                var dims = flux_lycoris_sgl_slot_dims(slot, D, F)
                ad.append(new_loha_adapter(dims[0], dims[1], rank, alpha, s))
                active.append(True)
            else:
                ad.append(_dummy_loha())
                active.append(False)
            s += 1
    return FluxLoHaSet(ad^, active^, num_double, num_single, rank)


def flux_loha_carrier_set(set: FluxLoHaSet, D: Int, F: Int) raises -> FluxLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ad.append(loha_carrier_adapter(set.ad[i]))
        else:
            var dims = _flux_flat_dims(i, set.num_double, D, F)
            ad.append(_inactive_carrier(dims[0], dims[1]))
    return FluxLoraSet(ad^, set.num_double, set.num_single, set.rank)


def flux_loha_carrier_total_bytes(set: FluxLoHaSet, D: Int, F: Int) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = loha_carrier_r_eff(set.ad[i])
            elems += r * set.ad[i].in_f + set.ad[i].out_f * r
        else:
            var dims = _flux_flat_dims(i, set.num_double, D, F)
            elems += dims[0] + dims[1]
    return elems * 2


struct FluxLoHaGrads(Movable):
    var g: List[LoHaGrads]

    def __init__(out self, var g: List[LoHaGrads]):
        self.g = g^


def flux_loha_chain_all(
    set: FluxLoHaSet, d_a: List[List[Float32]], d_b: List[List[Float32]]
) raises -> FluxLoHaGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("flux_loha_chain_all: grad list count mismatch")
    var g = List[LoHaGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            g.append(loha_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            g.append(_empty_loha_grads())
    return FluxLoHaGrads(g^)


def flux_loha_grad_norm(grads: FluxLoHaGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _loha_grads_sqsum(grads.g[i])
    return sqrt(s)


def flux_loha_clip_grads(mut grads: FluxLoHaGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _loha_grads_scale(grads.g[i], clip_scale)


def flux_loha_adamw_step(
    mut set: FluxLoHaSet, grads: FluxLoHaGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            loha_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


def flux_loha_zero_leg_l1(set: FluxLoHaSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref lo = set.ad[i]
        for j in range(len(lo.w2a)):
            var v = Float64(lo.w2a[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    return s


def save_flux_loha(set: FluxLoHaSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoHa]()
    for i in range(len(set.ad)):
        if set.active[i]:
            named.append(NamedLoHa(_flux_flat_prefix(i, set.num_double), set.ad[i].copy()))
    return save_loha_peft(named, path, ctx)
