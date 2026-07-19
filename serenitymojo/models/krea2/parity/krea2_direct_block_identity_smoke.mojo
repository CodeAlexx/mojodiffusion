# serenitymojo/models/krea2/parity/krea2_direct_block_identity_smoke.mojo
#
# Direct DoRA/OFT Krea2 block plumbing gate. At initialization, direct DoRA and
# OFT are identity substitutions for the frozen projection weights, so the full
# block forward should match the existing no-adapter LoRA block path without
# allocating dense full-delta carrier tensors.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.krea2.krea2_lokr_stack import KREA2_SLOTS, K2LOKR_TGT_ALL
from serenitymojo.models.krea2.krea2_direct_lycoris_stack import (
    build_krea2_direct_dora_set_from_weights,
    build_krea2_direct_oft_set,
    krea2_direct_dora_blocks_to_device,
    krea2_direct_oft_blocks_to_device,
    krea2_direct_dora_trainable_bytes,
    krea2_direct_oft_trainable_bytes,
    krea2_direct_dense_carrier_bytes,
)
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockLora,
    krea2_single_stream_block_lora,
    krea2_single_stream_block_dora,
    krea2_single_stream_block_oft,
    krea2_single_stream_block_lora_backward_dev,
    krea2_single_stream_block_dora_backward_dev,
    krea2_single_stream_block_oft_backward_dev,
)


comptime TArc = ArcPointer[Tensor]
comptime HEADS = 2
comptime KVHEADS = 1
comptime HEADDIM = 128
comptime FEATURES = HEADS * HEADDIM
comptime HALF = HEADDIM // 2
comptime L = 4
comptime MLPDIM = 512
comptime RANK = 4
comptime ALPHA = Float32(8.0)
comptime OFT_BLOCK = 4
comptime EPS = Float32(1.0e-5)
comptime COS_BAR = 0.99999
comptime NREL_BAR = 5.0e-4


def _randn(n: Int, seed: UInt64, scale: Float32, bias: Float32 = Float32(0.0)) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale + bias)
    return out^


def _fill(n: Int, value: Float32) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(value)
    return out^


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^


def _t(var values: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(values^, shape^, STDtype.F32, ctx))


def _tile_rope(value: Float32, heads: Int, ctx: DeviceContext) raises -> Tensor:
    var out = List[Float32]()
    for _l in range(L):
        for _h in range(heads):
            for _c in range(HALF):
                out.append(value)
    return Tensor.from_host(out^, _shape2(L * heads, HALF), STDtype.F32, ctx)


def _empty_lora() -> Krea2BlockLora:
    return Krea2BlockLora(
        Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None),
    )


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cos: len mismatch")
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        raise Error("cos: zero vector")
    return dot / (sqrt(na) * sqrt(nb))


def _nrel(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("nrel: len mismatch")
    var d = 0.0
    var n = 0.0
    for i in range(len(a)):
        var dd = Float64(a[i]) - Float64(b[i])
        d += dd * dd
        n += Float64(b[i]) * Float64(b[i])
    if n == 0.0:
        return sqrt(d)
    return sqrt(d / n)


def _max_abs(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error("max_abs: len mismatch")
    var mx = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        var ad = d if d >= 0.0 else -d
        if ad > mx:
            mx = ad
    return mx


def _l1(a: List[Float32]) -> Float64:
    var out = Float64(0.0)
    for i in range(len(a)):
        var x = Float64(a[i])
        out += x if x >= 0.0 else -x
    return out


def _l1_t(name: String, ad: Optional[TArc], ctx: DeviceContext) raises -> Float64:
    if not ad:
        raise Error(String("missing direct grad: ") + name)
    return _l1(ad.value()[].to_host(ctx))


def _require_pos(name: String, value: Float64) raises:
    print("  ", name, " l1=", value)
    if value <= 0.0:
        raise Error(String("zero direct grad: ") + name)


def _check(name: String, got: List[Float32], expected: List[Float32]) raises:
    var c = _cos(got, expected)
    var n = _nrel(got, expected)
    var m = _max_abs(got, expected)
    print("  ", name, " cos=", c, " nrel=", n, " max_abs=", m)
    if c < COS_BAR or n > NREL_BAR:
        raise Error(String("GATE FAIL: ") + name)


def _block_weight_list(
    wq: List[Float32], wk: List[Float32], wv: List[Float32],
    gate_w: List[Float32], wo: List[Float32],
    mlp_gate_w: List[Float32], mlp_up_w: List[Float32],
    mlp_down_w: List[Float32],
) -> List[List[Float32]]:
    var weights = List[List[Float32]]()
    weights.append(wq.copy())
    weights.append(wk.copy())
    weights.append(wv.copy())
    weights.append(gate_w.copy())
    weights.append(wo.copy())
    weights.append(mlp_gate_w.copy())
    weights.append(mlp_up_w.copy())
    weights.append(mlp_down_w.copy())
    return weights^


def main() raises:
    var ctx = DeviceContext()
    print("==== krea2_direct_block_identity_smoke ====")
    print("HEADS=", HEADS, " KVHEADS=", KVHEADS, " HEADDIM=", HEADDIM,
          " FEATURES=", FEATURES, " L=", L, " MLPDIM=", MLPDIM)

    var wq = _randn(HEADS * HEADDIM * FEATURES, UInt64(1001), Float32(0.08))
    var wk = _randn(KVHEADS * HEADDIM * FEATURES, UInt64(1002), Float32(0.08))
    var wv = _randn(KVHEADS * HEADDIM * FEATURES, UInt64(1003), Float32(0.08))
    var gate_w = _randn(FEATURES * FEATURES, UInt64(1004), Float32(0.08))
    var wo = _randn(FEATURES * FEATURES, UInt64(1005), Float32(0.08))
    var mlp_gate_w = _randn(MLPDIM * FEATURES, UInt64(1006), Float32(0.06))
    var mlp_up_w = _randn(MLPDIM * FEATURES, UInt64(1007), Float32(0.06))
    var mlp_down_w = _randn(FEATURES * MLPDIM, UInt64(1008), Float32(0.06))

    var qnorm = _fill(HEADDIM, Float32(0.0))
    var knorm = _fill(HEADDIM, Float32(0.0))
    var prenorm = _fill(FEATURES, Float32(0.0))
    var postnorm = _fill(FEATURES, Float32(0.0))
    var mod_lin = _randn(6 * FEATURES, UInt64(1010), Float32(0.02))

    var w = Krea2BlockWeights(
        _t(wq.copy(), _shape2(HEADS * HEADDIM, FEATURES), ctx),
        _t(wk.copy(), _shape2(KVHEADS * HEADDIM, FEATURES), ctx),
        _t(wv.copy(), _shape2(KVHEADS * HEADDIM, FEATURES), ctx),
        _t(gate_w.copy(), _shape2(FEATURES, FEATURES), ctx),
        _t(wo.copy(), _shape2(FEATURES, FEATURES), ctx),
        _t(mlp_gate_w.copy(), _shape2(MLPDIM, FEATURES), ctx),
        _t(mlp_up_w.copy(), _shape2(MLPDIM, FEATURES), ctx),
        _t(mlp_down_w.copy(), _shape2(FEATURES, MLPDIM), ctx),
        _t(qnorm^, _shape1(HEADDIM), ctx),
        _t(knorm^, _shape1(HEADDIM), ctx),
        _t(prenorm^, _shape1(FEATURES), ctx),
        _t(postnorm^, _shape1(FEATURES), ctx),
        _t(mod_lin^, _shape1(6 * FEATURES), ctx),
    )

    var x = _t(
        _randn(L * FEATURES, UInt64(2001), Float32(0.15)),
        _shape3(1, L, FEATURES), ctx,
    )
    var vec = Tensor.from_host(
        _randn(6 * FEATURES, UInt64(2002), Float32(0.02)),
        _shape2(1, 6 * FEATURES), STDtype.F32, ctx,
    )
    var cos0 = Tensor.from_host(_fill(L * HALF, Float32(1.0)), _shape2(L, HALF), STDtype.F32, ctx)
    var sin0 = Tensor.from_host(_fill(L * HALF, Float32(0.0)), _shape2(L, HALF), STDtype.F32, ctx)
    var cos_q = _tile_rope(Float32(1.0), HEADS, ctx)
    var sin_q = _tile_rope(Float32(0.0), HEADS, ctx)
    var cos_k = _tile_rope(Float32(1.0), KVHEADS, ctx)
    var sin_k = _tile_rope(Float32(0.0), KVHEADS, ctx)

    var base = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
        x.copy(), vec, w, _empty_lora(),
        cos0, sin0, cos_q, sin_q, cos_k, sin_k, EPS, ctx,
    )
    var base_h = base.out[].to_host(ctx)

    var weights = _block_weight_list(
        wq, wk, wv, gate_w, wo, mlp_gate_w, mlp_up_w, mlp_down_w,
    )
    if len(weights) != KREA2_SLOTS:
        raise Error("krea2 direct block smoke: expected 8 block weights")
    var dense = krea2_direct_dense_carrier_bytes(
        1, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM, K2LOKR_TGT_ALL,
    )

    var dora_set = build_krea2_direct_dora_set_from_weights(
        weights.copy(), 1, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM,
        RANK, ALPHA, K2LOKR_TGT_ALL, UInt64(3001), False,
    )
    var dora_blocks = krea2_direct_dora_blocks_to_device(dora_set, 1, K2LOKR_TGT_ALL, ctx)
    print("[direct-dora] trainable_bytes=", krea2_direct_dora_trainable_bytes(dora_set),
          " dense_carrier_bytes=", dense)
    var dora = krea2_single_stream_block_dora[L, HEADS, KVHEADS, HEADDIM](
        x.copy(), vec, w, dora_blocks.blocks[0],
        cos0, sin0, cos_q, sin_q, cos_k, sin_k, EPS, ctx,
    )
    _check("dora block output", dora.out[].to_host(ctx), base_h)

    var oft_set = build_krea2_direct_oft_set(
        1, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM,
        OFT_BLOCK, K2LOKR_TGT_ALL,
    )
    var oft_blocks = krea2_direct_oft_blocks_to_device(oft_set, 1, K2LOKR_TGT_ALL, ctx)
    print("[direct-oft] trainable_bytes=", krea2_direct_oft_trainable_bytes(oft_set),
          " dense_carrier_bytes=", dense)
    var oft = krea2_single_stream_block_oft[L, HEADS, KVHEADS, HEADDIM](
        x.copy(), vec, w, oft_blocks.blocks[0],
        cos0, sin0, cos_q, sin_q, cos_k, sin_k, EPS, ctx,
    )
    _check("oft block output", oft.out[].to_host(ctx), base_h)

    var d_out = Tensor.from_host(
        _randn(L * FEATURES, UInt64(4001), Float32(0.05)),
        _shape3(1, L, FEATURES), STDtype.F32, ctx,
    )
    var base_g = krea2_single_stream_block_lora_backward_dev[L, HEADS, KVHEADS, HEADDIM](
        d_out.clone(ctx), vec, w, _empty_lora(), base.saved,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx,
    )
    var dora_g = krea2_single_stream_block_dora_backward_dev[L, HEADS, KVHEADS, HEADDIM](
        d_out.clone(ctx), vec, w, dora_blocks.blocks[0], dora.saved,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx,
    )
    var oft_g = krea2_single_stream_block_oft_backward_dev[L, HEADS, KVHEADS, HEADDIM](
        d_out.clone(ctx), vec, w, oft_blocks.blocks[0], oft.saved,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx,
    )
    var base_dx = base_g.d_x[].to_host(ctx)
    _check("dora block d_x", dora_g.d_x[].to_host(ctx), base_dx)
    _check("oft block d_x", oft_g.d_x[].to_host(ctx), base_dx)

    var dora_db = (
        _l1_t("dora wq d_B", dora_g.wq.d_b, ctx)
        + _l1_t("dora wo d_B", dora_g.wo.d_b, ctx)
        + _l1_t("dora mlp_down d_B", dora_g.mlp_down_w.d_b, ctx)
    )
    var dora_dm = (
        _l1_t("dora wq d_m", dora_g.wq.d_m, ctx)
        + _l1_t("dora wo d_m", dora_g.wo.d_m, ctx)
        + _l1_t("dora mlp_down d_m", dora_g.mlp_down_w.d_m, ctx)
    )
    var oft_dv = (
        _l1_t("oft wq d_vec", oft_g.wq.d_vec, ctx)
        + _l1_t("oft wo d_vec", oft_g.wo.d_vec, ctx)
        + _l1_t("oft mlp_down d_vec", oft_g.mlp_down_w.d_vec, ctx)
    )
    _require_pos("dora selected d_B", dora_db)
    _require_pos("dora selected d_m", dora_dm)
    _require_pos("oft selected d_vec", oft_dv)

    print("ALL GATES PASS -- krea2_direct_block_identity_smoke")
