# models/wan22/parity/wan22_direct_lycoris_projection_smoke.mojo
#
# Wan2.2-specific direct DoRA/OFT projection gate. This does not enable the live
# trainer yet; it proves the model-specific slot/order/byte-estimate surface that
# the streamed block forward/backward will call instead of full-delta carriers.

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.training.dora_save import read_dora_module
from serenitymojo.training.dora_adapter import dora_forward, dora_backward
from serenitymojo.training.oft_onetrainer import oft_ot_forward, oft_ot_backward
from serenitymojo.models.wan22.wan22_direct_lycoris_stack import (
    W_SA_Q, WAN22_DIRECT_SLOTS,
    wan22_direct_slot_count, wan22_direct_slot_prefix,
    empty_wan22_direct_dora_set,
    empty_wan22_direct_oft_set,
    wan22_direct_dense_carrier_bytes,
    wan22_direct_dora_trainable_bytes_estimate,
    wan22_direct_oft_trainable_bytes_estimate,
    wan22_direct_dora_preflight,
    wan22_direct_oft_preflight,
    build_wan22_direct_dora_set_from_weights,
    wan22_direct_dora_append_block_weights,
    build_wan22_direct_oft_set,
    wan22_direct_oft_append_block,
    wan22_direct_dora_projection_forward,
    wan22_direct_dora_projection_backward,
    wan22_direct_oft_projection_forward,
    wan22_direct_oft_projection_backward,
    wan22_direct_dora_zero_grads,
    wan22_direct_dora_scatter_slot_grad,
    wan22_direct_dora_grad_norm,
    wan22_direct_dora_adamw_step,
    wan22_direct_dora_zero_leg_l1,
    wan22_direct_dora_trainable_bytes,
    wan22_direct_oft_zero_grads,
    wan22_direct_oft_scatter_slot_grad,
    wan22_direct_oft_grad_norm,
    wan22_direct_oft_adamw_step,
    wan22_direct_oft_vec_l1,
    wan22_direct_oft_trainable_bytes,
    save_wan22_direct_dora,
    save_wan22_direct_oft,
)


comptime COS_BAR = 0.999999
comptime NREL_BAR = 2.0e-4
comptime VRAM_24_GIB = 24 * 1024 * 1024 * 1024


def _randn(n: Int, seed: UInt64, scale: Float32, bias: Float32 = Float32(0.0)) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale + bias)
    return out^


def _add_bias(var y: List[Float32], bias: List[Float32], M: Int, out_f: Int) raises -> List[Float32]:
    if len(y) != M * out_f or len(bias) != out_f:
        raise Error("test _add_bias: shape mismatch")
    for m in range(M):
        for o in range(out_f):
            y[m * out_f + o] += bias[o]
    return y^


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


def _check(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        raise Error("check: len mismatch")
    var na = 0.0
    var nb = 0.0
    for i in range(len(a)):
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na < 1.0e-24 and nb < 1.0e-24:
        print("  ", name, " both zero")
        return
    var c = _cos(a, b)
    var n = _nrel(a, b)
    print("  ", name, " cos=", c, " nrel=", n)
    if c < COS_BAR or n > NREL_BAR:
        raise Error(String("GATE FAIL: ") + name)


def _check_real_wan22_bytes() raises:
    var blocks = 40
    var dim = 5120
    var rank = 16
    var oft_block = 4
    var dense = wan22_direct_dense_carrier_bytes(blocks, dim)
    var dora_direct = wan22_direct_dora_trainable_bytes_estimate(blocks, dim, rank, False)
    var oft_direct = wan22_direct_oft_trainable_bytes_estimate(blocks, dim, oft_block)
    print("[wan22-direct-bytes] slots=", wan22_direct_slot_count(blocks))
    print("  dense_carrier_bytes=", dense, " direct_dora_bytes=", dora_direct, " direct_oft_bytes=", oft_direct)
    if dense <= VRAM_24_GIB:
        raise Error("wan22 direct smoke: dense carrier unexpectedly fits 24 GiB")
    if dora_direct >= VRAM_24_GIB:
        raise Error("wan22 direct smoke: DoRA direct trainable estimate exceeds 24 GiB")
    if oft_direct >= VRAM_24_GIB:
        raise Error("wan22 direct smoke: OFT direct trainable estimate exceeds 24 GiB")
    var dora_pf = wan22_direct_dora_preflight(blocks, dim, rank, VRAM_24_GIB, False)
    var oft_pf = wan22_direct_oft_preflight(blocks, dim, oft_block, VRAM_24_GIB)
    if dora_pf != dora_direct or oft_pf != oft_direct:
        raise Error("wan22 direct smoke: preflight byte mismatch")
    var dora_failed = False
    try:
        _ = wan22_direct_dora_preflight(blocks, dim, rank, 1024, False)
    except:
        dora_failed = True
    if not dora_failed:
        raise Error("wan22 direct smoke: DoRA tiny-budget preflight did not fail")
    var oft_failed = False
    try:
        _ = wan22_direct_oft_preflight(blocks, dim, oft_block, 1024)
    except:
        oft_failed = True
    if not oft_failed:
        raise Error("wan22 direct smoke: OFT tiny-budget preflight did not fail")


def _check_prefixes() raises:
    if wan22_direct_slot_count(1) != WAN22_DIRECT_SLOTS:
        raise Error("wan22 direct smoke: slot count mismatch")
    if wan22_direct_slot_prefix(3, W_SA_Q) != "blocks.3.self_attn.q":
        raise Error("wan22 direct smoke: self q prefix mismatch")
    if wan22_direct_slot_prefix(3, 7) != "blocks.3.cross_attn.o":
        raise Error("wan22 direct smoke: cross o prefix mismatch")


def _check_streaming_block_init() raises:
    var blocks = 2
    var dim = 32
    var rank = 4
    var oft_block = 4
    var keep_w = List[Float32]()

    var dora = empty_wan22_direct_dora_set()
    for bi in range(blocks):
        var block_weights = List[List[Float32]]()
        for slot in range(WAN22_DIRECT_SLOTS):
            var w = _randn(dim * dim, UInt64(900 + bi * WAN22_DIRECT_SLOTS + slot), 0.25)
            if bi == 1 and slot == W_SA_Q:
                keep_w = w.copy()
            block_weights.append(w^)
        wan22_direct_dora_append_block_weights(
            dora, bi, block_weights^, dim, rank, Float32(2.0),
            UInt64(1200 + bi * WAN22_DIRECT_SLOTS), False,
        )

    if len(dora.ad) != wan22_direct_slot_count(blocks):
        raise Error("wan22 direct smoke: streaming DoRA slot count mismatch")
    if dora.prefix[WAN22_DIRECT_SLOTS + W_SA_Q] != "blocks.1.self_attn.q":
        raise Error("wan22 direct smoke: streaming DoRA prefix mismatch")
    var dense = wan22_direct_dense_carrier_bytes(blocks, dim)
    var dbytes = wan22_direct_dora_trainable_bytes(dora)
    print("[wan22-direct-stream-init] dora_slots=", len(dora.ad), " direct_bytes=", dbytes, " dense_bytes=", dense)
    if dbytes >= dense:
        raise Error("wan22 direct smoke: streaming DoRA direct state is not below dense carrier")

    var oft = empty_wan22_direct_oft_set()
    for bi in range(blocks):
        wan22_direct_oft_append_block(oft, bi, dim, oft_block)
    if len(oft.ad) != wan22_direct_slot_count(blocks):
        raise Error("wan22 direct smoke: streaming OFT slot count mismatch")
    if oft.prefix[WAN22_DIRECT_SLOTS + W_SA_Q] != "blocks.1.self_attn.q":
        raise Error("wan22 direct smoke: streaming OFT prefix mismatch")
    var obytes = wan22_direct_oft_trainable_bytes(oft)
    print("[wan22-direct-stream-init] oft_slots=", len(oft.ad), " direct_bytes=", obytes, " dense_bytes=", dense)
    if obytes >= dense:
        raise Error("wan22 direct smoke: streaming OFT direct state is not below dense carrier")

    var M = 2
    var slot_idx = WAN22_DIRECT_SLOTS + W_SA_Q
    var x = _randn(M * dim, 1300, 0.8)
    var bias = _randn(dim, 1301, 0.1)
    var d_y = _randn(M * dim, 1302, 0.5)
    var yg = wan22_direct_dora_projection_forward(dora, slot_idx, x.copy(), keep_w.copy(), bias.copy(), M)
    var yg_ref = dora_forward(x.copy(), keep_w.copy(), dora.ad[slot_idx], M)
    yg_ref = _add_bias(yg_ref^, bias.copy(), M, dim)
    _check(String("stream dora projection+bias"), yg, yg_ref)
    var dg = wan22_direct_dora_projection_backward(dora, slot_idx, d_y.copy(), x.copy(), keep_w.copy(), M)
    var dg_ref = dora_backward(d_y.copy(), x.copy(), keep_w.copy(), dora.ad[slot_idx], M)
    _check(String("stream dora d_B"), dg.d_b, dg_ref.d_b)
    _check(String("stream dora d_x"), dg.d_x, dg_ref.d_x)

    var yo = wan22_direct_oft_projection_forward(oft, slot_idx, x.copy(), keep_w.copy(), bias.copy(), M)
    var yo_ref = oft_ot_forward(x.copy(), oft.ad[slot_idx].vec.copy(), keep_w.copy(), M, dim, dim, oft_block, dim // oft_block)
    yo_ref = _add_bias(yo_ref^, bias.copy(), M, dim)
    _check(String("stream oft projection+bias"), yo, yo_ref)
    var og = wan22_direct_oft_projection_backward(oft, slot_idx, d_y.copy(), x.copy(), keep_w.copy(), M)
    var og_ref = oft_ot_backward(d_y.copy(), x.copy(), oft.ad[slot_idx].vec.copy(), keep_w.copy(), M, dim, dim, oft_block, dim // oft_block)
    _check(String("stream oft d_vec"), og.d_vec, og_ref.d_vec)
    _check(String("stream oft d_x"), og.d_x, og_ref.d_x)


def _check_dora_projection(ctx: DeviceContext) raises:
    var dim = 64
    var rank = 4
    var M = 3
    var weights = List[List[Float32]]()
    for slot in range(WAN22_DIRECT_SLOTS):
        weights.append(_randn(dim * dim, UInt64(100 + slot), 0.35))
    var dora = build_wan22_direct_dora_set_from_weights(
        weights, 1, dim, rank, Float32(2.0), UInt64(77), False,
    )
    var x = _randn(M * dim, 300, 0.8)
    var bias = _randn(dim, 301, 0.1)
    var d_y = _randn(M * dim, 302, 0.5)
    var slot = W_SA_Q
    var y = wan22_direct_dora_projection_forward(dora, slot, x.copy(), weights[slot].copy(), bias.copy(), M)
    var y_ref = dora_forward(x.copy(), weights[slot].copy(), dora.ad[slot], M)
    y_ref = _add_bias(y_ref^, bias.copy(), M, dim)
    _check(String("dora projection+bias"), y, y_ref)
    var g = wan22_direct_dora_projection_backward(dora, slot, d_y.copy(), x.copy(), weights[slot].copy(), M)
    var g_ref = dora_backward(d_y.copy(), x.copy(), weights[slot].copy(), dora.ad[slot], M)
    _check(String("dora d_A"), g.d_a, g_ref.d_a)
    _check(String("dora d_B"), g.d_b, g_ref.d_b)
    _check(String("dora d_m"), g.d_m, g_ref.d_m)
    _check(String("dora d_x"), g.d_x, g_ref.d_x)

    var all_g = wan22_direct_dora_zero_grads(dora)
    wan22_direct_dora_scatter_slot_grad(all_g, slot, g)
    var n = wan22_direct_dora_grad_norm(all_g)
    print("  dora set_grad_norm=", n, " direct_bytes=", wan22_direct_dora_trainable_bytes(dora))
    if n <= 0.0:
        raise Error("wan22 direct smoke: DoRA set grad norm stayed zero")
    var before = wan22_direct_dora_zero_leg_l1(dora)
    wan22_direct_dora_adamw_step(
        dora, all_g, 1, Float32(1.0e-3),
        Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.0),
    )
    var after = wan22_direct_dora_zero_leg_l1(dora)
    print("  dora zero_leg_l1 ", before, " -> ", after)
    if not (before == 0.0 and after > 0.0):
        raise Error("wan22 direct smoke: DoRA direct set did not step")
    var nmods = save_wan22_direct_dora(dora, String("/tmp/wan22_direct_dora_smoke.safetensors"), ctx)
    print("  dora save modules=", nmods)
    if nmods != WAN22_DIRECT_SLOTS:
        raise Error("wan22 direct smoke: DoRA save module count mismatch")
    var rb = read_dora_module(
        String("blocks.0.self_attn.q"), String("/tmp/wan22_direct_dora_smoke.safetensors"), ctx,
    )
    if rb.in_f != dim or rb.out_f != dim or rb.rank != rank:
        raise Error("wan22 direct smoke: DoRA readback shape mismatch")
    if rb.wd_on_out:
        raise Error("wan22 direct smoke: DoRA readback lost OneTrainer input-axis magnitude")


def _check_oft_projection(ctx: DeviceContext) raises:
    var dim = 64
    var block_size = 4
    var r = dim // block_size
    var M = 3
    var oft = build_wan22_direct_oft_set(1, dim, block_size)
    var W = _randn(dim * dim, 700, 0.35)
    var x = _randn(M * dim, 701, 0.8)
    var bias = _randn(dim, 702, 0.1)
    var d_y = _randn(M * dim, 703, 0.5)
    var slot = W_SA_Q
    var y = wan22_direct_oft_projection_forward(oft, slot, x.copy(), W.copy(), bias.copy(), M)
    var y_ref = oft_ot_forward(x.copy(), oft.ad[slot].vec.copy(), W.copy(), M, dim, dim, block_size, r)
    y_ref = _add_bias(y_ref^, bias.copy(), M, dim)
    _check(String("oft projection+bias"), y, y_ref)
    var g = wan22_direct_oft_projection_backward(oft, slot, d_y.copy(), x.copy(), W.copy(), M)
    var g_ref = oft_ot_backward(d_y.copy(), x.copy(), oft.ad[slot].vec.copy(), W.copy(), M, dim, dim, block_size, r)
    _check(String("oft d_vec"), g.d_vec, g_ref.d_vec)
    _check(String("oft d_x"), g.d_x, g_ref.d_x)

    var all_g = wan22_direct_oft_zero_grads(oft)
    wan22_direct_oft_scatter_slot_grad(all_g, slot, g)
    var n = wan22_direct_oft_grad_norm(all_g)
    print("  oft set_grad_norm=", n, " direct_bytes=", wan22_direct_oft_trainable_bytes(oft))
    if n <= 0.0:
        raise Error("wan22 direct smoke: OFT set grad norm stayed zero")
    var before = wan22_direct_oft_vec_l1(oft)
    wan22_direct_oft_adamw_step(
        oft, all_g, 1, Float32(1.0e-3),
        Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.0),
    )
    var after = wan22_direct_oft_vec_l1(oft)
    print("  oft vec_l1 ", before, " -> ", after)
    if not (before == 0.0 and after > 0.0):
        raise Error("wan22 direct smoke: OFT direct set did not step")
    var nmods = save_wan22_direct_oft(oft, String("/tmp/wan22_direct_oft_smoke.safetensors"), ctx)
    print("  oft save modules=", nmods)
    if nmods != WAN22_DIRECT_SLOTS:
        raise Error("wan22 direct smoke: OFT save module count mismatch")


def main() raises:
    var ctx = DeviceContext()
    _check_prefixes()
    _check_real_wan22_bytes()
    _check_streaming_block_init()
    _check_dora_projection(ctx)
    _check_oft_projection(ctx)
    print("ALL GATES PASS -- wan22_direct_lycoris_projection_smoke")
