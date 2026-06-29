# models/flux/parity/flux_direct_lycoris_projection_smoke.mojo
#
# Proves shared Flux/Chroma direct DoRA/OFT slot metadata, byte preflight,
# compact target construction, projection wrappers, optimizer movement, and save
# names. This is not the live GPU block lowering gate.

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.training.dora_save import read_dora_module
from serenitymojo.training.dora_adapter import dora_forward, dora_backward
from serenitymojo.training.oft_onetrainer import oft_ot_forward, oft_ot_backward
from serenitymojo.models.flux.lora_block import (
    DBL_STREAM_SLOTS, SGL_SLOTS, D_MLP0, S_PMLP,
)
from serenitymojo.models.flux.flux_stack_lora import DBL_SLOTS_PER_BLOCK
from serenitymojo.models.flux.flux_lycoris_stack import (
    FLUX_LYCORIS_TGT_ATTN, FLUX_LYCORIS_TGT_ALL,
)
from serenitymojo.models.flux.flux_direct_lycoris_stack import (
    FLUX_DIRECT_24_GIB,
    flux_direct_total_slots,
    flux_direct_flat_dims,
    flux_direct_flat_prefix,
    flux_direct_active_slot_count,
    empty_flux_direct_dora_set,
    empty_flux_direct_oft_set,
    flux_direct_dense_carrier_bytes,
    flux_direct_dora_trainable_bytes_estimate,
    flux_direct_oft_trainable_bytes_estimate,
    flux_direct_dora_preflight,
    flux_direct_oft_preflight,
    build_flux_direct_dora_set_from_weights,
    build_flux_direct_oft_set,
    flux_direct_dora_projection_forward,
    flux_direct_dora_projection_backward,
    flux_direct_oft_projection_forward,
    flux_direct_oft_projection_backward,
    flux_direct_dora_zero_grads,
    flux_direct_dora_scatter_slot_grad,
    flux_direct_dora_grad_norm,
    flux_direct_dora_adamw_step,
    flux_direct_dora_zero_leg_l1,
    flux_direct_dora_trainable_bytes,
    flux_direct_oft_zero_grads,
    flux_direct_oft_scatter_slot_grad,
    flux_direct_oft_grad_norm,
    flux_direct_oft_adamw_step,
    flux_direct_oft_vec_l1,
    flux_direct_oft_trainable_bytes,
    save_flux_direct_dora,
    save_flux_direct_oft,
)


comptime COS_BAR = 0.999999
comptime NREL_BAR = 2.0e-4
comptime DORA_SAVE_PATH = "/tmp/flux_direct_dora_smoke.safetensors"
comptime OFT_SAVE_PATH = "/tmp/flux_direct_oft_smoke.safetensors"


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


def _max_abs(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error("_max_abs: len mismatch")
    var mx = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        var ad = d if d >= 0.0 else -d
        if ad > mx:
            mx = ad
    return mx


def _make_weights(num_double: Int, num_single: Int, D: Int, F: Int, seed: UInt64) raises -> List[List[Float32]]:
    var weights = List[List[Float32]]()
    var s = seed
    for flat in range(flux_direct_total_slots(num_double, num_single)):
        var dims = flux_direct_flat_dims(flat, num_double, D, F)
        weights.append(_randn(dims[0] * dims[1], s, Float32(0.35)))
        s += 1
    return weights^


def _check_prefixes_and_counts() raises:
    if flux_direct_active_slot_count(2, 2, FLUX_LYCORIS_TGT_ATTN) != 22:
        raise Error("flux direct smoke: attn target count mismatch")
    if flux_direct_active_slot_count(2, 2, FLUX_LYCORIS_TGT_ALL) != 34:
        raise Error("flux direct smoke: all target count mismatch")
    if flux_direct_flat_prefix(0, 2) != "lora_transformer_transformer_blocks_0_attn_to_q":
        raise Error("flux direct smoke: double prefix mismatch")
    var sflat = 2 * DBL_SLOTS_PER_BLOCK
    if flux_direct_flat_prefix(sflat + S_PMLP, 2) != "lora_transformer_single_transformer_blocks_0_proj_mlp":
        raise Error("flux direct smoke: single prefix mismatch")


def _check_real_flux_bytes() raises:
    var nd = 19
    var ns = 38
    var D = 3072
    var F = 12288
    var rank = 16
    var oft_block = 4
    var targets = FLUX_LYCORIS_TGT_ALL
    var dense = flux_direct_dense_carrier_bytes(nd, ns, D, F, targets)
    var dora_direct = flux_direct_dora_trainable_bytes_estimate(
        nd, ns, D, F, rank, targets, False,
    )
    var oft_direct = flux_direct_oft_trainable_bytes_estimate(
        nd, ns, D, F, oft_block, targets,
    )
    print("[flux-direct-bytes] slots=", flux_direct_active_slot_count(nd, ns, targets))
    print("  dense_carrier_bytes=", dense, " direct_dora_bytes=", dora_direct, " direct_oft_bytes=", oft_direct)
    if dense <= FLUX_DIRECT_24_GIB:
        raise Error("flux direct smoke: dense carrier unexpectedly fits 24 GiB for all targets")
    if dora_direct >= FLUX_DIRECT_24_GIB:
        raise Error("flux direct smoke: DoRA direct trainable estimate exceeds 24 GiB")
    if oft_direct >= FLUX_DIRECT_24_GIB:
        raise Error("flux direct smoke: OFT direct trainable estimate exceeds 24 GiB")
    var dora_pf = flux_direct_dora_preflight(nd, ns, D, F, rank, targets, FLUX_DIRECT_24_GIB, False)
    var oft_pf = flux_direct_oft_preflight(nd, ns, D, F, oft_block, targets, FLUX_DIRECT_24_GIB)
    if dora_pf != dora_direct or oft_pf != oft_direct:
        raise Error("flux direct smoke: preflight byte mismatch")


def _check_builder_compact_targets() raises:
    var D = 32
    var F = 48
    var nd = 2
    var ns = 2
    var weights = _make_weights(nd, ns, D, F, UInt64(7000))
    var dora = build_flux_direct_dora_set_from_weights(
        weights, nd, ns, D, F, 4, Float32(2.0),
        FLUX_LYCORIS_TGT_ATTN, UInt64(8000), False,
    )
    if len(dora.ad) != flux_direct_active_slot_count(nd, ns, FLUX_LYCORIS_TGT_ATTN):
        raise Error("flux direct smoke: compact DoRA attn target count mismatch")
    if dora.prefix[4] != "lora_transformer_transformer_blocks_0_attn_add_q_proj":
        raise Error("flux direct smoke: compact DoRA txt-q prefix mismatch")
    var oft = build_flux_direct_oft_set(nd, ns, D, F, 4, FLUX_LYCORIS_TGT_ATTN)
    if len(oft.ad) != len(dora.ad):
        raise Error("flux direct smoke: compact OFT attn target count mismatch")
    if oft.prefix[16] != "lora_transformer_single_transformer_blocks_0_attn_to_q":
        raise Error("flux direct smoke: compact OFT single prefix mismatch")


def _check_streaming_projection(ctx: DeviceContext) raises:
    var D = 32
    var F = 48
    var nd = 2
    var ns = 2
    var rank = 4
    var block_size = 4
    var targets = FLUX_LYCORIS_TGT_ALL
    var weights = _make_weights(nd, ns, D, F, UInt64(1000))
    var active_slots = flux_direct_active_slot_count(nd, ns, targets)

    var dora = build_flux_direct_dora_set_from_weights(
        weights, nd, ns, D, F, rank, Float32(2.0), targets, UInt64(2000), False,
    )
    if len(dora.ad) != active_slots:
        raise Error("flux direct smoke: DoRA active slot count mismatch")
    var dense = flux_direct_dense_carrier_bytes(nd, ns, D, F, targets)
    var dbytes = flux_direct_dora_trainable_bytes(dora)
    print("[flux-direct-stream-init] dora_slots=", len(dora.ad), " direct_bytes=", dbytes, " dense_bytes=", dense)
    if dbytes >= dense:
        raise Error("flux direct smoke: DoRA direct state is not below dense carrier")

    var oft = build_flux_direct_oft_set(nd, ns, D, F, block_size, targets)
    if len(oft.ad) != active_slots:
        raise Error("flux direct smoke: OFT active slot count mismatch")
    var obytes = flux_direct_oft_trainable_bytes(oft)
    print("[flux-direct-stream-init] oft_slots=", len(oft.ad), " direct_bytes=", obytes, " dense_bytes=", dense)
    if obytes >= dense:
        raise Error("flux direct smoke: OFT direct state is not below dense carrier")

    var M = 3
    var compact_slot = D_MLP0
    var weight_idx = D_MLP0
    var dims = flux_direct_flat_dims(weight_idx, nd, D, F)
    var x = _randn(M * dims[0], UInt64(3001), Float32(0.8))
    var bias = _randn(dims[1], UInt64(3002), Float32(0.1))
    var d_y = _randn(M * dims[1], UInt64(3003), Float32(0.5))
    var W = weights[weight_idx].copy()

    var yd = flux_direct_dora_projection_forward(dora, compact_slot, x.copy(), W.copy(), bias.copy(), M)
    var yd_ref = dora_forward(x.copy(), W.copy(), dora.ad[compact_slot], M)
    yd_ref = _add_bias(yd_ref^, bias.copy(), M, dims[1])
    _check(String("stream dora rectangular projection+bias"), yd, yd_ref)
    var dg = flux_direct_dora_projection_backward(dora, compact_slot, d_y.copy(), x.copy(), W.copy(), M)
    var dg_ref = dora_backward(d_y.copy(), x.copy(), W.copy(), dora.ad[compact_slot], M)
    _check(String("stream dora rectangular d_A"), dg.d_a, dg_ref.d_a)
    _check(String("stream dora rectangular d_B"), dg.d_b, dg_ref.d_b)
    _check(String("stream dora rectangular d_x"), dg.d_x, dg_ref.d_x)

    var all_dg = flux_direct_dora_zero_grads(dora)
    flux_direct_dora_scatter_slot_grad(all_dg, compact_slot, dg)
    var dn = flux_direct_dora_grad_norm(all_dg)
    print("  dora set_grad_norm=", dn)
    if dn <= 0.0:
        raise Error("flux direct smoke: DoRA set grad norm stayed zero")
    var d_before = flux_direct_dora_zero_leg_l1(dora)
    flux_direct_dora_adamw_step(
        dora, all_dg, 1, Float32(1.0e-3),
        Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.0),
    )
    var d_after = flux_direct_dora_zero_leg_l1(dora)
    print("  dora zero_leg_l1 ", d_before, " -> ", d_after)
    if not (d_before == 0.0 and d_after > 0.0):
        raise Error("flux direct smoke: DoRA direct set did not step")
    var dmods = save_flux_direct_dora(dora, String(DORA_SAVE_PATH), ctx)
    if dmods != active_slots:
        raise Error("flux direct smoke: DoRA save module count mismatch")
    var rb = read_dora_module(dora.prefix[compact_slot], String(DORA_SAVE_PATH), ctx)
    if rb.in_f != dims[0] or rb.out_f != dims[1] or rb.rank != rank:
        raise Error("flux direct smoke: DoRA readback shape mismatch")
    if rb.wd_on_out:
        raise Error("flux direct smoke: DoRA readback lost OneTrainer input-axis magnitude")
    if _max_abs(rb.m, dora.ad[compact_slot].m) > Float32(1.0e-6):
        raise Error("flux direct smoke: DoRA magnitude readback mismatch")

    var yo = flux_direct_oft_projection_forward(oft, compact_slot, x.copy(), W.copy(), bias.copy(), M)
    var yo_ref = oft_ot_forward(
        x.copy(), oft.ad[compact_slot].vec.copy(), W.copy(),
        M, dims[0], dims[1], block_size, dims[0] // block_size,
    )
    yo_ref = _add_bias(yo_ref^, bias.copy(), M, dims[1])
    _check(String("stream oft rectangular projection+bias"), yo, yo_ref)
    var og = flux_direct_oft_projection_backward(oft, compact_slot, d_y.copy(), x.copy(), W.copy(), M)
    var og_ref = oft_ot_backward(
        d_y.copy(), x.copy(), oft.ad[compact_slot].vec.copy(), W.copy(),
        M, dims[0], dims[1], block_size, dims[0] // block_size,
    )
    _check(String("stream oft rectangular d_vec"), og.d_vec, og_ref.d_vec)
    _check(String("stream oft rectangular d_x"), og.d_x, og_ref.d_x)

    var all_og = flux_direct_oft_zero_grads(oft)
    flux_direct_oft_scatter_slot_grad(all_og, compact_slot, og)
    var on = flux_direct_oft_grad_norm(all_og)
    print("  oft set_grad_norm=", on)
    if on <= 0.0:
        raise Error("flux direct smoke: OFT set grad norm stayed zero")
    var o_before = flux_direct_oft_vec_l1(oft)
    flux_direct_oft_adamw_step(
        oft, all_og, 1, Float32(1.0e-3),
        Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.0),
    )
    var o_after = flux_direct_oft_vec_l1(oft)
    print("  oft vec_l1 ", o_before, " -> ", o_after)
    if not (o_before == 0.0 and o_after > 0.0):
        raise Error("flux direct smoke: OFT direct set did not step")
    var omods = save_flux_direct_oft(oft, String(OFT_SAVE_PATH), ctx)
    if omods != active_slots:
        raise Error("flux direct smoke: OFT save module count mismatch")
    _ = SafeTensors.open(String(OFT_SAVE_PATH))


def main() raises:
    var ctx = DeviceContext()
    _check_prefixes_and_counts()
    _check_real_flux_bytes()
    _check_builder_compact_targets()
    _check_streaming_projection(ctx)
    print("ALL GATES PASS -- flux_direct_lycoris_projection_smoke")
