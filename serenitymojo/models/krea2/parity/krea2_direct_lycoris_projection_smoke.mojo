# models/krea2/parity/krea2_direct_lycoris_projection_smoke.mojo
#
# Proves Krea2 direct DoRA/OFT slot metadata, byte preflight, compact targeted
# set construction, projection wrappers, optimizer movement, and save names.
# This is not the live Krea2 GPU block lowering gate.

from std.collections import List, Optional
from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.dora_save import read_dora_module
from serenitymojo.training.dora_adapter import dora_forward, dora_backward
from serenitymojo.training.oft_onetrainer import oft_ot_forward, oft_ot_backward
from serenitymojo.models.krea2.krea2_lokr_stack import (
    KREA2_SLOTS, K2LOKR_TGT_ATTN, K2LOKR_TGT_ALL, krea2_lokr_slot_dims,
)
from serenitymojo.models.krea2.krea2_direct_lycoris_stack import (
    KREA2_DIRECT_24_GIB,
    K2D_WQ, K2D_MLP_UP, K2D_MLP_DOWN,
    krea2_direct_active_slot_count, krea2_direct_slot_prefix,
    empty_krea2_direct_dora_set, empty_krea2_direct_oft_set,
    krea2_direct_dense_carrier_bytes,
    krea2_direct_dora_trainable_bytes_estimate,
    krea2_direct_oft_trainable_bytes_estimate,
    krea2_direct_dora_preflight,
    krea2_direct_oft_preflight,
    build_krea2_direct_dora_set_from_weights,
    krea2_direct_dora_append_block_weights,
    build_krea2_direct_oft_set,
    krea2_direct_oft_append_block,
    krea2_direct_dora_projection_forward,
    krea2_direct_dora_projection_backward,
    krea2_direct_dora_set_to_device,
    krea2_direct_dora_blocks_to_device,
    krea2_direct_dora_projection_forward_resident,
    krea2_direct_dora_projection_backward_resident,
    krea2_direct_dora_projection_forward_device,
    krea2_direct_dora_projection_backward_device,
    krea2_direct_oft_projection_forward,
    krea2_direct_oft_projection_backward,
    krea2_direct_oft_set_to_device,
    krea2_direct_oft_blocks_to_device,
    krea2_direct_oft_projection_forward_resident,
    krea2_direct_oft_projection_backward_resident,
    krea2_direct_oft_projection_forward_device,
    krea2_direct_oft_projection_backward_device,
    krea2_direct_dora_zero_grads,
    krea2_direct_dora_scatter_slot_grad,
    krea2_direct_dora_grad_norm,
    krea2_direct_dora_adamw_step,
    krea2_direct_dora_zero_leg_l1,
    krea2_direct_dora_trainable_bytes,
    krea2_direct_oft_zero_grads,
    krea2_direct_oft_scatter_slot_grad,
    krea2_direct_oft_grad_norm,
    krea2_direct_oft_adamw_step,
    krea2_direct_oft_vec_l1,
    krea2_direct_oft_trainable_bytes,
    save_krea2_direct_dora,
    save_krea2_direct_oft,
)
from serenitymojo.models.krea2.krea2_block import (
    krea2_block_direct_dora_projection_forward,
    krea2_block_direct_dora_projection_backward_dev,
    krea2_block_direct_oft_projection_forward,
    krea2_block_direct_oft_projection_backward_dev,
)


comptime COS_BAR = 0.999999
comptime NREL_BAR = 2.0e-4
comptime DORA_SAVE_PATH = "/tmp/krea2_direct_dora_smoke.safetensors"
comptime OFT_SAVE_PATH = "/tmp/krea2_direct_oft_smoke.safetensors"


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


def _dora_delta_zero_dm_per_input(
    d_y: List[Float32], x: List[Float32], w: List[Float32], m_resident: List[Float32],
    M: Int, in_f: Int, out_f: Int, eps: Float32,
) raises -> List[Float32]:
    if len(d_y) != M * out_f or len(x) != M * in_f or len(w) != out_f * in_f:
        raise Error("test _dora_delta_zero_dm_per_input: shape mismatch")
    if len(m_resident) != in_f:
        raise Error("test _dora_delta_zero_dm_per_input: m shape mismatch")
    var out = List[Float32]()
    for i in range(in_f):
        var acc = Float32(0.0)
        var den = m_resident[i] + eps
        for o in range(out_f):
            var d_wpdora = Float32(0.0)
            for row in range(M):
                d_wpdora += d_y[row * out_f + o] * x[row * in_f + i]
            acc += d_wpdora * w[o * in_f + i] / den
        out.append(acc)
    return out^


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


def _make_weights(
    num_blocks: Int, D: Int, F: Int, qdim: Int, kvdim: Int, seed: UInt64,
) raises -> List[List[Float32]]:
    var weights = List[List[Float32]]()
    var s = seed
    for _bi in range(num_blocks):
        for slot in range(KREA2_SLOTS):
            var dims = krea2_lokr_slot_dims(slot, D, F, qdim, kvdim)
            weights.append(_randn(dims[0] * dims[1], s, Float32(0.35)))
            s += 1
    return weights^


def _check_prefixes_and_counts() raises:
    if krea2_direct_active_slot_count(2, K2LOKR_TGT_ATTN) != 10:
        raise Error("krea2 direct smoke: attn target count mismatch")
    if krea2_direct_active_slot_count(2, K2LOKR_TGT_ALL) != 16:
        raise Error("krea2 direct smoke: all target count mismatch")
    if krea2_direct_slot_prefix(3, K2D_WQ) != "diffusion_model.blocks.3.attn.wq":
        raise Error("krea2 direct smoke: wq prefix mismatch")
    if krea2_direct_slot_prefix(3, K2D_MLP_DOWN) != "diffusion_model.blocks.3.mlp.down":
        raise Error("krea2 direct smoke: mlp.down prefix mismatch")


def _check_real_krea2_bytes() raises:
    var blocks = 28
    var D = 6144
    var F = 16384
    var qdim = 6144
    var kvdim = 1536
    var rank = 16
    var oft_block = 4
    var targets = K2LOKR_TGT_ALL
    var dense = krea2_direct_dense_carrier_bytes(blocks, D, F, qdim, kvdim, targets)
    var dora_direct = krea2_direct_dora_trainable_bytes_estimate(
        blocks, D, F, qdim, kvdim, rank, targets, False,
    )
    var oft_direct = krea2_direct_oft_trainable_bytes_estimate(
        blocks, D, F, qdim, kvdim, oft_block, targets,
    )
    print("[krea2-direct-bytes] slots=", krea2_direct_active_slot_count(blocks, targets))
    print("  dense_carrier_bytes=", dense, " direct_dora_bytes=", dora_direct, " direct_oft_bytes=", oft_direct)
    if dense <= KREA2_DIRECT_24_GIB:
        raise Error("krea2 direct smoke: dense carrier unexpectedly fits 24 GiB for all targets")
    if dora_direct >= KREA2_DIRECT_24_GIB:
        raise Error("krea2 direct smoke: DoRA direct trainable estimate exceeds 24 GiB")
    if oft_direct >= KREA2_DIRECT_24_GIB:
        raise Error("krea2 direct smoke: OFT direct trainable estimate exceeds 24 GiB")
    var dora_pf = krea2_direct_dora_preflight(
        blocks, D, F, qdim, kvdim, rank, targets, KREA2_DIRECT_24_GIB, False,
    )
    var oft_pf = krea2_direct_oft_preflight(
        blocks, D, F, qdim, kvdim, oft_block, targets, KREA2_DIRECT_24_GIB,
    )
    if dora_pf != dora_direct or oft_pf != oft_direct:
        raise Error("krea2 direct smoke: preflight byte mismatch")

    var dora_failed = False
    try:
        _ = krea2_direct_dora_preflight(blocks, D, F, qdim, kvdim, rank, targets, 1024, False)
    except:
        dora_failed = True
    if not dora_failed:
        raise Error("krea2 direct smoke: DoRA tiny-budget preflight did not fail")
    var oft_failed = False
    try:
        _ = krea2_direct_oft_preflight(blocks, D, F, qdim, kvdim, oft_block, targets, 1024)
    except:
        oft_failed = True
    if not oft_failed:
        raise Error("krea2 direct smoke: OFT tiny-budget preflight did not fail")


def _check_streaming_append_and_projection(ctx: DeviceContext) raises:
    var D = 32
    var F = 48
    var qdim = 40
    var kvdim = 16
    var blocks = 2
    var rank = 4
    var block_size = 4
    var targets = K2LOKR_TGT_ALL
    var weights = _make_weights(blocks, D, F, qdim, kvdim, UInt64(1000))
    var active_slots = krea2_direct_active_slot_count(blocks, targets)

    var dora = empty_krea2_direct_dora_set()
    for bi in range(blocks):
        var block_weights = List[List[Float32]]()
        for slot in range(KREA2_SLOTS):
            block_weights.append(weights[bi * KREA2_SLOTS + slot].copy())
        krea2_direct_dora_append_block_weights(
            dora, bi, block_weights^, D, F, qdim, kvdim, rank, Float32(2.0),
            targets, UInt64(2000 + bi * KREA2_SLOTS), False,
        )
    if len(dora.ad) != active_slots:
        raise Error("krea2 direct smoke: streaming DoRA active slot count mismatch")
    if dora.prefix[KREA2_SLOTS + K2D_WQ] != "diffusion_model.blocks.1.attn.wq":
        raise Error("krea2 direct smoke: streaming DoRA prefix mismatch")
    var dense = krea2_direct_dense_carrier_bytes(blocks, D, F, qdim, kvdim, targets)
    var dbytes = krea2_direct_dora_trainable_bytes(dora)
    print("[krea2-direct-stream-init] dora_slots=", len(dora.ad), " direct_bytes=", dbytes, " dense_bytes=", dense)
    if dbytes >= dense:
        raise Error("krea2 direct smoke: streaming DoRA direct state is not below dense carrier")

    var oft = empty_krea2_direct_oft_set()
    for bi in range(blocks):
        krea2_direct_oft_append_block(oft, bi, D, F, qdim, kvdim, block_size, targets)
    if len(oft.ad) != active_slots:
        raise Error("krea2 direct smoke: streaming OFT active slot count mismatch")
    if oft.prefix[KREA2_SLOTS + K2D_WQ] != "diffusion_model.blocks.1.attn.wq":
        raise Error("krea2 direct smoke: streaming OFT prefix mismatch")
    var obytes = krea2_direct_oft_trainable_bytes(oft)
    print("[krea2-direct-stream-init] oft_slots=", len(oft.ad), " direct_bytes=", obytes, " dense_bytes=", dense)
    if obytes >= dense:
        raise Error("krea2 direct smoke: streaming OFT direct state is not below dense carrier")

    var M = 3
    var compact_slot = K2D_MLP_UP
    var weight_idx = K2D_MLP_UP
    var dims = krea2_lokr_slot_dims(K2D_MLP_UP, D, F, qdim, kvdim)
    var x = _randn(M * dims[0], UInt64(3001), Float32(0.8))
    var bias = _randn(dims[1], UInt64(3002), Float32(0.1))
    var d_y = _randn(M * dims[1], UInt64(3003), Float32(0.5))
    var W = weights[weight_idx].copy()

    var yd = krea2_direct_dora_projection_forward(dora, compact_slot, x.copy(), W.copy(), bias.copy(), M)
    var yd_ref_nobias = dora_forward(x.copy(), W.copy(), dora.ad[compact_slot], M)
    var yd_ref = _add_bias(yd_ref_nobias.copy(), bias.copy(), M, dims[1])
    _check(String("stream dora rectangular projection+bias"), yd, yd_ref)
    var dg = krea2_direct_dora_projection_backward(dora, compact_slot, d_y.copy(), x.copy(), W.copy(), M)
    var dg_ref = dora_backward(d_y.copy(), x.copy(), W.copy(), dora.ad[compact_slot], M)
    _check(String("stream dora rectangular d_A"), dg.d_a, dg_ref.d_a)
    _check(String("stream dora rectangular d_B"), dg.d_b, dg_ref.d_b)
    _check(String("stream dora rectangular d_x"), dg.d_x, dg_ref.d_x)

    var dora_dev_slots = krea2_direct_dora_set_to_device(dora, ctx)
    if len(dora_dev_slots.slots) != active_slots:
        raise Error("krea2 direct smoke: resident DoRA slot count mismatch")
    if dora_dev_slots.slots[compact_slot].m[].dtype() != STDtype.BF16:
        raise Error("krea2 direct smoke: resident DoRA magnitude must be BF16")
    var dora_dev_ref = dora.ad[compact_slot].copy()
    dora_dev_ref.m = dora_dev_slots.slots[compact_slot].m[].to_host(ctx)
    var dm_dev_ref = _dora_delta_zero_dm_per_input(
        d_y.copy(), x.copy(), W.copy(), dora_dev_ref.m.copy(),
        M, dims[0], dims[1], dora_dev_ref.eps,
    )

    var xd_dev = Tensor.from_host(x.copy(), [M, dims[0]], STDtype.F32, ctx)
    var wd_dev = Tensor.from_host(W.copy(), [dims[1], dims[0]], STDtype.F32, ctx)
    var bd_dev = Tensor.from_host(bias.copy(), [dims[1]], STDtype.F32, ctx)
    var yd_dev = krea2_direct_dora_projection_forward_device(
        dora, compact_slot, xd_dev, wd_dev, M, ctx, Optional[Tensor](bd_dev^),
    )
    _check(String("stream dora device projection+bias"), yd_dev.to_host(ctx), yd_ref)

    var xd_dev_b = Tensor.from_host(x.copy(), [M, dims[0]], STDtype.F32, ctx)
    var wd_dev_b = Tensor.from_host(W.copy(), [dims[1], dims[0]], STDtype.F32, ctx)
    var dy_dora_dev = Tensor.from_host(d_y.copy(), [M, dims[1]], STDtype.F32, ctx)
    var dg_dev = krea2_direct_dora_projection_backward_device(
        dora, compact_slot, dy_dora_dev, xd_dev_b, wd_dev_b, M, ctx,
    )
    _check(String("stream dora device d_A"), dg_dev.d_a.to_host(ctx), dg_ref.d_a)
    _check(String("stream dora device d_B"), dg_dev.d_b.to_host(ctx), dg_ref.d_b)
    _check(String("stream dora device d_m"), dg_dev.d_m.to_host(ctx), dm_dev_ref)
    _check(String("stream dora device d_x"), dg_dev.d_x.to_host(ctx), dg_ref.d_x)

    var xdr = Tensor.from_host(x.copy(), [M, dims[0]], STDtype.F32, ctx)
    var wdr = Tensor.from_host(W.copy(), [dims[1], dims[0]], STDtype.F32, ctx)
    var bdr = Tensor.from_host(bias.copy(), [dims[1]], STDtype.F32, ctx)
    var yd_res = krea2_direct_dora_projection_forward_resident(
        dora_dev_slots.slots[compact_slot], xdr, wdr, M, ctx, Optional[Tensor](bdr^),
    )
    _check(String("stream dora resident projection+bias"), yd_res.to_host(ctx), yd_ref)
    var xdr_b = Tensor.from_host(x.copy(), [M, dims[0]], STDtype.F32, ctx)
    var wdr_b = Tensor.from_host(W.copy(), [dims[1], dims[0]], STDtype.F32, ctx)
    var dyr = Tensor.from_host(d_y.copy(), [M, dims[1]], STDtype.F32, ctx)
    var dg_res = krea2_direct_dora_projection_backward_resident(
        dora_dev_slots.slots[compact_slot], dyr, xdr_b, wdr_b, M, ctx,
    )
    _check(String("stream dora resident d_A"), dg_res.d_a.to_host(ctx), dg_ref.d_a)
    _check(String("stream dora resident d_B"), dg_res.d_b.to_host(ctx), dg_ref.d_b)
    _check(String("stream dora resident d_m"), dg_res.d_m.to_host(ctx), dm_dev_ref)
    _check(String("stream dora resident d_x"), dg_res.d_x.to_host(ctx), dg_ref.d_x)

    var dora_blocks = krea2_direct_dora_blocks_to_device(dora, blocks, targets, ctx)
    if len(dora_blocks.blocks) != blocks:
        raise Error("krea2 direct smoke: resident DoRA block count mismatch")
    if not dora_blocks.blocks[0].mlp_up_w:
        raise Error("krea2 direct smoke: resident DoRA block lost mlp_up slot")
    var xdb = Tensor.from_host(x.copy(), [M, dims[0]], STDtype.F32, ctx)
    var wdb = Tensor.from_host(W.copy(), [dims[1], dims[0]], STDtype.F32, ctx)
    var yd_blk = krea2_block_direct_dora_projection_forward(
        xdb, wdb, dora_blocks.blocks[0].mlp_up_w, M, ctx,
    )
    _check(String("stream dora block-hook projection"), yd_blk.to_host(ctx), yd_ref_nobias)
    var xdb_b = Tensor.from_host(x.copy(), [M, dims[0]], STDtype.F32, ctx)
    var wdb_b = Tensor.from_host(W.copy(), [dims[1], dims[0]], STDtype.F32, ctx)
    var dyb = Tensor.from_host(d_y.copy(), [M, dims[1]], STDtype.F32, ctx)
    var dg_blk = krea2_block_direct_dora_projection_backward_dev(
        dyb, xdb_b, wdb_b, dora_blocks.blocks[0].mlp_up_w,
        M, dims[0], dims[1], ctx,
    )
    if not dg_blk.dora.d_a or not dg_blk.dora.d_b or not dg_blk.dora.d_m:
        raise Error("krea2 direct smoke: resident DoRA block-hook missing grads")
    _check(String("stream dora block-hook d_A"), dg_blk.dora.d_a.value()[].to_host(ctx), dg_ref.d_a)
    _check(String("stream dora block-hook d_B"), dg_blk.dora.d_b.value()[].to_host(ctx), dg_ref.d_b)
    _check(String("stream dora block-hook d_m"), dg_blk.dora.d_m.value()[].to_host(ctx), dm_dev_ref)
    _check(String("stream dora block-hook d_x"), dg_blk.d_x.to_host(ctx), dg_ref.d_x)

    var all_dg = krea2_direct_dora_zero_grads(dora)
    krea2_direct_dora_scatter_slot_grad(all_dg, compact_slot, dg)
    var dn = krea2_direct_dora_grad_norm(all_dg)
    print("  dora set_grad_norm=", dn)
    if dn <= 0.0:
        raise Error("krea2 direct smoke: DoRA set grad norm stayed zero")
    var d_before = krea2_direct_dora_zero_leg_l1(dora)
    krea2_direct_dora_adamw_step(
        dora, all_dg, 1, Float32(1.0e-3),
        Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.0),
    )
    var d_after = krea2_direct_dora_zero_leg_l1(dora)
    print("  dora zero_leg_l1 ", d_before, " -> ", d_after)
    if not (d_before == 0.0 and d_after > 0.0):
        raise Error("krea2 direct smoke: DoRA direct set did not step")
    var dmods = save_krea2_direct_dora(dora, String(DORA_SAVE_PATH), ctx)
    if dmods != active_slots:
        raise Error("krea2 direct smoke: DoRA save module count mismatch")
    var rb = read_dora_module(dora.prefix[compact_slot], String(DORA_SAVE_PATH), ctx)
    if rb.in_f != dims[0] or rb.out_f != dims[1] or rb.rank != rank:
        raise Error("krea2 direct smoke: DoRA readback shape mismatch")
    if rb.wd_on_out:
        raise Error("krea2 direct smoke: DoRA readback lost OneTrainer input-axis magnitude")
    if _max_abs(rb.m, dora.ad[compact_slot].m) > Float32(1.0e-6):
        raise Error("krea2 direct smoke: DoRA magnitude readback mismatch")
    print("  dora save/reopen modules=", dmods, " wd_on_out=", rb.wd_on_out)

    var yo = krea2_direct_oft_projection_forward(oft, compact_slot, x.copy(), W.copy(), bias.copy(), M)
    var yo_ref_nobias = oft_ot_forward(
        x.copy(), oft.ad[compact_slot].vec.copy(), W.copy(),
        M, dims[0], dims[1], block_size, dims[0] // block_size,
    )
    var yo_ref = _add_bias(yo_ref_nobias.copy(), bias.copy(), M, dims[1])
    _check(String("stream oft rectangular projection+bias"), yo, yo_ref)
    var og = krea2_direct_oft_projection_backward(oft, compact_slot, d_y.copy(), x.copy(), W.copy(), M)
    var og_ref = oft_ot_backward(
        d_y.copy(), x.copy(), oft.ad[compact_slot].vec.copy(), W.copy(),
        M, dims[0], dims[1], block_size, dims[0] // block_size,
    )
    _check(String("stream oft rectangular d_vec"), og.d_vec, og_ref.d_vec)
    _check(String("stream oft rectangular d_x"), og.d_x, og_ref.d_x)

    var oft_dev_slots = krea2_direct_oft_set_to_device(oft, ctx)
    if len(oft_dev_slots.slots) != active_slots:
        raise Error("krea2 direct smoke: resident OFT slot count mismatch")
    if oft_dev_slots.slots[compact_slot].vec[].dtype() != STDtype.BF16:
        raise Error("krea2 direct smoke: resident OFT vec must be BF16")
    var oft_vec_dev_ref = oft_dev_slots.slots[compact_slot].vec[].to_host(ctx)
    var yo_dev_ref_nobias = oft_ot_forward(
        x.copy(), oft_vec_dev_ref.copy(), W.copy(),
        M, dims[0], dims[1], block_size, dims[0] // block_size,
    )
    var yo_dev_ref = _add_bias(yo_dev_ref_nobias.copy(), bias.copy(), M, dims[1])
    var og_dev_ref = oft_ot_backward(
        d_y.copy(), x.copy(), oft_vec_dev_ref.copy(), W.copy(),
        M, dims[0], dims[1], block_size, dims[0] // block_size,
    )

    var x_dev = Tensor.from_host(x.copy(), [M, dims[0]], STDtype.F32, ctx)
    var w_dev = Tensor.from_host(W.copy(), [dims[1], dims[0]], STDtype.F32, ctx)
    var bias_dev = Tensor.from_host(bias.copy(), [dims[1]], STDtype.F32, ctx)
    var yo_dev = krea2_direct_oft_projection_forward_device(
        oft, compact_slot, x_dev, w_dev, M, ctx, Optional[Tensor](bias_dev^),
    )
    _check(String("stream oft device projection+bias"), yo_dev.to_host(ctx), yo_dev_ref)

    var x_dev_b = Tensor.from_host(x.copy(), [M, dims[0]], STDtype.F32, ctx)
    var w_dev_b = Tensor.from_host(W.copy(), [dims[1], dims[0]], STDtype.F32, ctx)
    var dy_dev = Tensor.from_host(d_y.copy(), [M, dims[1]], STDtype.F32, ctx)
    var og_dev = krea2_direct_oft_projection_backward_device(
        oft, compact_slot, dy_dev, x_dev_b, w_dev_b, M, ctx,
    )
    _check(String("stream oft device d_vec"), og_dev.d_vec.to_host(ctx), og_dev_ref.d_vec)
    _check(String("stream oft device d_x"), og_dev.d_x.to_host(ctx), og_dev_ref.d_x)

    var xor = Tensor.from_host(x.copy(), [M, dims[0]], STDtype.F32, ctx)
    var wor = Tensor.from_host(W.copy(), [dims[1], dims[0]], STDtype.F32, ctx)
    var bor = Tensor.from_host(bias.copy(), [dims[1]], STDtype.F32, ctx)
    var yo_res = krea2_direct_oft_projection_forward_resident(
        oft_dev_slots.slots[compact_slot], xor, wor, M, ctx, Optional[Tensor](bor^),
    )
    _check(String("stream oft resident projection+bias"), yo_res.to_host(ctx), yo_dev_ref)
    var xor_b = Tensor.from_host(x.copy(), [M, dims[0]], STDtype.F32, ctx)
    var wor_b = Tensor.from_host(W.copy(), [dims[1], dims[0]], STDtype.F32, ctx)
    var dyor = Tensor.from_host(d_y.copy(), [M, dims[1]], STDtype.F32, ctx)
    var og_res = krea2_direct_oft_projection_backward_resident(
        oft_dev_slots.slots[compact_slot], dyor, xor_b, wor_b, M, ctx,
    )
    _check(String("stream oft resident d_vec"), og_res.d_vec.to_host(ctx), og_dev_ref.d_vec)
    _check(String("stream oft resident d_x"), og_res.d_x.to_host(ctx), og_dev_ref.d_x)

    var oft_blocks = krea2_direct_oft_blocks_to_device(oft, blocks, targets, ctx)
    if len(oft_blocks.blocks) != blocks:
        raise Error("krea2 direct smoke: resident OFT block count mismatch")
    if not oft_blocks.blocks[0].mlp_up_w:
        raise Error("krea2 direct smoke: resident OFT block lost mlp_up slot")
    var xob = Tensor.from_host(x.copy(), [M, dims[0]], STDtype.F32, ctx)
    var wob = Tensor.from_host(W.copy(), [dims[1], dims[0]], STDtype.F32, ctx)
    var yo_blk = krea2_block_direct_oft_projection_forward(
        xob, wob, oft_blocks.blocks[0].mlp_up_w, M, ctx,
    )
    _check(String("stream oft block-hook projection"), yo_blk.to_host(ctx), yo_dev_ref_nobias)
    var xob_b = Tensor.from_host(x.copy(), [M, dims[0]], STDtype.F32, ctx)
    var wob_b = Tensor.from_host(W.copy(), [dims[1], dims[0]], STDtype.F32, ctx)
    var dyob = Tensor.from_host(d_y.copy(), [M, dims[1]], STDtype.F32, ctx)
    var og_blk = krea2_block_direct_oft_projection_backward_dev(
        dyob, xob_b, wob_b, oft_blocks.blocks[0].mlp_up_w,
        M, dims[0], dims[1], ctx,
    )
    if not og_blk.oft.d_vec:
        raise Error("krea2 direct smoke: resident OFT block-hook missing grad")
    _check(String("stream oft block-hook d_vec"), og_blk.oft.d_vec.value()[].to_host(ctx), og_dev_ref.d_vec)
    _check(String("stream oft block-hook d_x"), og_blk.d_x.to_host(ctx), og_dev_ref.d_x)

    var all_og = krea2_direct_oft_zero_grads(oft)
    krea2_direct_oft_scatter_slot_grad(all_og, compact_slot, og)
    var on = krea2_direct_oft_grad_norm(all_og)
    print("  oft set_grad_norm=", on)
    if on <= 0.0:
        raise Error("krea2 direct smoke: OFT set grad norm stayed zero")
    var o_before = krea2_direct_oft_vec_l1(oft)
    krea2_direct_oft_adamw_step(
        oft, all_og, 1, Float32(1.0e-3),
        Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.0),
    )
    var o_after = krea2_direct_oft_vec_l1(oft)
    print("  oft vec_l1 ", o_before, " -> ", o_after)
    if not (o_before == 0.0 and o_after > 0.0):
        raise Error("krea2 direct smoke: OFT direct set did not step")
    var omods = save_krea2_direct_oft(oft, String(OFT_SAVE_PATH), ctx)
    if omods != active_slots:
        raise Error("krea2 direct smoke: OFT save module count mismatch")
    _ = SafeTensors.open(String(OFT_SAVE_PATH))


def _check_builder_compact_targets() raises:
    var D = 32
    var F = 48
    var qdim = 40
    var kvdim = 16
    var blocks = 2
    var weights = _make_weights(blocks, D, F, qdim, kvdim, UInt64(7000))
    var dora = build_krea2_direct_dora_set_from_weights(
        weights, blocks, D, F, qdim, kvdim, 4, Float32(2.0),
        K2LOKR_TGT_ATTN, UInt64(8000), False,
    )
    if len(dora.ad) != krea2_direct_active_slot_count(blocks, K2LOKR_TGT_ATTN):
        raise Error("krea2 direct smoke: compact DoRA attn target count mismatch")
    if dora.prefix[4] != "diffusion_model.blocks.0.attn.wo":
        raise Error("krea2 direct smoke: compact DoRA attn prefix mismatch")
    var oft = build_krea2_direct_oft_set(
        blocks, D, F, qdim, kvdim, 4, K2LOKR_TGT_ATTN,
    )
    if len(oft.ad) != len(dora.ad):
        raise Error("krea2 direct smoke: compact OFT attn target count mismatch")
    if oft.prefix[5] != "diffusion_model.blocks.1.attn.wq":
        raise Error("krea2 direct smoke: compact OFT attn prefix mismatch")


def main() raises:
    var ctx = DeviceContext()
    _check_prefixes_and_counts()
    _check_real_krea2_bytes()
    _check_builder_compact_targets()
    _check_streaming_append_and_projection(ctx)
    print("ALL GATES PASS -- krea2_direct_lycoris_projection_smoke")
